"""Track-2 submission (D-040): stage 1 = mean of the 5 b0 fold models, then R2 refinement of the stage-1 candidates
that overlap a b0 instance ("R2b0"). Every step mirrors the validated code path (scripts/nested_r2b0_eval.py R2nb0):

  1 stage 1: each b0 model -> full-res probability (seg.logits_to_full_prob; bf16 autocast on GPU) rounded to float16
    (train_seg.validate stored it so); float32 mean over the models rounded to float16 (nested_stage1 ens).
  2 existence: B = seg.prob_to_instances(p16 -> float32, PostCfg(0.5, 500)).
  3 prob_u8 = rint(p16 * 255); candidates = 8-connected components of prob_u8 / 255 > CAND_THR with >= CAND_MIN px,
    numbered in component order (refine._build_one); a candidate is kept iff it overlaps B's union by >= 1 px.
  4 R2: windows refine.windows_for_box(bbox); inputs refine.make_input('r2', img, prob_u8, cand, j, y, x); sigmoid of
    the model output (bf16 autocast on GPU); per-candidate mean over its windows, rounded to float16
    (train_refine.predict_val).
  5 instances = refine.assemble(kept, thr 0.5, min_area 300) -> COCO RLE, ids <stem>_<k> (0-based), every mask
    round-tripped and checked disjoint.
  6 (test mode) cleanup of multi-part instances, never dropping the largest part: parts < 20 px (D-051), then parts
    < 5 % of the instance area (D-058). --min_part 0 --island_frac 0 reproduces the D-041 CSV.

  test  : CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/predict_r2b0.py test --out outputs/submissions/r2b0_5fold.csv
  check : CUDA_VISIBLE_DEVICES="" taskset -c 0-3 .venv/bin/python scripts/predict_r2b0.py check --fold 2 --n 8
          (a) assembly from the STORED validation stage 1 + stored R2n probabilities == nested_r2b0_eval R2nb0 (exact);
          (b) R2n_k recomputed here vs stored refined probabilities; (c) stage 1 recomputed from the 4 inner models vs
          the stored b0n_ens mean. (b)/(c) differ only numerically on CPU (fp32 here, bf16 GPU in validation).
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402

from filseg import refine as RF  # noqa: E402
from filseg import rle as R  # noqa: E402
from filseg import seg  # noqa: E402

EXPS = ROOT / "outputs" / "exps"
B0_THR, B0_MIN, R2_THR, R2_MIN = 0.5, 500, 0.5, 300
MIN_PART = 20      # D-051: in a multi-part instance drop 8-connected parts < 20 px (never the largest part)
ISLAND_FRAC = 0.05 # D-058: then drop remaining parts < 5 % of the instance area (MAGFiLO QA rule; never the largest)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["test", "check"])
    p.add_argument("--b0_exp", default="b0_r101_s1024")
    p.add_argument("--r2", default=str(EXPS / "R2_final_context" / "all" / "last.pt"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "submissions" / "r2b0_5fold_t05_a500_r2a300.csv"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--fold", type=int, default=2, help="check: outer fold")
    p.add_argument("--n", type=int, default=8, help="check: number of validation files")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--min_part", type=int, default=MIN_PART, help="fragment cleanup (D-051); 0 = off (D-041 CSV)")
    p.add_argument("--island_frac", type=float, default=ISLAND_FRAC, help="island cleanup (D-058); 0 = off (D-051 CSV)")
    return p.parse_args()


# ----------------------------------------------------------------------------- models
def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_b0(path, dev):
    import torch
    m = seg.build_model("resnet101", pretrained=None)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(dev).eval()


def load_r2(path, dev):
    """R2 = ResNet-101 U-Net with a 10-channel first conv (refine.build_model without the ImageNet download)."""
    import torch
    m = seg.build_model("resnet101", pretrained=None)
    old = m.encoder.conv1
    m.encoder.conv1 = torch.nn.Conv2d(RF.VARIANT_CH["r2"], old.out_channels, old.kernel_size, old.stride, old.padding,
                                      bias=False)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(dev).eval()


def autocast(dev):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")


# ----------------------------------------------------------------------------- pipeline
def stage1(models, img_u8, dev):
    """Steps 1: float16 mean of the per-model float16 probabilities (validation rounding)."""
    import torch
    x = torch.from_numpy(seg.to_model_space(img_u8, 1024)[None]).to(dev)
    acc = None
    with torch.no_grad():
        for m in models:
            with autocast(dev):
                y = m(seg.normalize(x)[:, None]).float()
            p = seg.logits_to_full_prob(y[:, 0])[0].astype(np.float16).astype(np.float32)
            acc = p if acc is None else acc + p
    return (acc / len(models)).astype(np.float16)


def candidates(p16):
    """Steps 2-3: prob_u8, candidate map (refine._build_one numbering), boxes, kept ids (overlap with B)."""
    prob_u8 = np.clip(np.rint(p16.astype(np.float32) * 255), 0, 255).astype(np.uint8)
    p = prob_u8.astype(np.float32) / 255.0
    n, lab, stats = cv2.connectedComponentsWithStats((p > RF.CAND_THR).astype(np.uint8), connectivity=8)[:3]
    keep_cc = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= RF.CAND_MIN]
    lut = np.zeros(n, np.uint16)
    for j, k in enumerate(keep_cc, 1):
        lut[k] = j
    cmap = lut[lab]
    boxes = {}
    for j, k in enumerate(keep_cc, 1):
        x, y, w, h, a = (int(v) for v in stats[k])
        boxes[j] = (y, x, y + h, x + w)
    B = seg.prob_to_instances(p16.astype(np.float32), seg.PostCfg(B0_THR, B0_MIN))
    inb = np.bincount(cmap[B.union()], minlength=len(keep_cc) + 1)
    kept = [j for j in range(1, len(keep_cc) + 1) if inb[j] > 0]
    return prob_u8, cmap, boxes, kept, B


def refine_probs(r2, img_u8, prob_u8, cmap, boxes, kept, dev, bs=16):
    """Step 4: {cid: (y0, x0, float16 mean prob over the candidate's windows)} for the kept candidates."""
    import torch
    items = [(j, y, x) for j in kept for (y, x) in RF.windows_for_box(*boxes[j])]
    acc = {}
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            xb = torch.from_numpy(np.stack([RF.make_input("r2", img_u8, prob_u8, cmap, j, y, x) for j, y, x in chunk]))
            with autocast(dev):
                pb = torch.sigmoid(r2(xb.to(dev)).float())[:, 0].cpu().numpy()
            for p, (j, y, x) in zip(pb, chunk):
                acc.setdefault(j, []).append((y, x, p))
    out = {}
    for j, lst in acc.items():
        y0 = min(y for y, _, _ in lst); x0 = min(x for _, x, _ in lst)
        y1 = max(y for y, _, _ in lst) + RF.WIN; x1 = max(x for _, x, _ in lst) + RF.WIN
        s = np.zeros((y1 - y0, x1 - x0), np.float32); c = np.zeros_like(s)
        for y, x, p in lst:
            s[y - y0:y - y0 + RF.WIN, x - x0:x - x0 + RF.WIN] += p
            c[y - y0:y - y0 + RF.WIN, x - x0:x - x0 + RF.WIN] += 1
        out[j] = (y0, x0, (s / np.maximum(c, 1)).astype(np.float16))
    return out


def _drop_parts(mask, thr_fn):
    n, lab, st = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)[:3]
    if n <= 2:
        return mask
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= thr_fn(areas)
    keep[1 + int(np.argmax(areas))] = True
    return keep[lab]


def drop_small_parts(mask, min_part=MIN_PART, island_frac=ISLAND_FRAC):
    """Cleanup of multi-part instances (the largest part is never dropped):
    step 1 (D-051) drop 8-connected parts < min_part px; step 2 (D-058) drop parts < island_frac of the remaining
    instance area. min_part = 0 and island_frac = 0 reproduce the D-041 CSV; island_frac = 0 the D-051 CSV."""
    if min_part:
        mask = _drop_parts(mask, lambda a: min_part)
    if island_frac:
        mask = _drop_parts(mask, lambda a: island_frac * a.sum())
    return mask


def assemble(cp16, kept):
    """Step 5 (float16 refined probabilities -> float32, as the evaluation reads them)."""
    return RF.assemble({j: (y0, x0, p.astype(np.float32)) for j, (y0, x0, p) in cp16.items() if j in kept},
                       thr=R2_THR, min_area=R2_MIN)


# ----------------------------------------------------------------------------- modes
def cmd_test(a):
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = device()
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("FILSEG_THREADS", "4")))
    b0 = [load_b0(EXPS / a.b0_exp / f"fold{k}" / "last.pt", dev) for k in range(5)]
    r2 = load_r2(a.r2, dev)
    files = sorted(p.name for p in seg.D.TEST_IMG_DIR.glob("*.jpeg"))
    if a.limit:
        files = files[:a.limit]
    rows, stats, t0 = [], [], time.time()
    for i, f in enumerate(files):
        stem = Path(f).stem
        img = seg.D.read_image(seg.D.test_image_path(f))
        p16 = stage1(b0, img, dev)
        prob_u8, cmap, boxes, kept, B = candidates(p16)
        inst = assemble(refine_probs(r2, img, prob_u8, cmap, boxes, kept, dev), kept)
        cover = np.zeros((seg.H, seg.W), bool)
        for k in range(len(inst)):
            m = drop_small_parts(inst.full_mask(k), a.min_part, a.island_frac)
            assert not (cover & m).any(), f"overlap {stem}_{k}"
            cover |= m
            counts = R.encode(m)
            assert np.array_equal(R.decode(counts).astype(bool), m), f"RLE roundtrip {stem}_{k}"
            rows.append((R.make_filament_id(stem, k), counts))
        stats.append(dict(stem=stem, n_b0=len(B), n_cand=len(boxes), n_kept=len(kept), n_inst=len(inst)))
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(files)} images, {len(rows)} filaments, {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = R.write_submission(rows, out)
    df = R.read_submission(out)
    assert len(df) == n and df[R.DEFAULT_FORMAT.id_col].is_unique
    for fid, counts in zip(df[R.DEFAULT_FORMAT.id_col], df[R.DEFAULT_FORMAT.rle_col]):
        assert R.decode(counts).sum() > 0, fid
    meta = dict(pipeline="R2b0 (D-040): 5-fold b0 mean -> b0-confirmed candidates -> R2_final -> assemble a300",
                b0_exp=a.b0_exp, r2=a.r2, device=str(dev), n_images=len(files), n_rows=n,
                n_images_with_pred=int(df["stem"].nunique()), seconds=round(time.time() - t0, 1),
                mean_kept_candidates=float(np.mean([s["n_kept"] for s in stats])), path=str(out))
    json.dump(meta, open(str(out) + ".meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


def cmd_check(a):
    import torch
    import nested_r2b0_eval as NE
    torch.set_num_threads(int(os.environ.get("FILSEG_THREADS", "4")))
    dev = device()
    k = a.fold
    files = (EXPS / "b0n_ens" / f"fold{k}" / "val_files.txt").read_text().split()[:a.n]
    ens = np.load(EXPS / "b0n_ens" / f"fold{k}" / "val_probs_last.npz")
    z = np.load(EXPS / "R2n_context" / f"fold{k}" / "val_cand_probs.npz")
    by = {}
    for key in z.files:
        st, j, y0, x0 = key.split("|")
        by.setdefault(st, {})[int(j)] = (int(y0), int(x0), z[key])
    cfg = dict(ens_exp="b0n_ens", r2_exp="R2n_context", b0_exp="b0_r101_s1024",
               cache_prefix=str(ROOT / "outputs" / "cache" / "refine_nested_o"))
    inner = [load_b0(EXPS / "b0n_r101_s1024" / f"hold{min(k, j)}{max(k, j)}" / "last.pt", dev) for j in range(5) if j != k]
    r2 = load_r2(EXPS / "R2n_context" / f"fold{k}" / "last.pt", dev)
    res = []
    for f in files:
        stem = f[:-5]
        img = seg.D.read_image(seg.D.train_image_path(f))
        # (a) exact path equivalence from the stored inputs
        p16 = ens[stem]
        prob_u8, cmap, boxes, kept, B = candidates(p16)
        mine = assemble(by.get(stem, {}), kept)
        ref = NE.instances(stem, k, cfg)[0]["R2nb0"]
        exact = len(mine) == len(ref) and np.array_equal(mine.union(), ref.union()) and \
            sorted(mine.areas.tolist()) == sorted(ref.areas.tolist())
        # (b) R2 recomputed here vs stored refined probabilities (same stage 1)
        cp = refine_probs(r2, img, prob_u8, cmap, boxes, kept, dev)
        dmax = max((float(np.abs(cp[j][2].astype(np.float32) - by[stem][j][2].astype(np.float32)).max())
                    for j in kept if j in cp and j in by.get(stem, {})), default=0.0)
        mine_b = assemble(cp, kept)
        iou_b = _union_iou(mine_b, ref)
        # (c) stage 1 recomputed from the 4 inner models vs the stored mean
        p16_c = stage1(inner, img, dev)
        dp = float(np.abs(p16_c.astype(np.float32) - p16.astype(np.float32)).max())
        flips = int(((p16_c.astype(np.float32) > 0.5) != (p16.astype(np.float32) > 0.5)).sum())
        res.append(dict(stem=stem, n_kept=len(kept), n_inst=len(ref), a_exact=bool(exact), b_max_abs_dprob=round(dmax, 4),
                        b_union_iou=round(iou_b, 4), c_max_abs_dprob=round(dp, 4), c_pixels_flipped_at_0_5=flips))
        print(res[-1], flush=True)
    out = ROOT / "outputs" / "nested" / f"predict_r2b0_check_fold{k}.json"
    json.dump(dict(device=str(dev), fold=k, files=res, a_all_exact=all(r["a_exact"] for r in res)), open(out, "w"), indent=1)
    print("a_all_exact", all(r["a_exact"] for r in res), "->", out)


def _union_iou(A, B):
    a, b = A.union(), B.union()
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


if __name__ == "__main__":
    a = parse()
    {"test": cmd_test, "check": cmd_check}[a.mode](a)
