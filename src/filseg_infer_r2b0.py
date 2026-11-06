"""Self-contained inference of the final pipeline: 5-fold stage 1 -> existence gate -> R2_final refinement of the
confirmed candidates -> assembly -> cleanup -> SE2 (src/filseg_infer_se.py).

Dependencies: torch, torchvision, numpy, opencv-python (plus src/filseg_infer.py from this folder).
No segmentation_models_pytorch, no pycocotools, no scipy.

The functions below are line-by-line ports of the validated code (src/filseg/refine.py: windows_for_box,
make_input('r2'), orientation, _ctx, assemble; scripts/predict_r2b0.py: stage1, candidates, refine_probs; its `check`
command proved the port identical to the nested-validation code path); constants are identical.

  1 stage 1: each of the 5 fold models -> full-resolution probability (filseg_infer: crop [120:1928]^2 -> 1024^2 ->
    U-Net -> logits upsampled bilinearly to 1808^2 -> sigmoid -> zero 2048^2 canvas), each rounded to float16;
    float32 mean over the 5 models, rounded to float16 (= p16).
  2 existence gate: B = 8-connected components of p16 > 0.5 with >= 500 px.
  3 prob_u8 = rint(p16 * 255); candidates = 8-connected components of prob_u8 / 255 > 0.3 with >= 20 px, numbered
    1, 2, ... in component-label order; a candidate is kept iff it overlaps B's union by >= 1 px.
  4 R2_final (U-Net ResNet-101, 10 input channels) refines every kept candidate on native 384x384 windows covering
    its bounding box (+32 px margin); the 10 channels are image, this candidate, stage-1 probability, other
    candidates, orientation cos2t / sin2t of the probability, and a 4x wider context (image, probability, this
    candidate, other candidates) downsampled to 384; the per-candidate mean of sigmoid outputs over its windows is
    rounded to float16.
  5 assembly (refine.assemble): per-candidate masks p > 0.5; candidates whose masks overlap with IoU > 0.5 are merged;
    pixel conflicts go to the higher probability; instances < 300 px are dropped -> disjoint instances.
  6 cleanup: in a multi-part instance, detached parts < 20 px, then parts < 5 % of the instance, are removed (the
    largest part is always kept).
  7 SE2 (filseg_infer_se): the small fragments that step 2 discards are redrawn by R2_final, scored, and added iff
    the score exceeds 0.12744 and they overlap no instance.
  Output: COCO compressed RLE with filseg_infer's writer, ids '<image stem>_<k>' (0-based), every mask round-tripped
  and checked disjoint.

Precision: bf16 autocast only on CUDA GPUs with compute capability >= 8 (the A6000 reference run); otherwise fp32.
fp32 and bf16 round differently, so a CPU run is close to, not bit-identical with, the GPU reference CSV
(reference/submission_r2b0_se2.csv).
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch

import filseg_infer as K

H = W = 2048
B0_THR, B0_MIN = 0.5, 500            # existence (stage-1 instances)
CAND_THR, CAND_MIN = 0.3, 20         # candidates
WIN = 384                            # native refinement window
CTX = 4 * WIN                        # context window side (1536 px), downsampled by 4 to WIN
MARGIN = 32                          # bbox margin of the windows
R2_THR, R2_MIN, MERGE_IOU = 0.5, 300, 0.5
R2_CHANNELS = 10
R2_BATCH = 16                        # windows per R2 forward pass (as predict_r2b0.refine_probs)
MIN_PART = 20                        # v1.2 cleanup step 1 (D-051): drop parts < 20 px of a multi-part instance
ISLAND_FRAC = 0.05                   # v1.2 cleanup step 2 (D-058): then parts < 5 % of the instance area


# ----------------------------------------------------------------------------- models
def device_and_precision():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 autocast only on GPUs with native bf16 (compute capability >= 8), exactly as filseg_infer.run
    bf16 = dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 8
    return dev, bf16


def load_r2(path, device):
    """R2_final: the filseg_infer U-Net (ResNet-101) with a 10-channel first convolution."""
    return K.load_models([path], device, in_channels=R2_CHANNELS)[0]


# ----------------------------------------------------------------------------- step 1: stage 1
def stage1(models, img_u8, device, bf16):
    """float16 mean of the per-model float16 probabilities (2048^2), one image at a time."""
    x = torch.from_numpy(K.to_model_space(img_u8)[None]).to(device)
    acc = None
    with torch.no_grad():
        for m in models:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                y = m(K.normalize(x)[:, None]).float()
            p = K.logits_to_full_prob(y[:, 0])[0].astype(np.float16).astype(np.float32)
            acc = p if acc is None else acc + p
    return (acc / len(models)).astype(np.float16)


# ----------------------------------------------------------------------------- steps 2-3: existence, candidates
def candidates(p16):
    """-> prob_u8, candidate label map (uint16), {cid: bbox}, kept cids (overlap with B), B (list of (bbox, crop))."""
    prob_u8 = np.clip(np.rint(p16.astype(np.float32) * 255), 0, 255).astype(np.uint8)
    p = prob_u8.astype(np.float32) / 255.0
    n, lab, stats = cv2.connectedComponentsWithStats((p > CAND_THR).astype(np.uint8), connectivity=8)[:3]
    keep_cc = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= CAND_MIN]
    lut = np.zeros(n, np.uint16)
    for j, k in enumerate(keep_cc, 1):
        lut[k] = j
    cmap = lut[lab]
    boxes = {}
    for j, k in enumerate(keep_cc, 1):
        x, y, w, h, a = (int(v) for v in stats[k])
        boxes[j] = (y, x, y + h, x + w)
    B = K.components(p16.astype(np.float32), B0_THR, B0_MIN)
    b_union = np.zeros((H, W), bool)
    for (y0, x0, y1, x1), c in B:
        b_union[y0:y1, x0:x1] |= c
    inb = np.bincount(cmap[b_union], minlength=len(keep_cc) + 1)
    kept = [j for j in range(1, len(keep_cc) + 1) if inb[j] > 0]
    return prob_u8, cmap, boxes, kept, B


# ----------------------------------------------------------------------------- step 4: R2 inputs and refinement
def windows_for_box(y0, x0, y1, x1):
    """Native windows (top-left y, x) covering a candidate bbox (refine.windows_for_box)."""
    def starts(a0, a1):
        lo, hi = a0 - MARGIN, a1 + MARGIN
        if hi - lo <= WIN:
            c = (lo + hi) // 2 - WIN // 2
            return [int(np.clip(c, 0, H - WIN))]
        st = list(range(lo, hi - WIN + 1, WIN - 128))
        if st[-1] + WIN < hi:
            st.append(hi - WIN)
        return sorted({int(np.clip(s, 0, H - WIN)) for s in st})
    return [(y, x) for y in starts(y0, y1) for x in starts(x0, x1)]


def orientation(prob_win):
    """Coherence-weighted axis orientation (cos 2t, sin 2t) from the structure tensor of the stage-1 probability."""
    p = cv2.GaussianBlur(prob_win, (0, 0), 2.0)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 6.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 6.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 6.0)
    num = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = num / (jxx + jyy + 1e-6)
    c2 = -(jxx - jyy) / (num + 1e-9)   # axis = gradient direction + 90 deg -> cos2t, sin2t change sign
    s2 = -(2 * jxy) / (num + 1e-9)
    return (coh * c2).astype(np.float32), (coh * s2).astype(np.float32)


def _ctx(arr2d, y, x, interp, fill=0):
    """4x context window centred on the native window, zero-padded, downsampled to WIN."""
    cy, cx = y + WIN // 2, x + WIN // 2
    y0, x0 = cy - CTX // 2, cx - CTX // 2
    out = np.full((CTX, CTX), fill, arr2d.dtype)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(H, y0 + CTX), min(W, x0 + CTX)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr2d[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (WIN, WIN), interpolation=interp)


def _norm_img(a, gamma=1.0, gain=1.0, bias=0.0):
    # refine.make_input's norm_img at inference (no augmentation: gamma 1, gain 1, bias 0; kept to mirror its arithmetic)
    a = np.clip(a.astype(np.float32) / 255.0, 0, 1) ** gamma * gain + bias
    return (a - 0.30) / 0.22


def make_input(img, prob_u8, cand, cid, y, x):
    """The 10 input channels (float32, 10 x WIN x WIN) for one (candidate, window); refine.make_input('r2', ...)."""
    sl = (slice(y, y + WIN), slice(x, x + WIN))
    f = np.ascontiguousarray                       # no flips at inference
    im = f(_norm_img(img[sl]))
    cw = cand[sl]
    this = f((cw == cid).astype(np.float32))
    pw = f(prob_u8[sl].astype(np.float32) / 255.0)
    oth = f(((cw > 0) & (cw != cid)).astype(np.float32))
    oc, os_ = orientation(pw)
    cand_full_this = (cand == cid)
    ctx_img = f(_norm_img(_ctx(img, y, x, cv2.INTER_AREA)))
    ctx_p = f(_ctx(prob_u8, y, x, cv2.INTER_AREA).astype(np.float32) / 255.0)
    ctx_this = f(_ctx(cand_full_this.astype(np.uint8) * 255, y, x, cv2.INTER_AREA).astype(np.float32) / 255.0)
    ctx_oth = f(_ctx(((cand > 0) & ~cand_full_this).astype(np.uint8) * 255, y, x, cv2.INTER_AREA).astype(np.float32) / 255.0)
    return np.stack([im, this, pw, oth, oc, os_, ctx_img, ctx_p, ctx_this, ctx_oth]).astype(np.float32)


def refine_probs(r2, img_u8, prob_u8, cmap, boxes, kept, device, bf16, bs=R2_BATCH):
    """{cid: (y0, x0, float16 mean probability over the candidate's windows)} for the kept candidates."""
    items = [(j, y, x) for j in kept for (y, x) in windows_for_box(*boxes[j])]
    acc = {}
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            xb = torch.from_numpy(np.stack([make_input(img_u8, prob_u8, cmap, j, y, x) for j, y, x in chunk]))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                pb = torch.sigmoid(r2(xb.to(device)).float())[:, 0].cpu().numpy()
            for p, (j, y, x) in zip(pb, chunk):
                acc.setdefault(j, []).append((y, x, p))
    out = {}
    for j, lst in acc.items():
        y0 = min(y for y, _, _ in lst); x0 = min(x for _, x, _ in lst)
        y1 = max(y for y, _, _ in lst) + WIN; x1 = max(x for _, x, _ in lst) + WIN
        s = np.zeros((y1 - y0, x1 - x0), np.float32); c = np.zeros_like(s)
        for y, x, p in lst:
            s[y - y0:y - y0 + WIN, x - x0:x - x0 + WIN] += p
            c[y - y0:y - y0 + WIN, x - x0:x - x0 + WIN] += 1
        out[j] = (y0, x0, (s / np.maximum(c, 1)).astype(np.float16))
    return out, len(items)


# ----------------------------------------------------------------------------- step 5: instance assembly
def assemble(cand_probs, thr=R2_THR, min_area=R2_MIN, merge_iou=MERGE_IOU):
    """refine.assemble. cand_probs: {cid: (y0, x0, float32 prob crop)} -> list of (bbox, crop), disjoint.

    Same-instance duplicates (mask IoU > merge_iou) are merged by union-find, pixel conflicts go to the higher
    probability (ties to the earlier group), instances < min_area are dropped. The bounding boxes of the final label
    map are taken per group inside the group's box, which equals scipy.ndimage.find_objects on that map (each group
    writes its label only inside its own box)."""
    ids = [c for c, (_, _, p) in cand_probs.items() if (p > thr).any()]
    masks = {}
    for c in ids:
        y0, x0, p = cand_probs[c]
        m = p > thr
        ys, xs = np.where(m)
        masks[c] = (y0 + ys.min(), x0 + xs.min(), y0 + ys.max() + 1, x0 + xs.max() + 1)
    parent = {c: c for c in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def crop(c, box):
        y0, x0, p = cand_probs[c]
        by0, bx0, by1, bx1 = box
        out = np.zeros((by1 - by0, bx1 - bx0), np.float32)
        sy0, sx0 = max(by0, y0), max(bx0, x0)
        sy1, sx1 = min(by1, y0 + p.shape[0]), min(bx1, x0 + p.shape[1])
        if sy1 > sy0 and sx1 > sx0:
            out[sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0] = p[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
        return out

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            A, B = masks[a], masks[b]
            box = (min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3]))
            if A[2] <= B[0] or B[2] <= A[0] or A[3] <= B[1] or B[3] <= A[1]:
                continue
            ma, mb = crop(a, box) > thr, crop(b, box) > thr
            inter = np.logical_and(ma, mb).sum()
            if inter and inter / np.logical_or(ma, mb).sum() > merge_iou:
                parent[find(a)] = find(b)
    groups = {}
    for c in ids:
        groups.setdefault(find(c), []).append(c)
    best_p = np.zeros((H, W), np.float32)
    best_id = np.zeros((H, W), np.int32)
    group_box = {}
    for gi, (_, members) in enumerate(sorted(groups.items()), 1):
        box = (min(masks[c][0] for c in members), min(masks[c][1] for c in members),
               max(masks[c][2] for c in members), max(masks[c][3] for c in members))
        group_box[gi] = box
        pg = np.max([crop(c, box) for c in members], axis=0)
        y0, x0, y1, x1 = box
        sub_p, sub_i = best_p[y0:y1, x0:x1], best_id[y0:y1, x0:x1]
        upd = (pg > thr) & (pg > sub_p)
        sub_p[upd] = pg[upd]
        sub_i[upd] = gi
    out = []
    for gi in range(1, len(group_box) + 1):
        y0, x0, y1, x1 = group_box[gi]
        sub = best_id[y0:y1, x0:x1] == gi
        rows, cols = np.flatnonzero(sub.any(axis=1)), np.flatnonzero(sub.any(axis=0))
        if rows.size == 0:                    # every pixel of this group went to other groups
            continue
        ty0, ty1, tx0, tx1 = y0 + rows[0], y0 + rows[-1] + 1, x0 + cols[0], x0 + cols[-1] + 1
        c = best_id[ty0:ty1, tx0:tx1] == gi
        if c.sum() < max(min_area, 1):
            continue
        out.append(((int(ty0), int(tx0), int(ty1), int(tx1)), c))
    return out


# ----------------------------------------------------------------------------- v1.2 cleanup (port of predict_r2b0)
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
    """Cleanup of multi-part instances (the largest part is never dropped): drop 8-connected parts < min_part px,
    then parts < island_frac of the remaining instance area. 0 / 0 gives the original R2b0 masks (D-041)."""
    if min_part:
        mask = _drop_parts(mask, lambda a: min_part)
    if island_frac:
        mask = _drop_parts(mask, lambda a: island_frac * a.sum())
    return mask


def write_rows(out_csv, per_image):
    """per_image: [(stem, base_rows, extra_rows)] in image order. Base rows and then extra (SE) rows per image; an
    image with extra rows but no base rows is written after all other images (sorted), as the submitted CSV."""
    rows, tail = [], []
    for stem, base, extra in per_image:
        if base:
            rows += base + extra
        elif extra:
            tail.append((stem, extra))
    for _, extra in sorted(tail):
        rows += extra
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["filament_id", "segmentation_rle"])
        wr.writerows(rows)
    assert len({r[0] for r in rows}) == len(rows), "duplicate filament_id"
    return rows


# ----------------------------------------------------------------------------- one image, end to end
def predict_image(img_u8, b0_models, r2, device, bf16):
    """All intermediate results for one 2048^2 uint8 image; 'instances' is the list of (bbox, crop) to submit."""
    p16 = stage1(b0_models, img_u8, device, bf16)
    prob_u8, cmap, boxes, kept, B = candidates(p16)
    cp16, n_windows = refine_probs(r2, img_u8, prob_u8, cmap, boxes, kept, device, bf16)
    inst = assemble({j: (y0, x0, p.astype(np.float32)) for j, (y0, x0, p) in cp16.items() if j in kept})
    return dict(p16=p16, prob_u8=prob_u8, cand=cmap, boxes=boxes, kept=kept, existence=B, n_windows=n_windows,
                instances=inst)


def run(test_dir, b0_weight_paths, r2_weight_path, se2_weight_path, out_csv, limit=0, verify=True, log=print,
        keep=None, min_part=MIN_PART, island_frac=ISLAND_FRAC, se_tau=None):
    """Predict all *.jpeg in test_dir with the final pipeline, write the submission CSV, return a summary dict.

    keep: optional dict; if given, keep[stem] receives the per-image counts (stage-1 instances, candidates, kept,
    windows, instances, SE2 additions). min_part / island_frac: the cleanup. se_tau: the SE2 threshold
    (default filseg_infer_se.SE2_TAU = 0.12744)."""
    import time
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device, bf16 = device_and_precision()
    b0 = K.load_models(b0_weight_paths, device)
    r2 = load_r2(r2_weight_path, device)
    import filseg_infer_se as SE
    se = SE.load_scorer(se2_weight_path, device)
    se_tau = SE.SE2_TAU if se_tau is None else se_tau
    files = sorted(p for p in Path(test_dir).iterdir() if p.suffix.lower() in (".jpeg", ".jpg"))
    if limit:
        files = files[:limit]
    log(f"device={device} bf16_autocast={bf16} stage-1 models={len(b0)} refinement model=R2_final images={len(files)}")
    per_image, n_rows, n_se, t0 = [], 0, 0, time.time()
    for i, f in enumerate(files):
        img = K.read_image(f)
        res = predict_image(img, b0, r2, device, bf16)
        comps = res["instances"]
        cover = np.zeros((H, W), bool)
        base, extra = [], []
        for k, ((y0, x0, y1, x1), c) in enumerate(comps):
            full = np.zeros((H, W), bool)
            full[y0:y1, x0:x1] = c
            full = drop_small_parts(full, min_part, island_frac)
            counts = K.encode(full)
            if verify:
                assert np.array_equal(K.mask_from_counts(counts).astype(bool), full), f"RLE roundtrip {f.stem}_{k}"
            assert not (cover & full).any(), f"overlap {f.stem}_{k}"
            cover |= full
            base.append((f"{f.stem}_{k}", counts))
        # SE2: redraw the small fragments with R2_final, score the redrawn masks, add the accepted ones
        fj, gate = SE.fragment_candidates(res["p16"], res["cand"], res["kept"])
        fcp, _ = refine_probs(r2, img, res["prob_u8"], res["cand"], res["boxes"], fj, device, bf16)
        frags = assemble({j: (y0, x0, p.astype(np.float32)) for j, (y0, x0, p) in fcp.items()}, min_area=100)
        add, _ = SE.additions_refined(se, frags, img, res["p16"], gate, cover, se_tau, device)
        for j, m in enumerate(add):
            counts = K.encode(m)
            if verify:
                assert np.array_equal(K.mask_from_counts(counts).astype(bool), m), f"RLE roundtrip SE2 {f.stem}"
            extra.append((f"{f.stem}_{len(base) + j}", counts))
        per_image.append((f.stem, base, extra))
        n_rows += len(base) + len(extra); n_se += len(extra)
        if keep is not None:
            keep[f.stem] = dict(n_b0=len(res["existence"]), n_cand=len(res["boxes"]), n_kept=len(res["kept"]),
                                n_windows=res["n_windows"], n_inst=len(comps), n_se=len(extra))
        log(f"{i + 1}/{len(files)} images, {n_rows} filaments ({n_se} added by SE2), {time.time() - t0:.0f} s")
    rows = write_rows(out_csv, per_image)
    n_with = sum(bool(b or e) for _, b, e in per_image)
    return dict(images=len(files), rows=len(rows), images_with_predictions=n_with, se_added=n_se, device=str(device),
                bf16=bf16, csv=str(out_csv))
