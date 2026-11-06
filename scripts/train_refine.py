"""Train/evaluate the native-resolution refinement model R2 (10 channels: crop + structural context).

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_refine.py --variant r2 --fold 1
  .venv/bin/python scripts/train_refine.py --variant r2 --fold 1 --dump 8      # render 8 training samples, exit

Training windows: candidates of files in folds != k; validation: candidates of fold k (b0 OOF stage 1).
Nested CV (D-035): --cache outputs/cache/refine_nested_o<k> --exp R2n_context --fold k  (stage 1 of the outer-train
files = inner OOF of models that never saw fold k; stage 1 of fold k = mean of those 4 inner models).
Final model: --all --exp R2_final_context (all 707 files, refine_v1 = 5-fold b0 OOF stage 1, no validation).
The random stream (window order, annotation-set choice, flips, intensity) is seeded per window index.
Outputs (outputs/exps/<exp>/fold<k>/): config.json, log.jsonl, last.pt, val_cand_probs.npz (per-candidate refined
full-extent probability crops for post-hoc assembly), val_instances_t0.5_a0.json (RLE per stem), summary.json.
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse
import json
import math
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from filseg import refine as RF  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.metrics import EvalConfig, _eval_item, aggregate, build_items, single_thread_worker  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["r2"], required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--exp", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dump", type=int, default=0)
    p.add_argument("--cache", default=str(RF.CACHE), help="refine cache dir (stage-1 source), default refine_v1")
    p.add_argument("--all", action="store_true", help="train on ALL files of the cache, no validation (final model)")
    p.add_argument("--inner", type=int, default=-1,
                   help="SE2 inner model: also hold out fold INNER; train on folds outside {fold, inner}, validate on "
                        "fold INNER (its val_cand_probs.npz = refinements of fold INNER by a model that never saw "
                        "--fold or INNER); output dir fold<fold>_in<inner>")
    return p.parse_args()


class WinSet(torch.utils.data.Dataset):
    def __init__(self, idx, files, variant, train=True, cache=RF.CACHE):
        self.idx, self.variant, self.train, self.cache = idx, variant, train, Path(cache)
        self.items = []
        for f in files:
            m = idx["meta"][f]
            fi = idx["files"].index(f)
            for (j, y, x) in m["windows"]:
                self.items.append((fi, f, j, y, x, m["set_rows"]))
        self.arr = None

    def __len__(self):
        return len(self.items)

    def _open(self):
        if self.arr is None:
            self.arr = {k: np.load(self.cache / f"{k}.npy", mmap_mode="r") for k in ("images", "prob_u8", "cand", "gt")}

    def __getitem__(self, i):
        self._open()
        fi, f, j, y, x, rows = self.items[i]
        rng = np.random.default_rng((torch.initial_seed() + i * 7919) % 2**63) if self.train else None
        # one annotation set per window visit, flips and intensity, drawn from a per-index seed
        u = rng.random(6) if self.train else np.zeros(6)
        row = rows[int(u[0] * len(rows)) % len(rows)] if self.train else rows[0]
        fh, fv = bool(u[1] < 0.5), bool(u[2] < 0.5)
        gamma = 0.8 + 0.45 * u[3] if self.train else 1.0
        gain = 0.85 + 0.30 * u[4] if self.train else 1.0
        bias = -0.06 + 0.12 * u[5] if self.train else 0.0
        a = self.arr
        xin = RF.make_input(self.variant, a["images"][fi], a["prob_u8"][fi], a["cand"][fi], j, y, x,
                            flip_h=fh and self.train, flip_v=fv and self.train, gain=gain, gamma=gamma, bias=bias)
        gid = self.idx["meta"][f]["match"][f"{j}|{row}"]
        t = (a["gt"][row][y:y + RF.WIN, x:x + RF.WIN] == gid).astype(np.float32) if gid else \
            np.zeros((RF.WIN, RF.WIN), np.float32)
        if self.train:
            if fh:
                t = t[:, ::-1]
            if fv:
                t = t[::-1]
        return torch.from_numpy(xin), torch.from_numpy(np.ascontiguousarray(t))[None], i


def worker_init(_):
    cv2.setNumThreads(1)


def soft_dice(lg, t, eps=1.0):
    p = torch.sigmoid(lg)
    inter = (p * t).flatten(1).sum(1)
    den = p.flatten(1).sum(1) + t.flatten(1).sum(1)
    return (1 - (2 * inter + eps) / (den + eps)).mean()


@torch.no_grad()
def predict_val(model, idx, files, variant, device, bs=16, cache=RF.CACHE):
    """Per-candidate refined probability over the union of its windows (averaged where windows overlap)."""
    model.eval()
    ds = WinSet(idx, files, variant, train=False, cache=cache)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=6, worker_init_fn=worker_init)
    acc = {}
    for xb, _, ii in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pb = torch.sigmoid(model(xb.to(device)).float())[:, 0].cpu().numpy()
        for p, i in zip(pb, ii.numpy()):
            fi, f, j, y, x, _ = ds.items[i]
            acc.setdefault((f, j), []).append((y, x, p))
    out = {}
    for (f, j), lst in acc.items():
        y0 = min(y for y, _, _ in lst); x0 = min(x for _, x, _ in lst)
        y1 = max(y for y, _, _ in lst) + RF.WIN; x1 = max(x for _, x, _ in lst) + RF.WIN
        s = np.zeros((y1 - y0, x1 - x0), np.float32); c = np.zeros_like(s)
        for y, x, p in lst:
            s[y - y0:y - y0 + RF.WIN, x - x0:x - x0 + RF.WIN] += p
            c[y - y0:y - y0 + RF.WIN, x - x0:x - x0 + RF.WIN] += 1
        out[(f, j)] = (y0, x0, (s / np.maximum(c, 1)).astype(np.float16))
    model.train()
    return out


def evaluate(cand_probs, files, pool, thr=0.5, min_area=0):
    gts = seg.gt_records(files)
    preds = {}
    for f in files:
        cp = {j: (y0, x0, p.astype(np.float32)) for (ff, j), (y0, x0, p) in cand_probs.items() if ff == f}
        preds[Path(f).stem] = RF.assemble(cp, thr=thr, min_area=min_area)
    cfg = EvalConfig(compute_boundary=False, compute_pair_boundary=False, compute_thin=True)
    rows = list(pool.imap(partial(_eval_item, cfg=cfg, return_instances=False), build_items(gts, preds), chunksize=4))
    return aggregate(pd.DataFrame([r[0] for r in rows])), preds


def main():
    a = parse()
    import multiprocessing as mp
    single_thread_worker()
    pool = mp.get_context("forkserver").Pool(8, initializer=single_thread_worker)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    cache = Path(a.cache)
    idx = json.load(open(cache / "index.json"))
    if a.all:
        tr_files, va_files = sorted(idx["files"]), []
    elif a.inner >= 0:
        assert a.inner != a.fold
        tr_files = sorted(f for f in idx["files"] if idx["fold"][f] not in (a.fold, a.inner))
        va_files = sorted(f for f in idx["files"] if idx["fold"][f] == a.inner)
    else:
        tr_files = sorted(f for f in idx["files"] if idx["fold"][f] != a.fold)
        va_files = sorted(f for f in idx["files"] if idx["fold"][f] == a.fold)
    exp = a.exp or "R2_refine_context"
    out = ROOT / "outputs" / "exps" / exp / ("all" if a.all else f"fold{a.fold}_in{a.inner}" if a.inner >= 0 else f"fold{a.fold}")
    out.mkdir(parents=True, exist_ok=True)
    ds = WinSet(idx, tr_files, a.variant, train=True, cache=cache)
    if a.dump:
        dump(ds, a, out)
        return
    device = torch.device("cuda")
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                     worker_init_fn=worker_init, persistent_workers=True, pin_memory=True,
                                     generator=torch.Generator().manual_seed(a.seed))
    model = RF.build_model(a.variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl); warm = min(len(dl), 200)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    cfg = vars(a) | dict(exp=exp, n_train_windows=len(ds), n_val_files=len(va_files), win=RF.WIN, ctx=RF.CTX,
                         cand_thr=RF.CAND_THR, cand_min=RF.CAND_MIN, channels=RF.VARIANT_CH[a.variant],
                         stage1=idx["stage1"], cache_dir=str(cache), exist_exp=idx.get("exist_exp", "b0_r101_s1024"),
                         train_folds=sorted({idx["fold"][f] for f in tr_files}), torch=torch.__version__)
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    (out / "val_files.txt").write_text("\n".join(va_files))
    (out / "train_files.txt").write_text("\n".join(tr_files))
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        tl, n, te = 0.0, 0, time.time()
        for batch in dl:
            xb, tb = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(xb)
            lg = lg.float()
            loss = F.binary_cross_entropy_with_logits(lg, tb) + soft_dice(lg, tb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item(); n += 1
        rec = dict(epoch=ep, train_loss=tl / n, epoch_s=time.time() - te, peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
        if (ep % 10 == 0 or ep == a.epochs) and not va_files:
            torch.save(model.state_dict(), out / "last.pt")
        elif ep % 10 == 0 or ep == a.epochs:
            cp = predict_val(model, idx, va_files, a.variant, device, cache=cache)
            s, _ = evaluate(cp, va_files, pool)
            rec.update({k: s[k] for k in ("PQ_pooled_host", "SQ_pooled", "RQ_pooled", "TP", "FP", "FN", "n_split_gt",
                                          "n_merge_pred", "thin_region_recall_pooled_px", "thin_region_precision_pooled_px")})
            torch.save(model.state_dict(), out / "last.pt")
            if ep == a.epochs:
                np.savez_compressed(out / "val_cand_probs.npz",
                                    **{f"{Path(f).stem}|{j}|{y0}|{x0}": p for (f, j), (y0, x0, p) in cp.items()})
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        with open(out / "log.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    json.dump(dict(train_minutes=(time.time() - t0) / 60, peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30),
              open(out / "summary.json", "w"))
    print("DONE", flush=True)


def dump(ds, a, out):
    """Render training samples (all channels + target) for a visual alignment check."""
    rng = np.random.default_rng(2026)
    d = out / "dump"; d.mkdir(exist_ok=True)
    for k in rng.choice(len(ds), a.dump, replace=False):
        x, t, _ = ds[int(k)]
        x = x.numpy(); t = t.numpy()[0]
        tiles = []
        for c in range(x.shape[0]):
            v = x[c]; v = (v - v.min()) / (v.max() - v.min() + 1e-6)
            tiles.append((v * 255).astype(np.uint8))
        tiles.append((t * 255).astype(np.uint8))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(d / f"sample_{int(k)}.png"), grid)
    print("dumped to", d)


if __name__ == "__main__":
    main()
