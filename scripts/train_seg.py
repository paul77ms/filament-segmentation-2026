"""Train the baseline full-disk U-Net (semantic filament probability) on one CV fold.

Usage (GPU chosen by the caller):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_seg.py --exp b0_r101_s1024 --fold 0
  ... --subset 8 --epochs 60 --val_every 20   # overfit sanity check (train == val files)
  ... --exp b0n_r101_s1024 --holdout 0,1 --val_every 60   # nested inner model: trains on folds 2-4 (D-035)

Selection: the checkpoint with the best validation PQ_pooled_host at the FIXED default
post-processing (thr 0.5, min_area 0) is kept as best.pt; a small post-processing grid is logged
for information only (tuning it on the same fold is optimistic and is reported as such).
"""
import os

# Must precede the numpy import: on this shared host THP is in "madvise" mode and heavily
# fragmented; numpy's default MADV_HUGEPAGE on large arrays then triggers direct compaction
# (measured: eval threads at ~100% system time, validation >20 min). Disable it.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from functools import partial  # noqa: E402

from filseg import seg  # noqa: E402
from filseg.metrics import (EvalConfig, _eval_item, aggregate, build_items,  # noqa: E402
                            single_thread_worker)

FOLDS = ROOT / "outputs" / "splits" / "folds_v1.csv"
# In-training validation stays light (2 post-processing settings, no thin-region metric);
# the post-processing is chosen afterwards by nested CV (scripts/tune_post.py).
POST_GRID = [(0.5, 0), (0.5, 300)]
DEFAULT_POST = (0.5, 0)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--encoder", default="resnet101")
    p.add_argument("--pretrained", default="local")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dice_w", type=float, default=1.0)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--subset", type=int, default=0, help="sanity: train AND validate on the first N train files")
    p.add_argument("--rot", type=float, default=15.0)
    p.add_argument("--no_aug", action="store_true")
    p.add_argument("--no_best", action="store_true", help="do not write best.* (with --val_every = epochs they "
                   "duplicate last.*; saves ~0.8 GB per nested run)")
    p.add_argument("--holdout", default="", help="nested CV (D-035): comma-separated folds left out of training, all "
                   "validated (val_probs_last.npz covers them); output dir hold<folds>; overrides --fold")
    return p.parse_args()


def select_files(folds: pd.DataFrame, fold: int, holdout: str = "", subset: int = 0):
    """(train files, val files, run dir name). holdout '1,3' -> train on folds {0,2,4}, validate on folds {1,3}."""
    if subset:
        tr = sorted(folds.file_name)[:subset]
        return tr, tr, f"sanity{subset}"
    if holdout:
        hold = sorted({int(f) for f in holdout.split(",")})
        assert len(hold) >= 1 and set(hold) <= set(folds.fold), hold
        return (sorted(folds.file_name[~folds.fold.isin(hold)]), sorted(folds.file_name[folds.fold.isin(hold)]),
                "hold" + "".join(map(str, hold)))
    return sorted(folds.file_name[folds.fold != fold]), sorted(folds.file_name[folds.fold == fold]), f"fold{fold}"


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, cache_dir, files, aug=True, rot=15.0, seed=0):
        self.idx = json.load(open(cache_dir / "index.json"))
        self.cache_dir = cache_dir
        self.X = None
        self.T = None
        frow = {f: i for i, f in enumerate(self.idx["files"])}
        self.items = [(frow[f], self.idx["file_sets"][f]) for f in files]
        self.aug, self.rot, self.seed = aug, rot, seed
        self._main_rng = None

    def _rng(self):  # used only with num_workers=0
        if self._main_rng is None:
            self._main_rng = np.random.default_rng(self.seed)
        return self._main_rng

    def __len__(self):
        return len(self.items)

    def _open(self):
        if self.X is None:
            self.X = np.load(self.cache_dir / "images.npy", mmap_mode="r")
            self.T = np.load(self.cache_dir / "targets.npy", mmap_mode="r")

    def __getitem__(self, i):
        self._open()
        xi, set_rows = self.items[i]
        rng = _WORKER_RNG if _WORKER_RNG is not None else self._rng()
        x = np.asarray(self.X[xi], np.float32) / 255.0
        srow = set_rows[rng.integers(len(set_rows))]   # one annotation set of the image per visit (seeded)
        t = np.asarray(self.T[srow], np.float32) / 255.0
        if self.aug:
            if rng.random() < 0.5:
                x, t = x[:, ::-1], t[:, ::-1]
            if rng.random() < 0.5:
                x, t = x[::-1], t[::-1]
            if self.rot > 0 and rng.random() < 0.5:
                s = x.shape[0]
                M = cv2.getRotationMatrix2D(((s - 1) / 2, (s - 1) / 2), float(rng.uniform(-self.rot, self.rot)), 1.0)
                x = cv2.warpAffine(np.ascontiguousarray(x), M, (s, s), flags=cv2.INTER_LINEAR, borderValue=0)
                t = cv2.warpAffine(np.ascontiguousarray(t), M, (s, s), flags=cv2.INTER_LINEAR, borderValue=0)
            if rng.random() < 0.8:
                g = rng.uniform(0.8, 1.25)
                x = np.clip(x, 0, 1) ** g
                x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.06, 0.06)
            if rng.random() < 0.2:
                x = cv2.GaussianBlur(np.ascontiguousarray(x), (0, 0), rng.uniform(0.4, 1.2))
            if rng.random() < 0.2:
                x = x + rng.normal(0, rng.uniform(0.005, 0.03), x.shape).astype(np.float32)
        x = (np.ascontiguousarray(x, np.float32) - 0.30) / 0.22
        t = np.ascontiguousarray(t, np.float32)[None]
        return torch.from_numpy(x)[None], torch.from_numpy(t)


_WORKER_RNG = None


def worker_init(wid):
    """Per-worker numpy Generator derived from torch.initial_seed(), which the DataLoader derives
    from its seeded generator (--seed) and the worker id. Index->worker assignment is deterministic,
    so augmentation and annotation-set sampling are reproducible for a given seed and worker count."""
    global _WORKER_RNG
    s = torch.initial_seed()
    _WORKER_RNG = np.random.default_rng(s)
    np.random.seed(s % 2**32)
    random.seed(s)
    cv2.setNumThreads(1)


def soft_dice_per_sample(logits, t, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * t).flatten(1).sum(1)
    den = p.flatten(1).sum(1) + t.flatten(1).sum(1)
    return 1 - (2 * inter + eps) / (den + eps)


def soft_dice_loss(logits, t, eps=1.0):
    return soft_dice_per_sample(logits, t, eps).mean()


def make_eval_pool(n=8):
    """Created BEFORE CUDA init / DataLoader threads. 'forkserver' children do not inherit the
    trainer's threads and locks (a plain fork pool deadlocked here in futex_wait)."""
    import multiprocessing as mp
    return mp.get_context("forkserver").Pool(n, initializer=single_thread_worker)


def eval_records(pool, gts, preds, cfg):
    rows = list(pool.imap(partial(_eval_item, cfg=cfg, return_instances=False), build_items(gts, preds), chunksize=4))
    df = pd.DataFrame([r[0] for r in rows])
    return aggregate(df), df


@torch.no_grad()
def validate(model, X, val_rows, val_files, gts, device, pool, bs=4):
    """Returns (summary at default post, grid dict, per-image df at default post, probs, infer seconds)."""
    model.eval()
    probs = {}
    t0 = time.time()
    for i in range(0, len(val_rows), bs):
        xb = torch.from_numpy(np.stack([X[r] for r in val_rows[i:i + bs]])).to(device)
        x = seg.normalize(xb)[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(x).float()
        full = [seg.logits_to_full_prob(y[:, c]) for c in range(y.shape[1])]
        for j, f in enumerate(val_files[i:i + bs]):
            probs[Path(f).stem] = full[0][j].astype(np.float16)
    torch.cuda.synchronize()
    t_inf = time.time() - t0
    cfg = EvalConfig(compute_boundary=False, compute_pair_boundary=False, compute_thin=False)
    grid = {}
    out_default = None
    for thr in sorted({t for t, _ in POST_GRID}):
        ccs = {k: cv2.connectedComponentsWithStats((v > thr).astype(np.uint8), connectivity=8)[:3]
               for k, v in probs.items()}
        for ma in sorted({a for t, a in POST_GRID if t == thr}):
            preds = {k: seg.labels_to_instances(*c, min_area=ma) for k, c in ccs.items()}
            s, df = eval_records(pool, gts, preds, cfg)
            grid[f"t{thr}_a{ma}"] = {k: s[k] for k in ("PQ_pooled_host", "PQ_pooled", "SQ_pooled", "RQ_pooled",
                                                       "n_pred", "n_split_gt", "n_merge_pred")}
            if (thr, ma) == DEFAULT_POST:
                out_default = (s, df)
    model.train()
    return out_default[0], grid, out_default[1], probs, t_inf


def main():
    a = parse()
    pool = make_eval_pool(8)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    folds = pd.read_csv(FOLDS)
    assert not (a.holdout and a.subset), "--holdout is not combined with --subset"
    tr_files, va_files, run_dir = select_files(folds, a.fold, a.holdout, a.subset)
    out = ROOT / "outputs" / "exps" / a.exp / run_dir
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = seg.build_cache(a.size)
    idx = json.load(open(cache_dir / "index.json"))
    frow = {f: i for i, f in enumerate(idx["files"])}
    X = np.load(cache_dir / "images.npy", mmap_mode="r")
    val_rows = [frow[f] for f in va_files]
    gts = seg.gt_records(va_files)
    fold_of = dict(zip(folds.file_name, folds.fold))
    cfg = vars(a) | dict(n_train_files=len(tr_files), n_val_files=len(va_files), n_val_sets=len(gts),
                         train_folds=sorted({int(fold_of[f]) for f in tr_files}),
                         val_folds=sorted({int(fold_of[f]) for f in va_files}),
                         torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0), cudnn_benchmark=True, post_default=DEFAULT_POST)
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    (out / "val_files.txt").write_text("\n".join(va_files))
    (out / "train_files.txt").write_text("\n".join(tr_files))

    ds = TrainSet(cache_dir, tr_files, aug=not a.no_aug, rot=a.rot)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                     worker_init_fn=worker_init, persistent_workers=True, pin_memory=True,
                                     generator=torch.Generator().manual_seed(a.seed))
    if a.subset:  # repeat the tiny subset so an 'epoch' has a useful number of steps
        dl = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([ds] * max(1, 64 // len(ds))),
                                         batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                         worker_init_fn=worker_init, persistent_workers=True,
                                         generator=torch.Generator().manual_seed(a.seed))
    model = seg.build_model(a.encoder, a.pretrained, classes=1)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * len(dl)
    warm = min(len(dl), 200)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    log = []
    best = -1.0
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        te = time.time()
        for batch in dl:
            x, t = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x)
            lg = lg.float()
            loss = F.binary_cross_entropy_with_logits(lg[:, :1], t[:, :1]) + a.dice_w * soft_dice_loss(lg[:, :1], t[:, :1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item(); n += 1
        rec = dict(epoch=ep, train_loss=tl / max(n, 1), lr=sched.get_last_lr()[0], epoch_s=time.time() - te,
                   peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
        if ep % a.val_every == 0 or ep == a.epochs:
            s, grid, df, probs, t_inf = validate(model, X, val_rows, va_files, gts, device, pool)
            rec.update({k: s[k] for k in ("PQ_pooled_host", "PQ_pooled", "SQ_pooled", "RQ_pooled", "TP", "FP", "FN",
                                          "n_split_gt", "n_merge_pred", "semantic_dice_pooled_px")})
            rec["val_infer_s_per_img"] = t_inf / len(val_rows)
            rec["grid"] = grid
            torch.save(model.state_dict(), out / "last.pt")
            if ep == a.epochs:  # fixed-epoch model = primary (unbiased) checkpoint
                df.to_csv(out / "val_per_image_last.csv", index=False)
                np.savez_compressed(out / "val_probs_last.npz", **{k: v for k, v in probs.items()})
            if s["PQ_pooled_host"] > best:
                best = s["PQ_pooled_host"]
                if not a.no_best:
                    torch.save(model.state_dict(), out / "best.pt")
                    df.to_csv(out / "val_per_image_best.csv", index=False)
                    np.savez_compressed(out / "val_probs_best.npz", **{k: v for k, v in probs.items()})
                rec["is_best"] = True
        log.append(rec)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items() if k != "grid"}),
              flush=True)
        with open(out / "log.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    summary = dict(best_PQ_pooled_host=best, train_minutes=(time.time() - t_start) / 60,
                   peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
