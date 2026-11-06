"""Experiments R1/R2: instance-conditioned refinement at NATIVE resolution around stage-1 candidates.

Stage 1 = b0 OOF probabilities (each training image's probability comes from the b0 fold model that did not see
it, so candidate errors are realistic). Candidates = 8-connected components of p > CAND_THR with >= CAND_MIN px.
For each candidate, native-resolution windows (WIN x WIN) cover its bounding box. The target of a (candidate,
annotation set) pair is the FULL GT instance of that set that best matches the candidate (max IoU; empty if no
GT overlaps) -- i.e. "which filament is this candidate part of, and what is its whole extent here?".

Variants (identical windows, targets, training; only the inputs differ):
  r1: [image, this-candidate mask]                                     (crop-only)
  r2: r1 + [stage-1 prob, other candidates, orientation cos2t, sin2t]  (native)
          + [image, stage-1 prob, this candidate, other candidates]     (4x wider context window, 1536 -> WIN)
Instances: per-candidate refined masks (p > thr); candidates whose masks overlap with IoU > 0.5 are the same
instance (merged); remaining pixel conflicts go to the higher probability. No learned keep/drop rescoring.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import data as D
from . import seg
from .metrics import InstanceSet

WIN = 384
CTX = 4 * WIN          # context window side (1536 px), downsampled by 4 to WIN
MARGIN = 32
CAND_THR, CAND_MIN = 0.3, 20
CACHE = D.PROJECT_ROOT / "outputs" / "cache" / "refine_v1"
H = W = 2048
VARIANT_CH = {"r1": 2, "r2": 10}


# ----------------------------------------------------------------------------- cache
def windows_for_box(y0, x0, y1, x1):
    """Native windows (top-left y, x) covering a candidate bbox."""
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


def _build_one(args):
    fi, fname, prob_u8_path, img_path, cand_path, gt_path, set_rows, sets, write_static = args
    cv2.setNumThreads(1)
    pr = np.load(prob_u8_path, mmap_mode="r+")
    cm = np.load(cand_path, mmap_mode="r+")
    gm = np.load(gt_path, mmap_mode="r+") if write_static else None
    if write_static:   # images / GT maps do not depend on stage 1 (shared with refine_v1 when static_from is used)
        img = np.load(img_path, mmap_mode="r+")
        img[fi] = D.read_image(D.train_image_path(fname))
    p = pr[fi].astype(np.float32) / 255.0
    n, lab, stats = cv2.connectedComponentsWithStats((p > CAND_THR).astype(np.uint8), connectivity=8)[:3]
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= CAND_MIN]
    lut = np.zeros(n, np.uint16)
    for j, k in enumerate(keep, 1):
        lut[k] = j
    cmap = lut[lab]
    cm[fi] = cmap
    boxes = []
    for j, k in enumerate(keep, 1):
        x, y, w, h, a = (int(v) for v in stats[k])
        boxes.append((j, y, x, y + h, x + w, a))
    match = {}
    for row, (sid, G) in zip(set_rows, sets):
        g = np.zeros((H, W), np.uint8)
        for i in range(len(G)):
            y0, x0, y1, x1 = G.boxes[i]
            g[y0:y1, x0:x1][G.crops[i]] = i + 1
        if write_static:
            gm[row] = g
        # best GT instance (max IoU) per candidate for this set
        for (j, y0, x0, y1, x1, a) in boxes:
            sub_c = cmap[y0:y1, x0:x1] == j
            ids, cnt = np.unique(g[y0:y1, x0:x1][sub_c], return_counts=True)
            best, best_iou = 0, 0.0
            for gid, c in zip(ids, cnt):
                if gid == 0:
                    continue
                iou = c / (a + G.areas[gid - 1] - c)
                if iou > best_iou:
                    best, best_iou = int(gid), float(iou)
            match[f"{j}|{row}"] = best
    wins = [(j, y, x) for (j, y0, x0, y1, x1, a) in boxes for (y, x) in windows_for_box(y0, x0, y1, x1)]
    return fi, dict(boxes=boxes, match=match, windows=wins, set_rows=list(set_rows))


def file_sets(files):
    """GT records of the files, file -> its annotation-set ids, and the cache's GT row order (set_ids)."""
    gts_all = seg.gt_records(files)
    sets_by_file = {}
    for rid in sorted(gts_all):
        sets_by_file.setdefault(rid.split("-", 1)[1] + ".jpeg", []).append(rid)
    return gts_all, sets_by_file, [rid for f in files for rid in sets_by_file[f]]


def build_cache(stage1_exp="b0_r101_s1024", n_workers=12, cache=None, stage1_probs=None, files=None,
                static_from=None, extra=None):
    """Full-res memmaps (images, stage-1 prob as uint8, candidate label maps, GT instance maps per set) +
    per-file candidate boxes, candidate->GT matches and windows. Atomic: index.json written last.

    Default (refine_v1): stage 1 = the b0 OOF probability of `stage1_exp` (fold-k model for fold-k files).
    Nested CV (D-035): stage1_probs = callable(file_name) -> float32 2048^2 probability (any source), cache = target
    directory, files = subset (default all 707), static_from = an existing cache over the SAME files whose images.npy
    and gt.npy (independent of stage 1) are symlinked instead of rewritten, extra = dict merged into index.json
    (e.g. which models produced each file's stage 1)."""
    import multiprocessing as mp
    cache = Path(cache) if cache is not None else CACHE
    if (cache / "index.json").exists():
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    folds = __import__("pandas").read_csv(D.PROJECT_ROOT / "outputs" / "splits" / "folds_v1.csv")
    files = sorted(files) if files is not None else sorted(folds.file_name)
    fold_of = dict(zip(folds.file_name, folds.fold))
    gts_all, sets_by_file, set_ids = file_sets(files)
    row_of = {rid: i for i, rid in enumerate(set_ids)}
    paths = {k: cache / f"{k}.npy" for k in ("images", "prob_u8", "cand", "gt")}
    for q in paths.values():     # leftovers of a crashed build: never write THROUGH a symlink (it may point to refine_v1)
        if q.is_symlink():
            q.unlink()
    write_static = static_from is None
    if write_static:
        np.lib.format.open_memmap(paths["images"], "w+", np.uint8, (len(files), H, W))
        np.lib.format.open_memmap(paths["gt"], "w+", np.uint8, (len(set_ids), H, W))
    else:
        src = json.load(open(Path(static_from) / "index.json"))
        assert src["files"] == files and src["set_ids"] == set_ids, "static_from cache covers different files/sets"
        for k in ("images", "gt"):
            if paths[k].exists():
                paths[k].unlink()
            paths[k].symlink_to((Path(static_from) / f"{k}.npy").resolve())
    P = np.lib.format.open_memmap(paths["prob_u8"], "w+", np.uint8, (len(files), H, W))
    np.lib.format.open_memmap(paths["cand"], "w+", np.uint16, (len(files), H, W))
    if stage1_probs is None:   # stage-1 OOF probabilities of one K-fold experiment
        pos = {f: i for i, f in enumerate(files)}
        for f in range(5):
            z = np.load(D.PROJECT_ROOT / "outputs" / "exps" / stage1_exp / f"fold{f}" / "val_probs_last.npz")
            for k in z.files:
                if k + ".jpeg" in pos:
                    P[pos[k + ".jpeg"]] = np.clip(np.rint(z[k].astype(np.float32) * 255), 0, 255).astype(np.uint8)
    else:
        for i, fn in enumerate(files):
            P[i] = np.clip(np.rint(np.asarray(stage1_probs(fn), np.float32) * 255), 0, 255).astype(np.uint8)
    P.flush(); del P
    jobs = [(fi, fn, str(paths["prob_u8"]), str(paths["images"]), str(paths["cand"]), str(paths["gt"]),
             [row_of[r] for r in sets_by_file[fn]], [(r, gts_all[r]) for r in sets_by_file[fn]], write_static)
            for fi, fn in enumerate(files)]
    meta = {}
    with mp.get_context("forkserver").Pool(n_workers) as pool:
        for fi, m in pool.imap_unordered(_build_one, jobs, chunksize=2):
            meta[files[fi]] = m
    index = dict(files=files, fold={f: int(fold_of[f]) for f in files}, set_ids=set_ids, meta=meta,
                 win=WIN, ctx=CTX, cand_thr=CAND_THR, cand_min=CAND_MIN, stage1=stage1_exp,
                 static_from=str(static_from) if static_from else None, **(extra or {}))
    json.dump(index, open(cache / "index.json", "w"))
    return cache


# ----------------------------------------------------------------------------- inputs
def orientation(prob_win: np.ndarray):
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
    """4x context window centred on the native window, padded, downsampled to WIN."""
    cy, cx = y + WIN // 2, x + WIN // 2
    y0, x0 = cy - CTX // 2, cx - CTX // 2
    out = np.full((CTX, CTX), fill, arr2d.dtype)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(H, y0 + CTX), min(W, x0 + CTX)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr2d[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (WIN, WIN), interpolation=interp)


def make_input(variant, img, prob_u8, cand, cid, y, x, flip_h=False, flip_v=False, gain=1.0, gamma=1.0, bias=0.0):
    """Channels (float32, C x WIN x WIN) for one (candidate, window); img/prob_u8/cand are full-res 2048^2 arrays."""
    sl = (slice(y, y + WIN), slice(x, x + WIN))

    def f(a):
        if flip_h:
            a = a[:, ::-1]
        if flip_v:
            a = a[::-1]
        return np.ascontiguousarray(a)

    def norm_img(a):
        a = np.clip(a.astype(np.float32) / 255.0, 0, 1) ** gamma * gain + bias
        return (a - 0.30) / 0.22

    im = f(norm_img(img[sl]))
    cw = cand[sl]
    this = f((cw == cid).astype(np.float32))
    ch = [im, this]
    if variant == "r2":
        pw = f(prob_u8[sl].astype(np.float32) / 255.0)
        oth = f(((cw > 0) & (cw != cid)).astype(np.float32))
        oc, os_ = orientation(pw)
        cand_full_this = (cand == cid)
        ctx_img = f(norm_img(_ctx(img, y, x, cv2.INTER_AREA)))
        ctx_p = f(_ctx(prob_u8, y, x, cv2.INTER_AREA).astype(np.float32) / 255.0)
        ctx_this = f(_ctx(cand_full_this.astype(np.uint8) * 255, y, x, cv2.INTER_AREA).astype(np.float32) / 255.0)
        ctx_oth = f(_ctx(((cand > 0) & ~cand_full_this).astype(np.uint8) * 255, y, x, cv2.INTER_AREA).astype(np.float32) / 255.0)
        ch += [pw, oth, oc, os_, ctx_img, ctx_p, ctx_this, ctx_oth]
    return np.stack(ch).astype(np.float32)


# ----------------------------------------------------------------------------- model
def build_model(variant):
    import torch
    m = seg.build_model("resnet101", pretrained="local")   # 1-channel ImageNet init (RGB summed)
    c = VARIANT_CH[variant]
    if c != 1:
        old = m.encoder.conv1
        new = torch.nn.Conv2d(c, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :1] = old.weight      # image channel keeps the ImageNet filters; extra channels start at 0
        m.encoder.conv1 = new
    return m


# ----------------------------------------------------------------------------- instance formation
def assemble(cand_probs: dict, thr=0.5, min_area=0, merge_iou=0.5):
    """cand_probs: {cid: (y0, x0, prob_crop)} refined full-extent probability per candidate.
    Returns an InstanceSet (disjoint): same-instance duplicates (mask IoU > merge_iou) merged by union-find,
    pixel conflicts resolved by the higher probability, instances < min_area dropped."""
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
    for gi, (_, members) in enumerate(sorted(groups.items()), 1):
        box = (min(masks[c][0] for c in members), min(masks[c][1] for c in members),
               max(masks[c][2] for c in members), max(masks[c][3] for c in members))
        pg = np.max([crop(c, box) for c in members], axis=0)
        y0, x0, y1, x1 = box
        sub_p, sub_i = best_p[y0:y1, x0:x1], best_id[y0:y1, x0:x1]
        upd = (pg > thr) & (pg > sub_p)
        sub_p[upd] = pg[upd]
        sub_i[upd] = gi
    from scipy import ndimage
    objs = ndimage.find_objects(best_id)
    boxes, crops = [], []
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        c = best_id[sl] == k
        if c.sum() < max(min_area, 1):
            continue
        boxes.append((sl[0].start, sl[1].start, sl[0].stop, sl[1].stop))
        crops.append(c)
    return InstanceSet._build((H, W), boxes, crops)
