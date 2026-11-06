#!/usr/bin/env python
"""Task D1 -- data audit (statistics) for MAGFiLO Kaggle 2026 train/test.

CPU only.  Outputs (outputs/audit/):
  images.csv       one row per train annotation set (image entry) + file-level image stats
  instances.csv    one row per train annotation
  test_images.csv  one row per test image (image stats only)
  audit_stats.json aggregated statistics used in docs/data_audit.md

Memory: masks are handled per annotation as compressed RLE or bbox crops; at most one
full-size decoded mask per worker at a time.  'spine' and 'category_id' are only counted.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/data_audit.py [--workers 16] [--limit N]
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root (path-independent)
import os
import sys

# glibc malloc tuning: on this shared host first-touch page faults are slow (kcompactd);
# keeping freed memory in the heap gives ~4x speed-up for repeated 2048^2 allocations.
_ENV = {
    "MALLOC_MMAP_THRESHOLD_": "4294967296",
    "MALLOC_TRIM_THRESHOLD_": "68719476736",
    "MALLOC_TOP_PAD_": "268435456",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
if os.environ.get("FILSEG_AUDIT_REEXEC") != "1":
    _env = dict(os.environ)
    _env.update(_ENV)
    _env["FILSEG_AUDIT_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)

import argparse
import hashlib
import io
import json
import math
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_util
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.data import (  # noqa: E402
    IMG_H,
    IMG_W,
    TEST_IMG_DIR,
    TRAIN_IMG_DIR,
    TRAIN_JSON,
    ann_polygon,
    ann_to_mask_crop,
    ann_to_rle,
    crop_window,
    disk_radius_map,
    estimate_disk,
    group_by_file,
    index_coco,
    load_coco,
    parse_filename,
    parse_image_id,
)

cv2.setNumThreads(1)
OUT = Path(_REPO + "/outputs/audit")
SHIFTS = np.array([-2, -1, -0.5, 0, 0.5, 1, 2], dtype=np.float64)
DS_FACTORS = (2, 4, 8)
SQRT2 = math.sqrt(2.0)
RING_PX = 5

# globals for workers
G = {}


def _init_worker(anns_by_image_id, sets_by_file):
    cv2.setNumThreads(1)
    G["anns"] = anns_by_image_id
    G["sets"] = sets_by_file


# --------------------------------------------------------------------------------------
# image level
# --------------------------------------------------------------------------------------
def image_stats(path: Path):
    raw = path.read_bytes()
    row = {
        "file_name": path.name,
        "file_bytes": len(raw),
        "md5": hashlib.md5(raw).hexdigest(),
        "jpeg_eoi_ok": raw[-2:] == b"\xff\xd9",
    }
    dt, site = parse_filename(path.name)
    row["datetime"] = dt.isoformat(sep=" ")
    row["site"] = site
    # PIL: mode, EXIF, truncation
    try:
        im = Image.open(io.BytesIO(raw))
        row["pil_format"] = im.format
        row["pil_mode"] = im.mode
        row["pil_progressive"] = int(bool(im.info.get("progressive", 0) or im.info.get("progression", 0)))
        exif = im.getexif()
        row["exif_n_tags"] = len(exif)
        row["exif_orientation"] = exif.get(274, None)
        im.load()
        pil_arr = np.asarray(im)
        row["pil_decode_ok"] = True
    except Exception as e:  # corrupt / truncated
        row["pil_decode_ok"] = False
        row["pil_error"] = repr(e)
        pil_arr = None
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
    row["cv2_decode_ok"] = a is not None
    if a is None:
        return row, None, None
    row["h"], row["w"] = a.shape[:2]
    row["channels"] = 1 if a.ndim == 2 else a.shape[2]
    row["dtype"] = str(a.dtype)
    if a.ndim == 3:
        a = a[..., 0]
    col = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    row["color_decode_max_ch_diff"] = int(max(np.abs(col[..., 0].astype(np.int16) - col[..., 1]).max(),
                                              np.abs(col[..., 0].astype(np.int16) - col[..., 2]).max()))
    row["color_vs_gray_max_diff"] = int(np.abs(col[..., 0].astype(np.int16) - a).max())
    row["pil_vs_cv2_max_diff"] = int(np.abs(pil_arr.astype(np.int16) - a).max()) if pil_arr is not None and pil_arr.shape == a.shape else -1
    del col, pil_arr
    row["min"] = int(a.min())
    row["max"] = int(a.max())
    row["mean"] = float(a.mean())
    row["std"] = float(a.std())
    row["frac_0"] = float((a == 0).mean())
    row["frac_255"] = float((a == 255).mean())

    disk = estimate_disk(a)
    row["disk_ok"] = bool(disk.get("ok", False))
    for k in ("cx", "cy", "r", "fit_rms", "inlier_frac", "valid_rays"):
        row[f"disk_{k}"] = disk.get(k, np.nan)
    if not disk.get("ok", False):
        return row, a, disk
    cx, cy, R = disk["cx"], disk["cy"], disk["r"]
    h, w = a.shape
    row["disk_in_frame"] = bool(cx - R >= -0.5 and cy - R >= -0.5 and cx + R <= w - 0.5 and cy + R <= h - 0.5)
    r = disk_radius_map(cx, cy, h, w)
    in_disk = r <= R
    row["off_disk_frac"] = float(1.0 - in_disk.mean())
    row["disk_frac_outside_frame"] = float(max(0.0, 1.0 - in_disk.sum() / (math.pi * R * R)))
    inner = r <= 0.9 * R
    v = a[inner].astype(np.float32)
    row["disk_mean"] = float(v.mean())
    row["disk_median"] = float(np.median(v))
    row["disk_p1"], row["disk_p99"] = (float(x) for x in np.percentile(v, [1, 99]))
    row["disk_std"] = float(v.std())
    row["disk_rel_std"] = row["disk_std"] / max(row["disk_mean"], 1e-6)
    row["disk_frac_saturated"] = float((v >= 255).mean())
    annulus = (r > R + 3) & (r < 1.1 * R)
    row["offlimb_mean"] = float(a[annulus].mean())
    centre = r < 0.3 * R
    edge = (r > 0.85 * R) & (r < 0.9 * R)
    row["limb_darkening_ratio"] = float(np.median(a[edge]) / max(np.median(a[centre]), 1e-6))
    # sharpness: Laplacian variance inside 0.9 R (raw and brightness-normalised)
    lap = cv2.Laplacian(a.astype(np.float32), cv2.CV_32F, ksize=3)
    lv = float(lap[inner].var())
    row["lap_var"] = lv
    row["lap_var_norm"] = lv / max(row["disk_median"], 1e-6) ** 2 * 1e4
    del lap, r, in_disk, inner
    # flattened (radial-profile-normalised) image on a 4x block-averaged copy
    f = 4
    small = cv2.resize(a, (w // f, h // f), interpolation=cv2.INTER_AREA).astype(np.float32)
    rs = disk_radius_map((cx - (f - 1) / 2) / f, (cy - (f - 1) / 2) / f, h // f, w // f)
    Rs = R / f
    bins = np.floor(rs).astype(np.int32)
    on = rs <= 0.98 * Rs
    nb = int(math.ceil(0.98 * Rs)) + 1
    prof = np.asarray(ndi.median(small, labels=np.where(on, bins + 1, 0), index=np.arange(1, nb + 1)), dtype=np.float32)
    prof = np.nan_to_num(prof, nan=np.nanmedian(prof))
    flat = np.ones_like(small)
    flat[on] = small[on] / np.maximum(prof[bins[on]], 1e-3)
    in85 = rs <= 0.85 * Rs
    in90 = rs <= 0.9 * Rs
    row["flat_rms"] = float(flat[in90].std())  # small-scale+large-scale relative contrast
    row["flat_p1"], row["flat_p99"] = (float(x) for x in np.percentile(flat[in90], [1, 99]))
    big = cv2.GaussianBlur(flat, (0, 0), 16)  # sigma 16 small px = 64 full-res px
    row["nonuniformity"] = float(big[in85].std())
    row["nonuniformity_range"] = float(np.percentile(big[in85], 99) - np.percentile(big[in85], 1))
    hp = flat - big
    row["flat_hp_rms"] = float(hp[in85].std())  # texture contrast after removing large scale
    return row, a, disk


# --------------------------------------------------------------------------------------
# polygon checks
# --------------------------------------------------------------------------------------
def _orient(p, q, r):
    return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])


def _onseg(p, q, r):
    return (np.minimum(p[..., 0], q[..., 0]) <= r[..., 0]) & (r[..., 0] <= np.maximum(p[..., 0], q[..., 0])) & \
           (np.minimum(p[..., 1], q[..., 1]) <= r[..., 1]) & (r[..., 1] <= np.maximum(p[..., 1], q[..., 1]))


def polygon_checks(P: np.ndarray) -> dict:
    out = {"n_vertices": int(len(P))}
    out["closed_explicit"] = bool(len(P) > 1 and np.all(P[0] == P[-1]))
    d = np.diff(P, axis=0)
    out["n_dup_consecutive"] = int(np.sum(np.all(d == 0, axis=1)))
    Q = P[np.r_[True, np.any(d != 0, axis=1)]]
    if len(Q) > 1 and np.all(Q[0] == Q[-1]):
        Q = Q[:-1]
    m = len(Q)
    out["n_unique_vertices"] = int(m)
    u, cnt = np.unique(Q, axis=0, return_counts=True)
    out["n_repeated_vertices"] = int(np.sum(cnt > 1))
    # shoelace signed area (polygon area, independent of rasterisation)
    x, y = Q[:, 0], Q[:, 1]
    sa = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    out["poly_signed_area"] = sa
    if m < 4:
        out["n_self_cross"] = 0
        out["n_self_touch"] = 0
        return out
    A = Q
    B = np.roll(Q, -1, axis=0)
    I, J = np.triu_indices(m, k=2)
    keep = ~((I == 0) & (J == m - 1))
    I, J = I[keep], J[keep]
    p1, p2, p3, p4 = A[I], B[I], A[J], B[J]
    # cheap bbox rejection
    bb = (np.maximum(np.minimum(p1[:, 0], p2[:, 0]), np.minimum(p3[:, 0], p4[:, 0])) <=
          np.minimum(np.maximum(p1[:, 0], p2[:, 0]), np.maximum(p3[:, 0], p4[:, 0]))) & \
         (np.maximum(np.minimum(p1[:, 1], p2[:, 1]), np.minimum(p3[:, 1], p4[:, 1])) <=
          np.minimum(np.maximum(p1[:, 1], p2[:, 1]), np.maximum(p3[:, 1], p4[:, 1])))
    p1, p2, p3, p4 = p1[bb], p2[bb], p3[bb], p4[bb]
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    proper = (d1 * d2 < 0) & (d3 * d4 < 0)
    touch = ~proper & (((d1 == 0) & _onseg(p3, p4, p1)) | ((d2 == 0) & _onseg(p3, p4, p2)) |
                       ((d3 == 0) & _onseg(p1, p2, p3)) | ((d4 == 0) & _onseg(p1, p2, p4)))
    out["n_self_cross"] = int(proper.sum())
    out["n_self_touch"] = int(touch.sum())
    return out


# --------------------------------------------------------------------------------------
# mask morphology
# --------------------------------------------------------------------------------------
_K8 = np.ones((3, 3), np.uint8)


def _neighbors(s: np.ndarray):
    """s bool (H,W) with zero border -> list of 8 shifted arrays in circular order N,NE,E,SE,S,SW,W,NW."""
    p = np.pad(s, 1)
    H, W = s.shape
    offs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    return [p[1 + dy: 1 + dy + H, 1 + dx: 1 + dx + W] for dy, dx in offs]


def skeleton_metrics(mask: np.ndarray, dt: np.ndarray) -> dict:
    sk = skeletonize(mask.astype(bool))
    out = {"skel_px": int(sk.sum())}
    s = sk
    orth = int((s[:, :-1] & s[:, 1:]).sum() + (s[:-1, :] & s[1:, :]).sum())
    d1 = s[:-1, :-1] & s[1:, 1:] & ~(s[:-1, 1:] | s[1:, :-1])
    d2 = s[:-1, 1:] & s[1:, :-1] & ~(s[:-1, :-1] | s[1:, 1:])
    out["skel_len"] = float(orth + SQRT2 * (int(d1.sum()) + int(d2.sum())))
    nbrs = _neighbors(s)
    nb = np.sum(nbrs, axis=0)
    cn = np.zeros(s.shape, np.int32)
    for k in range(8):
        cn += (~nbrs[k]) & nbrs[(k + 1) % 8]
    endp = s & (cn == 1)
    junc = s & (cn >= 3)
    out["n_endpoints"] = int(endp.sum())
    wmed = float(np.median(2.0 * dt[sk] - 1.0)) if sk.any() else 0.0
    if junc.any():
        jd = cv2.dilate(junc.astype(np.uint8), _K8)
        nj, _ = cv2.connectedComponents(jd, connectivity=8)
        out["n_branchpoints"] = int(nj - 1)
        rest = (s & (jd == 0)).astype(np.uint8)
        nl, lab, st, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
        jd2 = cv2.dilate(jd, _K8).astype(bool)
        touch_lbl = set(np.unique(lab[jd2 & (rest > 0)]).tolist()) - {0}
        end_lbl = set(np.unique(lab[endp & (rest > 0)]).tolist()) - {0}
        term = [st[l, cv2.CC_STAT_AREA] for l in (touch_lbl & end_lbl)]
        out["n_terminal_branches"] = int(len(term))
        out["n_terminal_branches_ge10"] = int(sum(t >= 10 for t in term))
        out["n_terminal_branches_ge20"] = int(sum(t >= 20 for t in term))
        # width-relative filter: spurs created by polygon corners of wide bodies are ~w/2 long
        out["n_terminal_branches_gt_w"] = int(sum(t >= max(10.0, wmed) for t in term))
    else:
        out["n_branchpoints"] = 0
        out["n_terminal_branches"] = 0
        out["n_terminal_branches_ge10"] = 0
        out["n_terminal_branches_ge20"] = 0
        out["n_terminal_branches_gt_w"] = 0
    if out["skel_px"] > 0:
        lw = 2.0 * dt[sk] - 1.0  # local width (px) at skeleton pixel
        out["width_dt_med"] = float(np.median(lw))
        out["width_dt_p10"] = float(np.percentile(lw, 10))
        out["width_dt_p90"] = float(np.percentile(lw, 90))
        out["width_dt_max"] = float(lw.max())
        out["frac_skel_w_lt4"] = float((lw < 4).mean())
        out["frac_skel_w_lt8"] = float((lw < 8).mean())
    else:
        for k in ("width_dt_med", "width_dt_p10", "width_dt_p90", "width_dt_max", "frac_skel_w_lt4", "frac_skel_w_lt8"):
            out[k] = np.nan
    return out, sk


def holes_and_cc(mask: np.ndarray) -> dict:
    n8, _ = cv2.connectedComponents(mask, connectivity=8)
    n4, _ = cv2.connectedComponents(mask, connectivity=4)
    bg = (mask == 0).astype(np.uint8)
    nb, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    holes = [l for l in range(1, nb) if l not in border]
    return {
        "n_cc8": int(n8 - 1),
        "n_cc4": int(n4 - 1),
        "n_holes": int(len(holes)),
        "hole_area": int(sum(st[l, cv2.CC_STAT_AREA] for l in holes)),
    }


_RING_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))


def contrast(img_crop: np.ndarray, own: np.ndarray, exclude: np.ndarray | None):
    dil = cv2.dilate(own, _RING_K)
    ring = (dil > 0) & (own == 0)
    if exclude is not None:
        r2 = ring & (exclude == 0)
        if r2.sum() >= 10:
            ring = r2
    ins = img_crop[own > 0]
    rv = img_crop[ring]
    if ins.size == 0 or rv.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    im, rm, rs = float(ins.mean()), float(rv.mean()), float(rv.std())
    return im, rm, rm - im, (rm - im) / max(rs, 1e-6)


def crop_of(ann_like_poly: np.ndarray, pad: int, align: int = 1):
    """window for an arbitrary (N,2) polygon."""
    fake = {"segmentation": [ann_like_poly.reshape(-1).tolist()]}
    return crop_window(fake, pad=pad, align=align), fake


# --------------------------------------------------------------------------------------
# per train file worker
# --------------------------------------------------------------------------------------
def process_train_file(file_name: str):
    t0 = time.time()
    img_row, img, disk = image_stats(TRAIN_IMG_DIR / file_name)
    imgf = img.astype(np.float32)
    cx, cy, R = disk["cx"], disk["cy"], disk["r"]
    set_rows, inst_rows, shift_rows = [], [], []
    for image_id in G["sets"][file_name]:
        anns = G["anns"][image_id]
        rles = [ann_to_rle(a) for a in anns]
        areas = np.array([float(mask_util.area(r)) for r in rles])
        union = mask_util.merge(rles, intersect=False)
        union_area = float(mask_util.area(union))
        union_full = mask_util.decode(union)
        n = len(anns)
        iou = np.asarray(mask_util.iou(rles, rles, [0] * n)) if n > 1 else np.zeros((1, 1))
        inter = np.zeros((n, n))
        if n > 1:
            inter = iou * (areas[:, None] + areas[None, :]) / (1 + iou)
            np.fill_diagonal(inter, 0)
            np.fill_diagonal(iou, 0)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if iou[i, j] > 0]
        pid = parse_image_id(image_id)
        srow = {
            "image_id": image_id,
            "file_name": file_name,
            "group": pid["group"],
            "annotator": pid["annotator"],
            "n_instances": n,
            "sum_area": float(areas.sum()),
            "union_area": union_area,
            "union_frac_image": union_area / (IMG_H * IMG_W),
            "union_frac_disk": union_area / (math.pi * R * R),
            "double_counted_px": float(areas.sum() - union_area),
            "n_overlap_pairs": len(pairs),
            "n_overlap_pairs_iou_gt_0p5": int(sum(iou[i, j] > 0.5 for i, j in pairs)),
            "max_pair_iou": float(iou.max()) if n > 1 else 0.0,
        }
        cat = Counter(a["category_id"] for a in anns)
        for c in (1, 2, 3, 4):
            srow[f"n_cat{c}"] = cat.get(c, 0)
        set_rows.append(srow)

        for i, (a, rle) in enumerate(zip(anns, rles)):
            P = ann_polygon(a)
            row = {"image_id": image_id, "file_name": file_name, "ann_id": a["id"], "ann_idx": i,
                   "category_id": a["category_id"], "n_spine_pts": len(a["spine"]) // 2,
                   "seg_type": "polygon" if isinstance(a["segmentation"], list) else "rle",
                   "n_polygons": len(a["segmentation"]) if isinstance(a["segmentation"], list) else 0,
                   "iscrowd": a["iscrowd"]}
            row.update(polygon_checks(P))
            row["area_json"] = float(a["area"])
            row["area"] = float(areas[i])
            row["area_diff"] = row["area"] - row["area_json"]
            bj = np.asarray(a["bbox"], dtype=np.float64)
            pe = np.array([P[:, 0].min(), P[:, 1].min(), P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min()])
            mb = np.asarray(mask_util.toBbox(rle), dtype=np.float64)
            row.update({"bbox_x": bj[0], "bbox_y": bj[1], "bbox_w": bj[2], "bbox_h": bj[3],
                        "bbox_vs_poly_maxdiff": float(np.abs(bj - pe).max()),
                        "bbox_vs_mask_maxdiff": float(np.abs(bj - mb).max()),
                        "mask_bbox_x": mb[0], "mask_bbox_y": mb[1], "mask_bbox_w": mb[2], "mask_bbox_h": mb[3]})
            row["bbox_json_contains_mask"] = bool(bj[0] <= mb[0] + 1 and bj[1] <= mb[1] + 1 and
                                                  bj[0] + bj[2] >= mb[0] + mb[2] - 1 and bj[1] + bj[3] >= mb[1] + mb[3] - 1)
            row["bbox_aspect"] = float(max(bj[2], bj[3]) / max(min(bj[2], bj[3]), 1e-6))
            row["bbox_diag"] = float(math.hypot(bj[2], bj[3]))
            # crop aligned to 8 so block downsampling matches full-image downsampling
            win = crop_window(a, pad=16, align=8)
            x0, y0, x1, y1 = win
            full = mask_util.decode(rle)
            own = np.ascontiguousarray(full[y0:y1, x0:x1])  # exact pycocotools rasterisation
            row["crop_contains_mask"] = bool(int(own.sum()) == int(areas[i]))
            del full
            fast = ann_to_mask_crop(a, win, exact=False)  # translated-polygon rasterisation
            row["crop_fast_ndiff"] = int((fast != own).sum())
            del fast
            row.update(holes_and_cc(own))
            dt = cv2.distanceTransform(own, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            sm, sk = skeleton_metrics(own, dt)
            row.update(sm)
            row["mean_width"] = row["area"] / max(row["skel_len"], 1.0)
            row["mean_width_px_count"] = row["area"] / max(row["skel_px"], 1)
            row["max_inscribed_diam"] = float(2 * dt.max() - 1) if own.any() else np.nan
            # limb geometry
            ys, xs = np.nonzero(own)
            gx, gy = xs + x0, ys + y0
            rr = np.hypot(gx - cx, gy - cy)
            ccx, ccy = float(gx.mean()), float(gy.mean())
            rc = math.hypot(ccx - cx, ccy - cy)
            row.update({"centroid_x": ccx, "centroid_y": ccy, "centroid_r_frac": rc / R,
                        "dist_to_limb_px": R - rc, "min_dist_to_limb_px": float(R - rr.max()),
                        "off_disk_px": int((rr > R).sum())})
            # contrast (inside vs 5 px ring, ring excludes all instances of this set)
            ic = imgf[y0:y1, x0:x1]
            uc = union_full[y0:y1, x0:x1]
            im_, rm_, c_, snr_ = contrast(ic, own, uc)
            row.update({"mean_inside": im_, "mean_ring": rm_, "contrast": c_,
                        "rel_contrast": c_ / rm_ if rm_ == rm_ and rm_ > 0 else np.nan, "contrast_snr": snr_})
            _, _, c0, _ = contrast(ic, own, None)
            row["contrast_noexcl"] = c0
            # swapped (y, x) hypothesis
            Psw = P[:, ::-1].copy()
            wsw, fsw = crop_of(Psw, pad=16)
            msw = ann_to_mask_crop(fsw, wsw, exact=False)
            _, _, csw, _ = contrast(imgf[wsw[1]:wsw[3], wsw[0]:wsw[2]], msw, None)
            row["contrast_swapped_noexcl"] = csw
            # sub-pixel / pixel shift scan of mean inside intensity
            sh = np.full((len(SHIFTS), len(SHIFTS)), np.nan)
            for iy, dy in enumerate(SHIFTS):
                for ix, dx in enumerate(SHIFTS):
                    Ps = (P + np.array([dx, dy])).reshape(-1).tolist()
                    ms = ann_to_mask_crop({"segmentation": [Ps]}, win, exact=False)
                    if ms.any():
                        sh[iy, ix] = float(ic[ms > 0].mean())
            shift_rows.append(sh - sh[3, 3])
            # overlaps within the same annotation set
            if n > 1:
                others = mask_util.merge([rles[j] for j in range(n) if j != i], intersect=False)
                ov = float(mask_util.area(mask_util.merge([rle, others], intersect=True)))
                row["overlap_with_other_in_set_px"] = ov
                row["n_overlap_partners"] = int((iou[i] > 0).sum())
                row["max_iou_with_other_in_set"] = float(iou[i].max())
            else:
                row["overlap_with_other_in_set_px"] = 0.0
                row["n_overlap_partners"] = 0
                row["max_iou_with_other_in_set"] = 0.0
            # resolution: best-case reconstruction from a 1/f-resolution soft mask
            m = own.astype(np.float32)
            H, W = m.shape
            for fct in DS_FACTORS:
                low = cv2.resize(m, (W // fct, H // fct), interpolation=cv2.INTER_AREA)
                up = cv2.resize(low, (W, H), interpolation=cv2.INTER_LINEAR) >= 0.5
                nn = cv2.resize((low >= 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                o = own > 0
                row[f"ds{fct}_iou_soft"] = float((up & o).sum() / max((up | o).sum(), 1))
                row[f"ds{fct}_iou_hard"] = float((nn & o).sum() / max((nn | o).sum(), 1))
                row[f"ds{fct}_skel_recall_soft"] = float(up[sk].mean()) if sk.any() else np.nan
                row[f"ds{fct}_vanish"] = bool(low.max() < 0.5)
            inst_rows.append(row)
        del union_full
    img_row["n_sets"] = len(G["sets"][file_name])
    img_row["proc_sec"] = time.time() - t0
    return img_row, set_rows, inst_rows, np.stack(shift_rows) if shift_rows else np.zeros((0, 7, 7))


def process_test_file(file_name: str):
    t0 = time.time()
    row, _, _ = image_stats(TEST_IMG_DIR / file_name)
    row["proc_sec"] = time.time() - t0
    return row


# --------------------------------------------------------------------------------------
def pct(x, qs=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {"mean": float(x.mean()), "n": int(x.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only first N train/test files (smoke test)")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time()

    coco = load_coco(TRAIN_JSON)
    images_by_id, anns_by_image_id, ids_by_file = index_coco(coco)
    train_files = sorted(ids_by_file)
    test_files = sorted(os.listdir(TEST_IMG_DIR))
    disk_train = sorted(os.listdir(TRAIN_IMG_DIR))
    if args.limit:
        train_files = train_files[: args.limit]
        test_files = test_files[: max(1, args.limit // 4)]
    workers = min(args.workers, 16)

    # largest files first for load balance
    order = sorted(train_files, key=lambda f: -sum(len(anns_by_image_id[i]) for i in ids_by_file[f]))
    img_rows, set_rows, inst_rows, shifts = [], [], [], []
    with Pool(workers, initializer=_init_worker, initargs=(anns_by_image_id, ids_by_file)) as pool:
        for k, (ir, sr, inr, sh) in enumerate(pool.imap_unordered(process_train_file, order, chunksize=1)):
            img_rows.append(ir)
            set_rows += sr
            inst_rows += inr
            shifts.append(sh)
            if (k + 1) % 50 == 0:
                print(f"train {k + 1}/{len(order)}  {time.time() - T0:.0f}s", flush=True)
        test_rows = list(pool.imap_unordered(process_test_file, test_files, chunksize=2))
    print(f"workers done {time.time() - T0:.0f}s", flush=True)

    img_df = pd.DataFrame(img_rows).set_index("file_name")
    set_df = pd.DataFrame(set_rows)
    inst_df = pd.DataFrame(inst_rows)
    test_df = pd.DataFrame(test_rows).sort_values("file_name").reset_index(drop=True)
    shifts = np.concatenate(shifts, 0)

    # ---------------- per-file multi-set info + join image stats ----------------
    n_inst = set_df.set_index("image_id")["n_instances"]
    file_info = {}
    for f in train_files:
        ids = ids_by_file[f]
        file_info[f] = {"n_sets_for_file": len(ids), "set_ids_for_file": ";".join(ids),
                        "inst_counts_for_file": ";".join(str(int(n_inst[i])) for i in ids),
                        "groups_for_file": ";".join(sorted({parse_image_id(i)["group"] for i in ids}))}
    fi = pd.DataFrame.from_dict(file_info, orient="index")
    set_df = set_df.join(fi, on="file_name")
    set_df["datetime"] = [images_by_id[i]["date_captured"] for i in set_df["image_id"]]
    img_cols = [c for c in img_df.columns if c not in ("n_sets",)]
    set_df = set_df.join(img_df[img_cols].add_prefix("img_"), on="file_name")
    set_df = set_df.sort_values(["file_name", "image_id"]).reset_index(drop=True)
    inst_df = inst_df.sort_values(["file_name", "image_id", "ann_idx"]).reset_index(drop=True)

    set_df.to_csv(out / "images.csv", index=False)
    inst_df.to_csv(out / "instances.csv", index=False)

    # ---------------- time / site analysis ----------------
    def dts(files):
        return pd.to_datetime(pd.Series([parse_filename(f)[0] for f in files]))

    tr_t = dts(train_files).values.astype("datetime64[s]").astype(np.int64)
    te_t = dts(test_df["file_name"]).values.astype("datetime64[s]").astype(np.int64)
    tr_sorted = np.sort(tr_t)
    def nearest_gap(t, pool_sorted, exclude_self=False):
        j = np.searchsorted(pool_sorted, t)
        cands = []
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < len(pool_sorted):
                d = abs(pool_sorted[jj] - t)
                if exclude_self and d == 0:
                    continue
                cands.append(d)
        return min(cands) if cands else np.nan
    test_df["nearest_train_gap_h"] = [nearest_gap(t, tr_sorted) / 3600 for t in te_t]
    test_df.to_csv(out / "test_images.csv", index=False)
    # train-train: gap to nearest *other* train file
    tt_gap_h = []
    for t in tr_t:
        j = np.searchsorted(tr_sorted, t)
        c = [abs(tr_sorted[jj] - t) for jj in (j - 2, j - 1, j, j + 1, j + 2) if 0 <= jj < len(tr_sorted)]
        c = sorted(c)
        tt_gap_h.append((c[1] if len(c) > 1 else np.nan) / 3600)  # c[0] == 0 is itself
    tt_gap_h = np.array(tt_gap_h)
    tr_days = Counter(str(parse_filename(f)[0].date()) for f in train_files)
    te_days = Counter(str(parse_filename(f)[0].date()) for f in test_df["file_name"])

    # ---------------- aggregate stats ----------------
    S = {"runtime_sec": None, "n_workers": workers}
    S["json"] = {
        "n_image_entries": len(coco["images"]), "n_annotations": len(coco["annotations"]),
        "n_unique_files": len(ids_by_file), "n_files_on_disk_train": len(disk_train),
        "files_in_json_not_on_disk": sorted(set(ids_by_file) - set(disk_train)),
        "files_on_disk_not_in_json": sorted(set(disk_train) - set(ids_by_file)),
        "info": coco["info"], "categories": coco["categories"], "licenses": coco["licenses"],
        "image_keys": sorted(coco["images"][0].keys()), "annotation_keys": sorted(coco["annotations"][0].keys()),
        "date_captured_matches_filename": int(sum(
            parse_filename(im["file_name"])[0].strftime("%Y-%m-%d %H:%M:%S") == im["date_captured"] for im in coco["images"])),
        "image_id_stem_matches_filename": int(sum(parse_image_id(im["id"])["stem"] == Path(im["file_name"]).stem for im in coco["images"])),
        "image_entries_without_annotations": int(sum(1 for im in coco["images"] if im["id"] not in anns_by_image_id)),
        "n_unique_prefixes": len({parse_image_id(im["id"])["prefix"] for im in coco["images"]}),
        "n_unique_groups": len({parse_image_id(im["id"])["group"] for im in coco["images"]}),
        "sets_per_file": dict(Counter(len(v) for v in ids_by_file.values())),
        "train_test_filename_overlap": len(set(disk_train) & set(test_files)),
    }
    # group consistency for multi-set files
    multi = {f: ids_by_file[f] for f in train_files if len(ids_by_file[f]) > 1}
    S["json"]["multi_set_files_same_group"] = int(sum(len({parse_image_id(i)["group"] for i in ids}) == 1 for ids in multi.values()))
    S["json"]["multi_set_files_distinct_annotator_slots"] = int(sum(len({parse_image_id(i)["annotator"] for i in ids}) == len(ids) for ids in multi.values()))
    S["json"]["files_per_prefix"] = pct([sum(1 for im in coco["images"] if parse_image_id(im["id"])["prefix"] == p)
                                          for p in {parse_image_id(im["id"])["prefix"] for im in coco["images"]}])
    S["json"]["annotator_slot_counts"] = dict(Counter(parse_image_id(im["id"])["annotator"] for im in coco["images"]))
    cnt_diff = []
    for f, ids in multi.items():
        c = [int(n_inst[i]) for i in ids]
        cnt_diff.append(max(c) - min(c))
    S["json"]["multi_set_instance_count_range"] = pct(cnt_diff) | {"frac_equal_counts": float(np.mean(np.array(cnt_diff) == 0))}

    def img_summary(df):
        d = {"n": len(df)}
        for c in ("file_bytes", "min", "max", "mean", "std", "frac_0", "frac_255", "disk_cx", "disk_cy", "disk_r",
                  "disk_fit_rms", "disk_inlier_frac", "off_disk_frac", "disk_frac_outside_frame", "disk_median",
                  "disk_rel_std", "offlimb_mean", "limb_darkening_ratio", "lap_var", "lap_var_norm", "flat_rms",
                  "flat_hp_rms", "nonuniformity", "nonuniformity_range", "flat_p1"):
            if c in df:
                d[c] = pct(df[c])
        for c in ("pil_mode", "pil_format", "channels", "dtype", "exif_orientation", "exif_n_tags", "pil_progressive",
                  "jpeg_eoi_ok", "pil_decode_ok", "cv2_decode_ok", "color_decode_max_ch_diff", "color_vs_gray_max_diff",
                  "pil_vs_cv2_max_diff", "disk_ok", "disk_in_frame", "site"):
            if c in df:
                d[c] = {str(k): int(v) for k, v in Counter(df[c].tolist()).items()}
        d["shape"] = {str(k): int(v) for k, v in Counter(zip(df["h"], df["w"])).items()}
        d["year"] = {str(k): int(v) for k, v in sorted(Counter(pd.to_datetime(df["datetime"]).dt.year).items())}
        d["n_duplicate_md5"] = int(df["md5"].duplicated().sum())
        return d

    tr_img = img_df.reset_index()
    S["train_images"] = img_summary(tr_img)
    S["test_images"] = img_summary(test_df)
    S["train_test_md5_overlap"] = len(set(tr_img["md5"]) & set(test_df["md5"]))
    # outliers (robust z) for quality proxies, train+test
    allimg = pd.concat([tr_img.assign(split="train"), test_df.assign(split="test")], ignore_index=True)
    outl = {}
    for c, sign in (("lap_var_norm", -1), ("flat_hp_rms", -1), ("nonuniformity", 1), ("disk_fit_rms", 1),
                    ("offlimb_mean", 1), ("disk_rel_std", 1)):
        v = allimg[c].astype(float)
        med = v.median()
        mad = 1.4826 * (v - med).abs().median()
        z = sign * (v - med) / max(mad, 1e-9)
        allimg[f"z_{c}"] = z
        top = allimg.loc[z.sort_values(ascending=False).index[:10], ["file_name", "split", c]]
        outl[c] = {"median": float(med), "mad": float(mad), "n_z_gt_3": int((z > 3).sum()), "n_z_gt_5": int((z > 5).sum()),
                   "worst10": top.to_dict(orient="records")}
    S["image_quality_outliers"] = outl
    S["time"] = {
        "test_nearest_train_gap_h": pct(test_df["nearest_train_gap_h"]),
        "test_with_train_within_1h": int((test_df["nearest_train_gap_h"] <= 1).sum()),
        "test_with_train_within_24h": int((test_df["nearest_train_gap_h"] <= 24).sum()),
        "test_with_train_same_day": int(sum(1 for f in test_df["file_name"] if str(parse_filename(f)[0].date()) in tr_days)),
        "train_nearest_other_train_gap_h": pct(tt_gap_h),
        "train_files_with_other_train_within_1h": int((tt_gap_h <= 1).sum()),
        "train_files_with_other_train_within_24h": int((tt_gap_h <= 24).sum()),
        "train_days_with_multiple_files": int(sum(v > 1 for v in tr_days.values())),
        "train_n_days": len(tr_days), "test_n_days": len(te_days),
        "train_range": [str(min(parse_filename(f)[0] for f in train_files)), str(max(parse_filename(f)[0] for f in train_files))],
        "test_range": [str(min(parse_filename(f)[0] for f in test_df["file_name"])), str(max(parse_filename(f)[0] for f in test_df["file_name"]))],
    }

    # annotation-level
    I = inst_df
    A = {}
    A["seg_type"] = dict(Counter(I["seg_type"]))
    A["n_polygons"] = {str(k): int(v) for k, v in Counter(I["n_polygons"]).items()}
    A["iscrowd"] = {str(k): int(v) for k, v in Counter(I["iscrowd"]).items()}
    A["category"] = {str(k): int(v) for k, v in sorted(Counter(I["category_id"]).items())}
    A["n_spine_pts"] = pct(I["n_spine_pts"])
    A["n_spine_pts_lt2"] = int((I["n_spine_pts"] < 2).sum())
    A["n_vertices"] = pct(I["n_vertices"])
    A["closed_explicit"] = int(I["closed_explicit"].sum())
    A["with_dup_consecutive"] = int((I["n_dup_consecutive"] > 0).sum())
    A["with_repeated_vertices"] = int((I["n_repeated_vertices"] > 0).sum())
    A["with_self_cross"] = int((I["n_self_cross"] > 0).sum())
    A["self_cross_count"] = pct(I.loc[I["n_self_cross"] > 0, "n_self_cross"])
    A["with_self_touch"] = int((I["n_self_touch"] > 0).sum())
    A["poly_orientation_ccw_in_image_coords"] = int((I["poly_signed_area"] > 0).sum())
    A["area_exact_match"] = int((I["area_diff"] == 0).sum())
    A["area_abs_diff"] = pct(I["area_diff"].abs())
    A["area_rel_diff"] = pct((I["area_diff"].abs() / I["area_json"].clip(lower=1)))
    A["bbox_vs_poly_maxdiff"] = pct(I["bbox_vs_poly_maxdiff"])
    A["bbox_vs_mask_maxdiff"] = pct(I["bbox_vs_mask_maxdiff"])
    A["crop_contains_mask_all"] = bool(I["crop_contains_mask"].all())
    A["crop_fast_n_inexact"] = int((I["crop_fast_ndiff"] > 0).sum())
    A["crop_fast_ndiff_when_inexact"] = pct(I.loc[I["crop_fast_ndiff"] > 0, "crop_fast_ndiff"])
    A["bbox_json_ne_poly_extent"] = int((I["bbox_vs_poly_maxdiff"] > 1e-3).sum())
    A["bbox_json_ne_poly_extent_gt1px"] = int((I["bbox_vs_poly_maxdiff"] > 1).sum())
    A["bbox_json_contains_mask"] = int(I["bbox_json_contains_mask"].sum())
    A["area"] = pct(I["area"])
    A["area_lt"] = {str(t): int((I["area"] < t).sum()) for t in (10, 50, 100, 200, 500, 1000)}
    A["zero_area"] = int((I["area"] == 0).sum())
    A["bbox_w"] = pct(I["bbox_w"])
    A["bbox_h"] = pct(I["bbox_h"])
    A["bbox_long_side"] = pct(np.maximum(I["bbox_w"], I["bbox_h"]))
    A["bbox_aspect"] = pct(I["bbox_aspect"])
    A["n_cc8"] = {str(k): int(v) for k, v in sorted(Counter(I["n_cc8"]).items())}
    A["n_cc4"] = {str(k): int(v) for k, v in sorted(Counter(I["n_cc4"]).items())}
    A["n_holes"] = {str(k): int(v) for k, v in sorted(Counter(I["n_holes"]).items())}
    A["hole_area_when_present"] = pct(I.loc[I["n_holes"] > 0, "hole_area"])
    A["skel_len"] = pct(I["skel_len"])
    A["mean_width"] = pct(I["mean_width"])
    A["mean_width_lt"] = {str(t): float((I["mean_width"] < t).mean()) for t in (2, 3, 4, 6, 8, 12, 16)}
    A["width_dt_med"] = pct(I["width_dt_med"])
    A["width_dt_med_lt"] = {str(t): float((I["width_dt_med"] < t).mean()) for t in (2, 3, 4, 6, 8, 12, 16)}
    A["width_dt_p10"] = pct(I["width_dt_p10"])
    A["max_inscribed_diam"] = pct(I["max_inscribed_diam"])
    A["frac_skel_w_lt4"] = pct(I["frac_skel_w_lt4"])
    A["frac_skel_w_lt8"] = pct(I["frac_skel_w_lt8"])
    # pixel-weighted share of skeleton length that is thin
    sk = I["skel_px"].to_numpy(float)
    A["skel_weighted_frac_w_lt4"] = float(np.nansum(I["frac_skel_w_lt4"] * sk) / sk.sum())
    A["skel_weighted_frac_w_lt8"] = float(np.nansum(I["frac_skel_w_lt8"] * sk) / sk.sum())
    A["n_endpoints"] = pct(I["n_endpoints"])
    A["n_branchpoints"] = pct(I["n_branchpoints"])
    A["n_branchpoints_dist"] = {str(k): int(v) for k, v in sorted(Counter(I["n_branchpoints"].clip(upper=10)).items())}
    A["n_terminal_branches_ge10"] = pct(I["n_terminal_branches_ge10"])
    A["n_terminal_branches_gt_w"] = pct(I["n_terminal_branches_gt_w"])
    A["n_terminal_branches_gt_w_dist"] = {str(k): int(v) for k, v in sorted(Counter(I["n_terminal_branches_gt_w"].clip(upper=10)).items())}
    A["n_terminal_branches_ge10_dist"] = {str(k): int(v) for k, v in sorted(Counter(I["n_terminal_branches_ge10"].clip(upper=10)).items())}
    A["dist_to_limb_px"] = pct(I["dist_to_limb_px"])
    A["centroid_r_frac"] = pct(I["centroid_r_frac"])
    A["min_dist_to_limb_px"] = pct(I["min_dist_to_limb_px"])
    A["n_within_50px_of_limb"] = int((I["min_dist_to_limb_px"] < 50).sum())
    A["n_within_100px_of_limb"] = int((I["min_dist_to_limb_px"] < 100).sum())
    A["n_with_off_disk_px"] = int((I["off_disk_px"] > 0).sum())
    A["contrast"] = pct(I["contrast"])
    A["rel_contrast"] = pct(I["rel_contrast"])
    A["contrast_snr"] = pct(I["contrast_snr"])
    A["frac_inside_darker"] = float((I["contrast"] > 0).mean())
    A["contrast_noexcl"] = pct(I["contrast_noexcl"])
    A["contrast_swapped_noexcl"] = pct(I["contrast_swapped_noexcl"])
    A["frac_swapped_darker_than_orig"] = float((I["contrast_swapped_noexcl"] > I["contrast_noexcl"]).mean())
    A["frac_swapped_inside_darker"] = float((I["contrast_swapped_noexcl"] > 0).mean())
    # shift scan
    sm = np.nanmean(shifts, 0)
    smed = np.nanmedian(shifts, 0)
    am = Counter()
    for s in shifts:
        if np.isfinite(s).all():
            iy, ix = np.unravel_index(np.argmin(s), s.shape)
            am[(float(SHIFTS[ix]), float(SHIFTS[iy]))] += 1
    A["shift_scan"] = {"shifts": SHIFTS.tolist(), "rows_dy_cols_dx_mean_delta_inside": sm.round(4).tolist(),
                       "rows_dy_cols_dx_median_delta_inside": smed.round(4).tolist(),
                       "argmin_counts_top": [[list(k), v] for k, v in am.most_common(10)]}
    A["overlap"] = {
        "instances_with_overlap": int((I["overlap_with_other_in_set_px"] > 0).sum()),
        "overlap_px_when_present": pct(I.loc[I["overlap_with_other_in_set_px"] > 0, "overlap_with_other_in_set_px"]),
        "overlap_frac_of_area_when_present": pct((I["overlap_with_other_in_set_px"] / I["area"].clip(lower=1))[I["overlap_with_other_in_set_px"] > 0]),
        "max_iou_when_present": pct(I.loc[I["max_iou_with_other_in_set"] > 0, "max_iou_with_other_in_set"]),
        "sets_with_any_overlap": int((set_df["n_overlap_pairs"] > 0).sum()),
        "total_overlap_pairs": int(set_df["n_overlap_pairs"].sum()),
        "pairs_iou_gt_0p5": int(set_df["n_overlap_pairs_iou_gt_0p5"].sum()),
    }
    ds = {}
    for f in DS_FACTORS:
        ds[str(f)] = {
            "iou_soft": pct(I[f"ds{f}_iou_soft"]), "iou_hard": pct(I[f"ds{f}_iou_hard"]),
            "skel_recall_soft": pct(I[f"ds{f}_skel_recall_soft"]),
            "vanish": int(I[f"ds{f}_vanish"].sum()),
            "iou_soft_lt_0p5": int((I[f"ds{f}_iou_soft"] < 0.5).sum()),
            "iou_soft_lt_0p8": int((I[f"ds{f}_iou_soft"] < 0.8).sum()),
            "iou_hard_lt_0p5": int((I[f"ds{f}_iou_hard"] < 0.5).sum()),
            "area_weighted_iou_soft": float((I[f"ds{f}_iou_soft"] * I["area"]).sum() / I["area"].sum()),
        }
    A["downsample_best_case"] = ds
    S["annotations"] = A
    # per set
    S["sets"] = {
        "n_instances": pct(set_df["n_instances"]),
        "n_instances_dist": {str(k): int(v) for k, v in sorted(Counter(set_df["n_instances"].clip(upper=30)).items())},
        "union_frac_image": pct(set_df["union_frac_image"]),
        "union_frac_disk": pct(set_df["union_frac_disk"]),
        "union_area": pct(set_df["union_area"]),
    }
    S["proc_sec_per_train_file"] = pct(img_df["proc_sec"])
    S["runtime_sec"] = time.time() - T0
    with open(out / "audit_stats.json", "w") as fh:
        json.dump(S, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"done in {S['runtime_sec']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
