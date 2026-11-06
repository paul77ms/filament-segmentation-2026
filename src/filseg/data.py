"""COCO / image helpers for the MAGFiLO Kaggle 2026 filament data.

Conventions (verified in scripts/data_audit.py, see docs/data_audit.md):
- Images are 2048x2048 single-channel (PIL mode "L") baseline JPEGs -> read as uint8 HxW.
- annotation["segmentation"] is a list with exactly one polygon [x0, y0, x1, y1, ...] in
  (x, y) = (column, row) order, float pixel coordinates in the pycocotools convention
  (integer coordinates are pixel *corners*: polygon [0,10]^2 rasterises to pixels 0..9).
- Masks are produced exactly as pycocotools does (frPyObjects -> merge -> decode);
  annotation["area"] was computed by the organisers the same way.
- image["id"] = "<GGGG><AA>-<file stem>" (G = annotation batch/group, A = annotator slot).
  The same file can appear under several ids (independent annotation sets).

Never use annotation["spine"] or annotation["category_id"] to build training targets
(extra ground-truth metadata; see docs/problem_contract.md).
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

# Repository root (data, caches and outputs live below it); override with the FILSEG_ROOT environment variable.
PROJECT_ROOT = Path(os.environ.get("FILSEG_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = PROJECT_ROOT / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_JSON = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMG_DIR = DATA_ROOT / "train" / "train_images"
TEST_IMG_DIR = DATA_ROOT / "test" / "test_images"
IMG_H = IMG_W = 2048

# Site code -> GONG site. Only "Bh" = Big Bear is stated on the competition page;
# the others follow the GONG network naming and are an inference, not an official statement.
SITE_NAMES = {
    "Bh": "Big Bear",
    "Ch": "Cerro Tololo",
    "Lh": "Learmonth",
    "Mh": "Mauna Loa",
    "Th": "El Teide",
    "Uh": "Udaipur",
}


# --------------------------------------------------------------------------------------
# JSON / indexing
# --------------------------------------------------------------------------------------
def load_coco(path: str | Path = TRAIN_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def index_coco(coco: dict):
    """Return (images_by_id, anns_by_image_id, image_ids_by_file).

    images_by_id: image_id -> image dict (one per annotation set)
    anns_by_image_id: image_id -> list of annotation dicts (JSON order kept)
    image_ids_by_file: file_name -> sorted list of image_ids (annotation sets of that file)
    """
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image_id: dict[str, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)
    image_ids_by_file: dict[str, list] = defaultdict(list)
    for im in coco["images"]:
        image_ids_by_file[im["file_name"]].append(im["id"])
    for k in image_ids_by_file:
        image_ids_by_file[k].sort()
    return images_by_id, dict(anns_by_image_id), dict(image_ids_by_file)


def group_by_file(coco: dict) -> dict[str, list[dict]]:
    """file_name -> list of image entries (annotation sets), sorted by id."""
    out: dict[str, list] = defaultdict(list)
    for im in coco["images"]:
        out[im["file_name"]].append(im)
    return {k: sorted(v, key=lambda d: d["id"]) for k, v in out.items()}


def parse_filename(file_name: str):
    """'20140609195854Bh.jpeg' -> (datetime(2014,6,9,19,58,54), 'Bh')."""
    stem = Path(file_name).stem
    dt = datetime.strptime(stem[:14], "%Y%m%d%H%M%S")
    site = stem[14:]
    return dt, site


def parse_image_id(image_id: str) -> dict:
    """'040301-20140609195854Bh' -> {'prefix':'040301','group':'0403','annotator':'01','stem':...}."""
    prefix, stem = image_id.split("-", 1)
    return {"prefix": prefix, "group": prefix[:4], "annotator": prefix[4:], "stem": stem}


# --------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------
def read_image(path: str | Path) -> np.ndarray:
    """Read a grayscale JPEG as uint8 HxW (no colour conversion, no EXIF rotation)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise IOError(f"cannot decode {path}")
    if img.ndim == 3:  # not expected for this dataset (all files are 1-channel)
        img = img[..., 0]
    return img


def train_image_path(file_name: str) -> Path:
    return TRAIN_IMG_DIR / file_name


def test_image_path(file_name: str) -> Path:
    return TEST_IMG_DIR / file_name


# --------------------------------------------------------------------------------------
# Polygons / masks
# --------------------------------------------------------------------------------------
def ann_polygon(ann: dict) -> np.ndarray:
    """First (only) polygon as float64 array (N, 2) in (x, y) order."""
    seg = ann["segmentation"]
    if not isinstance(seg, list):
        raise TypeError("segmentation is RLE, not polygon")
    return np.asarray(seg[0], dtype=np.float64).reshape(-1, 2)


def ann_to_rle(ann: dict, h: int = IMG_H, w: int = IMG_W) -> dict:
    """Full-image compressed RLE, exactly as pycocotools/COCO.annToRLE."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_util.frPyObjects(seg, h, w)
        return mask_util.merge(rles)
    if isinstance(seg["counts"], list):  # uncompressed RLE
        return mask_util.frPyObjects(seg, h, w)
    return seg


def ann_to_mask(ann: dict, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    """Full-size uint8 {0,1} mask (4 MB at 2048^2 -- do not keep thousands in memory)."""
    return mask_util.decode(ann_to_rle(ann, h, w))


def polygon_extent(ann: dict):
    p = ann_polygon(ann)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def crop_window(ann: dict, pad: int = 16, align: int = 1, h: int = IMG_H, w: int = IMG_W):
    """Integer window (x0, y0, x1, y1) (x1/y1 exclusive) containing the polygon + pad.

    With align > 1 the window corners are multiples of `align` (useful for
    block-downsampling experiments that must match full-image downsampling).
    """
    xmin, ymin, xmax, ymax = polygon_extent(ann)
    x0 = int(math.floor((xmin - pad) / align) * align)
    y0 = int(math.floor((ymin - pad) / align) * align)
    x1 = int(math.ceil((xmax + 1 + pad) / align) * align)
    y1 = int(math.ceil((ymax + 1 + pad) / align) * align)
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def ann_to_mask_crop(ann: dict, window, exact: bool = True, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    """uint8 {0,1} mask of window (x0, y0, x1, y1) (x1/y1 exclusive).

    exact=True (default): decode the full-image pycocotools mask and slice it -> bit-identical to
    the official rasterisation (4 MB transient allocation).
    exact=False: rasterise the polygon translated by (-x0, -y0) directly at window size (fast).
    NOT bit-exact: pycocotools rounds 5x-upsampled vertex coordinates and the translated float
    coordinates can round differently at .5 boundaries; on the train set 76/8199 annotations
    differ by 1-6 pixels (docs/data_audit.md).  Use only for relative / approximate measures.
    """
    x0, y0, x1, y1 = window
    if exact:
        return np.ascontiguousarray(ann_to_mask(ann, h, w)[y0:y1, x0:x1])
    seg = ann["segmentation"]
    polys = []
    for poly in seg:
        p = np.asarray(poly, dtype=np.float64).copy()
        p[0::2] -= x0
        p[1::2] -= y0
        polys.append(p.tolist())
    rles = mask_util.frPyObjects(polys, y1 - y0, x1 - x0)
    return mask_util.decode(mask_util.merge(rles))


def set_union_rle(anns: list[dict], h: int = IMG_H, w: int = IMG_W):
    rles = [ann_to_rle(a, h, w) for a in anns]
    return mask_util.merge(rles, intersect=False), rles


# --------------------------------------------------------------------------------------
# Solar disk
# --------------------------------------------------------------------------------------
def _kasa_fit(x: np.ndarray, y: np.ndarray):
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = -(x * x + y * y)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r = math.sqrt(max(cx * cx + cy * cy - sol[2], 0.0))
    return cx, cy, r


# Prior disk geometry: rounded median of the fits over all 887 train+test images
# (cx 1022.98, cy 1024.02, R 903.12; docs/data_audit.md).  Only seeds the ray search (+-10 % R).
DISK_PRIOR = (1023.0, 1024.0, 903.0)


def _limb_along_rays(a32: np.ndarray, cx0: float, cy0: float, r_lo: float, r_hi: float, n_rays: int):
    """For each ray: thin-bright-ring peak (1-D white top-hat) then steepest outward drop."""
    h, w = a32.shape
    th = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    rr = np.arange(r_lo, r_hi, 0.5, dtype=np.float32)
    mx = (cx0 + np.cos(th)[:, None] * rr[None, :]).astype(np.float32)
    my = (cy0 + np.sin(th)[:, None] * rr[None, :]).astype(np.float32)
    prof = cv2.remap(a32, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (mx.min(1) >= 0) & (my.min(1) >= 0) & (mx.max(1) <= w - 1) & (my.max(1) <= h - 1)
    kern = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2).astype(np.float32)
    kern /= kern.sum()
    prof = cv2.filter2D(prof, -1, kern[None, :], borderType=cv2.BORDER_REPLICATE)
    # white top-hat along r with a 25-sample (12.5 px) flat element: keeps the 3-6 px limb ring,
    # suppresses the broad disk / off-limb annulus and their step edges
    tophat = cv2.morphologyEx(prof, cv2.MORPH_TOPHAT, np.ones((1, 25), np.uint8))
    imax = tophat.argmax(1)
    peak = tophat[np.arange(n_rays), imax]
    g = np.diff(prof, axis=1)
    redge = np.empty(n_rays, dtype=np.float64)
    for i in range(n_rays):
        j0 = int(imax[i])
        seg = g[i, j0: j0 + 24]
        j = j0 + int(np.argmin(seg)) if seg.size else min(j0, len(rr) - 1)
        redge[i] = rr[min(j, len(rr) - 1)] + 0.25
    return th, redge, valid, peak


def estimate_disk(img: np.ndarray, n_rays: int = 720, prior=DISK_PRIOR, search_frac: float = 0.10) -> dict:
    """Estimate the solar disk (cx, cy, R) in pixel-index coordinates (pixel (row i, col j) -> (x=j, y=i)).

    GONG H-alpha JPEGs show a thin bright ring at the limb, an off-limb annulus whose brightness
    varies a lot between images (sometimes as bright as the limb) and zeros beyond ~1.15 R.
    Method: along n_rays rays from the prior centre, over [(1-s)R0, (1+s)R0], find the thin bright
    ring with a 1-D white top-hat, take the steepest outward drop within 12 px after it (limb =
    outer edge of the ring), robust Kasa circle fit (3*MAD rejection); then repeat the ray search
    around the fitted centre within +-15 px and refit.  Rays leaving the frame are skipped.
    ok=False when fewer than 30 rays survive or the final fit RMS > 4 px.
    """
    a32 = img.astype(np.float32)
    cx, cy, r = prior
    out = {"ok": False}
    for it, (lo, hi) in enumerate([((1 - search_frac), (1 + search_frac)), (None, None), (None, None)]):
        if it == 0:
            r_lo, r_hi = lo * r, hi * r
        else:
            r_lo, r_hi = r - 15.0, r + 15.0
        th, redge, valid, peak = _limb_along_rays(a32, cx, cy, r_lo, r_hi, n_rays)
        ex = cx + np.cos(th) * redge
        ey = cy + np.sin(th) * redge
        sel = valid & (peak > 0)
        if sel.sum() < 30:
            return {"ok": False, "cx": float(cx), "cy": float(cy), "r": float(r), "fit_rms": np.nan,
                    "inlier_frac": 0.0, "valid_rays": int(valid.sum())}
        ncx, ncy, nr = _kasa_fit(ex[sel], ey[sel])
        for _ in range(6):
            res = np.hypot(ex - ncx, ey - ncy) - nr
            mad = np.median(np.abs(res[sel] - np.median(res[sel]))) * 1.4826
            tol = max(3 * mad, 1.5)
            new = valid & (peak > 0) & (np.abs(res) <= tol)
            if new.sum() < 30:
                break
            sel = new
            ncx, ncy, nr = _kasa_fit(ex[sel], ey[sel])
        res = np.hypot(ex[sel] - ncx, ey[sel] - ncy) - nr
        cx, cy, r = ncx, ncy, nr
        out = {
            "cx": float(cx), "cy": float(cy), "r": float(r),
            "fit_rms": float(np.sqrt(np.mean(res ** 2))),
            "inlier_frac": float(sel.sum() / max(valid.sum(), 1)),
            "valid_rays": int(valid.sum()),
        }
    out["ok"] = bool(out["fit_rms"] <= 4.0 and out["inlier_frac"] >= 0.6)
    return out


def disk_radius_map(cx: float, cy: float, h: int = IMG_H, w: int = IMG_W, x0: int = 0, y0: int = 0) -> np.ndarray:
    """Distance of each pixel centre (window offset x0, y0) to the disk centre."""
    yy, xx = np.mgrid[y0: y0 + h, x0: x0 + w].astype(np.float32)
    return np.hypot(xx - cx, yy - cy)
