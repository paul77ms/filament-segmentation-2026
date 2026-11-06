"""Instance-segmentation evaluation for solar filaments (internal reference implementation).

IMPORTANT: this is OUR standard-PQ reference plus diagnostics. The official Kaggle scorer was
not obtainable (kaggle.com unreachable through the proxy); see docs/metrics.md. Two matching
conventions are provided and named distinctly:

  * ``unique``    -- Kirillov et al. 2019 (CVPR, "Panoptic Segmentation") single-class PQ:
                     a (gt, pred) pair is a candidate iff IoU > 0.5 (strict). With
                     non-overlapping segments candidates are unique by construction; if a set
                     overlaps internally (possible here: masks are independent RLEs) and a row or
                     column has >1 candidate, a maximum-total-IoU one-to-one matching over the
                     candidates is used and the event is counted (``n_nonunique_cand``).
  * ``all_pairs`` -- "host-style" as DESCRIBED BY THIRD-PARTY RE-IMPLEMENTATIONS of the
                     organiser's Kaggle "Self-Evaluation Notebook" (not read by us): every pair
                     with IoU > 0.5 is a TP (no one-to-one constraint); FP = preds with no
                     candidate, FN = gts with no candidate; pooled over annotation-set records.

Instances are stored as bbox crops (``InstanceSet``) so that a 2048x2048 image with tens of
instances never needs an N x M loop over full-size masks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

__all__ = [
    "InstanceSet", "as_instance_set", "EvalConfig", "intersection_matrix", "iou_from_intersection",
    "MatchResult", "match_instances", "pq_counts", "split_merge_diagnostics", "thin_region",
    "inner_boundary", "evaluate_image", "evaluate_dataset", "aggregate", "stem_of_record", "build_items", "single_thread_worker",
]


# =========================================================================== instances
@dataclass
class InstanceSet:
    """A set of (possibly overlapping) binary instances on an (H, W) image, stored as bbox crops.

    boxes[i] = (y0, x0, y1, x1), half-open; crops[i] is a bool array of shape (y1-y0, x1-x0).
    Empty instances are kept (box (0,0,0,0), area 0): an empty predicted mask is still a row
    in a submission and is scored as an FP.
    """
    shape: tuple
    boxes: np.ndarray
    crops: list
    areas: np.ndarray

    def __len__(self) -> int:
        return len(self.crops)

    # ---------------------------------------------------------------- constructors
    @classmethod
    def empty(cls, shape=(2048, 2048)) -> "InstanceSet":
        return cls(tuple(int(s) for s in shape), np.zeros((0, 4), np.int64), [], np.zeros(0, np.int64))

    @staticmethod
    def _crop(m: np.ndarray, oy: int = 0, ox: int = 0):
        """Tight bbox crop of mask `m` (which sits at offset (oy, ox) in the full image)."""
        m = np.asarray(m) != 0
        rows = np.flatnonzero(m.any(axis=1))
        if rows.size == 0:
            return (0, 0, 0, 0), np.zeros((0, 0), bool)
        cols = np.flatnonzero(m.any(axis=0))
        y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        return (y0 + oy, x0 + ox, y1 + oy, x1 + ox), np.ascontiguousarray(m[y0:y1, x0:x1])

    @classmethod
    def _crop_rle(cls, r: dict):
        """Tight crop of an RLE dict, decoding only the column band it spans (filseg.rle.rle_to_crop)."""
        from .rle import rle_to_crop
        return rle_to_crop(r)

    @classmethod
    def from_masks(cls, masks, shape=None) -> "InstanceSet":
        """masks: sequence of (H, W) arrays or an (N, H, W) array. Nonzero = foreground."""
        masks = list(masks) if not isinstance(masks, np.ndarray) or masks.ndim == 3 else None
        if masks is None:
            raise TypeError("from_masks expects a sequence of 2-D masks or an (N, H, W) array")
        if shape is None:
            if not masks:
                raise ValueError("shape required for an empty mask list")
            shape = np.asarray(masks[0]).shape
        boxes, crops = [], []
        for m in masks:
            if np.asarray(m).shape != tuple(shape):
                raise ValueError(f"mask shape {np.asarray(m).shape} != {tuple(shape)}")
            b, c = cls._crop(m)
            boxes.append(b)
            crops.append(c)
        return cls._build(shape, boxes, crops)

    @classmethod
    def from_label_map(cls, lab: np.ndarray, ignore=(0,)) -> "InstanceSet":
        """Integer label image (0 = background) -> non-overlapping instances, sorted by label."""
        from scipy import ndimage
        lab = np.asarray(lab)
        if not np.issubdtype(lab.dtype, np.integer):
            raise TypeError("label map must be an integer array")
        ids = [int(i) for i in np.unique(lab) if int(i) not in ignore]
        boxes, crops = [], []
        if ids:
            if min(ids) > 0 and max(ids) <= 1_000_000:
                work, keys = lab, ids                       # find_objects works on labels directly
            else:                                           # remap arbitrary ints to 1..K
                lut = {i: k for k, i in enumerate(ids, 1)}
                uniq, inv = np.unique(lab, return_inverse=True)
                work = np.array([lut.get(int(u), 0) for u in uniq], np.int32)[inv].reshape(lab.shape)
                keys = list(range(1, len(ids) + 1))
            objs = ndimage.find_objects(np.where(np.isin(work, keys), work, 0).astype(np.int64))
            for k in keys:
                ys, xs = objs[k - 1]
                boxes.append((ys.start, xs.start, ys.stop, xs.stop))
                crops.append(np.ascontiguousarray(work[ys, xs] == k))
        return cls._build(lab.shape, boxes, crops)

    @classmethod
    def from_rles(cls, rles: Sequence, h: int = 2048, w: int = 2048) -> "InstanceSet":
        """rles: sequence of compressed counts strings or RLE dicts."""
        from .rle import _rle_dict
        boxes, crops = [], []
        for r in rles:
            b, c = cls._crop_rle(_rle_dict(r, h, w))
            boxes.append(b)
            crops.append(c)
        return cls._build((h, w), boxes, crops)

    @classmethod
    def from_polygons(cls, polygon_lists: Sequence, h: int = 2048, w: int = 2048) -> "InstanceSet":
        """polygon_lists: one COCO polygon list ([[x0,y0,...], ...]) per instance."""
        from .rle import polygons_to_rle
        boxes, crops = [], []
        for polys in polygon_lists:
            b, c = cls._crop_rle(polygons_to_rle(polys, h, w))
            boxes.append(b)
            crops.append(c)
        return cls._build((h, w), boxes, crops)

    @classmethod
    def from_coco_anns(cls, anns: Sequence[dict], h: int = 2048, w: int = 2048) -> "InstanceSet":
        """COCO annotations; segmentation decoded according to its actual type (polygon / RLE)."""
        from .rle import ann_to_rle
        boxes, crops = [], []
        for a in anns:
            b, c = cls._crop_rle(ann_to_rle(a, h, w))
            boxes.append(b)
            crops.append(c)
        return cls._build((h, w), boxes, crops)

    @classmethod
    def _build(cls, shape, boxes, crops) -> "InstanceSet":
        b = np.asarray(boxes, np.int64).reshape(-1, 4)
        areas = np.array([int(np.count_nonzero(c)) for c in crops], np.int64)
        return cls(tuple(int(s) for s in shape), b, list(crops), areas)

    # ---------------------------------------------------------------- views
    def full_mask(self, i: int) -> np.ndarray:
        out = np.zeros(self.shape, bool)
        y0, x0, y1, x1 = self.boxes[i]
        out[y0:y1, x0:x1] = self.crops[i]
        return out

    def paint(self, canvas: np.ndarray, i: int, crop: np.ndarray | None = None) -> None:
        """OR instance i (or a same-shaped derived crop) into a full-size bool canvas."""
        y0, x0, y1, x1 = self.boxes[i]
        if y1 > y0 and x1 > x0:
            canvas[y0:y1, x0:x1] |= self.crops[i] if crop is None else crop

    def union(self) -> np.ndarray:
        out = np.zeros(self.shape, bool)
        for i in range(len(self)):
            self.paint(out, i)
        return out

    def coverage(self) -> np.ndarray:
        """uint16 (H, W): number of instances covering each pixel (overlap diagnostic)."""
        out = np.zeros(self.shape, np.uint16)
        for i in range(len(self)):
            y0, x0, y1, x1 = self.boxes[i]
            if y1 > y0 and x1 > x0:
                out[y0:y1, x0:x1] += self.crops[i]
        return out

    def to_rles(self) -> list[str]:
        from .rle import encode
        return [encode(self.full_mask(i)) for i in range(len(self))]


def as_instance_set(x: Any, shape=(2048, 2048)) -> InstanceSet:
    """Accept InstanceSet | None | sequence of (H, W) masks | (N, H, W) array | sequence of RLE strings.

    A single 2-D array is ambiguous (binary mask vs label map) and is rejected: wrap it with
    InstanceSet.from_label_map or pass [mask].
    """
    if isinstance(x, InstanceSet):
        return x
    if x is None:
        return InstanceSet.empty(shape)
    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            return InstanceSet.from_masks(x, shape=x.shape[1:])
        raise TypeError("2-D array is ambiguous: use InstanceSet.from_label_map(...) or [mask]")
    x = list(x)
    if len(x) == 0:
        return InstanceSet.empty(shape)
    if isinstance(x[0], (str, bytes, dict)):
        return InstanceSet.from_rles(x, *shape)
    return InstanceSet.from_masks(x, shape=np.asarray(x[0]).shape)


# =========================================================================== config
@dataclass(frozen=True)
class EvalConfig:
    iou_thr: float = 0.5          # candidate iff IoU > iou_thr (strict, as in PQ)
    split_tau: float = 0.5        # containment fraction for split/merge diagnostics
    split_min_px: int = 10        # minimum intersection (px) for a fragment / contained gt
    miss_iou: float = 0.1         # gt with best IoU < miss_iou = 'missed'; pred = 'spurious'
    thin_radius: int = 3          # thin region = mask - opening(mask, cv2 ellipse SE (2r+1)^2) -> width < ~2r+1 px
    thin_tol_px: int = 0          # tolerance (px) for thin-region recall/precision coverage
    boundary_tol_px: int = 2      # boundary F-score tolerance d (Euclidean px)
    compute_thin: bool = True
    compute_boundary: bool = True
    compute_pair_boundary: bool = True


# =========================================================================== IoU
def intersection_matrix(a: InstanceSet, b: InstanceSet) -> np.ndarray:
    """(Na, Nb) int64 pixel intersections; only pairs with overlapping bboxes are evaluated."""
    na, nb = len(a), len(b)
    inter = np.zeros((na, nb), np.int64)
    if na == 0 or nb == 0:
        return inter
    B = b.boxes
    for i in range(na):
        if a.areas[i] == 0:
            continue
        y0, x0, y1, x1 = a.boxes[i]
        oy0 = np.maximum(y0, B[:, 0]); ox0 = np.maximum(x0, B[:, 1])
        oy1 = np.minimum(y1, B[:, 2]); ox1 = np.minimum(x1, B[:, 3])
        cand = np.flatnonzero((oy0 < oy1) & (ox0 < ox1) & (b.areas > 0))
        ca = a.crops[i]
        for j in cand:
            sy0, sx0, sy1, sx1 = int(oy0[j]), int(ox0[j]), int(oy1[j]), int(ox1[j])
            pa = ca[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
            by0, bx0 = B[j, 0], B[j, 1]
            pb = b.crops[j][sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0]
            inter[i, j] = np.count_nonzero(pa & pb)
    return inter


def iou_from_intersection(inter: np.ndarray, area_a: np.ndarray, area_b: np.ndarray) -> np.ndarray:
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return iou.astype(np.float64)


# =========================================================================== matching
@dataclass
class MatchResult:
    iou: np.ndarray                 # (Ng, Np)
    inter: np.ndarray               # (Ng, Np) int64
    pairs: list                     # [(g, p, iou)] counted as TP under `mode`
    gt_matched: np.ndarray          # (Ng,) bool
    pred_matched: np.ndarray        # (Np,) bool
    n_candidates: int               # number of pairs with IoU > thr
    unique_by_construction: bool    # every row and column has <= 1 candidate
    mode: str

    @property
    def tp(self) -> int:
        return len(self.pairs)

    @property
    def fp(self) -> int:
        return int((~self.pred_matched).sum())

    @property
    def fn(self) -> int:
        return int((~self.gt_matched).sum())

    @property
    def sum_iou(self) -> float:
        return float(sum(p[2] for p in self.pairs))


def match_instances(gt, pred, iou_thr: float = 0.5, mode: str = "unique",
                    inter: np.ndarray | None = None, shape=(2048, 2048)) -> MatchResult:
    """Match gt and pred instances.

    mode='unique'   : standard PQ (one-to-one). Candidates IoU > thr; if not unique by
                      construction, maximum-total-IoU assignment over candidates.
    mode='all_pairs': host-style per third-party description; every candidate is a TP.
    """
    if mode not in ("unique", "all_pairs"):
        raise ValueError(mode)
    g = as_instance_set(gt, shape)
    p = as_instance_set(pred, g.shape)
    if inter is None:
        inter = intersection_matrix(g, p)
    iou = iou_from_intersection(inter, g.areas, p.areas)
    cand = iou > iou_thr
    n_cand = int(cand.sum())
    unique = bool((cand.sum(1) <= 1).all() and (cand.sum(0) <= 1).all())
    gi, pj = np.nonzero(cand)
    if mode == "all_pairs" or unique:
        pairs = [(int(a), int(b), float(iou[a, b])) for a, b in zip(gi, pj)]
    else:
        from scipy.optimize import linear_sum_assignment
        rows, cols = np.unique(gi), np.unique(pj)
        w = np.where(cand[np.ix_(rows, cols)], iou[np.ix_(rows, cols)], 0.0)
        r, c = linear_sum_assignment(w, maximize=True)
        pairs = [(int(rows[a]), int(cols[b]), float(iou[rows[a], cols[b]]))
                 for a, b in zip(r, c) if cand[rows[a], cols[b]]]
    gm = np.zeros(len(g), bool)
    pm = np.zeros(len(p), bool)
    for a, b, _ in pairs:
        gm[a] = True
        pm[b] = True
    return MatchResult(iou, inter, pairs, gm, pm, n_cand, unique, mode)


def pq_counts(m: MatchResult) -> dict:
    """Per-image PQ terms. pq/sq/rq are NaN when undefined; pq_empty1 = 1.0 if both sets empty."""
    tp, fp, fn, s = m.tp, m.fp, m.fn, m.sum_iou
    den = tp + 0.5 * fp + 0.5 * fn
    pq = s / den if den > 0 else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, sum_iou=s,
                pq=pq, pq_empty1=(pq if den > 0 else 1.0),
                sq=(s / tp if tp > 0 else float("nan")),
                rq=(tp / den if den > 0 else float("nan")))


# =========================================================================== split / merge
def split_merge_diagnostics(inter: np.ndarray, gt_areas: np.ndarray, pred_areas: np.ndarray,
                            tau: float = 0.5, min_px: int = 10) -> dict:
    """Symmetric containment-based fragmentation / over-merge diagnostics.

    fragment[g, p]  : |p ∩ g| >= tau * |p|  and |p ∩ g| >= min_px   (p lies mostly inside g)
    contained[g, p] : |p ∩ g| >= tau * |g|  and |p ∩ g| >= min_px   (g lies mostly inside p)
    split gt g      : >= 2 fragments            (one gt -> many preds)
    merge pred p    : >= 2 contained gts        (many gts -> one pred)
    merge(gt, pred) == split(pred, gt) by construction.
    """
    ng, npred = inter.shape
    if ng == 0 or npred == 0:
        z = np.zeros(ng, np.int64)
        return dict(n_frag=z, split_gt=z.astype(bool), n_cont=np.zeros(npred, np.int64),
                    merge_pred=np.zeros(npred, bool), merged_gt=np.zeros(ng, bool),
                    fragment_pred=np.zeros(npred, bool))
    big = inter >= min_px
    frag = big & (inter >= tau * pred_areas[None, :])
    cont = big & (inter >= tau * gt_areas[:, None])
    n_frag = frag.sum(1)
    split_gt = n_frag >= 2
    n_cont = cont.sum(0)
    merge_pred = n_cont >= 2
    merged_gt = (cont & merge_pred[None, :]).any(1)
    fragment_pred = (frag & split_gt[:, None]).any(0)
    return dict(n_frag=n_frag, split_gt=split_gt, n_cont=n_cont, merge_pred=merge_pred,
                merged_gt=merged_gt, fragment_pred=fragment_pred)


# =========================================================================== morphology helpers
def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def thin_region(crop: np.ndarray, r: int) -> np.ndarray:
    """Pixels of `crop` removed by a morphological opening with the OpenCV elliptical structuring
    element cv2.getStructuringElement(MORPH_ELLIPSE, (2r+1, 2r+1)): parts narrower than about
    2r+1 px (thin branches, narrow necks, sharp tips). Computed from the MASK ONLY.
    NOTE (MV): this element is NOT the Euclidean disk {dy^2+dx^2 <= r^2} for r >= 2 (r=2: 17 vs 13 px,
    r=3: 33 vs 29 px); it is slightly larger. Behaviour unchanged; only the description was corrected."""
    if crop.size == 0 or r <= 0:
        return np.zeros_like(crop, bool)
    pad = r + 1
    m = cv2.copyMakeBorder(crop.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    op = cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(r), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (m.astype(bool) & ~op.astype(bool))[pad:-pad, pad:-pad]


def inner_boundary(crop: np.ndarray) -> np.ndarray:
    """Foreground pixels with a 4-neighbour outside the mask (mask XOR erosion by a 3x3 cross)."""
    if crop.size == 0:
        return np.zeros_like(crop, bool)
    m = cv2.copyMakeBorder(crop.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    er = cv2.erode(m, _CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (m.astype(bool) & ~er.astype(bool))[1:-1, 1:-1]


def _euclid_disk(d: int) -> np.ndarray:
    """Exact Euclidean disk {(dy, dx): dy^2 + dx^2 <= d^2} as a uint8 kernel."""
    yy, xx = np.mgrid[-d:d + 1, -d:d + 1]
    return (yy * yy + xx * xx <= d * d).astype(np.uint8)


def _within(target: np.ndarray, tol: int) -> np.ndarray:
    """Pixels whose Euclidean distance to the nearest True pixel of `target` is <= tol
    (tol=0 -> target itself). Exact on the integer grid (dilation by an exact Euclidean disk),
    evaluated only inside the padded bbox of `target` for speed."""
    if tol <= 0:
        return target
    out = np.zeros(target.shape, bool)
    rows = np.flatnonzero(target.any(1))
    if rows.size == 0:
        return out
    cols = np.flatnonzero(target.any(0))
    H, W = target.shape
    y0, y1 = max(rows[0] - tol, 0), min(rows[-1] + tol + 1, H)
    x0, x1 = max(cols[0] - tol, 0), min(cols[-1] + tol + 1, W)
    sub = target[y0:y1, x0:x1].astype(np.uint8)
    out[y0:y1, x0:x1] = cv2.dilate(sub, _euclid_disk(int(tol)), borderType=cv2.BORDER_CONSTANT,
                                   borderValue=0).astype(bool)
    return out


def _bf(b_gt: np.ndarray, b_pr: np.ndarray, d: int):
    """Boundary precision/recall/F and raw counts at tolerance d (Euclidean, <= d)."""
    n_g, n_p = int(b_gt.sum()), int(b_pr.sum())
    if n_g == 0 and n_p == 0:
        return dict(bf=np.nan, bp=np.nan, br=np.nan, bp_hit=0, bp_n=0, br_hit=0, br_n=0)
    hit_p = int((b_pr & _within(b_gt, d)).sum()) if (n_p and n_g) else 0
    hit_g = int((b_gt & _within(b_pr, d)).sum()) if (n_g and n_p) else 0
    bp = hit_p / n_p if n_p else 0.0
    br = hit_g / n_g if n_g else 0.0
    bf = 2 * bp * br / (bp + br) if (bp + br) > 0 else 0.0
    return dict(bf=bf, bp=bp, br=br, bp_hit=hit_p, bp_n=n_p, br_hit=hit_g, br_n=n_g)


def _pair_boundary_f(g: InstanceSet, i: int, p: InstanceSet, j: int, d: int) -> float:
    """Boundary F of one matched (gt i, pred j) pair, computed on their padded union bbox."""
    by0 = min(g.boxes[i, 0], p.boxes[j, 0]); bx0 = min(g.boxes[i, 1], p.boxes[j, 1])
    by1 = max(g.boxes[i, 2], p.boxes[j, 2]); bx1 = max(g.boxes[i, 3], p.boxes[j, 3])
    pad = d + 1
    H, W = by1 - by0 + 2 * pad, bx1 - bx0 + 2 * pad
    bg = np.zeros((H, W), bool)
    bp = np.zeros((H, W), bool)
    for S, k, canvas in ((g, i, bg), (p, j, bp)):
        y0, x0, y1, x1 = S.boxes[k]
        canvas[y0 - by0 + pad:y1 - by0 + pad, x0 - bx0 + pad:x1 - bx0 + pad] |= inner_boundary(S.crops[k])
    return float(_bf(bg, bp, d)["bf"])


# =========================================================================== per image
def evaluate_image(gt, pred, cfg: EvalConfig = EvalConfig(), image_key: str = "",
                   shape=(2048, 2048), return_instances: bool = True):
    """Evaluate one (annotation set, prediction set) record.

    Returns (row: dict, per_gt: list[dict], per_pred: list[dict]).
    """
    g = as_instance_set(gt, shape)
    p = as_instance_set(pred, g.shape)
    if g.shape != p.shape:
        raise ValueError(f"shape mismatch gt {g.shape} vs pred {p.shape}")
    ng, npr = len(g), len(p)
    inter = intersection_matrix(g, p)
    mu = match_instances(g, p, cfg.iou_thr, "unique", inter=inter)
    mh = match_instances(g, p, cfg.iou_thr, "all_pairs", inter=inter)
    iou = mu.iou
    row: dict = dict(image_key=image_key, n_gt=ng, n_pred=npr)
    row.update(pq_counts(mu))
    row.update({f"{k}_host": v for k, v in pq_counts(mh).items() if k in ("tp", "fp", "fn", "sum_iou", "pq")})
    row["n_candidates"] = mu.n_candidates
    row["n_nonunique_cand"] = mu.n_candidates - mu.tp
    row["unique_by_construction"] = mu.unique_by_construction

    # ---- per-instance best IoU / Dice
    gt_best_iou = iou.max(1) if npr else np.zeros(ng)
    gt_best_pred = iou.argmax(1) if npr else np.full(ng, -1)
    pr_best_iou = iou.max(0) if ng else np.zeros(npr)
    pr_best_gt = iou.argmax(0) if ng else np.full(npr, -1)
    # MV fix: argmax of an all-zero row is 0, which pointed at an unrelated instance; -1 = no overlap
    gt_best_pred = np.where(gt_best_iou > 0, gt_best_pred, -1)
    pr_best_gt = np.where(pr_best_iou > 0, pr_best_gt, -1)
    if npr and ng:
        dice = 2 * inter / np.maximum(g.areas[:, None] + p.areas[None, :], 1)
        gt_best_dice = dice.max(1)
    else:
        gt_best_dice = np.zeros(ng)
    row["gt_best_iou_sum"] = float(gt_best_iou.sum())
    row["gt_best_dice_sum"] = float(gt_best_dice.sum())
    row["pred_best_iou_sum"] = float(pr_best_iou.sum())

    # ---- split / merge / missed / spurious
    sm = split_merge_diagnostics(inter, g.areas, p.areas, cfg.split_tau, cfg.split_min_px)
    missed = gt_best_iou < cfg.miss_iou
    spurious = pr_best_iou < cfg.miss_iou
    row.update(n_split_gt=int(sm["split_gt"].sum()),
               n_split_fragments=int(sm["n_frag"][sm["split_gt"]].sum()),
               n_merge_pred=int(sm["merge_pred"].sum()),
               n_merged_gt=int(sm["merged_gt"].sum()),
               n_missed_gt=int(missed.sum()),
               n_spurious_pred=int(spurious.sum()))
    fn_cause = np.array([""] * ng, dtype=object)
    for k in range(ng):
        if mu.gt_matched[k]:
            continue
        # exclusive cause, precedence: merged > split > missed > low_iou
        fn_cause[k] = ("merged" if sm["merged_gt"][k] else "split" if sm["split_gt"][k]
                       else "missed" if missed[k] else "low_iou")
    fp_cause = np.array([""] * npr, dtype=object)
    for k in range(npr):
        if mu.pred_matched[k]:
            continue
        # exclusive cause, precedence: duplicate > merge > fragment > spurious > low_iou
        fp_cause[k] = ("duplicate" if pr_best_iou[k] > cfg.iou_thr else "merge" if sm["merge_pred"][k]
                       else "fragment" if sm["fragment_pred"][k] else "spurious" if spurious[k]
                       else "low_iou")
    for c in ("missed", "merged", "split", "low_iou"):
        row[f"fn_{c}"] = int((fn_cause == c).sum())
    for c in ("spurious", "duplicate", "merge", "fragment", "low_iou"):
        row[f"fp_{c}"] = int((fp_cause == c).sum())

    # ---- semantic (union) Dice / IoU and overlap diagnostics
    ug, up = g.union(), p.union()
    a_g, a_p, a_i = int(ug.sum()), int(up.sum()), int((ug & up).sum())
    row.update(union_gt_px=a_g, union_pred_px=a_p, union_inter_px=a_i,
               semantic_dice=(2 * a_i / (a_g + a_p) if (a_g + a_p) else np.nan),
               semantic_iou=(a_i / (a_g + a_p - a_i) if (a_g + a_p - a_i) else np.nan),
               gt_overlap_px=int(g.areas.sum()) - a_g,
               pred_overlap_px=int(p.areas.sum()) - a_p)

    # ---- thin region (from masks only)
    gt_thin_px = np.zeros(ng, np.int64); gt_thin_cov = np.zeros(ng, np.int64)
    pr_thin_px = np.zeros(npr, np.int64); pr_thin_in = np.zeros(npr, np.int64)
    if cfg.compute_thin:
        up_tol = _within(up, cfg.thin_tol_px)
        ug_tol = _within(ug, cfg.thin_tol_px)
        tg = np.zeros(g.shape, bool)
        for k in range(ng):
            t = thin_region(g.crops[k], cfg.thin_radius)
            y0, x0, y1, x1 = g.boxes[k]
            gt_thin_px[k] = t.sum()
            gt_thin_cov[k] = (t & up_tol[y0:y1, x0:x1]).sum()
            g.paint(tg, k, t)
        tp_ = np.zeros(p.shape, bool)
        for k in range(npr):
            t = thin_region(p.crops[k], cfg.thin_radius)
            y0, x0, y1, x1 = p.boxes[k]
            pr_thin_px[k] = t.sum()
            pr_thin_in[k] = (t & ug_tol[y0:y1, x0:x1]).sum()
            p.paint(tp_, k, t)
        # pixel-level (union) counts so overlapping instances are not double counted
        row.update(thin_gt_px=int(tg.sum()), thin_gt_covered_px=int((tg & up_tol).sum()),
                   thin_pred_px=int(tp_.sum()), thin_pred_in_gt_px=int((tp_ & ug_tol).sum()))
        row["thin_region_recall"] = row["thin_gt_covered_px"] / row["thin_gt_px"] if row["thin_gt_px"] else np.nan
        row["thin_region_precision"] = (row["thin_pred_in_gt_px"] / row["thin_pred_px"]
                                        if row["thin_pred_px"] else np.nan)

    # ---- boundary F (instance boundaries, image level) + per matched pair
    pair_bf: dict = {}
    if cfg.compute_boundary:
        bg = np.zeros(g.shape, bool)
        bp = np.zeros(p.shape, bool)
        for k in range(ng):
            g.paint(bg, k, inner_boundary(g.crops[k]))
        for k in range(npr):
            p.paint(bp, k, inner_boundary(p.crops[k]))
        b = _bf(bg, bp, cfg.boundary_tol_px)
        row.update(boundary_f=b["bf"], boundary_precision=b["bp"], boundary_recall=b["br"],
                   bnd_pred_hit=b["bp_hit"], bnd_pred_n=b["bp_n"], bnd_gt_hit=b["br_hit"], bnd_gt_n=b["br_n"])
        if cfg.compute_pair_boundary:
            for a, c, _ in mu.pairs:
                pair_bf[(a, c)] = _pair_boundary_f(g, a, p, c, cfg.boundary_tol_px)
            row["tp_boundary_f_sum"] = float(sum(pair_bf.values()))

    per_gt, per_pred = [], []
    if return_instances:
        gt_match = {a: (c, v) for a, c, v in mu.pairs}
        pr_match = {c: (a, v) for a, c, v in mu.pairs}
        for k in range(ng):
            y0, x0, y1, x1 = (int(v) for v in g.boxes[k])
            mp = gt_match.get(k)
            per_gt.append(dict(image_key=image_key, gt_idx=k, area=int(g.areas[k]), bbox_yxyx=(y0, x0, y1, x1),
                               best_iou=float(gt_best_iou[k]), best_dice=float(gt_best_dice[k]),
                               best_pred=int(gt_best_pred[k]), matched=mp is not None,
                               matched_pred=(mp[0] if mp else -1), matched_iou=(mp[1] if mp else np.nan),
                               tp_boundary_f=pair_bf.get((k, mp[0]), np.nan) if mp else np.nan,
                               n_fragments=int(sm["n_frag"][k]), is_split=bool(sm["split_gt"][k]),
                               in_merge=bool(sm["merged_gt"][k]), fn_cause=fn_cause[k],
                               thin_px=int(gt_thin_px[k]), thin_covered_px=int(gt_thin_cov[k])))
        for k in range(npr):
            y0, x0, y1, x1 = (int(v) for v in p.boxes[k])
            mp = pr_match.get(k)
            per_pred.append(dict(image_key=image_key, pred_idx=k, area=int(p.areas[k]), bbox_yxyx=(y0, x0, y1, x1),
                                 best_iou=float(pr_best_iou[k]), best_gt=int(pr_best_gt[k]),
                                 matched=mp is not None, matched_gt=(mp[0] if mp else -1),
                                 n_contained_gt=int(sm["n_cont"][k]), is_merge=bool(sm["merge_pred"][k]),
                                 is_fragment=bool(sm["fragment_pred"][k]), fp_cause=fp_cause[k],
                                 thin_px=int(pr_thin_px[k]), thin_in_gt_px=int(pr_thin_in[k])))
    return row, per_gt, per_pred


# =========================================================================== dataset
def _nanmean(x) -> float:
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _safe(a, b) -> float:
    return float(a / b) if b else float("nan")


def aggregate(df) -> dict:
    """Dataset summary from the per-image DataFrame (one row per annotation-set record)."""
    s = df.sum(numeric_only=True)
    out = dict(n_records=int(len(df)), n_gt=int(s["n_gt"]), n_pred=int(s["n_pred"]))
    for suf, (tp, fp, fn, si) in {"": ("tp", "fp", "fn", "sum_iou"),
                                  "_host": ("tp_host", "fp_host", "fn_host", "sum_iou_host")}.items():
        den = s[tp] + 0.5 * s[fp] + 0.5 * s[fn]
        out[f"PQ_pooled{suf}"] = _safe(s[si], den)
        out[f"SQ_pooled{suf}"] = _safe(s[si], s[tp])
        out[f"RQ_pooled{suf}"] = _safe(s[tp], den)
        out[f"TP{suf}"], out[f"FP{suf}"], out[f"FN{suf}"] = int(s[tp]), int(s[fp]), int(s[fn])
    out["PQ_img_mean"] = _nanmean(df["pq"])               # records with no gt and no pred excluded
    out["PQ_img_mean_empty1"] = float(df["pq_empty1"].mean())  # such records count as 1.0
    out["PQ_img_mean_host"] = _nanmean(df["pq_host"])
    out["n_records_both_empty"] = int(df["pq"].isna().sum())
    out["n_nonunique_cand"] = int(s["n_nonunique_cand"])
    out["n_records_nonunique"] = int((~df["unique_by_construction"].astype(bool)).sum())
    out["gt_best_iou_mean"] = _safe(s["gt_best_iou_sum"], s["n_gt"])
    out["gt_best_dice_mean"] = _safe(s["gt_best_dice_sum"], s["n_gt"])
    out["pred_best_iou_mean"] = _safe(s["pred_best_iou_sum"], s["n_pred"])
    out["semantic_dice_pooled_px"] = _safe(2 * s["union_inter_px"], s["union_gt_px"] + s["union_pred_px"])
    out["semantic_dice_img_mean"] = _nanmean(df["semantic_dice"])
    out["semantic_iou_img_mean"] = _nanmean(df["semantic_iou"])
    for c in ("n_split_gt", "n_split_fragments", "n_merge_pred", "n_merged_gt", "n_missed_gt", "n_spurious_pred",
              "fn_missed", "fn_merged", "fn_split", "fn_low_iou",
              "fp_spurious", "fp_duplicate", "fp_merge", "fp_fragment", "fp_low_iou",
              "gt_overlap_px", "pred_overlap_px"):
        out[c] = int(s[c])
    if "thin_gt_px" in s:
        out["thin_region_recall_pooled_px"] = _safe(s["thin_gt_covered_px"], s["thin_gt_px"])
        out["thin_region_precision_pooled_px"] = _safe(s["thin_pred_in_gt_px"], s["thin_pred_px"])
        out["thin_region_recall_img_mean"] = _nanmean(df["thin_region_recall"])
        out["thin_region_precision_img_mean"] = _nanmean(df["thin_region_precision"])
    if "bnd_gt_n" in s:
        bp = _safe(s["bnd_pred_hit"], s["bnd_pred_n"])
        br = _safe(s["bnd_gt_hit"], s["bnd_gt_n"])
        out["boundary_precision_pooled"], out["boundary_recall_pooled"] = bp, br
        out["boundary_f_pooled"] = (2 * bp * br / (bp + br)) if (bp + br) > 0 else float("nan")
        out["boundary_f_img_mean"] = _nanmean(df["boundary_f"])
        if "tp_boundary_f_sum" in s:
            out["tp_boundary_f_mean"] = _safe(s["tp_boundary_f_sum"], s["tp"])
    return out


def stem_of_record(record_id: str) -> str:
    """'040301-20140609195854Bh' -> '20140609195854Bh' (annotation-set record id -> image stem)."""
    return record_id.split("-", 1)[1] if "-" in record_id else record_id


def build_items(gt_records: dict, preds_by_stem: dict, shape=(2048, 2048)) -> list:
    """Pair every GT annotation-set record with the prediction of its image stem.

    gt_records: {record_id: gt instances}; preds_by_stem: {stem: pred instances}.
    A stem without predictions is scored as an empty prediction (all FN). Predictions for
    stems without any GT record are ignored here (they cannot be scored).
    """
    return [(rid, g, preds_by_stem.get(stem_of_record(rid), InstanceSet.empty(shape)))
            for rid, g in gt_records.items()]


def single_thread_worker() -> None:
    """Pool initializer: keep each worker on one core (OpenCV otherwise spawns a thread pool
    per process, which oversubscribes the shared machine)."""
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits  # optional
        threadpool_limits(1)
    except Exception:
        pass


def _eval_item(item, cfg: EvalConfig, return_instances: bool, shape=(2048, 2048)):
    key, gt, pred = item
    return evaluate_image(gt, pred, cfg, image_key=key, shape=shape, return_instances=return_instances)


def evaluate_dataset(items: Iterable, cfg: EvalConfig = EvalConfig(), *, n_workers: int = 1,
                     return_instances: bool = False, shape=(2048, 2048)):
    """items: iterable of (image_key, gt, pred). One item per ANNOTATION SET (record); when an
    image has several annotation sets, pass the same prediction once per set (this matches the
    third-party description of the host notebook). gt/pred: anything `as_instance_set` accepts.
    ``shape`` is the image size used for inputs that carry no shape (None / empty lists); added
    by MV so that non-2048 inputs with an empty side do not raise a shape mismatch.

    Returns (summary: dict, per_image: DataFrame) or, with return_instances=True,
    (summary, per_image, per_gt, per_pred).
    """
    import pandas as pd

    fn = partial(_eval_item, cfg=cfg, return_instances=return_instances, shape=tuple(shape))
    rows, pg, pp = [], [], []
    if n_workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(int(n_workers), 16), initializer=single_thread_worker) as pool:
            results = pool.imap(fn, items, chunksize=1)
            for r, a, b in results:
                rows.append(r); pg.extend(a); pp.extend(b)
    else:
        for it in items:
            r, a, b = fn(it)
            rows.append(r); pg.extend(a); pp.extend(b)
    df = pd.DataFrame(rows)
    summary = aggregate(df) if len(df) else {}
    summary["config"] = asdict(cfg)
    if return_instances:
        return summary, df, pd.DataFrame(pg), pd.DataFrame(pp)
    return summary, df
