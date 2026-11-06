"""Rubric diagnostics that replicate the organisers' self-evaluation notebook, plus named Dice variants.

Reference = the third-party VERBATIM copy of the organisers' "Self_Evaluation_Notebook",
``docs/thirdparty_metric/ooLemon0-0_self-evaluation-notebook_cells.py`` (provenance in its header and in
docs/sources_rules.md §2.2). "l.N" below = line N of that file. That copy is not proven to equal the
current Kaggle notebook or the hidden leaderboard scorer.

What the notebook shows besides PQ, and what this module returns for it
------------------------------------------------------------------------
(a) IoU and Dice histograms  (``plot_distribution`` l.318-371, called in cells 9/10, l.491/l.494, on
    ``overlap_df["iou_matrix"]`` / ``overlap_df["dice_matrix"]``):
      * records  = one row per GT annotator-image (``get_overlap_df`` l.240-245: the unique
        ``<annotator>-<stem>`` keys of the GT rows); each row is compared with ALL predictions of its image
        stem (``process_entry`` l.205, l.212), so a stem with k annotation sets contributes k rows that
        share one prediction set. Predictions on stems without GT rows are never looked at.
      * values   = the full n_gt x n_pred matrices of every row, raveled and concatenated in row order
        (l.336-338), then only values > 0 are kept (l.340): i.e. every (GT, pred) pair that overlaps
        by >= 1 px. IoU = |g∩p| / |g∪p| (l.96-100), Dice = 2|g∩p| / (|g|+|p|) (l.102-106), both 0 when the
        union is 0, computed in **float32** (layers are float32, l.181; torch matmul l.89).
      * binning  = ``np.histogram`` with 50 equal bins on [0, 1] (``np.linspace(0, 1, 51)``, l.342-347;
        numpy bins are [a, b) except the last, [0.98, 1.0], which is closed). The integer-bin branch for
        values > 1 (l.348-350) cannot trigger for IoU/Dice but is replicated.
      * aggregation = pooled over all pairs of all records; bar height = count / total count (l.353), so
        the bars sum to 1; the marked "Mean" is the plain mean of the pooled float32 values (l.359-360),
        i.e. pair-weighted, not a per-record mean.
(b) one-to-many / many-to-one overlap degrees (``plot_m2n_counts`` l.374-474, cell 11 l.497):
      * hit = float32 IoU > 0 (l.406), i.e. intersection >= 1 px (no IoU threshold, no containment);
      * per GT: number of overlapping predictions (row sums, l.408; label "1:n", l.459);
        per prediction: number of overlapping GTs (column sums, l.409; label "n:1", l.459);
      * a row with n_pred == 0 adds degree 0 for each of its GTs (l.402-404); a row with n_gt == 0 adds
        degree 0 for each of its predictions (l.398-400);
      * aggregation = ``Counter`` over all GTs / all predictions of all records (l.411-412); legend
        counts: GT hits = #GT with degree > 0, GT misses = degree 0, prediction hits / misses likewise
        (l.414-417).
    These counts are NOT the containment-based split/merge counters of ``filseg.metrics``.

Empty-GT records: in the notebook a record exists only if it has GT rows, so an annotation set with zero
filaments never appears (l.240-245) and its predictions are ignored. ``aggregate_diagnostics`` reproduces
this by default (``skip_empty_gt=True``, skipped records are counted). With ``skip_empty_gt=False`` such a
record is treated the way ``get_pq_score`` / ``plot_m2n_counts`` treat a row with n_gt == 0 (l.292-294,
l.398-400). All 1154 train annotation sets have >= 1 filament (docs/metrics_verification.md §3.3).

Exactness: intersections/areas are exact integers (``filseg.metrics.intersection_matrix``); the float32
IoU/Dice are formed exactly as torch does (float32 sums of integers < 2^24 are exact; one correctly-rounded
float32 division), so the values are bit-identical to ``get_overlap_matrices`` whenever |g| + |p| < 2^24
(always true for 2048 x 2048 images). Verified in tests/test_diagnostics.py against the copied notebook
function ``get_overlap_matrices`` (pure computation, executed on torch CPU).

Also returned: ``notebook_pq`` = the ``get_pq_score`` (l.261-315) value from the same float32 matrices
(cross-check against ``filseg.metrics`` ``PQ_pooled_host``).

Dice variants (our definitions, named explicitly; NONE is claimed to be the rubric's "mean Dice")
------------------------------------------------------------------------------------------------
  * ``dice_semantic_per_record``    per record: 2|U_g ∩ U_p| / (|U_g| + |U_p|) on the union masks of the
                                    record's GT set and prediction set; NaN when both unions are empty.
                                    Summary: mean over records with a defined value (+ the count excluded).
  * ``dice_semantic_pooled_px``     2 Σ_r |U_g ∩ U_p| / Σ_r (|U_g| + |U_p|) over records (pixel-pooled).
  * ``dice_matched_pairs_mean``     mean of the exact (float64) instance Dice over all (GT, pred) pairs
                                    with IoU > 0.5, pooled over records (pair-weighted). The pair set is
                                    the notebook's TP set (l.300: every pair with float32 IoU > 0.5, no
                                    one-to-one constraint); with pixel-disjoint instances in each set these
                                    pairs are automatically one-to-one.
  * the histogram mean of (a) (``dice_hist["mean"]``) is the notebook's own displayed mean Dice over all
    overlapping pairs.

``dice_torchmetrics_like`` is deliberately NOT provided: the rubric only names
``torchmetrics.segmentation.DiceScore``. Its value depends on constructor arguments (number of classes,
whether background is included, the averaging mode, the input format, the zero-division / empty-sample
handling and, in recent releases, the aggregation level) and on which tensors the organisers feed it
(per-image foreground union? per-instance masks? background channel?). None of this is published, and
torchmetrics is not installed in this project's .venv (checked 2026-10-27), so no configuration could be
checked against the library either; installing it into the shared environment used by running training
jobs was not done. Any function under that name would therefore be one of the variants above with a
misleading label.

API
---
``overlap_matrices_f32(gt, pred, inter=None) -> (iou32, dice32)``
``record_diagnostics(gt, pred, record_id="") -> RecordDiagnostics``
``notebook_scores(recs, "iou"|"dice") -> float64 array``  ``notebook_histogram(scores) -> dict``
``notebook_overlap_degrees(recs) -> (gt_degrees, pred_degrees)``  ``degree_distribution(degrees, side) -> dict``
``notebook_pq(recs) -> dict``  ``dice_semantic_per_record(rec)``  ``dice_semantic_pooled_px(recs)``
``dice_matched_pairs_mean(recs) -> (mean, n_pairs)``
``aggregate_diagnostics(items, skip_empty_gt=True, n_workers=1) -> (summary: dict, frames: dict[str, DataFrame])``
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Iterable

import numpy as np

from .metrics import InstanceSet, as_instance_set, intersection_matrix, single_thread_worker

__all__ = [
    "NOTEBOOK_COPY", "HIST_N_BINS", "RecordDiagnostics", "overlap_matrices_f32", "record_diagnostics",
    "notebook_scores", "notebook_histogram", "notebook_overlap_degrees", "degree_distribution",
    "notebook_pq", "dice_semantic_per_record", "dice_semantic_pooled_px", "dice_matched_pairs_mean",
    "aggregate_diagnostics", "to_jsonable",
]

NOTEBOOK_COPY = "docs/thirdparty_metric/ooLemon0-0_self-evaluation-notebook_cells.py"
HIST_N_BINS = 50            # l.345
NB_IOU_THRESHOLD = 0.5      # l.281 (strict '>' at l.300)
_F32_EXACT = 2 ** 24        # float32 represents every integer below this exactly


# =========================================================================== per record
@dataclass
class RecordDiagnostics:
    """Everything the notebook-style diagnostics need from one (GT set, prediction set) record."""
    record_id: str
    n_gt: int
    n_pred: int
    gt_areas: np.ndarray        # (n_gt,) int64
    pred_areas: np.ndarray      # (n_pred,) int64
    inter: np.ndarray           # (n_gt, n_pred) int64 exact intersections
    iou32: np.ndarray           # (n_gt, n_pred) float32, == notebook iou_matrix
    dice32: np.ndarray          # (n_gt, n_pred) float32, == notebook dice_matrix
    union_gt_px: int            # |U_g|
    union_pred_px: int          # |U_p|
    union_inter_px: int         # |U_g ∩ U_p|


def overlap_matrices_f32(gt, pred, inter: np.ndarray | None = None, shape=(2048, 2048)):
    """float32 (iou, dice) matrices, bit-identical to the notebook's ``get_overlap_matrices`` (l.56-108).

    Notebook: intersection = float32 matmul of 0/1 layers (l.89), areas = float32 row sums (l.91-92),
    union = areas_g + areas_p - intersection (l.94), iou = where(union == 0, 0, inter / union) (l.96-100),
    dice = where(union == 0, 0, 2 * inter / (areas_g + areas_p)) (l.102-106). All sums are of integers
    < 2^24 and therefore exact in float32; the only rounding is the final float32 division, reproduced here.
    """
    g = as_instance_set(gt, shape)
    p = as_instance_set(pred, g.shape)
    if inter is None:
        inter = intersection_matrix(g, p)
    if len(g) and len(p) and int(g.areas.max()) + int(p.areas.max()) >= _F32_EXACT:
        raise ValueError("areas too large for an exact float32 replica (|g| + |p| >= 2^24)")
    i32 = np.asarray(inter, np.int64).astype(np.float32)
    a32 = g.areas.astype(np.float32)[:, None]
    b32 = p.areas.astype(np.float32)[None, :]
    union = a32 + b32 - i32
    zero = np.float32(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union == 0, zero, i32 / union).astype(np.float32)
        dice = np.where(union == 0, zero, (np.float32(2.0) * i32) / (a32 + b32)).astype(np.float32)
    return iou, dice


def _union_counts(g: InstanceSet, p: InstanceSet):
    """(|U_g|, |U_p|, |U_g ∩ U_p|), painted only inside the joint bbox of all non-empty instances."""
    boxes = [g.boxes[k] for k in range(len(g)) if g.areas[k] > 0] + \
            [p.boxes[k] for k in range(len(p)) if p.areas[k] > 0]
    if not boxes:
        return 0, 0, 0
    B = np.asarray(boxes, np.int64)
    y0, x0, y1, x1 = int(B[:, 0].min()), int(B[:, 1].min()), int(B[:, 2].max()), int(B[:, 3].max())
    ug = np.zeros((y1 - y0, x1 - x0), bool)
    up = np.zeros_like(ug)
    for S, canvas in ((g, ug), (p, up)):
        for k in range(len(S)):
            if S.areas[k] == 0:
                continue
            by0, bx0, by1, bx1 = (int(v) for v in S.boxes[k])
            canvas[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] |= S.crops[k]
    return int(ug.sum()), int(up.sum()), int((ug & up).sum())


def record_diagnostics(gt, pred, record_id: str = "", shape=(2048, 2048)) -> RecordDiagnostics:
    """Per-record overlap data (one notebook ``overlap_df`` row, l.205-220, plus union-mask counts)."""
    g = as_instance_set(gt, shape)
    p = as_instance_set(pred, g.shape)
    if g.shape != p.shape:
        raise ValueError(f"shape mismatch gt {g.shape} vs pred {p.shape}")
    inter = intersection_matrix(g, p)
    iou, dice = overlap_matrices_f32(g, p, inter=inter)
    ug, up, ui = _union_counts(g, p)
    return RecordDiagnostics(str(record_id), len(g), len(p), g.areas.astype(np.int64).copy(),
                             p.areas.astype(np.int64).copy(), inter, iou, dice, ug, up, ui)


# =========================================================================== (a) histograms
def notebook_scores(recs: Iterable[RecordDiagnostics], which: str = "iou") -> np.ndarray:
    """Pooled float64 copy of the float32 scores > 0, in notebook order (l.336-340): records in the given
    order, each matrix raveled row-major (GT-major)."""
    if which not in ("iou", "dice"):
        raise ValueError(which)
    mats = [np.asarray(getattr(r, f"{which}32"), dtype=float).ravel() for r in recs]
    scores = np.concatenate(mats or [np.zeros(0)])
    return scores[scores > 0]


def notebook_histogram(scores: np.ndarray) -> dict:
    """Histogram exactly as ``plot_distribution`` computes it (l.342-360), returned instead of drawn.

    Returns bin_edges (51 values for IoU/Dice), counts, probabilities (= counts / counts.sum(), l.353),
    mean (= scores.mean(), l.360; NaN when there is no overlapping pair, where the notebook draws no mean
    line) and n_pairs.
    """
    scores = np.asarray(scores, dtype=float)
    x_max = 1.0 if scores.size == 0 or scores.max() <= 1.0 else scores.max()       # l.342
    if x_max <= 1.0:
        bins = np.linspace(0.0, x_max, HIST_N_BINS + 1)                            # l.345-346
    else:
        bins = np.arange(0, int(x_max) + 2)                                         # l.349 (unreachable here)
    counts, edges = np.histogram(scores, bins=bins)                                 # l.352
    probabilities = counts / counts.sum() if counts.sum() > 0 else counts           # l.353
    mean = float(scores.mean()) if scores.size > 0 else float("nan")               # l.359-360
    return dict(bin_edges=edges, counts=counts, probabilities=probabilities, mean=mean, n_pairs=int(scores.size))


# =========================================================================== (b) overlap degrees
def notebook_overlap_degrees(recs: Iterable[RecordDiagnostics]):
    """Per-GT and per-prediction overlap degrees exactly as ``plot_m2n_counts`` (l.389-409).

    Returns (gt_degrees, pred_degrees): flat int lists over all records, in notebook order.
    """
    gt_deg: list[int] = []
    pred_deg: list[int] = []
    for r in recs:
        if r.n_gt == 0:                                   # l.398-400
            pred_deg.extend([0] * r.n_pred)
            continue
        if r.n_pred == 0:                                 # l.402-404
            gt_deg.extend([0] * r.n_gt)
            continue
        hit = r.iou32 > 0                                 # l.406
        gt_deg.extend(hit.sum(axis=1).tolist())          # l.408  (1:n)
        pred_deg.extend(hit.sum(axis=0).tolist())        # l.409  (n:1)
    return gt_deg, pred_deg


def degree_distribution(degrees, side: str) -> dict:
    """Counter of degrees (l.411-412) + legend counts (l.414-417) + notebook tick labels (l.459).

    side='gt'  : labels "1:n" (one GT overlapping n predictions; "1:0" = GT miss)
    side='pred': labels "n:1" (one prediction overlapping n GTs; "0:1" = prediction miss)
    """
    if side not in ("gt", "pred"):
        raise ValueError(side)
    c = Counter(int(d) for d in degrees)
    counts = {d: int(c[d]) for d in sorted(c)}
    label = (lambda d: f"1:{d}") if side == "gt" else (lambda d: f"{d}:1")
    n = int(sum(counts.values()))
    return dict(counts_by_degree=counts,
                counts_by_label={label(d): v for d, v in counts.items()},
                hits=int(sum(v for d, v in counts.items() if d > 0)),   # l.415 / l.417
                misses=int(counts.get(0, 0)),                            # l.414 / l.416
                n=n,
                n_degree_ge2=int(sum(v for d, v in counts.items() if d >= 2)))  # derived, not shown by the notebook


# =========================================================================== PQ cross-check
def notebook_pq(recs: Iterable[RecordDiagnostics]) -> dict:
    """``get_pq_score`` (l.281-315) on the float32 IoU matrices: every pair with IoU > 0.5 is a TP (no
    one-to-one constraint), FP = predictions with no hit, FN = GTs with no hit, pooled; 0.0 if the
    denominator is 0 (l.310-313). Uses the same summation (built-in ``sum`` over the same ordered list)."""
    tp_iou: list[float] = []
    fp = fn = 0
    for r in recs:
        if r.n_gt == 0:                                   # l.292-294
            fp += r.n_pred
            continue
        if r.n_pred == 0:                                 # l.296-298
            fn += r.n_gt
            continue
        hit = r.iou32 > NB_IOU_THRESHOLD                  # l.300
        tp_iou.extend(r.iou32[hit].tolist())              # l.302
        fp += int((hit.sum(axis=0) == 0).sum())           # l.304, fp_count_hit l.133-134
        fn += int((hit.sum(axis=1) == 0).sum())           # l.305, fn_count_hit l.159-160
    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    s = sum(tp_iou)
    return dict(pq=(s / den if den > 0 else 0.0), tp=tp, fp=fp, fn=fn, sum_iou_f32=float(s))


# =========================================================================== Dice variants
def dice_semantic_per_record(rec: RecordDiagnostics) -> float:
    """2|U_g ∩ U_p| / (|U_g| + |U_p|) on the record's union masks; NaN when both unions are empty."""
    den = rec.union_gt_px + rec.union_pred_px
    return 2.0 * rec.union_inter_px / den if den > 0 else float("nan")


def dice_semantic_pooled_px(recs: Iterable[RecordDiagnostics]) -> float:
    """2 Σ|U_g ∩ U_p| / Σ(|U_g| + |U_p|) over records; NaN when the denominator is 0."""
    recs = list(recs)
    num = 2 * sum(r.union_inter_px for r in recs)
    den = sum(r.union_gt_px + r.union_pred_px for r in recs)
    return num / den if den > 0 else float("nan")


def _hit_pairs(r: RecordDiagnostics):
    if r.n_gt == 0 or r.n_pred == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.nonzero(r.iou32 > NB_IOU_THRESHOLD)


def dice_matched_pairs_mean(recs: Iterable[RecordDiagnostics]):
    """(mean, n_pairs): exact float64 Dice 2|g∩p|/(|g|+|p|) averaged over all pairs with IoU > 0.5 (the
    notebook's TP pairs, l.300), pooled over records. NaN if there is no such pair."""
    vals: list[float] = []
    for r in recs:
        gi, pj = _hit_pairs(r)
        for a, b in zip(gi, pj):
            vals.append(2.0 * int(r.inter[a, b]) / (int(r.gt_areas[a]) + int(r.pred_areas[b])))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


# =========================================================================== dataset
def _rec_item(item, shape):
    rid, gt, pred = item
    return record_diagnostics(gt, pred, record_id=rid, shape=shape)


def _frames(recs_all, used_flags, gt_deg_by_rec, pred_deg_by_rec, hists):
    import pandas as pd

    rows, pairs, gti, pri = [], [], [], []
    for r, used, gdeg, pdeg in zip(recs_all, used_flags, gt_deg_by_rec, pred_deg_by_rec):
        hit = r.iou32 > NB_IOU_THRESHOLD if (r.n_gt and r.n_pred) else np.zeros((r.n_gt, r.n_pred), bool)
        rows.append(dict(record_id=r.record_id, used=used, n_gt=r.n_gt, n_pred=r.n_pred,
                         n_overlap_pairs=int((r.inter > 0).sum()), n_hit_pairs=int(hit.sum()),
                         n_gt_miss=int(sum(1 for d in gdeg if d == 0)),
                         n_pred_miss=int(sum(1 for d in pdeg if d == 0)),
                         max_gt_degree=int(max(gdeg, default=0)), max_pred_degree=int(max(pdeg, default=0)),
                         union_gt_px=r.union_gt_px, union_pred_px=r.union_pred_px,
                         union_inter_px=r.union_inter_px, dice_semantic_per_record=dice_semantic_per_record(r)))
        gi, pj = np.nonzero(r.inter > 0)
        for a, b in zip(gi.tolist(), pj.tolist()):
            it, ga, pa = int(r.inter[a, b]), int(r.gt_areas[a]), int(r.pred_areas[b])
            pairs.append(dict(record_id=r.record_id, used=used, gt_idx=a, pred_idx=b, inter_px=it,
                              gt_area=ga, pred_area=pa, iou_f32=float(r.iou32[a, b]), dice_f32=float(r.dice32[a, b]),
                              iou=it / (ga + pa - it), dice=2.0 * it / (ga + pa), hit=bool(hit[a, b])))
        for k, d in enumerate(gdeg):
            gti.append(dict(record_id=r.record_id, used=used, gt_idx=k, area=int(r.gt_areas[k]), degree=int(d)))
        for k, d in enumerate(pdeg):
            pri.append(dict(record_id=r.record_id, used=used, pred_idx=k, area=int(r.pred_areas[k]), degree=int(d)))
    hrows = []
    for name, h in hists.items():
        e = h["bin_edges"]
        for k in range(len(h["counts"])):
            hrows.append(dict(metric=name, bin_lo=float(e[k]), bin_hi=float(e[k + 1]),
                              count=int(h["counts"][k]), probability=float(h["probabilities"][k])))
    return dict(records=pd.DataFrame(rows), pairs=pd.DataFrame(pairs), gt_instances=pd.DataFrame(gti),
                pred_instances=pd.DataFrame(pri), histograms=pd.DataFrame(hrows))


def aggregate_diagnostics(items: Iterable, *, skip_empty_gt: bool = True, n_workers: int = 1,
                          shape=(2048, 2048), return_records: bool = False):
    """Notebook-style rubric diagnostics + named Dice variants over (record_id, gt, pred) items.

    One item per GT annotation set (pass the same prediction once per set, as the notebook does).
    gt / pred: anything ``filseg.metrics.as_instance_set`` accepts.

    Returns (summary, frames); with return_records=True, (summary, frames, list[RecordDiagnostics]).
      summary: JSON-friendly after ``to_jsonable`` (iou_hist, dice_hist, overlap_degree, notebook_pq,
               dice_variants, record counts).
      frames : DataFrames 'records' (one row per item, column 'used'), 'pairs' (one row per overlapping
               (gt, pred) pair: exact and float32 IoU/Dice, 'hit' = IoU > 0.5), 'gt_instances' /
               'pred_instances' (overlap degree per instance; degrees of skipped records follow the
               n_gt == 0 rule but are excluded from the summary), 'histograms' (both histograms, long form).
    """
    items = list(items)
    fn = partial(_rec_item, shape=tuple(shape))
    if n_workers > 1 and len(items) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(int(n_workers), 16), initializer=single_thread_worker) as pool:
            recs_all = list(pool.imap(fn, items, chunksize=4))
    else:
        recs_all = [fn(it) for it in items]

    used_flags = [not (skip_empty_gt and r.n_gt == 0) for r in recs_all]
    recs = [r for r, u in zip(recs_all, used_flags) if u]

    iou_h = notebook_histogram(notebook_scores(recs, "iou"))
    dice_h = notebook_histogram(notebook_scores(recs, "dice"))
    gdeg_all, pdeg_all = [], []
    for r in recs_all:                                   # per-record degrees (same rule, l.398-409)
        g, p = notebook_overlap_degrees([r])
        gdeg_all.append(g)
        pdeg_all.append(p)
    gt_deg = [d for g, u in zip(gdeg_all, used_flags) if u for d in g]
    pred_deg = [d for p, u in zip(pdeg_all, used_flags) if u for d in p]
    assert (gt_deg, pred_deg) == tuple(notebook_overlap_degrees(recs))

    sem = [dice_semantic_per_record(r) for r in recs]
    sem_def = [v for v in sem if not math.isnan(v)]
    dmp, n_dmp = dice_matched_pairs_mean(recs)
    summary = dict(
        definitions=NOTEBOOK_COPY + " (third-party verbatim copy; see filseg.diagnostics docstring)",
        skip_empty_gt=bool(skip_empty_gt),
        n_records_input=len(recs_all), n_records_used=len(recs),
        n_records_skipped_empty_gt=int(sum(1 for u in used_flags if not u)),
        n_gt=int(sum(r.n_gt for r in recs)), n_pred=int(sum(r.n_pred for r in recs)),
        iou_hist=iou_h, dice_hist=dice_h,
        overlap_degree=dict(gt_one_to_many=degree_distribution(gt_deg, "gt"),
                            pred_many_to_one=degree_distribution(pred_deg, "pred")),
        notebook_pq=notebook_pq(recs),
        dice_variants=dict(
            dice_semantic_per_record_mean=(float(np.mean(sem_def)) if sem_def else float("nan")),
            dice_semantic_per_record_median=(float(np.median(sem_def)) if sem_def else float("nan")),
            dice_semantic_per_record_n=len(sem_def),
            dice_semantic_per_record_n_undefined=len(sem) - len(sem_def),
            dice_semantic_pooled_px=dice_semantic_pooled_px(recs),
            dice_matched_pairs_mean=dmp, dice_matched_pairs_n=n_dmp,
            dice_overlap_pairs_mean_notebook=dice_h["mean"],
        ),
    )
    frames = _frames(recs_all, used_flags, gdeg_all, pdeg_all, {"iou": iou_h, "dice": dice_h})
    if return_records:
        return summary, frames, recs_all
    return summary, frames


def to_jsonable(x):
    """numpy -> python types, NaN/inf -> None, dict keys -> str (for json.dump)."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x
