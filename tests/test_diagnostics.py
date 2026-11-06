"""Tests for filseg.diagnostics (notebook-style rubric diagnostics + named Dice variants).

Run:  .venv/bin/python tests/test_diagnostics.py            (CPU only, ~1 min; also pytest-compatible)

What is compared with what
  1. Hand-computed synthetic records (numbers derived on paper, see comments).
  2. Random small cases: filseg.diagnostics vs a DIRECT READING of the notebook
     (docs/thirdparty_metric/ooLemon0-0_self-evaluation-notebook_cells.py):
       * the copied notebook functions ``get_overlap_matrices`` and ``get_pq_score`` are EXECUTED
         (pure computation: torch / numpy arithmetic, no I/O, no plotting);
       * ``plot_distribution`` and ``plot_m2n_counts`` are NOT executed (they draw on matplotlib state);
         their computational lines are transcribed verbatim below (``_nb_distribution``, ``_nb_m2n``,
         with notebook line numbers) and run on the notebook's own float32 matrices.
     Also cross-checked against filseg.metrics (PQ_pooled_host, semantic Dice) and dense brute force.
  3. End to end at 2048 x 2048: the copied ``get_overlap_df`` (pure computation: pandas, pycocotools
     decode, torch) on submission-format DataFrames, incl. a 2-annotator image, an image without
     predictions, an annotation set without filaments and predictions on an image without GT.
If torch or the notebook copy is unavailable, parts 2-3 are skipped with a message.
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"   # shared host: avoid THP compaction stalls
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2  # noqa: E402

from filseg import rle  # noqa: E402
from filseg.diagnostics import (aggregate_diagnostics, dice_matched_pairs_mean, overlap_matrices_f32,  # noqa: E402
                                record_diagnostics, to_jsonable)
from filseg.metrics import EvalConfig, InstanceSet, evaluate_dataset  # noqa: E402

cv2.setNumThreads(1)
NB_PATH = os.path.join(ROOT, "docs", "thirdparty_metric", "ooLemon0-0_self-evaluation-notebook_cells.py")
EDGES = np.linspace(0.0, 1.0, 51)


def R(y0, x0, y1, x1, shape=(64, 64)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def S(masks, shape=(64, 64)):
    return InstanceSet.from_masks(masks, shape=shape)


def _bin(v):
    """index of the notebook bin (np.histogram on linspace(0, 1, 51)) holding value v."""
    return int(np.histogram([v], bins=EDGES)[0].argmax())


# =============================================================================== 1. hand-computed
def test_hand_computed():
    # record A (64x64):  g0 = rows 0-9, cols 0-9 (100 px);  g1 = rows 20-29, cols 0-9 (100 px)
    #   p0 = rows 0-9, cols 5-14 -> g0∩p0 = 50, union 150: IoU 1/3, Dice 0.5
    #   p1 = rows 0-9, cols 10-19 -> touches no GT (degree 0)
    #   p2 = g1 exactly             -> IoU 1, Dice 1 (the only IoU > 0.5 pair)
    #   p3 = rows 25-29, cols 0-9   -> g1∩p3 = 50, union 100: IoU exactly 0.5 (NOT a hit), Dice 2/3
    gA = [R(0, 0, 10, 10), R(20, 0, 30, 10)]
    pA = [R(0, 5, 10, 15), R(0, 10, 10, 20), R(20, 0, 30, 10), R(25, 0, 30, 10)]
    # record B: two 5x5 GTs, no prediction -> GT degrees 0, 0; semantic Dice 0
    gB = [R(40, 40, 45, 45), R(50, 50, 55, 55)]
    # record C: GT set without instances, 2 predictions -> dropped by default (as get_overlap_df)
    pC = [R(0, 0, 3, 3), R(10, 10, 12, 12)]
    # record D (many-to-one): g0 rows 0-3, g1 rows 6-9 (cols 0-9, 40 px each), pred rows 0-9 (100 px)
    #   -> pred degree 2 ("2:1"); IoU 0.4 each, Dice 80/140; plus an EMPTY pred mask (degree 0, IoU 0)
    gD = [R(0, 0, 4, 10), R(6, 0, 10, 10)]
    pD = [R(0, 0, 10, 10), np.zeros((64, 64), bool)]
    items = [("010101-A", S(gA), S(pA)), ("010201-B", S(gB), InstanceSet.empty((64, 64))),
             ("010301-C", InstanceSet.empty((64, 64)), S(pC)), ("010401-D", S(gD), S(pD))]
    s, fr = aggregate_diagnostics(items, shape=(64, 64))
    assert (s["n_records_input"], s["n_records_used"], s["n_records_skipped_empty_gt"]) == (4, 3, 1)
    assert (s["n_gt"], s["n_pred"]) == (6, 6)

    # (a) overlapping pairs: A (1/3, 1, 0.5), D (0.4, 0.4)  -> 5 IoU values; Dice (0.5, 1, 2/3, 4/7, 4/7)
    ih, dh = s["iou_hist"], s["dice_hist"]
    assert np.array_equal(ih["bin_edges"], EDGES) and ih["n_pairs"] == 5 and dh["n_pairs"] == 5
    exp = np.zeros(50, int)
    for v in (1 / 3, 1.0, 0.5, 0.4, 0.4):
        exp[_bin(np.float32(v))] += 1
    assert np.array_equal(ih["counts"], exp), ih["counts"]
    assert ih["counts"][16] == 1 and ih["counts"][25] == 1 and ih["counts"][49] == 1 and ih["counts"][20] == 2
    assert np.allclose(ih["probabilities"], exp / 5) and math.isclose(ih["probabilities"].sum(), 1.0)
    assert abs(ih["mean"] - (1 / 3 + 1 + 0.5 + 0.4 + 0.4) / 5) < 1e-7          # float32 values
    assert dh["counts"][25] == 1 and dh["counts"][33] == 1 and dh["counts"][49] == 1 and dh["counts"][28] == 2
    assert abs(dh["mean"] - (0.5 + 1 + 2 / 3 + 4 / 7 + 4 / 7) / 5) < 1e-7

    # (b) degrees.  GT: A g0 1, g1 2; B 0, 0; D 1, 1  -> {0: 2, 1: 3, 2: 1}
    #              pred: A 1, 0, 1, 1; D 2, 0       -> {0: 2, 1: 3, 2: 1}
    g, p = s["overlap_degree"]["gt_one_to_many"], s["overlap_degree"]["pred_many_to_one"]
    assert g["counts_by_degree"] == {0: 2, 1: 3, 2: 1} and g["counts_by_label"] == {"1:0": 2, "1:1": 3, "1:2": 1}
    assert (g["hits"], g["misses"], g["n_degree_ge2"]) == (4, 2, 1)
    assert p["counts_by_degree"] == {0: 2, 1: 3, 2: 1} and p["counts_by_label"] == {"0:1": 2, "1:1": 3, "2:1": 1}
    assert (p["hits"], p["misses"]) == (4, 2)

    # notebook PQ: TP = 1 (A g1-p2, IoU 1); FP = A p0, p1, p3 + D p0, p1 = 5; FN = A g0 + B 2 + D 2 = 5
    q = s["notebook_pq"]
    assert (q["tp"], q["fp"], q["fn"]) == (1, 5, 5) and q["pq"] == 1.0 / (1 + 2.5 + 2.5)

    # Dice variants.  A: |Ug| 200, |Up| 250 (p3 inside p2), |Ug∩Up| = 50 + 100 = 150 -> 300/450 = 2/3
    #                 B: 50, 0, 0 -> 0.    D: 80, 100, 80 -> 160/180 = 8/9
    dv = s["dice_variants"]
    assert math.isclose(dv["dice_semantic_per_record_mean"], (2 / 3 + 0 + 8 / 9) / 3, rel_tol=1e-12)
    assert math.isclose(dv["dice_semantic_pooled_px"], 2 * (150 + 0 + 80) / (450 + 50 + 180), rel_tol=1e-12)
    assert (dv["dice_semantic_per_record_n"], dv["dice_semantic_per_record_n_undefined"]) == (3, 0)
    assert dv["dice_matched_pairs_mean"] == 1.0 and dv["dice_matched_pairs_n"] == 1
    assert dv["dice_overlap_pairs_mean_notebook"] == dh["mean"]

    # frames
    rec = fr["records"].set_index("record_id")
    assert not rec.loc["010301-C", "used"] and rec.loc["010101-A", "n_overlap_pairs"] == 3
    assert rec.loc["010101-A", "n_hit_pairs"] == 1 and rec.loc["010401-D", "max_pred_degree"] == 2
    pr = fr["pairs"]
    assert len(pr) == 5 and pr["hit"].sum() == 1 and set(pr.record_id) == {"010101-A", "010401-D"}
    half = pr[(pr.record_id == "010101-A") & (pr.pred_idx == 3)].iloc[0]
    assert half.iou == 0.5 and half.iou_f32 == 0.5 and not half.hit                # strict > 0.5
    assert len(fr["histograms"]) == 100

    # skip_empty_gt=False: record C counts as notebook rows with n_gt == 0 (l.292-294, l.398-400)
    s2, _ = aggregate_diagnostics(items, shape=(64, 64), skip_empty_gt=False)
    assert s2["n_records_used"] == 4 and s2["overlap_degree"]["pred_many_to_one"]["counts_by_degree"] == {0: 4, 1: 3, 2: 1}
    assert (s2["notebook_pq"]["fp"], s2["iou_hist"]["n_pairs"]) == (7, 5)
    assert math.isclose(s2["dice_variants"]["dice_semantic_per_record_mean"], (2 / 3 + 0 + 0 + 8 / 9) / 4)

    # empty input / nothing overlapping
    s3, _ = aggregate_diagnostics([("x", S([R(0, 0, 2, 2)]), S([R(9, 9, 11, 11)]))], shape=(64, 64))
    assert s3["iou_hist"]["n_pairs"] == 0 and math.isnan(s3["iou_hist"]["mean"])
    assert s3["iou_hist"]["counts"].sum() == 0 and s3["notebook_pq"]["pq"] == 0.0
    assert math.isnan(s3["dice_variants"]["dice_matched_pairs_mean"])
    j = to_jsonable(s3)
    assert j["iou_hist"]["mean"] is None and j["overlap_degree"]["gt_one_to_many"]["counts_by_degree"] == {"0": 1}
    s4, _ = aggregate_diagnostics([("e", InstanceSet.empty((8, 8)), InstanceSet.empty((8, 8)))], shape=(8, 8),
                                  skip_empty_gt=False)
    assert math.isnan(s4["dice_variants"]["dice_semantic_per_record_mean"])
    assert s4["dice_variants"]["dice_semantic_per_record_n_undefined"] == 1 and s4["notebook_pq"]["pq"] == 0.0


def test_float32_rounding_hand():
    # g = 3 px, p = 7 px, overlap 1 px: IoU = 1/9, Dice = 2/10. float32 values, not float64.
    g = np.zeros((1, 12), bool); g[0, 0:3] = True
    p = np.zeros((1, 12), bool); p[0, 2:9] = True
    iou, dice = overlap_matrices_f32([g], [p], shape=(1, 12))
    assert iou.dtype == np.float32 and iou[0, 0] == np.float32(1) / np.float32(9) and float(iou[0, 0]) != 1 / 9
    assert dice[0, 0] == np.float32(2) / np.float32(10)
    # both masks empty: union 0 -> 0 (not NaN), as l.96-106
    iou0, dice0 = overlap_matrices_f32([np.zeros((1, 12), bool)], [np.zeros((1, 12), bool)], shape=(1, 12))
    assert iou0[0, 0] == 0 and dice0[0, 0] == 0


# =============================================================================== 2. random vs notebook
def _load_notebook():
    try:
        import torch  # noqa: F401
    except Exception:
        return None
    if not os.path.exists(NB_PATH):
        return None
    ns: dict = {}
    exec(compile(open(NB_PATH).read(), NB_PATH, "exec"), ns)    # defines functions; executes nothing else
    return ns


def _nb_distribution(matrices):
    """Verbatim transcription of the computational lines of plot_distribution (l.336-353, l.359-360).
    Plotting lines (plt.*) are omitted; nothing else is changed."""
    scores = np.concatenate(
        [np.asarray(m, dtype=float).ravel() for m in matrices] or [np.zeros(0)]
    )
    scores = scores[scores > 0]
    x_max = 1.0 if scores.size == 0 or scores.max() <= 1.0 else scores.max()
    if x_max <= 1.0:
        n_bins = 50
        bins = np.linspace(0.0, x_max, n_bins + 1)
    else:
        bins = np.arange(0, int(x_max) + 2)
    counts, edges = np.histogram(scores, bins=bins)
    probabilities = counts / counts.sum() if counts.sum() > 0 else counts
    mean_score = scores.mean() if scores.size > 0 else None
    return counts, edges, probabilities, mean_score


def _nb_m2n(overlap_df):
    """Verbatim transcription of the computational lines of plot_m2n_counts (l.390-417)."""
    gt_match_counts: list[int] = []
    pred_match_counts: list[int] = []
    for row in overlap_df.itertuples(index=False):
        iou_matrix = row.iou_matrix
        n_gt = row.n_gt
        n_pred = row.n_pred
        if n_gt == 0:
            pred_match_counts.extend([0] * n_pred)
            continue
        if n_pred == 0:
            gt_match_counts.extend([0] * n_gt)
            continue
        hit_matrix = iou_matrix > 0
        gt_match_counts.extend(hit_matrix.sum(dim=1).tolist())
        pred_match_counts.extend(hit_matrix.sum(dim=0).tolist())
    gt_degree_counts = Counter(gt_match_counts)
    pred_degree_counts = Counter(pred_match_counts)
    gt_miss_count = gt_degree_counts.get(0, 0)
    gt_hit_count = sum(count for degree, count in gt_degree_counts.items() if degree > 0)
    pred_miss_count = pred_degree_counts.get(0, 0)
    pred_hit_count = sum(count for degree, count in pred_degree_counts.items() if degree > 0)
    return gt_degree_counts, pred_degree_counts, (gt_hit_count, gt_miss_count, pred_miss_count, pred_hit_count)


def _gen_mask(rng, shape):
    h, w = shape
    m = np.zeros(shape, np.uint8)
    k = int(rng.integers(0, 6))
    if k == 0:
        y0, y1 = sorted(rng.integers(0, h + 1, 2)); x0, x1 = sorted(rng.integers(0, w + 1, 2))
        m[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = 1
    elif k == 1:
        pts = np.stack([rng.integers(-2, w + 2, 4), rng.integers(-2, h + 2, 4)], 1).astype(np.int32)
        cv2.polylines(m, [pts], False, 1, int(rng.integers(1, 4)))
    elif k == 2:
        cv2.ellipse(m, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                    (int(rng.integers(1, max(2, w // 2))), int(rng.integers(1, max(2, h // 2)))),
                    float(rng.uniform(0, 180)), 0, 360, 1, -1)
    elif k == 3:
        m[rng.random(shape) < 0.15] = 1
    elif k == 4:
        m[int(rng.integers(0, h)), int(rng.integers(0, w))] = 1
    # k == 5: empty mask
    return m.astype(bool)


def _gen_preds(rng, gts, shape):
    out = []
    for g in gts:
        r = rng.random()
        if r < 0.25:
            out.append(g.copy())
        elif r < 0.4:
            out += [g.copy(), g.copy()]                                              # duplicates
        elif r < 0.6:
            out.append(np.roll(g, int(rng.integers(-3, 4)), axis=int(rng.integers(0, 2))))
        elif r < 0.75:
            out.append(cv2.dilate(g.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
        elif r < 0.85:
            px = np.flatnonzero(g.ravel())
            if px.size >= 2:                                                         # split in two
                cut = int(rng.integers(1, px.size))
                for piece in (px[:cut], px[cut:]):
                    q = np.zeros(g.size, bool); q[piece] = True; out.append(q.reshape(shape))
    if len(gts) >= 2 and rng.random() < 0.4:
        out.append(gts[0] | gts[1])                                                  # merge
    out += [_gen_mask(rng, shape) for _ in range(int(rng.integers(0, 3)))]
    rng.shuffle(out)
    return out[:8]


def _random_records(n, seed):
    rng = np.random.default_rng(seed)
    shapes = [(1, 1), (1, 9), (9, 1), (16, 16), (23, 41), (48, 48)]
    recs = []
    for c in range(n):
        shape = shapes[int(rng.integers(0, len(shapes)))]
        gts = [_gen_mask(rng, shape) for _ in range(int(rng.integers(0, 6)))]
        preds = [] if rng.random() < 0.1 else _gen_preds(rng, gts, shape)
        if rng.random() < 0.08:
            gts = []
        recs.append((f"r{c}", shape, gts, preds))
    return recs


def test_random_vs_notebook(n: int = 600, seed: int = 20261027):
    nb = _load_notebook()
    if nb is None:
        print("SKIP notebook comparison (torch or notebook copy missing)")
        return None
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    # Records have different image shapes; that is fine because shapes only need to agree within a record
    # (every item is an InstanceSet, so the 'shape' argument of aggregate_diagnostics is never used).
    cases = _random_records(n, seed)
    nb_rows, items = [], []
    n_bits, n_overlap, n_ge2_gt, n_ge2_pred, n_empty_gt = 0, 0, 0, 0, 0
    for key, shape, gts, preds in cases:
        H, W = shape
        g = torch.from_numpy(np.stack(gts).astype(np.float32)) if gts else torch.zeros((0, H, W))
        p = torch.from_numpy(np.stack(preds).astype(np.float32)) if preds else torch.zeros((0, H, W))
        iou_t, dice_t = nb["get_overlap_matrices"](g, p)                         # executed (pure)
        gi, pi = InstanceSet.from_masks(gts, shape=shape), InstanceSet.from_masks(preds, shape=shape)
        iou, dice = overlap_matrices_f32(gi, pi, shape=shape)
        assert iou.dtype == np.float32 and iou_t.dtype == torch.float32
        assert np.array_equal(iou.view(np.uint32), iou_t.numpy().view(np.uint32)), key   # bit-identical
        assert np.array_equal(dice.view(np.uint32), dice_t.numpy().view(np.uint32)), key
        n_bits += iou.size
        n_overlap += int((iou > 0).sum())
        if gts and preds:
            hit = (iou > 0)
            n_ge2_gt += int((hit.sum(1) >= 2).sum()); n_ge2_pred += int((hit.sum(0) >= 2).sum())
        n_empty_gt += not gts
        nb_rows.append(dict(annotator_image=key, iou_matrix=iou_t, dice_matrix=dice_t, n_gt=len(gts), n_pred=len(preds)))
        items.append((key, gi, pi))
    assert n_overlap >= 1000 and n_ge2_gt >= 100 and n_ge2_pred >= 100 and n_empty_gt >= 20, \
        (n_overlap, n_ge2_gt, n_ge2_pred, n_empty_gt)

    for skip in (True, False):
        # skip=True <-> the notebook's overlap_df has no row for a GT set without filaments (l.240-245)
        df = pd.DataFrame([r for r in nb_rows if (r["n_gt"] > 0 or not skip)])
        s, fr, recs = aggregate_diagnostics(items, skip_empty_gt=skip, shape=(1, 1), return_records=True)
        assert s["n_records_used"] == len(df)
        for which, col in (("iou", "iou_matrix"), ("dice", "dice_matrix")):
            c, e, prob, mean = _nb_distribution(df[col])
            h = s[f"{which}_hist"]
            assert np.array_equal(h["counts"], c) and np.array_equal(h["bin_edges"], e), which
            assert np.array_equal(h["probabilities"], prob), which
            assert h["mean"] == mean, (which, h["mean"], mean)                     # bit-identical
        gdc, pdc, legend = _nb_m2n(df)
        g, p = s["overlap_degree"]["gt_one_to_many"], s["overlap_degree"]["pred_many_to_one"]
        assert g["counts_by_degree"] == dict(gdc) and p["counts_by_degree"] == dict(pdc), skip
        assert (g["hits"], g["misses"], p["misses"], p["hits"]) == legend
        pq_nb = nb["get_pq_score"](df)                                             # executed (pure)
        assert s["notebook_pq"]["pq"] == pq_nb, (s["notebook_pq"]["pq"], pq_nb)    # bit-identical

        # filseg.metrics host-style PQ (float64 IoU) agrees to float32 precision
        it = [x for x, r in zip(items, recs) if r.n_gt > 0 or not skip]
        sm, dfm = evaluate_dataset(it, EvalConfig(compute_thin=False, compute_boundary=False), shape=(1, 1))
        assert abs(sm["PQ_pooled_host"] - pq_nb) < 1e-6
        assert (sm["TP_host"], sm["FP_host"], sm["FN_host"]) == tuple(s["notebook_pq"][k] for k in ("tp", "fp", "fn"))
        # semantic Dice: independent implementation in filseg.metrics (full-image unions)
        a = fr["records"][fr["records"].used].reset_index(drop=True)
        assert np.allclose(a["dice_semantic_per_record"].to_numpy(float), dfm["semantic_dice"].to_numpy(float),
                           rtol=0, atol=1e-12, equal_nan=True)
        assert abs(s["dice_variants"]["dice_semantic_pooled_px"] - sm["semantic_dice_pooled_px"]) < 1e-12
    return dict(records=len(cases), matrix_entries=n_bits, overlapping_pairs=n_overlap,
                gt_degree_ge2=n_ge2_gt, pred_degree_ge2=n_ge2_pred, empty_gt_records=n_empty_gt)


def test_random_vs_dense_bruteforce(n: int = 300, seed: int = 11):
    """Degrees, overlapping-pair Dice and IoU>0.5 pair Dice from plain dense numpy masks."""
    for key, shape, gts, preds in _random_records(n, seed):
        if not gts:
            continue
        r = record_diagnostics(InstanceSet.from_masks(gts, shape=shape), InstanceSet.from_masks(preds, shape=shape),
                               record_id=key, shape=shape)
        for i, g in enumerate(gts):
            for j, p in enumerate(preds):
                it = int((g & p).sum()); un = int((g | p).sum()); sa = int(g.sum()) + int(p.sum())
                assert r.inter[i, j] == it
                assert r.iou32[i, j] == (np.float32(it) / np.float32(un) if un else 0)
                assert r.dice32[i, j] == (np.float32(2 * it) / np.float32(sa) if un else 0)
        if preds:
            deg_g = [sum(bool((g & p).any()) for p in preds) for g in gts]
            deg_p = [sum(bool((g & p).any()) for g in gts) for p in preds]
            assert (r.iou32 > 0).sum(1).tolist() == deg_g and (r.iou32 > 0).sum(0).tolist() == deg_p
        exp = [2 * int((g & p).sum()) / (int(g.sum()) + int(p.sum())) for g in gts for p in preds
               if 2 * int((g & p).sum()) > int((g | p).sum())]                        # exact IoU > 0.5
        m, k = dice_matched_pairs_mean([r])
        assert k == len(exp) and (k == 0 or abs(m - float(np.mean(exp))) < 1e-12), key
        ug = np.any(gts, 0); up = np.any(preds, 0) if preds else np.zeros(shape, bool)
        assert (r.union_gt_px, r.union_pred_px, r.union_inter_px) == (int(ug.sum()), int(up.sum()), int((ug & up).sum()))


def test_parallel_equals_serial():
    cases = [(k, InstanceSet.from_masks(g, shape=s), InstanceSet.from_masks(p, shape=s))
             for k, s, g, p in _random_records(80, 5) if s == (48, 48)]
    a, fa = aggregate_diagnostics(cases, shape=(48, 48))
    b, fb = aggregate_diagnostics(cases, shape=(48, 48), n_workers=3)
    assert to_jsonable(a) == to_jsonable(b)
    assert fa["pairs"].equals(fb["pairs"]) and fa["records"].equals(fb["records"])


# =============================================================================== 3. end to end 2048^2
def test_end_to_end_get_overlap_df():
    nb = _load_notebook()
    if nb is None:
        print("SKIP end-to-end notebook test (torch or notebook copy missing)")
        return None
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    rng = np.random.default_rng(3)
    H = W = 2048

    def blob(cy, cx, ry, rx, ang):
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (cx, cy), (rx, ry), ang, 0, 360, 1, -1)
        return m.astype(bool)

    def rect(y0, x0, y1, x1):
        return R(y0, x0, y1, x1, (H, W))

    s1, s2, s3, s4 = "20110101000000Bh", "20110102000000Ch", "20110103000000Lh", "20110104000000Mh"
    gA = [blob(500, 500, 8, 60, 20), rect(700, 800, 706, 1000), rect(1200, 380, 1205, 420)]
    gB = [blob(502, 505, 9, 55, 22), rect(700, 800, 706, 880), rect(700, 890, 706, 1000)]   # B splits A's 2nd
    pS1 = [blob(500, 498, 8, 58, 20), rect(699, 805, 707, 995), blob(1500, 1500, 10, 10, 0),
           np.zeros((H, W), bool),                                                # empty pred mask
           rect(1200, 380, 1205, 400), rect(1200, 400, 1205, 420)]                # pred split of gA[2]
    gC = [blob(300, 1300, 7, 70, 10), blob(310, 1420, 7, 30, 10)]                  # image without predictions
    pS3 = [blob(900, 900, 20, 20, 0)]                                              # GT set without filaments
    pS4 = [blob(100, 100, 5, 5, 0)]                                                # stem without any GT record
    gt_sets = [("010101-" + s1, gA), ("010102-" + s1, gB), ("020101-" + s2, gC), ("030101-" + s3, [])]
    preds = {s1: pS1, s3: pS3, s4: pS4}
    gt_rows = [(f"{rid}_{k}", rle.encode(m)) for rid, ms in gt_sets for k, m in enumerate(ms)]
    pr_rows = [(f"{st}_{k}", rle.encode(m)) for st, ms in preds.items() for k, m in enumerate(ms)]
    gt_df = pd.DataFrame(gt_rows, columns=["filament_id", "segmentation_rle"])
    pred_df = pd.DataFrame(pr_rows, columns=["filament_id", "segmentation_rle"])
    od = nb["get_overlap_df"](gt_df, pred_df)                                      # executed (pure)
    assert od["annotator_image"].tolist() == [r for r, ms in gt_sets if ms]      # empty set has no row

    items = [(rid, InstanceSet.from_masks(ms, shape=(H, W)) if ms else InstanceSet.empty((H, W)),
              InstanceSet.from_masks(preds[rid.split("-", 1)[1]], shape=(H, W)) if rid.split("-", 1)[1] in preds
              else InstanceSet.empty((H, W))) for rid, ms in gt_sets]
    s, fr = aggregate_diagnostics(items)
    assert s["n_records_skipped_empty_gt"] == 1 and s["n_records_used"] == 3
    for which, col in (("iou", "iou_matrix"), ("dice", "dice_matrix")):
        c, e, prob, mean = _nb_distribution(od[col])
        h = s[f"{which}_hist"]
        assert np.array_equal(h["counts"], c) and np.array_equal(h["probabilities"], prob) and h["mean"] == mean
    gdc, pdc, legend = _nb_m2n(od)
    g, p = s["overlap_degree"]["gt_one_to_many"], s["overlap_degree"]["pred_many_to_one"]
    assert g["counts_by_degree"] == dict(gdc) and p["counts_by_degree"] == dict(pdc)
    assert (g["hits"], g["misses"], p["misses"], p["hits"]) == legend
    assert s["notebook_pq"]["pq"] == nb["get_pq_score"](od)
    # predictions of s1 are reused for both annotators of s1 (12 pred degrees = 2 x 6), s3/s4 preds ignored
    assert p["n"] == 12 and g["n"] == 8 and g["counts_by_degree"].get(0, 0) >= 2
    # gA[2] is overlapped by 2 preds (1:2); pS1[1] overlaps gB[1] and gB[2] (2:1)
    assert g["counts_by_degree"].get(2, 0) >= 1 and p["counts_by_degree"].get(2, 0) >= 1
    return dict(iou_pairs=s["iou_hist"]["n_pairs"], gt_degrees=g["counts_by_label"], pred_degrees=p["counts_by_label"],
                pq=round(s["notebook_pq"]["pq"], 6))


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            out = fn()
            print(f"ok  {name}  ({time.time() - t:.1f}s){'  ' + str(out) if out is not None else ''}")
