"""Tests for filseg.metrics. Run:  PYTHONPATH=src .venv/bin/python tests/test_metrics.py [--skip-real]
(also pytest-compatible). The real-data smoke test (GT vs itself, inter-annotator PQ) writes
outputs/audit/m1_interannotator_pq_records.csv and m1_metric_smoke_summary.json."""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from pycocotools import mask as mask_util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from filseg import rle  # noqa: E402
from filseg.metrics import (EvalConfig, InstanceSet, evaluate_dataset, evaluate_image,  # noqa: E402
                            intersection_matrix, iou_from_intersection, match_instances,
                            single_thread_worker, split_merge_diagnostics, thin_region)

ANN = os.path.join(ROOT, "MAGFiLO_1.0_Kaggle_2026", "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
AUDIT = os.path.join(ROOT, "outputs", "audit")
S = (256, 256)


def rect(y0, x0, y1, x1, shape=S):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def close(a, b, tol=1e-12):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol


# ----------------------------------------------------------------------------- PQ core
def test_perfect_prediction():
    gts = [rect(10, 10, 40, 100), rect(60, 20, 70, 200), rect(100, 100, 180, 120)]
    row, pg, pp = evaluate_image(gts, [m.copy() for m in gts], shape=S)
    assert (row["tp"], row["fp"], row["fn"]) == (3, 0, 0)
    assert row["pq"] == 1.0 and row["sq"] == 1.0 and row["rq"] == 1.0 and row["pq_host"] == 1.0
    assert row["semantic_dice"] == 1.0 and row["boundary_f"] == 1.0
    assert row["n_split_gt"] == row["n_merge_pred"] == row["n_missed_gt"] == row["n_spurious_pred"] == 0
    assert row["thin_region_recall"] in (1.0,) or math.isnan(row["thin_region_recall"])
    s, df = evaluate_dataset([("a", gts, gts)])
    assert s["PQ_pooled"] == 1.0 and s["PQ_img_mean"] == 1.0 and s["tp_boundary_f_mean"] == 1.0
    assert s["gt_best_iou_mean"] == 1.0 and s["gt_best_dice_mean"] == 1.0


def test_split_in_two_halves():
    g = rect(10, 10, 30, 110)                       # 20 x 100 = 2000 px
    p1, p2 = rect(10, 10, 30, 60), rect(10, 60, 30, 110)
    row, pg, pp = evaluate_image([g], [p1, p2], shape=S)
    # each half: IoU = 1000 / 2000 = 0.5 exactly -> NOT a match (strict >)
    assert (row["tp"], row["fp"], row["fn"]) == (0, 2, 1), row
    assert row["pq"] == 0.0 and row["pq_host"] == 0.0
    assert row["n_split_gt"] == 1 and row["n_split_fragments"] == 2 and row["n_merge_pred"] == 0
    assert row["fn_split"] == 1 and row["fp_fragment"] == 2
    assert pg[0]["is_split"] and pg[0]["fn_cause"] == "split" and abs(pg[0]["best_iou"] - 0.5) < 1e-12
    # uneven split 60/40: big piece IoU 0.6 -> TP; small piece FP; gt still flagged split
    q1, q2 = rect(10, 10, 30, 70), rect(10, 70, 30, 110)
    row, pg, pp = evaluate_image([g], [q1, q2], shape=S)
    assert (row["tp"], row["fp"], row["fn"]) == (1, 1, 0) and row["n_split_gt"] == 1
    assert close(row["pq"], 0.6 / 1.5) and pp[1]["fp_cause"] == "fragment"


def test_merge_two_gts():
    g1, g2 = rect(10, 10, 30, 60), rect(10, 64, 30, 114)      # 1000 px each, 4 px gap
    p = rect(10, 10, 30, 114)                                  # covers both + gap: 2080 px
    row, pg, pp = evaluate_image([g1, g2], [p], shape=S)
    assert (row["tp"], row["fp"], row["fn"]) == (0, 1, 2), row
    assert row["n_merge_pred"] == 1 and row["n_merged_gt"] == 2 and row["n_split_gt"] == 0
    assert row["fn_merged"] == 2 and row["fp_merge"] == 1 and pp[0]["n_contained_gt"] == 2
    # unequal merge: big gt matched (IoU > .5), small gt FN with cause 'merged'
    G1, G2 = rect(10, 10, 40, 110), rect(50, 10, 54, 40)       # 3000 px, 120 px
    P = G1 | G2
    row, pg, pp = evaluate_image([G1, G2], [P], shape=S)
    assert (row["tp"], row["fp"], row["fn"]) == (1, 0, 1) and row["n_merge_pred"] == 1
    assert pg[1]["fn_cause"] == "merged" and close(row["pq"], (3000 / 3120) / 1.5)


def test_split_merge_symmetry():
    rng = np.random.default_rng(3)
    inter = rng.integers(0, 50, size=(6, 5))
    ga, pa = inter.sum(1) + rng.integers(0, 30, 6), inter.sum(0) + rng.integers(0, 30, 5)
    a = split_merge_diagnostics(inter, ga, pa, 0.5, 10)
    b = split_merge_diagnostics(inter.T, pa, ga, 0.5, 10)
    assert np.array_equal(a["split_gt"], b["merge_pred"]) and np.array_equal(a["merge_pred"], b["split_gt"])


def test_iou_exactly_half_not_matched():
    g = rect(0, 0, 10, 10)                                     # 100 px
    p = rect(0, 0, 10, 20)                                     # 200 px, contains g -> IoU = 100/200
    m = match_instances([g], [p], shape=S)
    assert m.iou[0, 0] == 0.5 and m.tp == 0 and m.fp == 1 and m.fn == 1
    p2 = np.zeros(S, bool); p2[0:10, 0:10] = True; p2[10, 0:9] = True  # 109 px -> IoU 100/109
    assert match_instances([g], [p2], shape=S).tp == 1
    p3 = rect(0, 0, 10, 10); p3[10, 0:99] = False; p3 = p3 | rect(10, 0, 11, 99)  # IoU 100/199 > .5
    assert match_instances([g], [p3], shape=S).tp == 1


def test_empty_cases():
    nan = float("nan")
    row, _, _ = evaluate_image([], [], shape=S)
    assert math.isnan(row["pq"]) and row["pq_empty1"] == 1.0 and math.isnan(row["semantic_dice"])
    row, _, _ = evaluate_image([rect(0, 0, 5, 5)], [], shape=S)
    assert (row["tp"], row["fp"], row["fn"], row["pq"]) == (0, 0, 1, 0.0) and row["fn_missed"] == 1
    row, _, _ = evaluate_image([], [rect(0, 0, 5, 5)], shape=S)
    assert (row["tp"], row["fp"], row["fn"], row["pq"]) == (0, 1, 0, 0.0) and row["fp_spurious"] == 1
    # an all-zero predicted mask is still an instance (FP)
    row, _, _ = evaluate_image([rect(0, 0, 5, 5)], [rect(0, 0, 5, 5), np.zeros(S, bool)], shape=S)
    assert (row["tp"], row["fp"], row["fn"]) == (1, 1, 0)
    # dataset of only empty records: pooled undefined (NaN), mean excludes, mean_empty1 = 1
    s, _ = evaluate_dataset([("e1", [], []), ("e2", None, None)])
    assert math.isnan(s["PQ_pooled"]) and math.isnan(s["PQ_img_mean"]) and s["PQ_img_mean_empty1"] == 1.0
    assert s["n_records_both_empty"] == 2
    del nan


def test_pooled_vs_image_mean():
    g = rect(10, 10, 30, 30)
    three = [g, rect(50, 50, 60, 90), rect(100, 10, 140, 20)]
    items = [("perfect3", three, three),
             ("miss+fp", [g], [rect(100, 100, 120, 120)]),
             ("empty", [], [])]
    s, df = evaluate_dataset(items)
    assert close(s["PQ_pooled"], 3.0 / (3 + 0.5 + 0.5))          # pooled: TP=3, FP=1, FN=1
    assert close(s["PQ_img_mean"], 0.5)                           # (1 + 0) / 2, empty excluded
    assert close(s["PQ_img_mean_empty1"], 2.0 / 3)                # (1 + 0 + 1) / 3
    assert close(s["SQ_pooled"] * s["RQ_pooled"], s["PQ_pooled"])


def test_duplicates_unique_vs_host():
    g = rect(10, 10, 30, 60)
    row, pg, pp = evaluate_image([g], [g.copy(), g.copy()], shape=S)
    # standard: one-to-one -> TP=1, FP=1 (duplicate), PQ = 1/1.5
    assert (row["tp"], row["fp"], row["fn"]) == (1, 1, 0) and close(row["pq"], 1 / 1.5)
    assert row["fp_duplicate"] == 1 and row["n_nonunique_cand"] == 1 and not row["unique_by_construction"]
    # host-style (as described by third parties): every IoU>0.5 pair is a TP -> PQ = 2/2
    assert (row["tp_host"], row["fp_host"], row["fn_host"]) == (2, 0, 0) and row["pq_host"] == 1.0
    assert row["pred_overlap_px"] == int(g.sum())
    # overlapping GTs (annotator overlap) + one pred: unique picks max total IoU
    g1, g2 = rect(10, 10, 30, 60), rect(10, 12, 30, 60)
    m = match_instances([g1, g2], [g1], shape=S)
    assert m.n_candidates == 2 and m.tp == 1 and m.pairs[0][0] == 0 and m.fn == 1


def test_iou_matrix_against_pycocotools_and_dense():
    rng = np.random.default_rng(7)
    shape = (300, 280)
    def blob():
        m = np.zeros(shape, bool)
        y, x = rng.integers(0, 260, 2)
        h, w = rng.integers(3, 60, 2)
        m[y:y + h, x:x + w] = True
        m &= rng.random(shape) < 0.9
        return m
    gts = [blob() for _ in range(9)] + [np.zeros(shape, bool)]
    prs = [blob() for _ in range(11)]
    G, P = InstanceSet.from_masks(gts), InstanceSet.from_masks(prs)
    inter = intersection_matrix(G, P)
    iou = iou_from_intersection(inter, G.areas, P.areas)
    dense = np.array([[np.logical_and(a, b).sum() for b in prs] for a in gts])
    assert np.array_equal(inter, dense)
    enc = lambda ms: [rle.encode_rle_dict(m) for m in ms]
    ref = np.asarray(mask_util.iou(enc(prs), enc(gts), [0] * len(gts))).T  # (n_gt, n_pred)
    assert np.allclose(iou, ref, atol=1e-12)
    # label-map constructor == masks constructor for disjoint instances
    lab = np.zeros(shape, np.int32)
    lab[5:20, 5:50] = 3; lab[100:140, 30:35] = 7; lab[200:210, 200:270] = 12
    L = InstanceSet.from_label_map(lab)
    M = InstanceSet.from_masks([lab == 3, lab == 7, lab == 12])
    assert np.array_equal(L.boxes, M.boxes) and np.array_equal(L.areas, M.areas)
    assert all(np.array_equal(a, b) for a, b in zip(L.crops, M.crops))
    # RLE constructor (toBbox-based crop) == masks constructor, incl. a run wrapping across
    # columns (bottom of column 3 -> top of column 4), where pycocotools toBbox is loose in y
    wrap = np.zeros(shape, bool); wrap[-2:, 3] = True; wrap[:2, 4] = True
    mid = np.zeros(shape, bool); mid[150:152, 3] = True; mid[150:153, 4] = True
    allm = prs + [wrap, mid, np.zeros(shape, bool)]
    R = InstanceSet.from_rles([rle.encode(m) for m in allm], *shape)
    M2 = InstanceSet.from_masks(allm)
    assert np.array_equal(R.boxes, M2.boxes) and np.array_equal(R.areas, M2.areas)
    assert all(np.array_equal(a, b) for a, b in zip(R.crops, M2.crops))
    assert np.array_equal(intersection_matrix(G, R)[:, :len(prs)], inter)


def test_within_is_exact_euclidean():
    from scipy import ndimage
    from filseg.metrics import _within
    rng = np.random.default_rng(5)
    t = np.zeros((90, 120), bool)
    t[rng.integers(0, 90, 25), rng.integers(0, 120, 25)] = True
    t[0, 0] = t[89, 119] = True                                    # image corners (clipping)
    edt = ndimage.distance_transform_edt(~t)
    for tol in (1, 2, 3, 5):
        assert np.array_equal(_within(t, tol), edt <= tol), tol


# ----------------------------------------------------------------------------- thin / boundary
def test_thin_region():
    body = rect(50, 20, 70, 200)                                  # 20 px wide body
    branch = rect(70, 100, 110, 102)                              # 2 px wide, 40 px long
    gt = body | branch
    t = thin_region(gt, 2)
    assert t[75:110, 100:102].all()                               # branch is thin
    assert not t[55:65, 30:190].any()                             # body interior is not thin
    cfg = EvalConfig(thin_radius=2)
    r_full, _, _ = evaluate_image([gt], [gt], cfg, shape=S)
    r_body, _, _ = evaluate_image([gt], [body], cfg, shape=S)
    assert r_full["thin_region_recall"] == 1.0 and r_full["thin_region_precision"] == 1.0
    assert r_body["thin_region_recall"] < 0.2                      # branch missed
    fake = body | rect(30, 150, 50, 152)                           # 2 px branch where no gt exists
    r_fake, _, _ = evaluate_image([gt], [fake], cfg, shape=S)
    assert r_fake["thin_region_precision"] < 0.5 < r_full["thin_region_precision"]
    # tolerance version accepts a 1-px shifted branch
    shifted = body | rect(70, 101, 110, 103)
    r0, _, _ = evaluate_image([gt], [shifted], EvalConfig(thin_tol_px=0), shape=S)
    r1, _, _ = evaluate_image([gt], [shifted], EvalConfig(thin_tol_px=1), shape=S)
    assert r1["thin_region_recall"] > r0["thin_region_recall"]


def test_boundary_f():
    g = rect(50, 50, 100, 150)
    r1, pg, _ = evaluate_image([g], [rect(51, 51, 101, 151)], EvalConfig(boundary_tol_px=2), shape=S)
    assert r1["boundary_f"] == 1.0 and pg[0]["tp_boundary_f"] == 1.0
    r5, pg5, _ = evaluate_image([g], [rect(55, 55, 105, 155)], EvalConfig(boundary_tol_px=2), shape=S)
    assert r5["boundary_f"] < 0.5 and pg5[0]["tp_boundary_f"] == r5["boundary_f"]


def test_parallel_equals_serial():
    rng = np.random.default_rng(11)
    items = []
    for k in range(6):
        gts = [rect(*sorted(rng.integers(0, 256, 2)), *sorted(rng.integers(0, 256, 2))) for _ in range(3)]
        prs = [np.roll(m, int(rng.integers(-5, 5)), axis=1) for m in gts]
        items.append((f"img{k}", gts, prs))
    s1, d1 = evaluate_dataset(items)
    s2, d2 = evaluate_dataset(items, n_workers=3)
    for k, v in s1.items():
        if isinstance(v, float):
            assert close(v, s2[k], 1e-12), k
    assert d1.equals(d2)


def test_submission_csv_end_to_end_multi_annotator():
    """masks -> CSV -> read -> InstanceSet.from_rles -> evaluate against 2 annotation sets of one stem."""
    import tempfile
    from filseg.metrics import build_items, stem_of_record
    shape = (128, 160)
    A = [rect(10, 10, 30, 80, shape), rect(60, 20, 70, 150, shape)]     # annotator 1
    B = [rect(10, 10, 30, 70, shape)]                                     # annotator 2 (one filament, shorter)
    pred = [rect(10, 10, 30, 80, shape), rect(60, 20, 70, 150, shape)]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.csv")
        rle.write_submission([(rle.make_filament_id("20140609195854Bh", k), rle.encode(m)) for k, m in enumerate(pred)], p)
        by = rle.group_submission_by_stem(rle.read_submission(p))
    preds = {s: InstanceSet.from_rles(c, *shape) for s, c in by.items()}
    gts = {"040301-20140609195854Bh": InstanceSet.from_masks(A), "040302-20140609195854Bh": InstanceSet.from_masks(B)}
    assert stem_of_record("040302-20140609195854Bh") == "20140609195854Bh"
    items = build_items(gts, preds, shape)
    s, df = evaluate_dataset(items)
    # record 1: 2 TP (IoU 1). record 2: pred0 vs B IoU = 1200/1400 -> TP; pred1 FP.
    iou_b = (20 * 60) / (20 * 70)
    assert (s["TP"], s["FP"], s["FN"]) == (3, 1, 0)
    assert close(s["PQ_pooled"], (2 + iou_b) / (3 + 0.5))
    # a stem with no predictions -> all FN
    s2, _ = evaluate_dataset(build_items(gts, {}, shape))
    assert (s2["TP"], s2["FN"], s2["PQ_pooled"]) == (0, 3, 0.0)


# ----------------------------------------------------------------------------- real data smoke
def _load_records():
    d = json.load(open(ANN))
    by_img = {im["id"]: [] for im in d["images"]}
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    by_stem: dict = {}
    for iid in by_img:
        by_stem.setdefault(iid.split("-", 1)[1], []).append(iid)
    return by_img, by_stem


def _stem_items(args):
    """Build InstanceSets for all annotation sets of one stem (crops only, small)."""
    stem, iids, anns = args
    return stem, {iid: InstanceSet.from_coco_anns(anns[iid]) for iid in iids}


def test_real_smoke(n_workers: int = 16):
    if not os.path.exists(ANN):
        print("SKIP real-data metric test: annotation file not found")
        return
    import multiprocessing as mp
    import pandas as pd

    t0 = time.time()
    by_img, by_stem = _load_records()
    jobs = [(s, iids, {i: by_img[i] for i in iids}) for s, iids in by_stem.items()]
    with mp.get_context("fork").Pool(min(n_workers, 16), initializer=single_thread_worker) as pool:
        sets = dict(pool.imap_unordered(_stem_items, jobs, chunksize=4))
    t_build = time.time() - t0

    # (1) every annotation set vs itself -> PQ must be exactly 1 (both conventions)
    t1 = time.time()
    self_items = [(iid, S_, S_) for s in sets.values() for iid, S_ in s.items()]
    s_self, d_self = evaluate_dataset(self_items, n_workers=n_workers)
    t_self = time.time() - t1
    assert s_self["PQ_pooled"] == 1.0 and s_self["PQ_pooled_host"] <= 1.0 + 1e-12
    print(f"[real] self-PQ over {len(self_items)} records: PQ_pooled={s_self['PQ_pooled']:.6f} "
          f"PQ_pooled_host={s_self['PQ_pooled_host']:.6f} nonunique_records={s_self['n_records_nonunique']} "
          f"({t_self:.1f}s, {t_self / len(self_items) * 1000:.0f} ms/record incl. thin+boundary)")

    # (2) inter-annotator: ordered pairs (A as GT, B as 'prediction') on multi-annotated stems
    t2 = time.time()
    pair_items = []
    for stem, s in sets.items():
        ids = sorted(s)
        for a in ids:
            for b in ids:
                if a != b:
                    pair_items.append((f"{a}|{b.split('-')[0]}", s[a], s[b]))
    s_ia, d_ia = evaluate_dataset(pair_items, n_workers=n_workers)
    t_ia = time.time() - t2
    os.makedirs(AUDIT, exist_ok=True)
    d_ia.to_csv(os.path.join(AUDIT, "m1_interannotator_pq_records.csv"), index=False)
    keys = ["n_records", "PQ_pooled", "SQ_pooled", "RQ_pooled", "PQ_pooled_host", "PQ_img_mean",
            "PQ_img_mean_host", "TP", "FP", "FN", "TP_host", "n_nonunique_cand", "gt_best_iou_mean",
            "semantic_dice_pooled_px", "semantic_dice_img_mean", "n_split_gt", "n_merge_pred", "n_missed_gt",
            "n_spurious_pred", "thin_region_recall_pooled_px", "thin_region_precision_pooled_px",
            "boundary_f_pooled", "tp_boundary_f_mean"]
    print(f"[real] inter-annotator ordered pairs: {len(pair_items)} records ({t_ia:.1f}s)")
    for k in keys:
        v = s_ia[k]
        print(f"    {k:34s} {v:.4f}" if isinstance(v, float) else f"    {k:34s} {v}")
    with open(os.path.join(AUDIT, "m1_metric_smoke_summary.json"), "w") as f:
        json.dump(dict(build_s=t_build, self_eval_s=t_self, interannotator_s=t_ia,
                       self_summary={k: v for k, v in s_self.items() if k != "config"},
                       interannotator_summary={k: v for k, v in s_ia.items()}), f, indent=1, default=float)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and k != "test_real_smoke"]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    if "--skip-real" not in sys.argv:
        test_real_smoke()
        print("real-data metric smoke test: OK")
