#!/usr/bin/env python
"""Task IA -- inter-annotator agreement on the multi-set train files (CPU only).

For every train file with >= 2 annotation sets, every ORDERED pair (A as GT, B as "prediction")
is scored with the verified metric module (filseg.metrics.evaluate_image, standard PQ with strict
IoU > 0.5, plus host-style all_pairs).  On top of that this script adds analysis-only measures:

  * refined cause of every unmatched A instance (exclusive, precedence order):
      merged_by_other  (>=2 A instances lie mostly inside one B instance; filseg 'merged')
      split_by_other   (A instance broken into >=2 B fragments; filseg 'split')
      absent_in_other  (< 10 % of the A instance is covered by ANY B instance -> existence)
      other_larger     (most-overlapping B instance covers >= 70 % of A, but A covers < 70 % of it)
      other_smaller    (A covers >= 70 % of the B instance, B covers < 70 % of A)
      partial_overlap  (both coverages < 70 %)
    (filseg's own exclusive fn_cause 'missed' = best IoU < 0.1 after the merged/split precedence -- not
    the non-exclusive counter n_missed_gt -- is also kept; it mixes existence with
    "small instance inside a much larger one", hence the coverage-based split.)
  * pixel decomposition of A-only pixels (U_A minus U_B): pixels of absent instances, and the rest
    binned by Euclidean distance to U_B: near (<= 2 px), mid (2-5 px], far (> 5 px).
  * matched (TP) pairs: IoU, Dice, width ratio (area / skeleton length), skeleton-length ratio,
    terminal-branch count difference (skeleton proxy from the audit, NOT barb GT), and the
    pixel decomposition of the pair XOR (near/mid/far w.r.t. the partner).
  * thin-region (mask minus opening by the OpenCV elliptical element
    cv2.getStructuringElement(MORPH_ELLIPSE, (2r+1, 2r+1)) = filseg.metrics.thin_region -- slightly larger
    than the Euclidean disk(r): 33 vs 29 px at r = 3 --, r = 3 and 2; computed from masks only) coverage
    with 2 px tolerance, restricted to A instances that are NOT absent in B, next to the same
    coverage for the non-thin "core" -> isolates thin-branch inclusion from existence.
  * RQ / PQ at IoU thresholds 0.3 ... 0.7 (one-to-one, max-total-IoU assignment; 0.5 == standard).

Sets are NEVER merged into a target (no union / intersection labels are produced); union masks are
used only to measure coverage.  'spine' and 'category_id' are not used at all.

Outputs (outputs/audit/):
  inter_annotator.csv            one row per ordered pair (1196 rows)
  inter_annotator_instances.csv  one row per (ordered pair, A instance)
  inter_annotator_sets.csv       one row per annotation set of a multi-set file (within-file contrasts)
  inter_annotator_examples.csv   example files per disagreement type (for later visualisation)
  inter_annotator_summary.json   all summary numbers quoted in docs/inter_annotator.md

Usage:
  PYTHONPATH=src .venv/bin/python scripts/inter_annotator.py [--workers 12] [--reuse]
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root (path-independent)

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, _REPO + "/src")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from filseg.data import index_coco, load_coco, parse_image_id  # noqa: E402
from filseg.metrics import (EvalConfig, InstanceSet, evaluate_image, intersection_matrix,  # noqa: E402
                            iou_from_intersection, match_instances, single_thread_worker,
                            thin_region)

PROJECT = Path(_REPO)
OUT = PROJECT / "outputs" / "audit"
INST_CSV = OUT / "instances.csv"

CFG = EvalConfig()                    # defaults: IoU > 0.5, tau 0.5, min 10 px, thin r 3, boundary d 2
ABSENT_COV = 0.10                     # A instance counted absent in B if < 10 % covered by U_B
CONTAIN = 0.70                        # coverage threshold for other_larger / other_smaller
TOL = 2                               # px tolerance for thin / core coverage
NEAR, MID = 2.0, 5.0                  # distance bins (px) for XOR pixel decomposition
THRS = (0.3, 0.4, 0.5, 0.6, 0.7)
SET_RAW_COLS = ["set_id", "file_name", "n_sets_file", "n_inst", "union_px", "sum_area", "width_set",
                "thin3_frac", "median_contrast", "term_per_inst"]
MIN_THIN_CC = 20                      # px; thin-region components smaller than this are corners / jitter
INST_COLS = ["area", "contrast", "contrast_snr", "mean_width", "skel_len", "centroid_r_frac",
             "n_terminal_branches_gt_w", "width_dt_med"]

# globals filled in main() before forking
ANNS: dict = {}
INST_ATTR: dict = {}


# ------------------------------------------------------------------------------------------ helpers
def _dist_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance of every pixel to the nearest True pixel of `mask` (0 inside)."""
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _bins(d: np.ndarray) -> tuple[int, int, int]:
    return int((d <= NEAR).sum()), int(((d > NEAR) & (d <= MID)).sum()), int((d > MID).sum())


def _set_precompute(S: InstanceSet) -> dict:
    U = S.union()
    T3 = np.zeros(S.shape, bool)
    T2 = np.zeros(S.shape, bool)
    L = np.zeros(S.shape, np.int16)       # instance label map (1-based; overlaps: last wins, 5 records)
    thin3_crops = []
    for k in range(len(S)):
        t3 = thin_region(S.crops[k], 3)
        t2 = thin_region(S.crops[k], 2)
        S.paint(T3, k, t3)
        S.paint(T2, k, t2)
        thin3_crops.append(t3)
        y0, x0, y1, x1 = S.boxes[k]
        L[y0:y1, x0:x1][S.crops[k]] = k + 1
    DI = cv2.distanceTransform(U.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)  # depth inside U
    RIM = U & ~T3 & (DI <= 3.0)           # non-thin pixels at the same depth range as thin pixels
    return dict(U=U, T3=T3, T2=T2, L=L, D=_dist_to(U), RIM=RIM, thin3=thin3_crops)


def _pair_geometry(SA: InstanceSet, i: int, SB: InstanceSet, j: int, thin_i: np.ndarray):
    """XOR decomposition + thin/core coverage for one matched pair, on the union bbox."""
    by0 = min(SA.boxes[i, 0], SB.boxes[j, 0]); bx0 = min(SA.boxes[i, 1], SB.boxes[j, 1])
    by1 = max(SA.boxes[i, 2], SB.boxes[j, 2]); bx1 = max(SA.boxes[i, 3], SB.boxes[j, 3])
    H, W = by1 - by0, bx1 - bx0
    a = np.zeros((H, W), bool); b = np.zeros((H, W), bool); ta = np.zeros((H, W), bool)
    y0, x0, y1, x1 = SA.boxes[i]
    a[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0] = SA.crops[i]
    ta[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0] = thin_i
    y0, x0, y1, x1 = SB.boxes[j]
    b[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0] = SB.crops[j]
    da, db = _dist_to(a), _dist_to(b)       # exact: all pixels of a / b are inside the bbox
    a_only = a & ~b
    b_only = b & ~a
    na = _bins(db[a_only]); nb = _bins(da[b_only])
    core = a & ~ta
    return dict(xor_near=na[0] + nb[0], xor_mid=na[1] + nb[1], xor_far=na[2] + nb[2],
                thin_px=int(ta.sum()), thin_cov2=int((ta & (db <= TOL)).sum()),
                core_px=int(core.sum()), core_cov2=int((core & (db <= TOL)).sum()))


def _attr(sid: str, k: int, col: str) -> float:
    return float(INST_ATTR[sid][col][k])


# ------------------------------------------------------------------------------------------ worker
def process_file(args):
    file_name, set_ids = args
    sets = {sid: InstanceSet.from_coco_anns(ANNS[sid]) for sid in set_ids}
    for sid, S in sets.items():                         # join check with the audit table
        if not np.array_equal(S.areas.astype(float), INST_ATTR[sid]["area"]):
            raise RuntimeError(f"instances.csv join mismatch for {sid}")
    pre = {sid: _set_precompute(S) for sid, S in sets.items()}
    pair_rows, inst_rows = [], []
    for A in set_ids:
        for B in set_ids:
            if A == B:
                continue
            SA, SB, PA, PB = sets[A], sets[B], pre[A], pre[B]
            key = f"{A}|{B}"
            row, per_gt, _ = evaluate_image(SA, SB, CFG, image_key=key, return_instances=True)
            inter = intersection_matrix(SA, SB)
            iou = iou_from_intersection(inter, SA.areas, SB.areas)
            ia, ib = parse_image_id(A), parse_image_id(B)
            out = dict(file_name=file_name, stem=ia["stem"], set_a=A, set_b=B, group=ia["group"],
                       prefix_a=ia["prefix"], prefix_b=ib["prefix"], slot_a=ia["annotator"],
                       slot_b=ib["annotator"], n_sets_file=len(set_ids))
            out.update({k: v for k, v in row.items() if k != "image_key"})
            out["count_diff_b_minus_a"] = row["n_pred"] - row["n_gt"]
            out["area_a_px"] = int(SA.areas.sum()); out["area_b_px"] = int(SB.areas.sum())
            out["union_area_ratio_b_over_a"] = row["union_pred_px"] / max(row["union_gt_px"], 1)
            mu = match_instances(SA, SB, CFG.iou_thr, "unique", inter=inter)
            tp_ious = np.array([p[2] for p in mu.pairs])
            out["matched_iou_mean"] = float(tp_ious.mean()) if tp_ious.size else np.nan
            out["matched_dice_mean"] = float((2 * tp_ious / (1 + tp_ious)).mean()) if tp_ious.size else np.nan
            for t in THRS:
                m = match_instances(SA, SB, t, "unique", inter=inter)
                den = m.tp + 0.5 * m.fp + 0.5 * m.fn
                out[f"tp_t{int(t * 100):02d}"] = m.tp
                out[f"sumiou_t{int(t * 100):02d}"] = m.sum_iou
                out[f"den_t{int(t * 100):02d}"] = den

            # ---- refined causes per A instance
            U_B, D_B = PB["U"], PB["D"]
            cause = {}
            cov = np.zeros(len(SA))
            for g in per_gt:
                k = g["gt_idx"]
                y0, x0, y1, x1 = SA.boxes[k]
                cov[k] = (SA.crops[k] & U_B[y0:y1, x0:x1]).sum() / max(SA.areas[k], 1)
                if g["matched"]:
                    c = "tp"
                elif g["fn_cause"] == "merged":
                    c = "merged_by_other"
                elif g["fn_cause"] == "split":
                    c = "split_by_other"
                elif cov[k] < ABSENT_COV:
                    c = "absent_in_other"
                else:
                    j = int(inter[k].argmax())
                    rg = inter[k, j] / max(SA.areas[k], 1)
                    rp = inter[k, j] / max(SB.areas[j], 1)
                    c = ("other_larger" if (rg >= CONTAIN and rp < CONTAIN) else
                         "other_smaller" if (rp >= CONTAIN and rg < CONTAIN) else "partial_overlap")
                cause[k] = c
            for c in ("tp", "merged_by_other", "split_by_other", "absent_in_other", "other_larger",
                      "other_smaller", "partial_overlap"):
                out[f"cause_{c}"] = sum(v == c for v in cause.values())

            # ---- pixel decomposition of A-only pixels
            absent_ids = [k + 1 for k, c in cause.items() if c == "absent_in_other"]
            a_only = PA["U"] & ~U_B
            absent_mask = np.isin(PA["L"], absent_ids) if absent_ids else np.zeros(SA.shape, bool)
            out["aonly_px"] = int(a_only.sum())
            out["aonly_absent_px"] = int((a_only & absent_mask).sum())
            n, m_, f = _bins(D_B[a_only & ~absent_mask])
            out["aonly_near_px"], out["aonly_mid_px"], out["aonly_far_px"] = n, m_, f

            # ---- thin vs core coverage (tol 2 px) over A instances present in B
            colo = PA["U"] & ~absent_mask
            within = D_B <= TOL
            for r, T in ((3, PA["T3"]), (2, PA["T2"])):
                t = T & colo
                out[f"colo_thin{r}_px"] = int(t.sum())
                out[f"colo_thin{r}_cov2_px"] = int((t & within).sum())
            core = colo & ~PA["T3"]
            out["colo_core_px"] = int(core.sum())
            out["colo_core_cov2_px"] = int((core & within).sum())
            rim = colo & PA["RIM"]                  # control at the same depth as thin pixels
            deep = core & ~PA["RIM"]
            out["colo_rim_px"] = int(rim.sum()); out["colo_rim_cov2_px"] = int((rim & within).sum())
            out["colo_deep_px"] = int(deep.sum()); out["colo_deep_cov2_px"] = int((deep & within).sum())
            # thin components (8-conn, >= MIN_THIN_CC px) of co-located A instances omitted by B
            tcc = (PA["T3"] & colo).astype(np.uint8)
            ncc, lab, st, _ = cv2.connectedComponentsWithStats(tcc, connectivity=8)
            big = np.flatnonzero(st[1:, cv2.CC_STAT_AREA] >= MIN_THIN_CC) + 1
            if big.size:
                covd = np.bincount(lab[within & (lab > 0)], minlength=ncc)[big] / st[big, cv2.CC_STAT_AREA]
            else:
                covd = np.zeros(0)
            out["thin_cc_n"] = int(big.size)
            out["thin_cc_omitted"] = int((covd < 0.2).sum())
            out["thin_cc_full"] = int((covd >= 0.8).sum())
            out["thin3_px_a"] = int(PA["T3"].sum())

            # ---- matched pairs: geometry ratios + XOR decomposition
            lw, ll, tb = [], [], []
            agg = defaultdict(int)
            pair_of = {a: (b, v) for a, b, v in mu.pairs}
            for g in per_gt:
                k = g["gt_idx"]
                r = dict(pair=key, file_name=file_name, set_a=A, set_b=B, prefix_a=ia["prefix"],
                         prefix_b=ib["prefix"], group=ia["group"], gt_idx=k, area=int(SA.areas[k]),
                         contrast=_attr(A, k, "contrast"), contrast_snr=_attr(A, k, "contrast_snr"),
                         mean_width=_attr(A, k, "mean_width"), skel_len=_attr(A, k, "skel_len"),
                         centroid_r_frac=_attr(A, k, "centroid_r_frac"),
                         n_term_a=_attr(A, k, "n_terminal_branches_gt_w"),
                         best_iou=g["best_iou"], best_dice=g["best_dice"], cov_by_other=float(cov[k]),
                         fn_cause_filseg=g["fn_cause"], cause=cause[k], n_fragments=g["n_fragments"])
                j = int(inter[k].argmax()) if inter.shape[1] and inter[k].max() > 0 else -1
                r["partner"] = j
                if j >= 0:
                    r["partner_rg"] = inter[k, j] / max(SA.areas[k], 1)
                    r["partner_rp"] = inter[k, j] / max(SB.areas[j], 1)
                    r["partner_iou"] = float(iou[k, j])
                    r["log_width_ratio"] = math.log(max(_attr(B, j, "mean_width"), 1e-6) /
                                                    max(_attr(A, k, "mean_width"), 1e-6))
                    r["log_len_ratio"] = math.log(max(_attr(B, j, "skel_len"), 1.0) /
                                                  max(_attr(A, k, "skel_len"), 1.0))
                    r["log_area_ratio"] = math.log(max(SB.areas[j], 1) / max(SA.areas[k], 1))
                    r["n_term_b"] = _attr(B, j, "n_terminal_branches_gt_w")
                if k in pair_of:
                    jj, v = pair_of[k]
                    geo = _pair_geometry(SA, k, SB, jj, PA["thin3"][k])
                    r.update({f"tp_{kk}": vv for kk, vv in geo.items()})
                    r["tp_iou"] = v
                    for kk, vv in geo.items():
                        agg[kk] += vv
                    lw.append(r["log_width_ratio"]); ll.append(r["log_len_ratio"])
                    tb.append(r["n_term_b"] - r["n_term_a"])
                inst_rows.append(r)
            for kk in ("xor_near", "xor_mid", "xor_far", "thin_px", "thin_cov2", "core_px", "core_cov2"):
                out[f"tp_{kk}"] = int(agg[kk])
            out["tp_median_log_width_ratio"] = float(np.median(lw)) if lw else np.nan
            out["tp_median_log_len_ratio"] = float(np.median(ll)) if ll else np.nan
            out["tp_mean_term_diff_b_minus_a"] = float(np.mean(tb)) if tb else np.nan
            pair_rows.append(out)
    set_rows = []
    for sid, S in sets.items():
        a = INST_ATTR[sid]
        set_rows.append(dict(set_id=sid, file_name=file_name, n_sets_file=len(set_ids),
                             n_inst=len(S), union_px=int(pre[sid]["U"].sum()), sum_area=float(a["area"].sum()),
                             width_set=float(a["area"].sum() / max(a["skel_len"].sum(), 1.0)),
                             thin3_frac=float(pre[sid]["T3"].sum() / max(pre[sid]["U"].sum(), 1)),
                             median_contrast=float(np.nanmedian(a["contrast"])),
                             term_per_inst=float(np.mean(a["n_terminal_branches_gt_w"]))))
    return pair_rows, inst_rows, set_rows


# ------------------------------------------------------------------------------------------ stats helpers
def q(x, ps=(10, 25, 50, 75, 90)):
    x = np.asarray(pd.Series(x).dropna(), float)
    if x.size == 0:
        return {}
    d = {f"p{p}": float(np.percentile(x, p)) for p in ps}
    d["mean"] = float(x.mean()); d["n"] = int(x.size)
    d["iqr"] = d.get("p75", np.nan) - d.get("p25", np.nan)
    return d


def bh(p):
    p = np.asarray(p, float)
    n = np.isfinite(p).sum()
    out = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out


def pooled(df, num="sum_iou", tp="tp", fp="fp", fn="fn"):
    s = df[[num, tp, fp, fn]].sum()
    den = s[tp] + 0.5 * s[fp] + 0.5 * s[fn]
    return dict(PQ=s[num] / den, SQ=s[num] / s[tp], RQ=s[tp] / den)


def boot_ci(df, fn, by="file_name", n=1000, seed=0):
    rng = np.random.default_rng(seed)
    keys = df[by].unique()
    groups = {k: g for k, g in df.groupby(by)}
    vals = []
    for _ in range(n):
        smp = rng.choice(keys, size=len(keys), replace=True)
        vals.append(fn(pd.concat([groups[k] for k in smp], ignore_index=True)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ------------------------------------------------------------------------------------------ analysis
def analyze(P: pd.DataFrame, I: pd.DataFrame, S: pd.DataFrame) -> dict:
    from scipy import stats
    R: dict = {}
    R["n_ordered_pairs"] = int(len(P))
    R["n_files"] = int(P.file_name.nunique())
    R["n_files_by_nsets"] = {int(k): int(v) for k, v in P.groupby("file_name").n_sets_file.first().value_counts().items()}

    # ---- PQ symmetry check and unordered pairs
    P = P.copy()
    key = P.apply(lambda r: "|".join(sorted([r.set_a, r.set_b])), axis=1)
    P = P.assign(upair=key)
    sym = P.groupby("upair").pq.agg(["min", "max"])
    R["pq_symmetry_max_abs_diff"] = float((sym["max"] - sym["min"]).abs().max())
    U = P.sort_values("set_a").groupby("upair").first().reset_index()     # one row per unordered pair
    R["n_unordered_pairs"] = int(len(U))

    R["pooled_ordered"] = pooled(P)
    R["pooled_ordered_host"] = pooled(P, "sum_iou_host", "tp_host", "fp_host", "fn_host")
    R["pooled_ci95_by_file"] = boot_ci(P, lambda d: pooled(d)["PQ"])
    R["pooled_unordered"] = pooled(U)
    # per-pair distributions (unordered pairs; PQ, SQ, RQ are symmetric in A<->B)
    R["dist_unordered_pairs"] = {m: q(U[m]) for m in ("pq", "sq", "rq", "semantic_dice", "matched_dice_mean",
                                                      "boundary_f")}
    # per-file mean over its pairs (each file weighted once)
    F = P.groupby("file_name").agg(pq=("pq", "mean"), sq=("sq", "mean"), rq=("rq", "mean"),
                                   n_sets=("n_sets_file", "first"), group=("group", "first"),
                                   n_mean=("n_gt", "mean"), sem_dice=("semantic_dice", "mean"))
    R["dist_file_mean"] = {m: q(F[m]) for m in ("pq", "sq", "rq", "sem_dice")}
    R["pq_by_nsets"] = {int(k): q(g.pq) for k, g in F.groupby("n_sets")}
    R["pq_zero_unordered_pairs"] = int((U.pq == 0).sum())
    R["pq_img_mean_ordered"] = float(P.pq.mean())
    # count / area disagreement (unordered: absolute values)
    R["abs_count_diff_unordered"] = q(U.count_diff_b_minus_a.abs())
    R["frac_unordered_equal_count"] = float((U.count_diff_b_minus_a == 0).mean())
    lar = np.log(U.union_area_ratio_b_over_a.clip(lower=1e-6))
    R["abs_log_union_area_ratio_unordered"] = q(np.abs(lar))
    R["union_area_ratio_max_over_min"] = q(np.exp(np.abs(lar)))

    # ---- IoU threshold sensitivity (pooled over ordered pairs)
    th = {}
    for t in THRS:
        s = f"{int(t * 100):02d}"
        th[s] = dict(RQ=float(P[f"tp_t{s}"].sum() / P[f"den_t{s}"].sum()),
                     PQ=float(P[f"sumiou_t{s}"].sum() / P[f"den_t{s}"].sum()))
    R["threshold_sensitivity"] = th
    gb = I.best_iou
    R["gt_best_iou_hist"] = {f"{lo:.1f}-{hi:.1f}": float(((gb >= lo) & (gb < hi)).mean())
                             for lo, hi in [(0, .1), (.1, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8),
                                            (.8, .9), (.9, 1.01)]}
    R["gt_best_iou_in_0.4_0.5"] = float(((gb > 0.4) & (gb <= 0.5)).mean())

    # ---- cause decomposition (every unmatched instance counted once, from its own set's view)
    causes = ["merged_by_other", "split_by_other", "absent_in_other", "other_larger", "other_smaller",
              "partial_overlap"]
    n_un = int(sum(P[f"cause_{c}"].sum() for c in causes))
    R["n_instances_scored"] = int(len(I))
    R["n_unmatched"] = n_un
    R["cause_counts"] = {c: int(P[f"cause_{c}"].sum()) for c in causes}
    R["cause_share_of_unmatched"] = {c: R["cause_counts"][c] / n_un for c in causes}
    R["filseg_fn_cause_counts"] = {c: int(P[f"fn_{c}"].sum()) for c in ("merged", "split", "missed", "low_iou")}
    R["filseg_missed_vs_absent"] = dict(
        missed=int((I.fn_cause_filseg == "missed").sum()),
        missed_and_absent=int(((I.fn_cause_filseg == "missed") & (I.cause == "absent_in_other")).sum()),
        missed_but_covered=int(((I.fn_cause_filseg == "missed") & (I.cause != "absent_in_other")).sum()))
    s = P[["sum_iou", "tp", "fp", "fn"]].sum()
    den = s.tp + 0.5 * s.fp + 0.5 * s.fn
    loss_rq = 1 - s.tp / den
    loss_sq = (s.tp / den) * (1 - s.sum_iou / s.tp)
    R["pq_deficit"] = dict(total=float(1 - s.sum_iou / den), from_unmatched=float(loss_rq),
                           from_iou_of_matched=float(loss_sq),
                           # over ORDERED pairs an unmatched instance is an FN in (A,B) and an FP in (B,A):
                           # it adds 0.5 + 0.5 = 1 to den -> cost 1/den per FN count (sum == from_unmatched)
                           by_cause={c: float(R["cause_counts"][c] / den) for c in causes})
    grp = {"(i) existence": ["absent_in_other"], "(ii) granularity split/merge": ["merged_by_other", "split_by_other"],
           "(iii) extent/boundary (co-located, IoU<=0.5)": ["other_larger", "other_smaller", "partial_overlap"]}
    R["cause_groups_share_of_unmatched"] = {g: sum(R["cause_share_of_unmatched"][c] for c in cs) for g, cs in grp.items()}
    R["cause_groups_pq_deficit"] = {g: sum(R["pq_deficit"]["by_cause"][c] for c in cs) for g, cs in grp.items()}
    R["cause_groups_pq_deficit"]["(iii-b) IoU<1 of matched pairs (SQ)"] = float(loss_sq)

    # ---- (i) existence: are absent instances faint / small / near the limb?
    ex = I.assign(absent=(I.cause == "absent_in_other"))
    comp = {}
    for col in ("contrast", "contrast_snr", "area", "mean_width", "skel_len", "centroid_r_frac"):
        a = ex.loc[ex.absent, col].dropna(); m = ex.loc[ex.cause == "tp", col].dropna()
        comp[col] = dict(median_absent=float(a.median()), median_tp=float(m.median()),
                         mwu_p=float(stats.mannwhitneyu(a, m).pvalue) if len(a) and len(m) else np.nan)
    R["absent_vs_tp"] = comp
    # P(absent | contrast quintile), P(absent | area quintile)
    for col in ("contrast", "area", "contrast_snr"):
        qs = pd.qcut(ex[col], 5, duplicates="drop")
        tab = ex.groupby(qs, observed=True).agg(p_absent=("absent", "mean"), p_tp=("cause", lambda s: (s == "tp").mean()),
                                                n=("absent", "size"))
        R[f"p_absent_by_{col}_quintile"] = {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in tab.iterrows()}
    # 2-way: faint AND small
    ex["faint"] = ex.contrast_snr < 1.0
    ex["small"] = ex.area < 500
    R["p_absent_faint_small"] = {f"faint={f},small={sm}": dict(p_absent=float(g.absent.mean()), n=int(len(g)))
                                 for (f, sm), g in ex.groupby(["faint", "small"])}
    # extra-instance pixel mass: share of all A-only pixels that come from absent instances
    tot_aonly = P.aonly_px.sum()
    R["aonly_pixel_decomposition"] = {k: float(P[f"aonly_{k}_px"].sum() / tot_aonly)
                                      for k in ("absent", "near", "mid", "far")}
    R["aonly_px_share_of_union_a"] = float(tot_aonly / P.union_gt_px.sum())

    # ---- prevalence: share of unordered pairs (both directions summed) showing each disagreement type
    Ug = P.groupby("upair").agg(absent=("cause_absent_in_other", "sum"),
                                gran=("cause_merged_by_other", "sum"), gran2=("cause_split_by_other", "sum"),
                                ext=("cause_other_larger", "sum"), ext2=("cause_other_smaller", "sum"),
                                ext3=("cause_partial_overlap", "sum"), thin_om=("thin_cc_omitted", "sum"),
                                n=("n_gt", "sum"), width=("tp_median_log_width_ratio", lambda x: np.nanmax(np.abs(x))
                                                         if x.notna().any() else np.nan))
    R["prevalence_unordered_pairs"] = {
        "(i) >=1 instance absent in the other set": float((Ug.absent > 0).mean()),
        "(ii) >=1 merge/split": float(((Ug.gran + Ug.gran2) > 0).mean()),
        "(iii) >=1 co-located instance with IoU<=0.5": float(((Ug.ext + Ug.ext2 + Ug.ext3) > 0).mean()),
        "(iii) median matched width ratio > 1.25x": float((Ug.width > math.log(1.25)).mean()),
        "(iv) >=1 thin component (>=20px) omitted": float((Ug.thin_om > 0).mean()),
        "no unmatched instance at all": float(((Ug.absent + Ug.gran + Ug.gran2 + Ug.ext + Ug.ext2 + Ug.ext3) == 0).mean())}

    # ---- (ii) split vs merge direction
    R["granularity"] = dict(
        n_merge_events=int(P.n_merge_pred.sum()), n_split_events=int(P.n_split_gt.sum()),
        unordered_pairs_with_any_merge_or_split=float(((U.n_merge_pred + U.n_split_gt) > 0).mean()),
        files_with_any_merge_or_split=float(P.groupby("file_name").apply(
            lambda d: ((d.n_merge_pred + d.n_split_gt) > 0).any()).mean()))
    # relation between instance-count difference and merges: B with fewer instances merges more?
    P["merge_minus_split"] = P.n_merge_pred - P.n_split_gt
    R["granularity"]["spearman_countdiff_vs_merge_minus_split"] = float(
        stats.spearmanr(P.count_diff_b_minus_a, P.merge_minus_split).statistic)

    # ---- (iii) extent / boundary: matched-pair geometry
    T = I[I.cause == "tp"]
    xs = T[["tp_xor_near", "tp_xor_mid", "tp_xor_far"]].sum()
    R["tp_pairs"] = dict(
        n=int(len(T)), iou=q(T.tp_iou), dice=q(2 * T.tp_iou / (1 + T.tp_iou)),
        abs_log_width_ratio=q(T.log_width_ratio.abs()), abs_log_len_ratio=q(T.log_len_ratio.abs()),
        frac_width_ratio_gt_1p25=float((T.log_width_ratio.abs() > math.log(1.25)).mean()),
        frac_len_ratio_gt_1p25=float((T.log_len_ratio.abs() > math.log(1.25)).mean()),
        xor_share={"near<=2px": float(xs.tp_xor_near / xs.sum()), "mid2-5px": float(xs.tp_xor_mid / xs.sum()),
                   "far>5px": float(xs.tp_xor_far / xs.sum())},
        spearman_iou_vs_abs_log_width=float(stats.spearmanr(T.tp_iou, T.log_width_ratio.abs()).statistic),
        spearman_iou_vs_abs_log_len=float(stats.spearmanr(T.tp_iou, T.log_len_ratio.abs()).statistic),
        spearman_iou_vs_mean_width=float(stats.spearmanr(T.tp_iou, T.mean_width).statistic))
    L = I[I.cause.isin(["other_larger", "other_smaller", "partial_overlap"])]
    R["lowiou_colocated"] = {c: dict(n=int((L.cause == c).sum()),
                                     median_abs_log_width=float(L.loc[L.cause == c, "log_width_ratio"].abs().median()),
                                     median_abs_log_len=float(L.loc[L.cause == c, "log_len_ratio"].abs().median()),
                                     median_partner_iou=float(L.loc[L.cause == c, "partner_iou"].median()))
                             for c in ("other_larger", "other_smaller", "partial_overlap")}
    # in other_larger / other_smaller: is the size gap mostly length or width?
    LS = L[L.cause.isin(["other_larger", "other_smaller"])]
    R["lowiou_size_gap_dominated_by"] = dict(
        length=float((LS.log_len_ratio.abs() > LS.log_width_ratio.abs()).mean()),
        width=float((LS.log_width_ratio.abs() >= LS.log_len_ratio.abs()).mean()), n=int(len(LS)))

    # ---- (iv) thin-branch inclusion (co-located instances only, tol 2 px)
    R["thin_vs_core_coverage_tol2"] = dict(
        thin_r3=float(P.colo_thin3_cov2_px.sum() / P.colo_thin3_px.sum()),
        thin_r2=float(P.colo_thin2_cov2_px.sum() / P.colo_thin2_px.sum()),
        core=float(P.colo_core_cov2_px.sum() / P.colo_core_px.sum()),
        tp_pairs_thin_r3=float(T.tp_thin_cov2.sum() / T.tp_thin_px.sum()),
        tp_pairs_core=float(T.tp_core_cov2.sum() / T.tp_core_px.sum()),
        filseg_thin_region_recall_tol0_pooled=float(P.thin_gt_covered_px.sum() / P.thin_gt_px.sum()))
    R["thin_vs_core_coverage_tol2"].update(
        rim_same_depth=float(P.colo_rim_cov2_px.sum() / P.colo_rim_px.sum()),
        deep=float(P.colo_deep_cov2_px.sum() / P.colo_deep_px.sum()),
        thin_cc_n=int(P.thin_cc_n.sum()), thin_cc_omitted_frac=float(P.thin_cc_omitted.sum() / P.thin_cc_n.sum()),
        thin_cc_fully_included_frac=float(P.thin_cc_full.sum() / P.thin_cc_n.sum()))
    T2 = T.assign(term_diff=T.n_term_b - T.n_term_a)
    R["tp_terminal_branch_diff"] = dict(
        note="skeleton terminal branches >= max(10px, median width) from the audit; proxy, not barb GT",
        abs_diff=q(T2.term_diff.abs()), frac_abs_diff_ge2=float((T2.term_diff.abs() >= 2).mean()),
        frac_equal=float((T2.term_diff == 0).mean()))
    # thin-region share per set (within-file) is in the set table

    # ---- annotator (6-digit prefix) and slot effects, within-file contrasts
    S = S.copy()
    S["prefix"] = S.set_id.str[:6]
    S["group"] = S.set_id.str[:4]
    S["slot"] = S.set_id.str[4:6]
    for col, lg in (("n_inst", True), ("union_px", True), ("width_set", True), ("thin3_frac", False),
                    ("term_per_inst", False), ("median_contrast", False)):
        v = np.log(S[col].clip(lower=1e-6)) if lg else S[col]
        S[f"v_{col}"] = v
        S[f"d_{col}"] = v - S.groupby("file_name")[f"v_{col}"].transform(
            lambda x: (x.sum() - x) / (len(x) - 1))  # value minus mean of the OTHER sets of the file
    # pairwise behaviour of set X as "B": merges/splits other sets' instances, absent/extra
    PB = P.groupby("set_b").agg(merge_by_x=("n_merge_pred", "sum"), split_by_x=("n_split_gt", "sum"),
                                skipped_by_x=("cause_absent_in_other", "sum"), n_other=("n_gt", "sum"),
                                pq_with_others=("pq", "mean"))
    PA_ = P.groupby("set_a").agg(extra_of_x=("cause_absent_in_other", "sum"), n_x=("n_gt", "sum"))
    S = S.merge(PB, left_on="set_id", right_index=True).merge(PA_, left_on="set_id", right_index=True)
    S["skip_rate"] = S.skipped_by_x / S.n_other.clip(lower=1)        # others' instances X leaves out
    S["extra_rate"] = S.extra_of_x / S.n_x.clip(lower=1)             # X's instances nobody else (pairwise) has
    S["merge_minus_split_per_other"] = (S.merge_by_x - S.split_by_x) / S.n_other.clip(lower=1)

    def effect_table(by):
        rows = []
        for k, g in S.groupby(by):
            r = {by: k, "n_sets": int(len(g)), "n_files": int(g.file_name.nunique())}
            for col in ("d_n_inst", "d_union_px", "d_width_set", "d_thin3_frac", "d_term_per_inst"):
                x = g[col].dropna()
                r[f"{col}_median"] = float(x.median()) if len(x) else np.nan
                r[f"{col}_p"] = (float(stats.wilcoxon(x).pvalue) if len(x) >= 6 and (x != 0).any() else np.nan)
            r["skip_rate"] = float(g.skip_rate.mean()); r["extra_rate"] = float(g.extra_rate.mean())
            r["merge_minus_split_per_other"] = float(g.merge_minus_split_per_other.mean())
            r["pq_with_others"] = float(g.pq_with_others.mean())
            rows.append(r)
        E = pd.DataFrame(rows)
        for col in ("d_n_inst", "d_union_px", "d_width_set", "d_thin3_frac", "d_term_per_inst"):
            E[f"{col}_q"] = bh(E[f"{col}_p"].values)
        return E

    EP = effect_table("prefix")
    ES = effect_table("slot")
    R["prefix_effects"] = EP.to_dict(orient="records")
    R["slot_effects"] = ES.to_dict(orient="records")
    sig = {}
    for col in ("d_n_inst", "d_union_px", "d_width_set", "d_thin3_frac", "d_term_per_inst"):
        e = EP[(EP[f"{col}_q"] < 0.05)]
        sig[col] = [dict(prefix=r.prefix, n_sets=r.n_sets, median=r[f"{col}_median"], q=r[f"{col}_q"])
                    for _, r in e.iterrows()]
    R["prefix_effects_significant_bh05"] = sig
    R["n_prefixes_tested"] = int(len(EP))
    R["n_prefixes_with_ge6_sets"] = int((EP.n_sets >= 6).sum())
    # spread of within-file deviations (how large are annotator-style differences overall)
    R["within_file_dev_abs"] = {c: q(S[c].abs()) for c in ("d_n_inst", "d_union_px", "d_width_set")}
    # consistency: same annotator ranks similarly across files? correlation of d_n_inst with d_union_px
    R["corr_dcount_dunion"] = float(stats.spearmanr(S.d_n_inst, S.d_union_px).statistic)
    R["corr_dcount_dwidth"] = float(stats.spearmanr(S.d_n_inst, S.d_width_set).statistic)
    # variance explained by prefix in d_n_inst / d_union_px (Kruskal-Wallis over prefixes with >= 6 sets)
    big = S[S.prefix.isin(EP.loc[EP.n_sets >= 6, "prefix"])]
    R["kw_prefix"] = {c: float(stats.kruskal(*[g[c].values for _, g in big.groupby("prefix")]).pvalue)
                      for c in ("d_n_inst", "d_union_px", "d_width_set", "d_thin3_frac")}

    # ---- batch (group) effect on agreement
    Fg = F.reset_index()
    Fg = Fg[Fg.group.isin(Fg.group.value_counts()[lambda s: s >= 5].index)]
    R["pq_by_group"] = {g: dict(n_files=int(len(d)), pq_mean=float(d.pq.mean()), pq_median=float(d.pq.median()),
                                n_inst_mean=float(d.n_mean.mean()))
                        for g, d in Fg.groupby("group")}
    R["kw_group_file_pq"] = float(stats.kruskal(*[d.pq.values for _, d in Fg.groupby("group")]).pvalue)
    R["kw_group_file_ninst"] = float(stats.kruskal(*[d.n_mean.values for _, d in Fg.groupby("group")]).pvalue)
    # PQ adjusted for log(n instances) (more instances -> more chances to disagree)
    X = np.c_[np.ones(len(Fg)), np.log(Fg.n_mean)]
    beta, *_ = np.linalg.lstsq(X, Fg.pq.values, rcond=None)
    Fg = Fg.assign(pq_resid=Fg.pq.values - X @ beta)
    R["pq_vs_log_ninst_slope"] = float(beta[1])
    R["spearman_file_pq_vs_ninst"] = float(stats.spearmanr(F.pq, F.n_mean).statistic)
    R["kw_group_file_pq_resid_logn"] = float(stats.kruskal(*[d.pq_resid.values for _, d in Fg.groupby("group")]).pvalue)
    # by slot pair inside groups (e.g. 01 vs 02)
    R["pq_by_slot_pair"] = {k: dict(n=int(len(g)), pq_mean=float(g.pq.mean()))
                            for k, g in U.assign(sp=U.slot_a + "-" + U.slot_b).groupby("sp")}

    # ---- reference "ceilings" (not upper bounds)
    rec = P.groupby("set_a").agg(pq_mean_other=("pq", "mean"), pq_best_other=("pq", "max"),
                                 n_sets=("n_sets_file", "first"))
    R["record_pq_mean_other"] = q(rec.pq_mean_other)
    R["record_pq_best_other"] = q(rec.pq_best_other)
    # pooled "best other" reference: pick, per GT record, the other set with highest pq, pool counts
    best = P.loc[P.groupby("set_a").pq.idxmax()]
    R["pooled_best_other"] = pooled(best)
    # medoid annotator of each 3-set file (oracle choice: the set with the highest mean PQ to the other
    # two), scored against the other two sets -- a human "consensus-like" predictor, NOT an upper bound
    P3 = P[P.n_sets_file == 3]
    mb = P3.groupby("set_b").pq.mean()
    med = mb.groupby(P3.groupby("set_b").file_name.first()).idxmax()
    R["pooled_3set_random_annotator"] = pooled(P3)
    R["pooled_3set_medoid_annotator"] = pooled(P3[P3.set_b.isin(med.values)])
    R["pooled_3set_anti_medoid"] = pooled(P3[P3.set_b.isin(mb.groupby(P3.groupby("set_b").file_name.first()).idxmin().values)])
    # support of each instance in 3-set files: in how many of the 2 other sets is it matched / present
    I3 = I[I.set_a.isin(P3.set_a)]
    sup = I3.groupby(["set_a", "gt_idx"]).agg(matched=("cause", lambda s: int((s == "tp").sum())),
                                              present=("cause", lambda s: int((s != "absent_in_other").sum())))
    R["support_3set"] = dict(
        n_instances=int(len(sup)),
        matched_in_k_of_2_others={int(k): float(v) for k, v in sup.matched.value_counts(normalize=True).sort_index().items()},
        present_in_k_of_2_others={int(k): float(v) for k, v in sup.present.value_counts(normalize=True).sort_index().items()})
    return R, S, EP, ES, F


def pick_examples(P: pd.DataFrame, I: pd.DataFrame, S: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Deterministic example files per disagreement type (file-level, all its pairs aggregated)."""
    F = P.groupby("file_name").agg(
        n_sets=("n_sets_file", "first"), sets=("set_a", lambda s: ";".join(sorted(set(s)))),
        n_inst=("n_gt", "sum"), pq=("pq", "mean"), sq=("sq", "mean"), rq=("rq", "mean"),
        absent=("cause_absent_in_other", "sum"), merged=("cause_merged_by_other", "sum"),
        split=("cause_split_by_other", "sum"), larger=("cause_other_larger", "sum"),
        smaller=("cause_other_smaller", "sum"), partial=("cause_partial_overlap", "sum"),
        tp=("tp", "sum"), colo_thin3_px=("colo_thin3_px", "sum"), colo_thin3_cov=("colo_thin3_cov2_px", "sum"),
        colo_core_px=("colo_core_px", "sum"), colo_core_cov=("colo_core_cov2_px", "sum"),
        width=("tp_median_log_width_ratio", lambda s: float(np.nanmax(np.abs(s))) if s.notna().any() else np.nan),
        count_range=("n_gt", lambda s: int(s.max() - s.min())))
    F["absent_rate"] = F.absent / F.n_inst
    F["gran_rate"] = (F.merged + F.split) / F.n_inst
    F["extent_rate"] = (F.larger + F.smaller + F.partial) / F.n_inst
    F["thin_gap"] = F.colo_core_cov / F.colo_core_px.clip(lower=1) - F.colo_thin3_cov / F.colo_thin3_px.clip(lower=1)
    ab = I[I.cause == "absent_in_other"].groupby("file_name").contrast_snr.median()
    F["absent_median_snr"] = ab.reindex(F.index)
    out = []
    used = set()

    def take(df, typ, why_cols, n=k):
        c = 0
        for fn, r in df.iterrows():
            if fn in used:
                continue
            used.add(fn)
            out.append(dict(type=typ, file_name=fn, sets=r.sets, n_sets=int(r.n_sets), pq_mean=round(r.pq, 3),
                            sq_mean=round(r.sq, 3) if r.sq == r.sq else np.nan, rq_mean=round(r.rq, 3),
                            why="; ".join(f"{c_}={r[c_]:.3g}" if isinstance(r[c_], float) else f"{c_}={r[c_]}"
                                          for c_ in why_cols)))
            c += 1
            if c >= n:
                break

    take(F[(F.absent >= 4)].sort_values(["absent_rate", "absent"], ascending=False),
         "existence (instance present in one set only)", ["absent", "n_inst", "absent_rate", "absent_median_snr"])
    take(F[(F.merged + F.split >= 4)].sort_values(["gran_rate"], ascending=False),
         "granularity (split vs merge)", ["merged", "split", "n_inst", "count_range"])
    G = F[(F.rq >= 0.5) & (F.tp >= 6)].sort_values("width", ascending=False)
    take(G, "boundary/width (matched but width differs)", ["width", "sq", "tp"])
    take(F[(F.larger + F.smaller + F.partial >= 4)].sort_values("extent_rate", ascending=False),
         "extent (co-located, IoU<=0.5: longer/shorter/shifted)", ["larger", "smaller", "partial", "n_inst"])
    take(F[(F.colo_thin3_px >= 3000)].sort_values("thin_gap", ascending=False),
         "thin-region inclusion (thin parts covered much less than core)", ["thin_gap", "colo_thin3_px"])
    take(F[F.n_inst >= 12].sort_values("pq", ascending=False), "high agreement (contrast case)", ["n_inst"], n=3)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reuse", action="store_true", help="reuse per-pair CSVs, only re-run the analysis")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N multi-set files")
    args = ap.parse_args()
    t0 = time.time()
    p_csv, i_csv, s_csv = OUT / "inter_annotator.csv", OUT / "inter_annotator_instances.csv", OUT / "inter_annotator_sets.csv"
    if not (args.reuse and p_csv.exists()):
        coco = load_coco()
        images_by_id, anns_by_id, ids_by_file = index_coco(coco)
        multi = {f: ids for f, ids in sorted(ids_by_file.items()) if len(ids) >= 2}
        if args.limit:
            multi = dict(list(multi.items())[: args.limit])
        need = {sid for ids in multi.values() for sid in ids}
        ANNS.update({sid: anns_by_id[sid] for sid in need})
        inst = pd.read_csv(INST_CSV, usecols=["image_id", "ann_idx"] + INST_COLS)
        inst = inst[inst.image_id.isin(need)].sort_values(["image_id", "ann_idx"])
        for sid, g in inst.groupby("image_id"):
            if not np.array_equal(g.ann_idx.values, np.arange(len(g))):
                raise RuntimeError(f"ann_idx not contiguous for {sid}")
            INST_ATTR[sid] = {c: g[c].to_numpy(float) for c in INST_COLS}
        del coco
        print(f"{len(multi)} multi-set files, {len(need)} sets; loading took {time.time() - t0:.1f}s", flush=True)
        items = list(multi.items())
        pair_rows, inst_rows, set_rows = [], [], []
        if args.workers > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(min(args.workers, 16), initializer=single_thread_worker) as pool:
                for n, (a, b, c) in enumerate(pool.imap_unordered(process_file, items, chunksize=1), 1):
                    pair_rows += a; inst_rows += b; set_rows += c
                    if n % 50 == 0:
                        print(f"  {n}/{len(items)} files  {time.time() - t0:.1f}s", flush=True)
        else:
            single_thread_worker()
            for it in items:
                a, b, c = process_file(it)
                pair_rows += a; inst_rows += b; set_rows += c
        P = pd.DataFrame(pair_rows).sort_values(["file_name", "set_a", "set_b"]).reset_index(drop=True)
        I = pd.DataFrame(inst_rows).sort_values(["file_name", "set_a", "set_b", "gt_idx"]).reset_index(drop=True)
        S = pd.DataFrame(set_rows).sort_values(["file_name", "set_id"]).reset_index(drop=True)
        P.to_csv(p_csv, index=False); I.to_csv(i_csv, index=False); S.to_csv(s_csv, index=False)
        print(f"computed {len(P)} ordered pairs, {len(I)} instance rows in {time.time() - t0:.1f}s", flush=True)
    else:
        P = pd.read_csv(p_csv, dtype={"group": str, "slot_a": str, "slot_b": str, "prefix_a": str, "prefix_b": str})
        I = pd.read_csv(i_csv, dtype={"group": str, "prefix_a": str, "prefix_b": str})
        S = pd.read_csv(s_csv)[SET_RAW_COLS]       # the file also holds derived columns from a previous run
    for c in ("group", "slot_a", "slot_b", "prefix_a", "prefix_b"):
        if c in P:
            P[c] = P[c].astype(str).str.zfill({"group": 4, "slot_a": 2, "slot_b": 2}.get(c, 6))
    R, S2, EP, ES, F = analyze(P, I, S)
    ex = pick_examples(P, I, S)
    ex.to_csv(OUT / "inter_annotator_examples.csv", index=False)
    S2.to_csv(s_csv, index=False)
    R["wall_sec"] = time.time() - t0
    R["config"] = dict(eval=CFG.__dict__, absent_cov=ABSENT_COV, contain=CONTAIN, tol=TOL, near=NEAR, mid=MID,
                       thresholds=THRS)

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    with open(OUT / "inter_annotator_summary.json", "w") as f:
        json.dump(_clean(R), f, indent=1)
    print(json.dumps(_clean({k: v for k, v in R.items() if k not in ("prefix_effects", "slot_effects")}), indent=1))
    print(EP.to_string())
    print(ES.to_string())
    print(ex.to_string())
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
