"""Task BH -- "beyond human quality": leave-one-annotator-out (LOAO) comparison of the model with every human.

Question answered: on the same reference annotations, does the model agree with the OTHER annotators of a
file better than annotator A of that file does?

Protocol (pre-defined before looking at the numbers; docs/beyond_human.md):
  * files: the 296 train files with >= 2 annotation sets (145 x 2, 151 x 3 sets).
  * for every file and every annotation set A of the file (743 "rater views"):
      reference records = every OTHER set R of the file, scored as a separate record (host-style pooling,
      as in the competition reading; filseg.metrics.evaluate_image, EvalConfig() defaults).
      raters scored against exactly these records:
        - "human"         : A's own instances used as predictions          (record key "A|R")
        - "b0"            : OOF b0 instances (thr 0.5, 8-conn CC, min_area 500), same keys "A|R"
        - "e5"            : OOF E5 (b0 + 1536-input mean) instances, same post   (information only)
        - "human_min500"  : A's instances with area < 500 px removed (sensitivity: the model's area filter
                            applied to the human; PQ/SQ/RQ only)
        - "b0_a0", "b0_a250": b0 at thr 0.5 with min_area 0 / 250 (post-processing sensitivity, information only)
    -> 1196 ordered (A, R) records per rater (= the 1196 ordered pairs of docs/inter_annotator.md).
    The model's score against R does not depend on A; it is repeated once per A != R so that the paired
    comparison uses exactly the same records (a 3-set reference is used twice, as for the humans).
  * Asymmetry: the model is out-of-fold (trained on annotations of OTHER files, 14-day grouped folds), never
    on this file; annotator A drew this file. The model's post-processing (thr/min_area) was chosen by
    nested CV to maximise pooled PQ; the humans did not optimise PQ.
  * Aggregates: pooled PQ / SQ / RQ (host convention: every IoU > 0.5 pair is a TP), paired difference
    model - human with a bootstrap CI over FILES (all records of a file move together; seed 2026), overall,
    per annotator id (6-digit prefix of A), per batch group (first 4 digits), separately for 2- and 3-set files.
  * Aspects (same records): thin_region recall/precision (filseg default r=3, tol 0; mask-derived, NOT barb
    accuracy), boundary F at 2 px, per-instance best-IoU distribution, FN causes (missed/split/merged/low_iou),
    FP causes (spurious/fragment/merge/low_iou), instance count bias, existence by contrast/area/limb strata,
    and (3-set files) recall of reference instances that the third annotator also marks ("consensus").

Never reads annotation 'spine' / 'category_id' (instances.csv is read with an explicit column list).
CPU only (the GPU is never touched; torch is not imported). Multiprocessing: forkserver, single-thread workers.

Usage:
  .venv/bin/python scripts/beyond_human.py --workers 12            # full run (~5-10 min on the shared box)
  .venv/bin/python scripts/beyond_human.py --reuse                 # analysis only from the saved tables
  .venv/bin/python scripts/beyond_human.py --reuse --figures       # + render the example figures
Outputs: outputs/beyond_human/{records,ref_instances,rater_instances,file_info,headline,per_annotator,per_group,
         aspects,aspects_per_annotator,existence_strata,gaps_per_rater_view,examples}.csv, summary.json,
         figures/*.png (+ figures_index.json). The 'BH1..BH5' target is evaluated in summary.json['target'].
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from filseg import data as D  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.metrics import (EvalConfig, InstanceSet, evaluate_image, intersection_matrix,  # noqa: E402
                            iou_from_intersection, single_thread_worker)

OUT = ROOT / "outputs" / "beyond_human"
EXPS = {"b0": "b0_r101_s1024", "e5": "E5_ens_b0_E4"}
PROBS = "val_probs_last.npz"
POST = seg.PostCfg(thr=0.5, min_area=500)
CFG = EvalConfig()  # IoU > 0.5 strict, thin r=3 tol 0, boundary tol 2 px
CFG_LIGHT = EvalConfig(compute_thin=False, compute_boundary=False, compute_pair_boundary=False)
SEED = 2026
MIN_SETS_ANNOTATOR = 15   # annotators with >= 15 rater views enter the "human median / best human" target
RATERS_MODEL = ("b0", "e5")
# post-processing sensitivity of b0 (information only): thr 0.5 with min_area 0 / 250 instead of 500
POST_VARIANTS = {"b0_a0": ("b0", 0), "b0_a250": ("b0", 250)}
ALL_MODEL_RATERS = RATERS_MODEL + tuple(POST_VARIANTS)

REC_COLS = ["n_gt", "n_pred", "tp", "fp", "fn", "sum_iou", "tp_host", "fp_host", "fn_host", "sum_iou_host",
            "pq_host", "gt_best_iou_sum", "pred_best_iou_sum", "n_split_gt", "n_split_fragments", "n_merge_pred",
            "n_merged_gt", "n_missed_gt", "n_spurious_pred", "fn_missed", "fn_merged", "fn_split", "fn_low_iou",
            "fp_spurious", "fp_duplicate", "fp_merge", "fp_fragment", "fp_low_iou", "union_gt_px", "union_pred_px",
            "union_inter_px", "thin_gt_px", "thin_gt_covered_px", "thin_pred_px", "thin_pred_in_gt_px",
            "boundary_f", "bnd_pred_hit", "bnd_pred_n", "bnd_gt_hit", "bnd_gt_n", "tp_boundary_f_sum"]
GT_COLS = ["gt_idx", "area", "best_iou", "matched", "fn_cause", "is_split", "in_merge", "n_fragments", "thin_px",
           "thin_covered_px", "matched_iou", "tp_boundary_f"]
PR_COLS = ["pred_idx", "area", "best_iou", "matched", "fp_cause", "is_fragment", "is_merge", "thin_px",
           "thin_in_gt_px"]
FEAT_COLS = ["image_id", "ann_idx", "area", "contrast", "contrast_snr", "centroid_r_frac", "mean_width",
             "frac_skel_w_lt4"]  # explicit list: never touches spine / category_id


# =============================================================================== worker side
_COCO_BY_IMG = None


def _anns_by_image():
    global _COCO_BY_IMG
    if _COCO_BY_IMG is None:
        by = {}
        for a in D.load_coco()["annotations"]:
            by.setdefault(a["image_id"], []).append(a)   # JSON order kept -> index k = ann_idx k
        _COCO_BY_IMG = by
    return _COCO_BY_IMG


def load_sets(set_ids):
    by = _anns_by_image()
    return {sid: InstanceSet.from_coco_anns(by.get(sid, [])) for sid in set_ids}


def load_prob(exp_key, fold, stem):
    z = np.load(ROOT / "outputs" / "exps" / EXPS[exp_key] / f"fold{fold}" / PROBS)
    return z[stem].astype(np.float32)


def min_area_filter(S: InstanceSet, a: int) -> InstanceSet:
    keep = [k for k in range(len(S)) if S.areas[k] >= a]
    return InstanceSet._build(S.shape, [tuple(S.boxes[k]) for k in keep], [S.crops[k] for k in keep])


def file_job(args):
    """All LOAO records of one multi-set file, for every rater."""
    file_name, stem, fold, set_ids = args
    t0 = time.time()
    S = load_sets(set_ids)
    probs = {k: load_prob(k, fold, stem) for k in RATERS_MODEL}
    M = {k: seg.prob_to_instances(probs[k], POST) for k in RATERS_MODEL}
    for k, (exp_k, ma) in POST_VARIANTS.items():
        M[k] = seg.prob_to_instances(probs[exp_k], seg.PostCfg(POST.thr, ma))
    del probs
    recs, gts, prs = [], [], []

    def add(rater, A, R, res):
        row, pg, pp = res
        base = dict(rater=rater, stem=stem, A=A, R=R, rec=f"{A}|{R}")
        recs.append({**base, **{c: row.get(c, np.nan) for c in REC_COLS}})
        for d in pg:
            gts.append({**base, **{c: d[c] for c in GT_COLS}})
        for d in pp:
            prs.append({**base, **{c: d[c] for c in PR_COLS}})

    model_eval = {(k, R): evaluate_image(S[R], P, CFG, image_key=R) for k, P in M.items() for R in set_ids}
    for A in set_ids:
        A500 = min_area_filter(S[A], POST.min_area)
        for R in set_ids:
            if R == A:
                continue
            add("human", A, R, evaluate_image(S[R], S[A], CFG, image_key=f"{A}|{R}"))
            r5, _, _ = evaluate_image(S[R], A500, CFG_LIGHT, image_key=f"{A}|{R}", return_instances=False)
            add("human_min500", A, R, (r5, [], []))
            for k in ALL_MODEL_RATERS:
                add(k, A, R, model_eval[(k, R)])
    info = dict(stem=stem, fold=fold, n_sets=len(set_ids), sec=time.time() - t0,
                **{f"n_inst_{k}": len(P) for k, P in M.items()})
    return recs, gts, prs, info


# =============================================================================== aggregation helpers
ADD = ["n_gt", "n_pred", "tp_host", "fp_host", "fn_host", "sum_iou_host", "tp", "tp_boundary_f_sum",
       "thin_gt_px", "thin_gt_covered_px", "thin_pred_px", "thin_pred_in_gt_px", "bnd_pred_hit", "bnd_pred_n",
       "bnd_gt_hit", "bnd_gt_n", "union_gt_px", "union_pred_px", "union_inter_px", "fn_missed", "fn_split",
       "fn_merged", "fn_low_iou", "n_split_gt", "n_merged_gt", "fp_spurious", "fp_fragment", "fp_merge",
       "fp_low_iou", "fp_duplicate"]
_I = {c: i for i, c in enumerate(ADD)}


def _div(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def _col(X, c):
    return X[..., _I[c]]


def _bf(X):
    bp = _div(_col(X, "bnd_pred_hit"), _col(X, "bnd_pred_n"))
    br = _div(_col(X, "bnd_gt_hit"), _col(X, "bnd_gt_n"))
    return _div(2 * bp * br, bp + br)


# additive (ratio-of-sums) record-level statistics; usable on (n_boot, k) or (k,) arrays
STATS = {
    "PQ": lambda X: _div(_col(X, "sum_iou_host"), _col(X, "tp_host") + .5 * _col(X, "fp_host") + .5 * _col(X, "fn_host")),
    "SQ": lambda X: _div(_col(X, "sum_iou_host"), _col(X, "tp_host")),
    "RQ": lambda X: _div(_col(X, "tp_host"), _col(X, "tp_host") + .5 * _col(X, "fp_host") + .5 * _col(X, "fn_host")),
    "thin_recall": lambda X: _div(_col(X, "thin_gt_covered_px"), _col(X, "thin_gt_px")),
    "thin_precision": lambda X: _div(_col(X, "thin_pred_in_gt_px"), _col(X, "thin_pred_px")),
    "boundary_F2": _bf,
    "semantic_dice_px": lambda X: _div(2 * _col(X, "union_inter_px"), _col(X, "union_gt_px") + _col(X, "union_pred_px")),
}


def rec_aspects(r: pd.DataFrame) -> dict:
    s = r[ADD].sum()
    X = s.values.astype(float)
    out = {k: float(f(X)) for k, f in STATS.items()}
    ng, npd = s["n_gt"], s["n_pred"]
    out.update(
        inst_recall=float(_div(ng - s["fn_host"], ng)), inst_precision=float(_div(npd - s["fp_host"], npd)),
        tp_boundary_F2=float(_div(s["tp_boundary_f_sum"], s["tp"])),
        fn_missed_rate=float(_div(s["fn_missed"], ng)), fn_split_rate=float(_div(s["fn_split"], ng)),
        fn_merged_rate=float(_div(s["fn_merged"], ng)), fn_low_iou_rate=float(_div(s["fn_low_iou"], ng)),
        split_gt_rate=float(_div(s["n_split_gt"], ng)), merged_gt_rate=float(_div(s["n_merged_gt"], ng)),
        fp_spurious_rate=float(_div(s["fp_spurious"], npd)), fp_fragment_rate=float(_div(s["fp_fragment"], npd)),
        fp_merge_rate=float(_div(s["fp_merge"], npd)), fp_low_iou_rate=float(_div(s["fp_low_iou"], npd)),
        count_ratio=float(_div(npd, ng)),
        count_log_bias=float(np.mean(np.log((r.n_pred.values + 1.0) / (r.n_gt.values + 1.0)))),
        count_abs_err=float(np.mean(np.abs(r.n_pred.values - r.n_gt.values))),
        n_records=int(len(r)), n_ref_inst=int(ng), n_rater_inst=int(npd))
    return out


def inst_aspects(g: pd.DataFrame, p: pd.DataFrame, faint_thr: float) -> dict:
    out = {}
    bi = g.best_iou.values
    out["ref_best_iou_median"] = float(np.median(bi)) if len(bi) else np.nan
    out["ref_best_iou_mean"] = float(np.mean(bi)) if len(bi) else np.nan
    out["ref_near_miss_share"] = float(np.mean((bi > 0.4) & (bi <= 0.5))) if len(bi) else np.nan
    out["ref_detected_rate"] = float(np.mean(bi >= 0.1)) if len(bi) else np.nan
    strata = {"faint": g.contrast_snr.values <= faint_thr, "small": g.area.values < 500,
              "ge500": g.area.values >= 500, "faint_ge500": (g.contrast_snr.values <= faint_thr) & (g.area.values >= 500),
              "large": g.area.values >= 2000, "limb": g.centroid_r_frac.values > 0.8,
              "thinshape": g.frac_skel_w_lt4.values >= 0.3}
    if "c_best_iou" in g:
        c = g.c_best_iou.values
        strata["consensus"] = c > 0.5          # third annotator also matches this reference instance
        strata["single_view"] = c < 0.1        # third annotator has nothing there
    for name, m in strata.items():
        m = np.asarray(m, bool)
        out[f"{name}_n"] = int(m.sum())
        out[f"{name}_matched_rate"] = float(np.mean(bi[m] > 0.5)) if m.any() else np.nan
        out[f"{name}_detected_rate"] = float(np.mean(bi[m] >= 0.1)) if m.any() else np.nan
    co = bi >= 0.1
    out["thin_recall_colocated"] = float(_div(g.thin_covered_px.values[co].sum(), g.thin_px.values[co].sum()))
    pb = p.best_iou.values
    out["rater_spurious_share"] = float(np.mean(pb < 0.1)) if len(pb) else np.nan
    cop = pb >= 0.1
    out["thin_precision_colocated"] = float(_div(p.thin_in_gt_px.values[cop].sum(), p.thin_px.values[cop].sum()))
    return out


def all_aspects(r, g, p, faint_thr):
    return {**rec_aspects(r), **inst_aspects(g, p, faint_thr)}


# direction: +1 higher is better, -1 lower is better, 0 closer to zero is better, None = information only
ASPECTS = {
    "PQ": 1, "SQ": 1, "RQ": 1, "inst_recall": 1, "inst_precision": 1,
    "thin_recall": 1, "thin_precision": 1, "thin_recall_colocated": 1, "thin_precision_colocated": 1,
    "boundary_F2": 1, "tp_boundary_F2": 1, "semantic_dice_px": 1,
    "ref_best_iou_median": 1, "ref_near_miss_share": -1, "ref_detected_rate": 1,
    "faint_detected_rate": 1, "faint_matched_rate": 1, "small_detected_rate": 1, "small_matched_rate": 1,
    "ge500_detected_rate": 1, "ge500_matched_rate": 1, "faint_ge500_detected_rate": 1, "faint_ge500_matched_rate": 1,
    "large_matched_rate": 1, "limb_detected_rate": 1, "limb_matched_rate": 1, "thinshape_matched_rate": 1,
    "consensus_matched_rate": 1, "consensus_detected_rate": 1, "single_view_detected_rate": None,
    "fn_missed_rate": -1, "fn_split_rate": -1, "fn_merged_rate": -1, "fn_low_iou_rate": -1,
    "split_gt_rate": -1, "merged_gt_rate": -1,
    "fp_spurious_rate": -1, "fp_fragment_rate": -1, "fp_merge_rate": -1, "fp_low_iou_rate": -1,
    "rater_spurious_share": -1, "count_log_bias": 0, "count_abs_err": -1, "count_ratio": None,
}
# aspects that enter the proposed target's "no aspect below the human median" condition
TARGET_ASPECTS = ["SQ", "RQ", "thin_recall", "thin_precision", "thin_recall_colocated", "boundary_F2",
                  "tp_boundary_F2", "ref_detected_rate", "faint_detected_rate", "small_detected_rate",
                  "limb_detected_rate", "consensus_matched_rate", "fn_missed_rate", "split_gt_rate",
                  "merged_gt_rate", "fp_spurious_rate", "fp_fragment_rate", "count_log_bias"]


def better(model, human, direction):
    if direction is None or not (np.isfinite(model) and np.isfinite(human)):
        return None
    if direction == 1:
        return bool(model > human)
    if direction == -1:
        return bool(model < human)
    return bool(abs(model) < abs(human))


def not_worse(model, human, direction):
    if direction is None or not (np.isfinite(model) and np.isfinite(human)):
        return None
    if direction == 1:
        return bool(model >= human)
    if direction == -1:
        return bool(model <= human)
    return bool(abs(model) <= abs(human))


def boot_weights(n_files, n_boot, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_files, size=(n_boot, n_files))
    return np.stack([np.bincount(row, minlength=n_files) for row in idx]).astype(float)


def file_sums(r: pd.DataFrame, files):
    return r.groupby("stem")[ADD].sum().reindex(files).fillna(0.0).values.astype(float)


def paired_boot(rh: pd.DataFrame, rm: pd.DataFrame, n_boot: int, stats=("PQ", "SQ", "RQ")) -> dict:
    """model - human on the SAME records; bootstrap over files (seed 2026)."""
    assert sorted(rh.rec) == sorted(rm.rec), "raters not on the same records"
    files = sorted(rh.stem.unique())
    H, M = file_sums(rh, files), file_sums(rm, files)
    W = boot_weights(len(files), n_boot)
    BH, BM = W @ H, W @ M
    out = {"n_files": len(files), "n_records": int(len(rh))}
    for s in stats:
        f = STATS[s]
        h, m = float(f(H.sum(0))), float(f(M.sum(0)))
        d = f(BM) - f(BH)
        d = d[np.isfinite(d)]
        out[s] = dict(human=h, model=m, diff=m - h,
                      ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else [np.nan, np.nan],
                      p_boot_model_better=float(np.mean(d > 0)) if len(d) else np.nan)
    return out


# =============================================================================== main
def build_tables(workers: int):
    import multiprocessing as mp
    folds = pd.read_csv(ROOT / "outputs" / "splits" / "folds_v1.csv")
    multi = folds[folds.n_sets >= 2].copy()
    assert len(multi) == 296 and (multi.n_sets == 2).sum() == 145 and (multi.n_sets == 3).sum() == 151
    jobs = []
    for _, row in multi.iterrows():
        stem = Path(row.file_name).stem
        fold = int(row.fold)
        for k in RATERS_MODEL:  # the OOF probabilities must come from the fold that held this file out
            vf = (ROOT / "outputs" / "exps" / EXPS[k] / f"fold{fold}" / "val_files.txt").read_text().split()
            assert row.file_name in vf, (k, row.file_name, fold)
        jobs.append((row.file_name, stem, fold, sorted(row.image_ids.split(";"))))
    single_thread_worker()
    t0 = time.time()
    R, G, P, INFO = [], [], [], []
    with mp.get_context("forkserver").Pool(workers, initializer=single_thread_worker) as pool:
        for i, (r, g, p, info) in enumerate(pool.imap_unordered(file_job, jobs, chunksize=1)):
            R += r; G += g; P += p; INFO.append(info)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)} files  {time.time() - t0:.0f}s", flush=True)
    rec, gt, pr, info = pd.DataFrame(R), pd.DataFrame(G), pd.DataFrame(P), pd.DataFrame(INFO)
    meta = multi.assign(stem=multi.file_name.map(lambda f: Path(f).stem))[["stem", "n_sets", "fold", "site"]]
    for df in (rec, gt, pr):
        df["prefix_A"] = df.A.str.slice(0, 6)
        df["prefix_R"] = df.R.str.slice(0, 6)
        df["group"] = df.A.str.slice(0, 4)
    rec = rec.merge(meta, on="stem", how="left")
    gt = gt.merge(meta[["stem", "n_sets"]], on="stem", how="left")
    pr = pr.merge(meta[["stem", "n_sets"]], on="stem", how="left")
    # reference-instance features (verified join: JSON order == ann_idx, areas must agree)
    feat = pd.read_csv(ROOT / "outputs" / "audit" / "instances.csv", usecols=FEAT_COLS)
    feat = feat.rename(columns={"image_id": "R", "ann_idx": "gt_idx", "area": "area_audit"})
    gt = gt.merge(feat, on=["R", "gt_idx"], how="left")
    assert gt.area_audit.notna().all() and (gt.area_audit == gt.area).all(), "instances.csv join failed"
    gt = gt.drop(columns="area_audit")
    # third-annotator support of each reference instance (3-set files): best IoU of g in R with set C
    hum = gt[gt.rater == "human"].set_index(["A", "R", "gt_idx"]).best_iou
    sets_of = rec.groupby("stem").R.agg(lambda s: sorted(set(s))).to_dict()
    cbest = []
    for stem, A, Rr, k in zip(gt.stem, gt.A, gt.R, gt.gt_idx):
        ss = sets_of[stem]
        if len(ss) == 3:
            C = [x for x in ss if x not in (A, Rr)][0]
            cbest.append(float(hum.loc[(C, Rr, k)]))
        else:
            cbest.append(np.nan)
    gt["c_best_iou"] = cbest
    OUT.mkdir(parents=True, exist_ok=True)
    rec.to_csv(OUT / "records.csv", index=False)
    gt.to_csv(OUT / "ref_instances.csv", index=False)
    pr.to_csv(OUT / "rater_instances.csv", index=False)
    info.to_csv(OUT / "file_info.csv", index=False)
    print(f"tables built in {time.time() - t0:.0f}s: {len(rec)} record rows, {len(gt)} ref-inst rows, {len(pr)} rater-inst rows")
    return rec, gt, pr


def subset(df, rater, mask=None):
    d = df[df.rater == rater]
    return d if mask is None else d[mask(d)]


def analyse(rec, gt, pr, n_boot: int):
    res = {}
    # ---------------------------------------------------------------- checks
    keys = {k: sorted(rec[rec.rater == k].rec) for k in ("human", "human_min500") + ALL_MODEL_RATERS}
    assert all(keys[k] == keys["human"] for k in keys), "raters are not evaluated on identical records"
    ng = rec.pivot_table(index="rec", columns="rater", values="n_gt")
    assert (ng.nunique(axis=1) == 1).all(), "reference instance counts differ between raters"
    assert len(keys["human"]) == 1196
    gref = gt[gt.rater == "human"][["stem", "R", "contrast_snr", "gt_idx"]].drop_duplicates(["R", "gt_idx"])
    faint_thr = float(np.quantile(gref.contrast_snr, 0.2))
    b0_unique = rec[rec.rater == "b0"].drop_duplicates("R")
    ev = json.load(open(ROOT / "outputs" / "exps" / "b0_r101_s1024" / "eval_oof_cc_f01234_t0.5_a500.json"))
    res["checks"] = dict(
        n_records_per_rater=len(keys["human"]), identical_record_keys=True, identical_n_gt=True,
        n_files=int(rec.stem.nunique()), n_rater_views=int(rec[rec.rater == "human"].A.nunique()),
        human_PQ_host_all_1196=rec_aspects(rec[rec.rater == "human"])["PQ"],
        human_PQ_reference_inter_annotator_doc=0.3399,
        b0_PQ_host_each_set_once=rec_aspects(b0_unique)["PQ"],
        b0_PQ_host_multi_set_reference_eval_oof=ev["multi_set_records"]["PQ_pooled_host"],
        faint_threshold_contrast_snr_q20=faint_thr,
        note="model rows depend only on R; LOAO pools each 3-set reference twice (once per A != R), like the humans' ordered pairs")
    # ---------------------------------------------------------------- headline
    scopes = {"all": lambda d: d.n_sets >= 2, "2set": lambda d: d.n_sets == 2, "3set": lambda d: d.n_sets == 3}
    rows = []
    res["headline"] = {}
    for sc, msk in scopes.items():
        rh = subset(rec, "human", msk)
        res["headline"][sc] = {}
        for m in ("b0", "e5", "human_min500") + tuple(POST_VARIANTS):
            pb = paired_boot(rh, subset(rec, m, msk), n_boot,
                             stats=("PQ", "SQ", "RQ") if m == "human_min500" else tuple(STATS))
            res["headline"][sc][m] = pb
            for s in [k for k in pb if k in STATS]:
                rows.append(dict(scope=sc, model=m, stat=s, n_files=pb["n_files"], n_records=pb["n_records"],
                                 human=pb[s]["human"], model_value=pb[s]["model"], diff=pb[s]["diff"],
                                 ci_lo=pb[s]["ci95"][0], ci_hi=pb[s]["ci95"][1],
                                 p_boot_model_better=pb[s]["p_boot_model_better"]))
    pd.DataFrame(rows).to_csv(OUT / "headline.csv", index=False)
    # ---------------------------------------------------------------- 3-set oracle: best other human per reference
    r3 = rec[(rec.n_sets == 3)]
    h3 = r3[r3.rater == "human"].copy()
    h3["pq_rec"] = h3.pq_host.fillna(0)
    best = h3.sort_values(["R", "pq_rec", "A"], ascending=[True, False, True]).drop_duplicates("R")
    worst = h3.sort_values(["R", "pq_rec", "A"], ascending=[True, True, True]).drop_duplicates("R")
    m3 = r3[r3.rater == "b0"].drop_duplicates("R").set_index("R").loc[best.R].reset_index()
    e3 = r3[r3.rater == "e5"].drop_duplicates("R").set_index("R").loc[best.R].reset_index()
    best_k = best.assign(rec=best.R)
    m3k = m3.assign(rec=m3.R)
    res["oracle_3set"] = dict(
        n_reference_records=int(len(best)),
        best_other_human_per_record=rec_aspects(best)["PQ"], worst_other_human_per_record=rec_aspects(worst)["PQ"],
        b0=rec_aspects(m3)["PQ"], e5=rec_aspects(e3)["PQ"],
        b0_minus_best_human=paired_boot(best_k, m3k, n_boot, stats=("PQ", "SQ", "RQ")),
        note="per reference set R of a 3-set file, the better (record PQ) of the two other humans is picked AFTER "
             "seeing the score (an oracle, not a realisable rater); model is scored once per R")
    # ---------------------------------------------------------------- per annotator / per group
    per_rows = []
    asp_human, asp_model, asp_e5 = {}, {}, {}
    for level, key in (("annotator", "prefix_A"), ("group", "group")):
        for sc, msk in scopes.items():
            rh_all = subset(rec, "human", msk)
            for val, rh in rh_all.groupby(key):
                rm = subset(rec, "b0", msk)
                rm = rm[rm[key] == val]
                re5 = subset(rec, "e5", msk)
                re5 = re5[re5[key] == val]
                pb = paired_boot(rh, rm, n_boot, stats=("PQ", "SQ", "RQ", "thin_recall", "thin_precision", "boundary_F2"))
                n_views = int(rh.A.nunique())
                row = dict(level=level, scope=sc, id=val, n_rater_views=n_views, n_files=pb["n_files"],
                           n_records=pb["n_records"])
                for s in ("PQ", "SQ", "RQ", "thin_recall", "thin_precision", "boundary_F2"):
                    row[f"{s}_human"] = pb[s]["human"]
                    row[f"{s}_b0"] = pb[s]["model"]
                    row[f"{s}_diff"] = pb[s]["diff"]
                    if s == "PQ":
                        row["PQ_diff_ci_lo"], row["PQ_diff_ci_hi"] = pb[s]["ci95"]
                        row["PQ_p_boot_model_better"] = pb[s]["p_boot_model_better"]
                row["PQ_e5"] = rec_aspects(re5)["PQ"]
                row["count_log_bias_human"] = rec_aspects(rh)["count_log_bias"]
                row["count_log_bias_b0"] = rec_aspects(rm)["count_log_bias"]
                per_rows.append(row)
                if level == "annotator" and sc == "all" and n_views >= MIN_SETS_ANNOTATOR:
                    recs_ = set(rh.rec)
                    gh = gt[(gt.rater == "human") & gt.rec.isin(recs_)]
                    ph = pr[(pr.rater == "human") & pr.rec.isin(recs_)]
                    gm = gt[(gt.rater == "b0") & gt.rec.isin(recs_)]
                    pm = pr[(pr.rater == "b0") & pr.rec.isin(recs_)]
                    ge = gt[(gt.rater == "e5") & gt.rec.isin(recs_)]
                    pe = pr[(pr.rater == "e5") & pr.rec.isin(recs_)]
                    asp_human[val] = all_aspects(rh, gh, ph, faint_thr)
                    asp_model[val] = all_aspects(rm, gm, pm, faint_thr)
                    asp_e5[val] = all_aspects(re5, ge, pe, faint_thr)
    per = pd.DataFrame(per_rows)
    per[per.level == "annotator"].drop(columns="level").to_csv(OUT / "per_annotator.csv", index=False)
    per[per.level == "group"].drop(columns="level").to_csv(OUT / "per_group.csv", index=False)
    # ---------------------------------------------------------------- aspects (pooled + per annotator)
    pooled = {}
    for rater in ("human",) + ALL_MODEL_RATERS:
        for sc, msk in scopes.items():
            r = subset(rec, rater, msk)
            g = gt[(gt.rater == rater) & gt.rec.isin(set(r.rec))]
            p = pr[(pr.rater == rater) & pr.rec.isin(set(r.rec))]
            pooled[(rater, sc)] = all_aspects(r, g, p, faint_thr)
    est = sorted(asp_human)
    arows = []
    for a, d in ASPECTS.items():
        hv = np.array([asp_human[x].get(a, np.nan) for x in est], float)
        mv = np.array([asp_model[x].get(a, np.nan) for x in est], float)
        ok = np.isfinite(hv) & np.isfinite(mv)
        beaten = [better(m_, h_, d) for m_, h_ in zip(mv[ok], hv[ok])]
        hmed = float(np.median(hv[np.isfinite(hv)])) if np.isfinite(hv).any() else np.nan
        if d == 0:  # closer to zero is better: median of |bias|
            hmed = float(np.median(np.abs(hv[np.isfinite(hv)])))
        hbest = (np.nan if d is None or not np.isfinite(hv).any() else
                 float(np.nanmax(hv)) if d == 1 else float(np.nanmin(hv)) if d == -1 else float(np.nanmin(np.abs(hv))))
        row = dict(aspect=a, direction={1: "higher", -1: "lower", 0: "closer_to_0", None: "info"}[d],
                   in_target=a in TARGET_ASPECTS)
        for sc in scopes:
            row[f"human_{sc}"] = pooled[("human", sc)].get(a)
            row[f"b0_{sc}"] = pooled[("b0", sc)].get(a)
            row[f"e5_{sc}"] = pooled[("e5", sc)].get(a)
        for v in POST_VARIANTS:
            row[f"{v}_all"] = pooled[(v, "all")].get(a)
        row.update(human_annotator_median=hmed, human_annotator_best=hbest,
                   human_annotator_min=float(np.nanmin(hv)) if np.isfinite(hv).any() else np.nan,
                   human_annotator_max=float(np.nanmax(hv)) if np.isfinite(hv).any() else np.nan,
                   n_annotators=int(ok.sum()),
                   b0_beats_annotator_on_own_records=(int(np.sum([b for b in beaten if b is not None]))
                                                      if d is not None else None),
                   b0_vs_pooled_human=better(row["b0_all"], row["human_all"], d),
                   b0_not_below_annotator_median=(not_worse(row["b0_all"], hmed, d) if d != 0 else
                                                  bool(abs(row["b0_all"]) <= hmed)) if d is not None else None,
                   e5_not_below_annotator_median=(not_worse(row["e5_all"], hmed, d) if d != 0 else
                                                  bool(abs(row["e5_all"]) <= hmed)) if d is not None else None)
        arows.append(row)
    aspects = pd.DataFrame(arows)
    aspects.to_csv(OUT / "aspects.csv", index=False)
    pd.DataFrame({x: {**{f"human_{k}": v for k, v in asp_human[x].items()},
                      **{f"b0_{k}": v for k, v in asp_model[x].items()}} for x in est}).T.to_csv(
        OUT / "aspects_per_annotator.csv")
    # ---------------------------------------------------------------- existence strata (reference instances)
    srows = []
    gq = gt.copy()
    gq["contrast_q"] = pd.qcut(gq.contrast_snr, 5, labels=False, duplicates="drop")
    gq["area_bin"] = pd.cut(gq.area, [0, 250, 500, 1000, 2000, 4000, 1e9], right=False)
    gq["limb_bin"] = pd.cut(gq.centroid_r_frac, [0, 0.4, 0.6, 0.8, 1.01], right=False)
    gq["third"] = np.where(gq.c_best_iou.isna(), "2set", np.where(gq.c_best_iou > 0.5, "C_matches",
                           np.where(gq.c_best_iou >= 0.1, "C_overlaps", "C_absent")))
    big = gq.area >= 500  # same strata restricted to reference instances the model's area filter cannot remove
    gq["contrast_q_ge500"] = gq.contrast_q.where(big)
    gq["limb_bin_ge500"] = gq.limb_bin.where(big)
    gq["third_ge500"] = gq.third.where(big)
    for col in ("contrast_q", "area_bin", "limb_bin", "third", "contrast_q_ge500", "limb_bin_ge500", "third_ge500"):
        for (val, rater), d in gq.groupby([col, "rater"], observed=True):
            if rater not in ("human", "b0", "e5"):
                continue
            srows.append(dict(stratum=col, value=str(val), rater=rater, n=len(d),
                              matched_rate=float((d.best_iou > 0.5).mean()),
                              detected_rate=float((d.best_iou >= 0.1).mean()),
                              best_iou_median=float(d.best_iou.median()),
                              thin_recall=float(_div(d.thin_covered_px.sum(), d.thin_px.sum()))))
    strata = pd.DataFrame(srows)
    strata.to_csv(OUT / "existence_strata.csv", index=False)
    # instance cross-tab (who matches each reference instance view)
    piv = gt[gt.rater.isin(["human", "b0"])].pivot_table(index=["rec", "gt_idx"], columns="rater", values="best_iou")
    hm, mm = piv.human > 0.5, piv.b0 > 0.5
    res["instance_crosstab_b0"] = dict(both=int((hm & mm).sum()), human_only=int((hm & ~mm).sum()),
                                       model_only=int((~hm & mm).sum()), neither=int((~hm & ~mm).sum()))
    # best-IoU distributions
    bins = [0, 1e-9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    lab = ["0", "(0,.1)", "[.1,.2)", "[.2,.3)", "[.3,.4)", "[.4,.5]", "(.5,.6)", "[.6,.7)", "[.7,.8)", "[.8,.9)", "[.9,1]"]
    dist = {}
    for side, df in (("reference_instances", gt), ("rater_instances", pr)):
        for rater in ("human", "b0", "e5"):
            v = df[df.rater == rater].best_iou.values
            # (.4,.5] and (.5,.6) split exactly at the strict 0.5 match threshold
            c = [np.mean(v == 0), np.mean((v > 0) & (v < 0.1))] + \
                [np.mean((v >= lo) & (v < hi)) for lo, hi in ((.1, .2), (.2, .3), (.3, .4))] + \
                [np.mean((v >= .4) & (v <= .5)), np.mean((v > .5) & (v < .6))] + \
                [np.mean((v >= lo) & (v < hi)) for lo, hi in ((.6, .7), (.7, .8), (.8, .9))] + [np.mean(v >= .9)]
            dist[f"{side}/{rater}"] = dict(n=int(len(v)), quantiles={q: float(np.quantile(v, q)) for q in (.1, .25, .5, .75, .9)},
                                           hist=dict(zip(lab, map(float, c))))
    res["best_iou_distribution"] = dist
    del bins
    # ---------------------------------------------------------------- examples: worst model-vs-human gaps
    ex = []
    for (stem, A), rh in rec[rec.rater == "human"].groupby(["stem", "A"]):
        rm = rec[(rec.rater == "b0") & (rec.stem == stem) & (rec.A == A)]
        ah, am = rec_aspects(rh), rec_aspects(rm)
        ex.append(dict(stem=stem, A=A, n_sets=int(rh.n_sets.iloc[0]), fold=int(rh.fold.iloc[0]),
                       refs=";".join(sorted(rh.R)), n_ref_inst=int(rh.n_gt.sum()), n_human_inst=int(rh.n_pred.iloc[0]),
                       n_b0_inst=int(rm.n_pred.iloc[0]), PQ_human=ah["PQ"], PQ_b0=am["PQ"], gap_b0_minus_human=am["PQ"] - ah["PQ"],
                       d_fn_missed=int(rm.fn_missed.sum() - rh.fn_missed.sum()),
                       d_split_fragment=int(rm.fn_split.sum() + rm.fp_fragment.sum() - rh.fn_split.sum() - rh.fp_fragment.sum()),
                       d_merge=int(rm.fn_merged.sum() + rm.fp_merge.sum() - rh.fn_merged.sum() - rh.fp_merge.sum()),
                       d_fp_spurious=int(rm.fp_spurious.sum() - rh.fp_spurious.sum()),
                       d_low_iou=int(rm.fn_low_iou.sum() + rm.fp_low_iou.sum() - rh.fn_low_iou.sum() - rh.fp_low_iou.sum()),
                       d_thin_recall=am["thin_recall"] - ah["thin_recall"]))
    ex = pd.DataFrame(ex)
    ex.to_csv(OUT / "gaps_per_rater_view.csv", index=False)
    elig = ex[ex.n_ref_inst >= 4].copy()
    picks, used = [], set()

    def take(df, why, n):
        k = 0
        for _, r in df.iterrows():
            if r.stem in used:
                continue
            used.add(r.stem)
            picks.append({**r.to_dict(), "why": why})
            k += 1
            if k >= n:
                break

    take(elig.sort_values("gap_b0_minus_human"), "largest PQ gap (model below human A)", 5)
    take(elig[elig.gap_b0_minus_human < 0].sort_values("d_split_fragment", ascending=False), "fragmentation excess", 2)
    take(elig[elig.gap_b0_minus_human < 0].sort_values("d_fn_missed", ascending=False), "missed (existence) excess", 2)
    take(elig[elig.gap_b0_minus_human < 0].sort_values("d_thin_recall"), "thin-region recall deficit", 1)
    take(elig.sort_values("gap_b0_minus_human", ascending=False), "contrast: model far above human A", 3)
    exdf = pd.DataFrame(picks)
    exdf.to_csv(OUT / "examples.csv", index=False)
    res["examples"] = exdf.to_dict(orient="records")
    res["gap_distribution_rater_views"] = dict(
        n=int(len(ex)), share_model_better=float((ex.gap_b0_minus_human > 0).mean()),
        share_model_worse=float((ex.gap_b0_minus_human < 0).mean()),
        quantiles={q: float(ex.gap_b0_minus_human.quantile(q)) for q in (.1, .25, .5, .75, .9)})
    # ---------------------------------------------------------------- per-file best annotator (oracle, same records)
    hpq = ex.groupby("stem").PQ_human.transform("max")
    best_views = set(zip(ex.stem[ex.PQ_human >= hpq - 1e-12], ex.A[ex.PQ_human >= hpq - 1e-12]))
    in_best = lambda d: [(s, a) in best_views for s, a in zip(d.stem, d.A)]  # noqa: E731
    rh_best = rec[(rec.rater == "human")][in_best(rec[rec.rater == "human"])]
    res["oracle_best_annotator_per_file"] = dict(
        note="per file, the annotator(s) with the highest PQ against the other set(s) of the file, picked AFTER seeing "
             "the scores (2-set files: both annotators tie, so this equals the plain comparison); the model is scored "
             "on exactly the same (A, R) records",
        **{m: paired_boot(rh_best, rec[rec.rater == m][in_best(rec[rec.rater == m])], n_boot) for m in RATERS_MODEL},
        **{f"{m}_3set": paired_boot(rh_best[rh_best.n_sets == 3],
                                    rec[(rec.rater == m) & (rec.n_sets == 3)][in_best(rec[(rec.rater == m) & (rec.n_sets == 3)])],
                                    n_boot) for m in RATERS_MODEL})
    # ---------------------------------------------------------------- target evaluation
    ann = per[(per.level == "annotator") & (per.scope == "all") & (per.n_rater_views >= MIN_SETS_ANNOTATOR)]
    best_ann = ann.sort_values("PQ_human", ascending=False).iloc[0]
    hl = res["headline"]
    tgt = {}
    for m in ("b0", "e5"):
        pq_all = hl["all"][m]["PQ"]["model"]
        own = ann.PQ_e5 if m == "e5" else ann.PQ_b0
        asp_col = f"{m}_not_below_annotator_median"
        blockers = aspects[aspects.in_target & (aspects[asp_col] == False)].aspect.tolist()  # noqa: E712
        ob = res["oracle_best_annotator_per_file"][m]["PQ"]
        tgt[m] = dict(
            BH1_beats_average_annotator=dict(
                rule="LOAO paired PQ diff (model - human A) > 0 with file-bootstrap 95% CI lower bound > 0, "
                     "in all, 2-set and 3-set files",
                values={sc: dict(diff=hl[sc][m]["PQ"]["diff"], ci95=hl[sc][m]["PQ"]["ci95"]) for sc in scopes},
                met=all(hl[sc][m]["PQ"]["ci95"][0] > 0 for sc in scopes)),
            BH2_beats_every_annotator_like_for_like=dict(
                rule=f"for every annotator id with >= {MIN_SETS_ANNOTATOR} rater views: model PQ on that annotator's "
                     "LOAO records > that annotator's PQ on the same records (point estimate)",
                n_annotators=int(len(ann)), n_beaten=int((own > ann.PQ_human).sum()),
                n_beaten_ci_lo_gt0=(int((ann.PQ_diff_ci_lo > 0).sum()) if m == "b0" else None),
                not_beaten=ann[own <= ann.PQ_human].id.tolist(),
                min_margin=float((own - ann.PQ_human).min()),
                met=bool((own > ann.PQ_human).all())),
            BH3_beats_best_annotator_of_each_file=dict(
                rule="model PQ > PQ of the per-file best annotator (oracle) on the same records, 95% CI lower bound > 0",
                human_oracle=ob["human"], model=ob["model"], diff=ob["diff"], ci95=ob["ci95"],
                met=bool(ob["ci95"][0] > 0)),
            info_3set_best_other_human_per_reference=dict(
                rule="3-set files: per reference R, the better of the two other humans (oracle) vs the model",
                model=res["oracle_3set"][m], oracle=res["oracle_3set"]["best_other_human_per_record"],
                margin=res["oracle_3set"][m] - res["oracle_3set"]["best_other_human_per_record"]),
            info_cross_record_best_annotator=dict(
                note="NOT like-for-like (different files / batch groups): model's overall LOAO PQ vs the highest "
                     "single-annotator LOAO PQ; group 0204 / 0501 files are easier for everyone",
                best_annotator=dict(id=best_ann.id, PQ=float(best_ann.PQ_human), n_views=int(best_ann.n_rater_views),
                                    model_PQ_on_its_records=float(best_ann.PQ_e5 if m == "e5" else best_ann.PQ_b0)),
                model_overall_PQ=pq_all, margin=pq_all - float(best_ann.PQ_human)),
            BH4_no_aspect_below_human_median=dict(
                rule="for every target aspect, model value pooled over all LOAO records is not worse than the "
                     f"median over annotators (>= {MIN_SETS_ANNOTATOR} views) of the annotators' own values",
                aspects=TARGET_ASPECTS, blocking_aspects=blockers, met=len(blockers) == 0),
            BH5_every_batch_group=dict(
                rule="model PQ >= human PQ (point estimate) in every batch group with >= 10 multi-set files",
                groups={r.id: dict(diff=float((r.PQ_e5 if m == "e5" else r.PQ_b0) - r.PQ_human), n_files=int(r.n_files))
                        for r in per[(per.level == "group") & (per.scope == "all") & (per.n_files >= 10)].itertuples()},
            ))
        g10 = per[(per.level == "group") & (per.scope == "all") & (per.n_files >= 10)]
        col = "PQ_e5" if m == "e5" else "PQ_b0"
        tgt[m]["BH5_every_batch_group"]["met"] = bool((g10[col] >= g10.PQ_human).all())
        tgt[m]["all_met"] = all(v["met"] for v in tgt[m].values() if isinstance(v, dict) and "met" in v)
    res["target"] = tgt
    res["pooled_aspects"] = {f"{r}/{sc}": v for (r, sc), v in pooled.items()}
    json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return res, aspects, per, exdf


# =============================================================================== figures
DISK_CX, DISK_CY, DISK_R = 1023.0, 1024.0, 903.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_A = (0, 230, 255)      # BGR yellow   : human A
COL_M = (255, 220, 0)      # BGR cyan     : model
COL_R = [(255, 0, 255), (60, 255, 60)]  # magenta / green: other sets
K3 = np.ones((3, 3), np.uint8)


def _stretch(crop, x0, y0):
    h, w = crop.shape
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
    on = (yy - DISK_CY) ** 2 + (xx - DISK_CX) ** 2 < (DISK_R - 5) ** 2
    v = crop[on] if on.sum() > 100 else crop.ravel()
    lo, hi = np.percentile(v, [0.5, 99.5])
    hi = max(hi, lo + 1)
    return np.clip((crop.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _union_win(S, win):
    x0, y0, w, h = win
    out = np.zeros((h, w), bool)
    for k in range(len(S)):
        by0, bx0, by1, bx1 = (int(v) for v in S.boxes[k])
        iy0, ix0, iy1, ix1 = max(by0, y0), max(bx0, x0), min(by1, y0 + h), min(bx1, x0 + w)
        if iy0 < iy1 and ix0 < ix1:
            out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] |= S.crops[k][iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
    return out


def _inst_win(S, win):
    x0, y0, w, h = win
    res = []
    for k in range(len(S)):
        by0, bx0, by1, bx1 = (int(v) for v in S.boxes[k])
        iy0, ix0, iy1, ix1 = max(by0, y0), max(bx0, x0), min(by1, y0 + h), min(bx1, x0 + w)
        if iy0 < iy1 and ix0 < ix1:
            m = np.zeros((h, w), bool)
            m[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = S.crops[k][iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
            if m.any():
                res.append(m)
    return res


def _outline(img, masks, color):
    for m in masks:  # 1 px ring just OUTSIDE each instance (instance pixels stay visible)
        ring = cv2.dilate(m.astype(np.uint8), K3).astype(bool) & ~m
        img[ring] = color


def _wrap(text, width, scale):
    words, lines, cur = text.split(" "), [], ""
    for wd in words:
        t = (cur + " " + wd) if cur else wd
        if cv2.getTextSize(t, FONT, scale, 1)[0][0] > width - 12 and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = t
    return lines + [cur]


def _header(width, lines, scale=0.42):
    lh = 17
    flat = [x for ln in lines for x in _wrap(ln, width, scale)]
    out = np.full((lh * len(flat) + 8, width, 3), 40, np.uint8)
    for i, ln in enumerate(flat):
        cv2.putText(out, ln, (6, lh * (i + 1)), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _grid(cells, ncol=3, gap=6):
    hh = max(_header(p.shape[1], t).shape[0] for t, p in cells)
    blocks = []
    for t, p in cells:
        hd = _header(p.shape[1], t)
        hd = np.vstack([hd, np.full((hh - hd.shape[0], p.shape[1], 3), 40, np.uint8)])
        blocks.append(np.vstack([hd, p]))
    ch, cw = max(b.shape[0] for b in blocks), max(b.shape[1] for b in blocks)
    pad = [np.pad(b, ((0, ch - b.shape[0]), (0, cw - b.shape[1]), (0, 0)), constant_values=255) for b in blocks]
    while len(pad) % ncol:
        pad.append(np.full((ch, cw, 3), 255, np.uint8))
    rows = []
    for i in range(0, len(pad), ncol):
        r = []
        for j, c in enumerate(pad[i:i + ncol]):
            r += [c] + ([np.full((ch, gap, 3), 255, np.uint8)] if j < ncol - 1 else [])
        rows.append(np.hstack(r))
    out = []
    for i, r in enumerate(rows):
        out += [r] + ([np.full((gap, r.shape[1], 3), 255, np.uint8)] if i < len(rows) - 1 else [])
    return np.vstack(out)


def _best_iou_rows(G, P):
    if len(G) == 0:
        return np.zeros(0)
    if len(P) == 0:
        return np.zeros(len(G))
    return iou_from_intersection(intersection_matrix(G, P), G.areas, P.areas).max(1)


def render_example(stem, fold, A, set_ids, mode, out_path, side_min=360, side_max=600, pad=40):
    """raw | human A | model b0 | other sets | b0 probability | overlay, native resolution (1 px = 1 px)."""
    img = D.read_image(D.train_image_path(stem + ".jpeg"))
    S = load_sets(set_ids)
    prob = load_prob("b0", fold, stem)
    M = seg.prob_to_instances(prob, POST)
    refs = [R for R in set_ids if R != A]
    cand = []  # largest disagreement: reference instance with the biggest |best IoU(human A) - best IoU(model)|
    for R in refs:
        hb, mb = _best_iou_rows(S[R], S[A]), _best_iou_rows(S[R], M)
        for k in range(len(S[R])):
            s = (hb[k] - mb[k]) if mode == "human_better" else (mb[k] - hb[k])
            cand.append((s * math.sqrt(S[R].areas[k]), R, k, hb[k], mb[k]))
    cand.sort(key=lambda t: -t[0])
    _, R0, k0, hb0, mb0 = cand[0]
    y0, x0, y1, x1 = (int(v) for v in S[R0].boxes[k0])
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    hh = int(np.clip(y1 - y0 + 2 * pad, side_min, side_max))
    ww = int(np.clip(x1 - x0 + 2 * pad, side_min, side_max))
    wy0 = int(np.clip(cy - hh // 2, 0, 2048 - hh))
    wx0 = int(np.clip(cx - ww // 2, 0, 2048 - ww))
    win = (wx0, wy0, ww, hh)
    base = _stretch(img[wy0:wy0 + hh, wx0:wx0 + ww], wx0, wy0)
    gray = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    pA = gray.copy(); _outline(pA, _inst_win(S[A], win), COL_A)
    pM = gray.copy(); _outline(pM, _inst_win(M, win), COL_M)
    pR = gray.copy()
    for i, R in enumerate(refs):
        _outline(pR, _inst_win(S[R], win), COL_R[i % 2])
    pc = prob[wy0:wy0 + hh, wx0:wx0 + ww]
    pP = cv2.applyColorMap(np.clip(pc * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    _outline(pP, [pc > 0.5], (255, 255, 255))
    pO = gray.copy()
    nref = np.zeros((hh, ww), np.uint8)
    for R in refs:
        nref += _union_win(S[R], win).astype(np.uint8)
    tint = pO.astype(np.float32)
    tint[nref >= 1] = tint[nref >= 1] * 0.55 + np.array((60, 60, 230), np.float32) * 0.45   # red-ish: >= 1 ref
    tint[nref >= 2] = tint[nref >= 2] * 0.6 + np.array((0, 0, 255), np.float32) * 0.4      # stronger: 2 refs
    pO = tint.astype(np.uint8)
    _outline(pO, [_union_win(S[A], win)], COL_A)
    _outline(pO, [_union_win(M, win)], COL_M)
    rh = [f"{A}|{R}" for R in refs]
    cells = [
        ([f"raw {stem}  crop x{wx0} y{wy0} {ww}x{hh} (native px)", f"mode={mode}; target ref inst {R0}#{k0}"], gray),
        ([f"HUMAN A = {A} (yellow, n={len(S[A])})", f"best IoU of target ref inst with A: {hb0:.2f}"], pA),
        ([f"MODEL b0 thr.5/a500 (cyan, n={len(M)})", f"best IoU of target ref inst with b0: {mb0:.2f}"], pM),
        (["OTHER SETS (references): " + ", ".join(refs), "magenta = first, green = second"], pR),
        (["b0 probability 0..1 (viridis)", "white = p 0.5 contour"], pP),
        (["overlay: red fill = refs (darker = 2 refs)", "yellow = human A, cyan = b0"], pO),
    ]
    cv2.imwrite(str(out_path), _grid(cells, ncol=3))
    return dict(stem=stem, A=A, mode=mode, window=win, target_ref=R0, target_idx=int(k0),
                human_best_iou=float(hb0), model_best_iou=float(mb0), records=rh, file=str(out_path))


def make_figures(exdf: pd.DataFrame, n_worse=4, n_better=2):
    fd = OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(ROOT / "outputs" / "splits" / "folds_v1.csv")
    sets = {Path(f).stem: sorted(s.split(";")) for f, s in zip(folds.file_name, folds.image_ids)}
    worse = exdf[exdf.gap_b0_minus_human < 0]
    # one per selection reason first (diverse failure types), then by gap
    order = pd.concat([worse.drop_duplicates("why"), worse]).drop_duplicates("stem").head(n_worse)
    better_ = exdf[exdf.gap_b0_minus_human > 0].head(n_better)
    out = []
    for mode, df in (("human_better", order), ("model_better", better_)):
        for _, r in df.iterrows():
            p = fd / f"{mode}_{r.stem}_{r.A.split('-')[0]}.png"
            info = render_example(r.stem, int(r.fold), r.A, sets[r.stem], mode, p)
            info.update(gap=float(r.gap_b0_minus_human), why=r.why, PQ_human=float(r.PQ_human), PQ_b0=float(r.PQ_b0))
            out.append(info)
            print("figure", p)
    json.dump(out, open(fd / "figures_index.json", "w"), indent=1, default=str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--reuse", action="store_true", help="reuse saved record/instance tables")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    single_thread_worker()
    if a.reuse:
        rec = pd.read_csv(OUT / "records.csv")
        gt = pd.read_csv(OUT / "ref_instances.csv")
        pr = pd.read_csv(OUT / "rater_instances.csv")
    else:
        rec, gt, pr = build_tables(min(a.workers, 16))
    t0 = time.time()
    res, aspects, per, exdf = analyse(rec, gt, pr, a.n_boot)
    print(f"analysis {time.time() - t0:.0f}s")
    print(json.dumps(res["checks"], indent=1, default=str))
    for sc in ("all", "2set", "3set"):
        for m in ("b0", "e5", "human_min500"):
            h = res["headline"][sc][m]
            print(sc, m, {s: (round(h[s]["human"], 4), round(h[s]["model"], 4), round(h[s]["diff"], 4),
                              [round(x, 4) for x in h[s]["ci95"]]) for s in ("PQ", "SQ", "RQ")})
    print(json.dumps({m: {k: v.get("met") for k, v in t.items() if isinstance(v, dict)} for m, t in res["target"].items()}))
    if a.figures:
        make_figures(exdf)


if __name__ == "__main__":
    main()
