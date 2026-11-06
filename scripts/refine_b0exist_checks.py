"""TASK REFINE (CPU only): adoption-status checks of the POST-HOC candidate 'b0 decides existence + R2 refines
extent / instance membership' (compare_paired method 'R2_refine_context:refineb0:0.5:300'; R1 analogue
'R1_refine_crop:refineb0:0.5:300') on folds 0 and 1, the only folds with R1/R2 models.

Only existing OOF predictions are used (b0 val_probs_last.npz, R1/R2 val_cand_probs.npz, refine_v1 candidate cache);
nothing is trained, the GPU is never used (CUDA_VISIBLE_DEVICES is forced empty), no test data or test metadata is
read except our OWN frozen test-prediction CSV (instance counts only, for the cost estimate of `timing`).

Methods (all instance sets on the 2048^2 grid; scored with filseg.metrics.evaluate_image, host pooling):
  b0                      : b0 OOF prob, thr 0.5, 8-conn CC, min_area 500 (the submitted baseline)
  b0_a300                 : same with min_area 300 (information only: isolates the min_area effect)
  {R2,R1}b0_{rule}_a{ma}  : refined masks of the stage-1 candidates (p>0.3, >=20 px) that b0 'confirms', assembled by
                            filseg.refine.assemble (merge IoU>0.5, conflicts by max prob), instances < ma dropped.
                            rule 'any'   : candidate overlaps the b0 (t0.5/a500) union by >= 1 px  (= compare_paired
                                           'refineb0', the configuration reported in D-029)
                            rule 'frac50': >= 50% of the candidate's pixels lie inside b0 (t0.5/a500) instances
                            ma in {0, 100, 300, 500}
  {R2,R1}full_a300        : all candidates refined (no b0 existence filter; D-029 'R full pipeline')
assemble(min_area=ma) == assemble(min_area=0) followed by dropping instances with area < max(ma, 1) (identical code
path: the area filter is the last step), so each (exp, rule) is assembled once.

Subcommands:
  table    per-record table (all methods, folds 0,1) + per-file candidate/window counts -> outputs/refine_candidate/
  loao     leave-one-annotator-out records (scripts/beyond_human.py protocol, its functions reused) on the multi-set
           files of folds 0,1 for human / b0 / R2b0 / R1b0 / R2full
  analyse  statistics (file + time-group bootstraps, sign tests, review set, counts, LOAO aspects) -> summary.json
  timing   CPU inference time (R2 per window, b0 per 1024^2 image) and test-set cost estimate; also checks that the
           CPU fp32 R2 output reproduces the stored (GPU bf16) refined probabilities

Usage (always CPU):
  CUDA_VISIBLE_DEVICES="" NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/refine_b0exist_checks.py table --workers 12
  ... loao --workers 12 ; ... analyse ; ... timing --threads 16
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # strict: no GPU (user instruction 2026-10-28)
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from filseg import refine as RF  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.metrics import (EvalConfig, InstanceSet, evaluate_image, intersection_matrix,  # noqa: E402
                            single_thread_worker)

OUT = ROOT / "outputs" / "refine_candidate"
FOLDS = (0, 1)
EXPS = {"R2": "R2_refine_context", "R1": "R1_refine_crop"}
RULES = ("any", "frac50")
MIN_AREAS = (0, 100, 300, 500)
MAIN = "R2b0_any_a300"          # the configuration reported in D-029 (fixed before this analysis)
MAIN_R1 = "R1b0_any_a300"
CFG = EvalConfig(compute_boundary=False, compute_pair_boundary=False, compute_thin=True)
SEED = 2026
N_BOOT = 2000


def method_names():
    names = ["b0", "b0_a300"]
    for t in EXPS:
        for r in RULES:
            names += [f"{t}b0_{r}_a{ma}" for ma in MIN_AREAS]
        names.append(f"{t}full_a300")
    return names


# ============================================================================ instance formation (worker side)
_W: dict = {}


def _cache():
    if "idx" not in _W:
        _W["idx"] = json.load(open(RF.CACHE / "index.json"))
        _W["fi"] = {f: i for i, f in enumerate(_W["idx"]["files"])}
        _W["cand"] = np.load(RF.CACHE / "cand.npy", mmap_mode="r")
    return _W


def _npz(path):
    key = str(path)
    if key not in _W:
        _W[key] = np.load(path)
        by = {}
        for k in _W[key].files:
            st, j, y0, x0 = k.split("|")
            by.setdefault(st, []).append((int(j), int(y0), int(x0), k))
        _W[key + "#by"] = by
    return _W[key], _W[key + "#by"]


def cand_probs(exp, fold, stem):
    z, by = _npz(ROOT / "outputs" / "exps" / exp / f"fold{fold}" / "val_cand_probs.npz")
    return {j: (y0, x0, z[k].astype(np.float32)) for j, y0, x0, k in by.get(stem, [])}


def b0_prob(fold, stem):
    key = f"b0#{fold}"
    if key not in _W:
        _W[key] = np.load(ROOT / "outputs" / "exps" / "b0_r101_s1024" / f"fold{fold}" / "val_probs_last.npz")
    return _W[key][stem].astype(np.float32)


def area_filter(S: InstanceSet, a: int) -> InstanceSet:
    keep = [k for k in range(len(S)) if S.areas[k] >= max(a, 1)]
    return InstanceSet._build(S.shape, [tuple(S.boxes[k]) for k in keep], [S.crops[k] for k in keep])


def b0_existence(stem, fold):
    """Candidate map, b0 instances and the two existence rules for one file."""
    W = _cache()
    fname = stem + ".jpeg"
    p = b0_prob(fold, stem)
    B0 = seg.prob_to_instances(p, seg.PostCfg(0.5, 500))
    b0u = B0.union()
    cmap = np.asarray(W["cand"][W["fi"][fname]])
    n = int(cmap.max())
    tot = np.bincount(cmap.ravel(), minlength=n + 1)
    inb = np.bincount(cmap[b0u], minlength=n + 1)
    keep = {"any": {j for j in range(1, n + 1) if inb[j] > 0},
            "frac50": {j for j in range(1, n + 1) if tot[j] and inb[j] / tot[j] >= 0.5}}
    return p, B0, cmap, keep, n


def all_instances(stem, fold):
    p, B0, cmap, keep, n = b0_existence(stem, fold)
    M = {"b0": B0, "b0_a300": seg.prob_to_instances(p, seg.PostCfg(0.5, 300))}
    for t, exp in EXPS.items():
        cp = cand_probs(exp, fold, stem)
        for r in RULES:
            S0 = RF.assemble({j: v for j, v in cp.items() if j in keep[r]}, thr=0.5, min_area=0)
            for ma in MIN_AREAS:
                M[f"{t}b0_{r}_a{ma}"] = area_filter(S0, ma)
        M[f"{t}full_a300"] = RF.assemble(cp, thr=0.5, min_area=300)
    W = _cache()
    wins = W["idx"]["meta"][stem + ".jpeg"]["windows"]
    info = dict(stem=stem, fold=fold, n_cand=n, n_keep_any=len(keep["any"]), n_keep_frac50=len(keep["frac50"]),
                n_win=len(wins), n_win_any=sum(1 for j, _, _ in wins if j in keep["any"]),
                n_win_frac50=sum(1 for j, _, _ in wins if j in keep["frac50"]), n_b0=len(B0))
    return M, info


# ============================================================================ FP classes (copy of fp_decomposition.job)
def _iou(a, b):
    inter = intersection_matrix(a, b).astype(np.float64)
    return inter, inter / np.maximum(a.areas[:, None] + b.areas[None, :] - inter, 1)


def fp_classes(rid, G, P, gts, B0):
    """Exclusive classes of the unmatched predictions of record rid (same rules/order as scripts/fp_decomposition.py)."""
    cls = dict(c_other_set_match=0, c_duplicate=0, c_fragment=0, c_background=0, c_background_shared_b0=0, c_other=0)
    if not len(P):
        return cls
    inter_r, iou_r = _iou(G, P) if len(G) else (np.zeros((0, len(P))), np.zeros((0, len(P))))
    matched_pred = (iou_r > 0.5).any(0) if len(G) else np.zeros(len(P), bool)
    matched_gt = (iou_r > 0.5).any(1) if len(G) else np.zeros(0, bool)
    other_match = np.zeros(len(P), bool)
    for k, g in gts.items():
        if k != rid and len(g):
            other_match |= (_iou(g, P)[1] > 0.5).any(0)
    in_any = np.zeros(len(P))
    for g in gts.values():
        if len(g):
            in_any = np.maximum(in_any, _iou(g, P)[0].sum(0) / np.maximum(P.areas, 1))
    in_b0 = (_iou(B0, P)[0].sum(0) / np.maximum(P.areas, 1)) if len(B0) else np.zeros(len(P))
    for j in np.where(~matched_pred)[0]:
        frac_r = inter_r[:, j] / max(P.areas[j], 1) if len(G) else np.zeros(0)
        if other_match[j]:
            cls["c_other_set_match"] += 1
        elif len(G) and (frac_r >= 0.3).any() and matched_gt[frac_r >= 0.3].any():
            cls["c_duplicate"] += 1
        elif len(G) and (frac_r >= 0.3).any():
            cls["c_fragment"] += 1
        elif in_any[j] < 0.1:
            cls["c_background"] += 1
            cls["c_background_shared_b0"] += int(in_b0[j] >= 0.1)
        else:
            cls["c_other"] += 1
    return cls


REC_KEYS = ["tp_host", "fp_host", "fn_host", "sum_iou_host", "n_gt", "n_pred", "fn_split", "fn_merged", "fn_missed",
            "fn_low_iou", "fp_fragment", "fp_merge", "fp_duplicate", "fp_spurious", "fp_low_iou", "n_split_gt",
            "n_split_fragments", "n_merged_gt", "n_merge_pred", "thin_gt_px", "thin_gt_covered_px", "thin_pred_px",
            "thin_pred_in_gt_px"]


def table_job(args):
    stem, fold, gts = args
    t0 = time.time()
    M, info = all_instances(stem, fold)
    rows = []
    for name, P in M.items():
        for rid, G in gts.items():
            r, _, _ = evaluate_image(G, P, CFG, image_key=rid, return_instances=False)
            row = dict(method=name, rid=rid, stem=stem, fold=fold, **{k: r.get(k, 0) for k in REC_KEYS})
            row.update(fp_classes(rid, G, P, gts, M["b0"]))
            rows.append(row)
    info["sec"] = time.time() - t0
    return rows, info


def split_table():
    sp = pd.read_csv(ROOT / "outputs" / "splits" / "folds_v1.csv")
    sp["stem"] = sp.file_name.str[:-5]
    return sp


def cmd_table(a):
    import multiprocessing as mp
    single_thread_worker()
    jobs = []
    for f in FOLDS:
        files = (ROOT / "outputs/exps/b0_r101_s1024" / f"fold{f}" / "val_files.txt").read_text().split()
        for e in EXPS.values():
            assert files == (ROOT / "outputs/exps" / e / f"fold{f}" / "val_files.txt").read_text().split()
        gts = seg.gt_records(files)
        by = {}
        for rid, g in gts.items():
            by.setdefault(rid.split("-", 1)[1], {})[rid] = g
        jobs += [(s, f, by[s]) for s in sorted(by)]
    if a.limit:
        jobs = jobs[:a.limit]
    t0 = time.time()
    R, I = [], []
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for i, (rows, info) in enumerate(pool.imap_unordered(table_job, jobs, chunksize=1)):
            R += rows
            I.append(info)
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(jobs)} files {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(R)
    sp = split_table().set_index("stem")
    df["group_id"] = df.stem.map(sp.group_id)
    df["n_sets"] = df.stem.map(sp.n_sets)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_limit{a.limit}" if a.limit else ""
    df.to_csv(OUT / f"records{tag}.csv", index=False)
    pd.DataFrame(I).to_csv(OUT / f"file_info{tag}.csv", index=False)
    print(f"table: {len(df)} rows, {df.stem.nunique()} files, {time.time() - t0:.0f}s")


# ============================================================================ LOAO (beyond_human protocol)
LOAO_RATERS = ("b0", "R2b0_any_a300", "R2b0_frac50_a300", "R1b0_any_a300", "R2full_a300")


def loao_job(args):
    import beyond_human as BH
    file_name, stem, fold, set_ids = args
    S = BH.load_sets(set_ids)
    M_all, _ = all_instances(stem, fold)
    M = {k: M_all[k] for k in LOAO_RATERS}
    recs, gts, prs = [], [], []

    def add(rater, A, R, res):
        row, pg, pp = res
        base = dict(rater=rater, stem=stem, A=A, R=R, rec=f"{A}|{R}")
        recs.append({**base, **{c: row.get(c, np.nan) for c in BH.REC_COLS}})
        for d in pg:
            gts.append({**base, **{c: d[c] for c in BH.GT_COLS}})
        for d in pp:
            prs.append({**base, **{c: d[c] for c in BH.PR_COLS}})

    model_eval = {(k, R): evaluate_image(S[R], P, BH.CFG, image_key=R) for k, P in M.items() for R in set_ids}
    for A in set_ids:
        for R in set_ids:
            if R == A:
                continue
            add("human", A, R, evaluate_image(S[R], S[A], BH.CFG, image_key=f"{A}|{R}"))
            for k in LOAO_RATERS:
                add(k, A, R, model_eval[(k, R)])
    return recs, gts, prs


def cmd_loao(a):
    import multiprocessing as mp
    import beyond_human as BH
    single_thread_worker()
    sp = split_table()
    multi = sp[(sp.n_sets >= 2) & sp.fold.isin(FOLDS)]
    jobs = [(r.file_name, r.stem, int(r.fold), sorted(r.image_ids.split(";"))) for r in multi.itertuples()]
    t0 = time.time()
    R, G, P = [], [], []
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for i, (r, g, p) in enumerate(pool.imap_unordered(loao_job, jobs, chunksize=1)):
            R += r; G += g; P += p
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} files {time.time() - t0:.0f}s", flush=True)
    rec, gt, pr = pd.DataFrame(R), pd.DataFrame(G), pd.DataFrame(P)
    meta = multi[["stem", "n_sets", "fold", "group_id"]]
    for df in (rec, gt, pr):
        df["prefix_A"] = df.A.str.slice(0, 6)
    rec = rec.merge(meta, on="stem", how="left")
    gt = gt.merge(meta[["stem", "n_sets"]], on="stem", how="left")
    feat = pd.read_csv(ROOT / "outputs" / "audit" / "instances.csv", usecols=BH.FEAT_COLS)
    feat = feat.rename(columns={"image_id": "R", "ann_idx": "gt_idx", "area": "area_audit"})
    gt = gt.merge(feat, on=["R", "gt_idx"], how="left")
    assert gt.area_audit.notna().all() and (gt.area_audit == gt.area).all(), "instances.csv join failed"
    gt = gt.drop(columns="area_audit")
    hum = gt[gt.rater == "human"].set_index(["A", "R", "gt_idx"]).best_iou
    sets_of = rec.groupby("stem").R.agg(lambda s: sorted(set(s))).to_dict()
    cb = []
    for stem, A, Rr, k in zip(gt.stem, gt.A, gt.R, gt.gt_idx):
        ss = sets_of[stem]
        cb.append(float(hum.loc[([x for x in ss if x not in (A, Rr)][0], Rr, k)]) if len(ss) == 3 else np.nan)
    gt["c_best_iou"] = cb
    OUT.mkdir(parents=True, exist_ok=True)
    rec.to_csv(OUT / "loao_records.csv", index=False)
    gt.to_csv(OUT / "loao_ref_instances.csv", index=False)
    pr.to_csv(OUT / "loao_rater_instances.csv", index=False)
    print(f"loao: {len(rec)} record rows, {rec.stem.nunique()} files, {time.time() - t0:.0f}s")


# ============================================================================ statistics
C4 = ["tp_host", "fp_host", "fn_host", "sum_iou_host"]


def pq_arr(X):
    """X[..., 4] = (tp, fp, fn, sum_iou) -> PQ (host pooling)."""
    den = X[..., 0] + 0.5 * X[..., 1] + 0.5 * X[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, X[..., 3] / np.where(den > 0, den, 1), np.nan)


def ratio(n, d):
    return float(n / d) if d else float("nan")


def boot_diff(A, B, units, n_boot=N_BOOT, seed=SEED, strata=None, extra=None):
    """Bootstrap of PQ(B) - PQ(A) over 'units' (files or time groups); A, B = record DataFrames.
    strata: optional unit -> stratum (resample units within each stratum, i.e. fold-stratified).
    extra: dict name -> (num_col, den_col) ratio statistics (diff B - A) bootstrapped with the same draws."""
    ua = A.groupby(units)[C4 + sum(([n, d] for n, d in (extra or {}).values()), [])].sum()
    ub = B.groupby(units)[C4 + sum(([n, d] for n, d in (extra or {}).values()), [])].sum().reindex(ua.index)
    keys = ua.index.values
    rng = np.random.default_rng(seed)
    if strata is None:
        idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    else:
        st = np.array([strata[k] for k in keys])
        parts = []
        for s in np.unique(st):
            pos = np.where(st == s)[0]
            parts.append(pos[rng.integers(0, len(pos), size=(n_boot, len(pos)))])
        idx = np.concatenate(parts, axis=1)
    out = {}
    Xa, Xb = ua[C4].values.astype(float), ub[C4].values.astype(float)
    SA, SB = Xa[idx].sum(1), Xb[idx].sum(1)
    d = pq_arr(SB) - pq_arr(SA)
    point = float(pq_arr(Xb.sum(0)) - pq_arr(Xa.sum(0)))
    out["PQ"] = dict(diff=point, ci95=[float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                     p_boot_le0=float(np.mean(d <= 0)), n_units=int(len(keys)))
    for name, (n, dd) in (extra or {}).items():
        na, da = ua[n].values[idx].sum(1), ua[dd].values[idx].sum(1)
        nb, db = ub[n].values[idx].sum(1), ub[dd].values[idx].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dx = nb / db - na / da
        pt = ratio(ub[n].sum(), ub[dd].sum()) - ratio(ua[n].sum(), ua[dd].sum())
        out[name] = dict(diff=pt, ci95=[float(np.nanpercentile(dx, 2.5)), float(np.nanpercentile(dx, 97.5))])
    return out


def sign_test(A, B, unit="stem"):
    from scipy.stats import binomtest
    pa = A.groupby(unit)[C4].sum()
    pb = B.groupby(unit)[C4].sum().reindex(pa.index)
    d = pd.Series(pq_arr(pb.values.astype(float)) - pq_arr(pa.values.astype(float)), index=pa.index).dropna()
    better, worse = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    p = binomtest(better, better + worse, 0.5).pvalue if better + worse else float("nan")
    return dict(n=int(len(d)), better=better, worse=worse, equal=int(len(d) - better - worse),
                mean_diff=float(d.mean()), median_diff=float(d.median()), sign_test_p=float(p)), d


def summary_counts(d):
    s = d.sum(numeric_only=True)
    out = dict(PQ=float(pq_arr(s[C4].values.astype(float))), SQ=ratio(s.sum_iou_host, s.tp_host),
               RQ=ratio(s.tp_host, s.tp_host + 0.5 * s.fp_host + 0.5 * s.fn_host),
               TP=int(s.tp_host), FP=int(s.fp_host), FN=int(s.fn_host), n_pred=int(s.n_pred),
               thin_recall=ratio(s.thin_gt_covered_px, s.thin_gt_px),
               thin_precision=ratio(s.thin_pred_in_gt_px, s.thin_pred_px))
    for k in ("fn_split", "fn_merged", "fn_missed", "fn_low_iou", "fp_fragment", "fp_merge", "fp_duplicate",
              "fp_spurious", "fp_low_iou", "n_split_gt", "n_split_fragments", "n_merged_gt", "n_merge_pred",
              "c_other_set_match", "c_duplicate", "c_fragment", "c_background", "c_background_shared_b0", "c_other"):
        if k in s:
            out[k] = int(s[k])
    return out


def compare(df, a_name, b_name, sp, review_stems):
    A = df[df.method == a_name].sort_values("rid").reset_index(drop=True)
    B = df[df.method == b_name].sort_values("rid").reset_index(drop=True)
    assert (A.rid.values == B.rid.values).all()
    thin = {"thin_recall": ("thin_gt_covered_px", "thin_gt_px"), "thin_precision": ("thin_pred_in_gt_px", "thin_pred_px")}
    fold_of_group = sp.groupby("group_id").fold.first().to_dict()
    res = dict(a=a_name, b=b_name, n_records=int(len(A)), n_files=int(A.stem.nunique()),
               A=summary_counts(A), B=summary_counts(B))
    res["file_boot"] = boot_diff(A, B, "stem", extra=thin)
    res["group_boot"] = boot_diff(A, B, "group_id", extra=thin)
    res["group_boot_fold_stratified"] = boot_diff(A, B, "group_id", strata=fold_of_group)
    res["per_fold"] = {}
    for f in FOLDS:
        a_, b_ = A[A.fold == f], B[B.fold == f]
        res["per_fold"][str(f)] = dict(A=float(pq_arr(a_[C4].sum().values.astype(float))),
                                       B=float(pq_arr(b_[C4].sum().values.astype(float))))
        res["per_fold"][str(f)]["diff"] = res["per_fold"][str(f)]["B"] - res["per_fold"][str(f)]["A"]
    st, d = sign_test(A, B, "stem")
    res["per_file_sign"] = st
    gs, gd = sign_test(A, B, "group_id")
    res["per_group_sign"] = gs
    # leave-one-group-out range of the pooled diff (influence of single 14-day groups)
    logo = []
    for g in A.group_id.unique():
        a_, b_ = A[A.group_id != g], B[B.group_id != g]
        logo.append(float(pq_arr(b_[C4].sum().values.astype(float)) - pq_arr(a_[C4].sum().values.astype(float))))
    res["leave_one_group_out_diff_range"] = [min(logo), max(logo)]
    ra, rb = A[A.stem.isin(review_stems)], B[B.stem.isin(review_stems)]
    res["review_set_f01"] = dict(n_files=int(ra.stem.nunique()), n_records=int(len(ra)),
                                 A=float(pq_arr(ra[C4].sum().values.astype(float))),
                                 B=float(pq_arr(rb[C4].sum().values.astype(float))))
    return res, d


def rule_check(cmp_b0):
    """D-019 conditions + per-file sign test, evaluated on folds 0,1 only (the rule is defined over 5 folds)."""
    A, B = cmp_b0["A"], cmp_b0["B"]
    return dict(
        pq_diff_ge_0=cmp_b0["file_boot"]["PQ"]["diff"] >= 0,
        thin_recall_higher=B["thin_recall"] > A["thin_recall"],
        thin_precision_higher=B["thin_precision"] > A["thin_precision"],
        review_set_f01_not_lower=cmp_b0["review_set_f01"]["B"] >= cmp_b0["review_set_f01"]["A"],
        per_file_sign_test_more_better=cmp_b0["per_file_sign"]["better"] > cmp_b0["per_file_sign"]["worse"],
        per_file_sign_test_p=cmp_b0["per_file_sign"]["sign_test_p"])


def loao_analysis(n_boot):
    import beyond_human as BH
    dt = {"prefix_A": str}
    rec = pd.read_csv(OUT / "loao_records.csv", dtype=dt)
    gt = pd.read_csv(OUT / "loao_ref_instances.csv", dtype=dt)
    pr = pd.read_csv(OUT / "loao_rater_instances.csv", dtype=dt)
    faint_thr = json.load(open(ROOT / "outputs/beyond_human/summary.json"))["checks"]["faint_threshold_contrast_snr_q20"]
    raters = ("human",) + LOAO_RATERS
    keys = {k: sorted(rec[rec.rater == k].rec) for k in raters}
    assert all(keys[k] == keys["human"] for k in raters)
    res = dict(n_files=int(rec.stem.nunique()), n_records=len(keys["human"]),
               n_rater_views=int(rec[rec.rater == "human"].A.nunique()), faint_threshold=faint_thr,
               note="folds 0,1 multi-set files only; full-data b0 results in docs/beyond_human.md (296 files)")
    asp = {}
    for k in raters:
        r = rec[rec.rater == k]
        asp[k] = BH.all_aspects(r, gt[gt.rater == k], pr[pr.rater == k], faint_thr)
    sel = ["PQ", "SQ", "RQ", "thin_recall", "thin_precision", "boundary_F2", "split_gt_rate", "merged_gt_rate",
           "fp_fragment_rate", "fp_spurious_rate", "fp_merge_rate", "fn_missed_rate", "small_detected_rate",
           "faint_detected_rate", "faint_ge500_detected_rate", "ref_detected_rate", "limb_detected_rate",
           "consensus_matched_rate", "count_log_bias", "inst_recall", "inst_precision"]
    res["aspects"] = {k: {s: asp[k].get(s) for s in sel} for k in raters}
    res["vs_human"] = {k: BH.paired_boot(rec[rec.rater == "human"], rec[rec.rater == k], n_boot,
                                         stats=("PQ", "SQ", "RQ", "thin_recall", "thin_precision"))
                       for k in LOAO_RATERS}
    # candidate vs b0 on the LOAO records (same records; bootstrap over files)
    b0 = rec[rec.rater == "b0"]
    res["vs_b0"] = {k: BH.paired_boot(b0, rec[rec.rater == k], n_boot,
                                      stats=("PQ", "SQ", "RQ", "thin_recall", "thin_precision"))
                    for k in LOAO_RATERS if k != "b0"}
    # D-025 deficit aspects: is the candidate closer to (or past) the pooled human level than b0?
    dirs = BH.ASPECTS
    deficit = ["split_gt_rate", "fp_fragment_rate", "small_detected_rate", "faint_detected_rate", "thin_recall",
               "merged_gt_rate", "fp_merge_rate", "thin_precision", "fp_spurious_rate"]
    res["deficits"] = {}
    for s in deficit:
        h = asp["human"][s]
        row = dict(direction={1: "higher", -1: "lower", 0: "closer_to_0"}[dirs[s]], human=h)
        for k in LOAO_RATERS:
            row[k] = asp[k][s]
            row[f"{k}_not_worse_than_human"] = BH.not_worse(asp[k][s], h, dirs[s])
        res["deficits"][s] = row
    # per annotator (rater views of folds 0,1): how many annotators does each model beat on PQ, own records
    per = []
    for pa, rh in rec[rec.rater == "human"].groupby("prefix_A"):
        row = dict(annotator=pa, n_views=int(rh.A.nunique()), n_records=int(len(rh)),
                   human=BH.rec_aspects(rh)["PQ"])
        for k in LOAO_RATERS:
            rm = rec[(rec.rater == k) & rec.rec.isin(set(rh.rec))]
            row[k] = BH.rec_aspects(rm)["PQ"]
        per.append(row)
    per = pd.DataFrame(per)
    per.to_csv(OUT / "loao_per_annotator.csv", index=False)
    big = per[per.n_views >= 5]
    res["per_annotator_ge5_views"] = dict(n_annotators=int(len(big)),
                                          **{f"{k}_beats": int((big[k] > big.human).sum()) for k in LOAO_RATERS})
    # 3-set oracle: best other human per reference (picked after seeing the score)
    r3 = rec[rec.n_sets == 3]
    h3 = r3[r3.rater == "human"].copy()
    h3["pq_rec"] = h3.pq_host.fillna(0)
    best = h3.sort_values(["R", "pq_rec", "A"], ascending=[True, False, True]).drop_duplicates("R")
    best_k = best.assign(rec=best.R)
    res["oracle_3set"] = {"n_reference_records": int(len(best)), "best_other_human": BH.rec_aspects(best)["PQ"]}
    for k in LOAO_RATERS:
        m3 = r3[r3.rater == k].drop_duplicates("R").set_index("R").loc[best.R].reset_index()
        res["oracle_3set"][k] = BH.paired_boot(best_k, m3.assign(rec=m3.R), n_boot, stats=("PQ",))["PQ"]
    return res


def cmd_analyse(a):
    df = pd.read_csv(OUT / "records.csv")
    info = pd.read_csv(OUT / "file_info.csv")
    sp = split_table()
    rc = pd.read_csv(ROOT / "outputs/audit/review_cases.csv")
    rc01 = rc[rc.fold_v1.isin(FOLDS)]
    review_stems = set(rc01.file_name.str[:-5])
    res = dict(folds=list(FOLDS), n_files=int(df.stem.nunique()), n_records=int(df[df.method == "b0"].shape[0]),
               n_time_groups=int(df.group_id.nunique()), n_boot=N_BOOT, seed=SEED,
               note="post-hoc candidate; 2 folds; all configurations reported, none chosen here")
    # consistency with the earlier compare_paired / fp_decomposition outputs
    old = json.load(open(ROOT / "outputs/comparisons/b0_r101_s1024-cc_vs_R2_refine_context-refineb0_f01_t0.5_a300_sNone_val_probs_last.json"))
    new_main = summary_counts(df[df.method == MAIN])
    new_b0 = summary_counts(df[df.method == "b0"])
    res["reproduces_compare_paired"] = dict(
        b0=[old["A"]["PQ"], new_b0["PQ"], old["A"]["TP"] == new_b0["TP"] and old["A"]["FP"] == new_b0["FP"]],
        R2b0_any_a300=[old["B"]["PQ"], new_main["PQ"], old["B"]["TP"] == new_main["TP"] and old["B"]["FP"] == new_main["FP"]])
    fpd = json.load(open(ROOT / "outputs/comparisons/fp_decomposition_f01.json"))
    fpk = {k: v for k, v in fpd.items() if k.startswith("R2_refine_context:refine:")}
    if fpk:
        v = list(fpk.values())[0]
        n2 = summary_counts(df[df.method == "R2full_a300"])
        res["reproduces_fp_decomposition_R2full"] = dict(
            old=v["FP_classes"], new={k[2:]: n2[k] for k in n2 if k.startswith("c_")})
    # (a, b, c, f) every method vs b0
    rows, diffs = [], {}
    res["vs_b0"] = {}
    for m in method_names():
        if m == "b0":
            continue
        r, d = compare(df, "b0", m, sp, review_stems)
        res["vs_b0"][m] = r
        diffs[m] = d
        fb, gb, gbs = r["file_boot"], r["group_boot"], r["group_boot_fold_stratified"]
        rows.append(dict(method=m, PQ=r["B"]["PQ"], dPQ=fb["PQ"]["diff"],
                         file_ci_lo=fb["PQ"]["ci95"][0], file_ci_hi=fb["PQ"]["ci95"][1],
                         group_ci_lo=gb["PQ"]["ci95"][0], group_ci_hi=gb["PQ"]["ci95"][1],
                         groupstrat_ci_lo=gbs["PQ"]["ci95"][0], groupstrat_ci_hi=gbs["PQ"]["ci95"][1],
                         d_f0=r["per_fold"]["0"]["diff"], d_f1=r["per_fold"]["1"]["diff"],
                         files_better=r["per_file_sign"]["better"], files_worse=r["per_file_sign"]["worse"],
                         sign_p=r["per_file_sign"]["sign_test_p"],
                         groups_better=r["per_group_sign"]["better"], groups_worse=r["per_group_sign"]["worse"],
                         thin_R=r["B"]["thin_recall"], thin_P=r["B"]["thin_precision"],
                         d_thin_R=fb["thin_recall"]["diff"], d_thin_R_lo=fb["thin_recall"]["ci95"][0],
                         d_thin_R_hi=fb["thin_recall"]["ci95"][1],
                         d_thin_P=fb["thin_precision"]["diff"], d_thin_P_lo=fb["thin_precision"]["ci95"][0],
                         d_thin_P_hi=fb["thin_precision"]["ci95"][1],
                         review_b0=r["review_set_f01"]["A"], review=r["review_set_f01"]["B"],
                         TP=r["B"]["TP"], FP=r["B"]["FP"], FN=r["B"]["FN"],
                         fn_split=r["B"]["fn_split"], fn_merged=r["B"]["fn_merged"], fp_fragment=r["B"]["fp_fragment"],
                         fp_merge=r["B"]["fp_merge"], fp_duplicate=r["B"]["fp_duplicate"],
                         logo_lo=r["leave_one_group_out_diff_range"][0], logo_hi=r["leave_one_group_out_diff_range"][1]))
    res["b0"] = new_b0
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "sensitivity_grid.csv", index=False)
    # (d) main vs R2 full
    r, _ = compare(df, "R2full_a300", MAIN, sp, review_stems)
    res["main_vs_R2full"] = r
    r, _ = compare(df, "R1full_a300", MAIN_R1, sp, review_stems)
    res["R1main_vs_R1full"] = r
    r, _ = compare(df, MAIN_R1, MAIN, sp, review_stems)
    res["R2main_vs_R1main"] = r
    r, _ = compare(df, "b0_a300", MAIN, sp, review_stems)
    res["main_vs_b0_a300"] = r
    # per-file diffs of the main candidate (for review / figures)
    dmain = diffs[MAIN].rename("dPQ_file").to_frame()
    dmain["review"] = dmain.index.isin(review_stems)
    dmain.join(sp.set_index("stem")[["fold", "group_id", "n_sets"]]).sort_values("dPQ_file").to_csv(
        OUT / "per_file_diff_main.csv")
    # (c) review set per case, main and R1 main
    rv = []
    for _, c in rc01.iterrows():
        st = c.file_name[:-5]
        row = dict(case_id=c.case_id, category=c.category, file=st, fold=int(c.fold_v1))
        for m in ("b0", MAIN, MAIN_R1, "R2b0_frac50_a300", "R2full_a300"):
            d_ = df[(df.method == m) & (df.stem == st)]
            row[m] = float(pq_arr(d_[C4].sum().values.astype(float)))
        rv.append(row)
    rv = pd.DataFrame(rv)
    rv.to_csv(OUT / "review_set_f01_cases.csv", index=False)
    res["review_cases_f01"] = rv.to_dict(orient="records")
    res["review_cases_f01_main_better_worse_equal"] = [int((rv[MAIN] > rv.b0 + 1e-9).sum()),
                                                       int((rv[MAIN] < rv.b0 - 1e-9).sum()),
                                                       int((abs(rv[MAIN] - rv.b0) <= 1e-9).sum())]
    res["rule_check_folds01"] = {m: rule_check(res["vs_b0"][m]) for m in (MAIN, MAIN_R1)}
    # (g-part) window counts
    res["windows"] = dict(files=int(len(info)), cand_per_file=float(info.n_cand.mean()),
                          win_per_file_all=float(info.n_win.mean()), win_per_file_any=float(info.n_win_any.mean()),
                          win_per_file_frac50=float(info.n_win_frac50.mean()),
                          keep_any_per_file=float(info.n_keep_any.mean()),
                          keep_frac50_per_file=float(info.n_keep_frac50.mean()), b0_inst_per_file=float(info.n_b0.mean()),
                          win_per_b0_instance_any=float(info.n_win_any.sum() / max(info.n_b0.sum(), 1)),
                          sec_per_file_table_job=float(info.sec.mean()))
    # (e) LOAO
    if (OUT / "loao_records.csv").exists():
        res["loao"] = loao_analysis(N_BOOT)
    json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(grid[["method", "PQ", "dPQ", "file_ci_lo", "file_ci_hi", "group_ci_lo", "group_ci_hi", "d_f0", "d_f1",
                "files_better", "files_worse", "sign_p", "groups_better", "groups_worse", "thin_R", "thin_P",
                "review_b0", "review"]].round(4).to_string())
    print(json.dumps({k: res[k] for k in ("reproduces_compare_paired", "rule_check_folds01", "windows",
                                          "review_cases_f01_main_better_worse_equal")}, indent=1, default=str))


# ============================================================================ timing (CPU)
def cmd_timing(a):
    import torch
    assert not torch.cuda.is_available(), "GPU must not be visible"
    torch.set_num_threads(a.threads)
    W = _cache()
    idx = W["idx"]
    arr = {k: np.load(RF.CACHE / f"{k}.npy", mmap_mode="r") for k in ("images", "prob_u8", "cand")}
    res = dict(threads=a.threads, torch=torch.__version__, cpu_count=os.cpu_count(),
               loadavg_before=os.getloadavg())
    # R2 fold-0 model on fold-0 validation windows of single-window candidates (comparable with stored probs)
    fold = 0
    model = RF.build_model("r2")
    sd = torch.load(ROOT / "outputs/exps/R2_refine_context/fold0/last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    va = [f for f in idx["files"] if idx["fold"][f] == fold]
    items = []
    for f in va:
        wins = idx["meta"][f]["windows"]
        cnt = {}
        for j, y, x in wins:
            cnt[j] = cnt.get(j, 0) + 1
        for j, y, x in wins:
            if cnt[j] == 1:
                items.append((f, j, y, x))
    rng = np.random.default_rng(SEED)
    pick = [items[i] for i in rng.choice(len(items), a.n_windows, replace=False)]
    t_in, xs = time.time(), []
    for f, j, y, x in pick:
        fi = W["fi"][f]
        xs.append(RF.make_input("r2", arr["images"][fi], arr["prob_u8"][fi], arr["cand"][fi], j, y, x))
    t_in = (time.time() - t_in) / len(pick)
    X = torch.from_numpy(np.stack(xs))
    with torch.inference_mode():
        model(X[:2])  # warm-up
        t0 = time.time()
        outs = []
        for i in range(0, len(X), a.bs):
            outs.append(torch.sigmoid(model(X[i:i + a.bs]))[:, 0].numpy())
        t_win = (time.time() - t0) / len(X)
    P = np.concatenate(outs)
    # compare with the stored GPU (bf16 autocast) outputs
    z, by = _npz(ROOT / "outputs/exps/R2_refine_context/fold0/val_cand_probs.npz")
    agree, maxd = [], []
    for (f, j, y, x), p in zip(pick, P):
        st = f[:-5]
        k = [kk for jj, yy, xx, kk in by[st] if jj == j][0]
        q = z[k].astype(np.float32)
        maxd.append(float(np.abs(q - p).max()))
        m1, m2 = p > 0.5, q > 0.5
        agree.append(float((m1 & m2).sum() / max((m1 | m2).sum(), 1)))
    res["r2_cpu_fp32"] = dict(n_windows=len(pick), batch=a.bs, sec_per_window_forward=t_win,
                              sec_per_window_input_build=t_in,
                              vs_stored_gpu_bf16=dict(max_abs_prob_diff_median=float(np.median(maxd)),
                                                      max_abs_prob_diff_max=float(np.max(maxd)),
                                                      mask_iou_t05_median=float(np.median(agree)),
                                                      mask_iou_t05_min=float(np.min(agree))))
    del model
    # b0 stage-1 forward at 1024^2 (one fold model; the test pipeline averages 5)
    m0 = seg.build_model("resnet101", pretrained=None)
    m0.load_state_dict(torch.load(ROOT / "outputs/exps/b0_r101_s1024/fold0/last.pt", map_location="cpu", weights_only=True))
    m0.eval()
    x = torch.randn(1, 1, 1024, 1024)
    with torch.inference_mode():
        m0(x)
        t0 = time.time()
        for _ in range(a.n_b0):
            m0(x)
        t_b0 = (time.time() - t0) / a.n_b0
    res["b0_cpu_fp32"] = dict(sec_per_image_per_model_1024=t_b0, n=a.n_b0)
    # test-set estimate: windows per test image from OOF ratios x our own test b0 instance counts
    summ = json.load(open(OUT / "summary.json")) if (OUT / "summary.json").exists() else None
    sub = pd.read_csv(ROOT / "outputs/submissions/frozen/b0_r101_s1024_5fold_last_t05_a500.csv")
    meta = json.load(open(ROOT / "outputs/submissions/frozen/b0_r101_s1024_5fold_last_t05_a500.csv.meta.json"))
    n_test = int(meta["n_images"])            # our own prediction file: one row per predicted instance
    n_inst_test = int(len(sub))
    res["test_estimate"] = dict(n_test_images_in_our_csv=n_test, n_rows_in_our_csv=n_inst_test)
    if summ:
        w = summ["windows"]
        n_win_any = w["win_per_b0_instance_any"] * n_inst_test
        n_win_all = w["win_per_file_all"] * n_test
        res["test_estimate"].update(
            est_windows_any=n_win_any, est_windows_all_candidates=n_win_all,
            minutes_R2_per_model_any=n_win_any * (t_win + t_in) / 60,
            minutes_R2_5models_any=5 * n_win_any * t_win / 60 + n_win_any * t_in / 60,
            minutes_b0_stage1_5models=5 * n_test * t_b0 / 60)
    res["loadavg_after"] = os.getloadavg()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "timing_cpu.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "loao", "analyse", "timing"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--n_windows", type=int, default=24)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--n_b0", type=int, default=2)
    a = ap.parse_args()
    assert a.workers <= 16 and a.threads <= 16
    {"table": cmd_table, "loao": cmd_loao, "analyse": cmd_analyse, "timing": cmd_timing}[a.cmd](a)


if __name__ == "__main__":
    main()
