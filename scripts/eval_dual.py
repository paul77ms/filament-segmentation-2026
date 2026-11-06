"""Dual-scorer evaluation of the R2 stage on the nested OOF (docs/improvement_plan.md §1-§2). CPU only.

Methods (outer-validation files of each fold; stage 1 = mean of the 4 inner models, nested_r2b0_eval.instances):
  b0_single, B_ens, B_ens_a300, R2nb0 (R2 refinement), R2nfull  - as in the pre-registered evaluation
  R2nb0_e300 : R2nb0 with the existence gate at min_area 300 (candidate overlaps a B_ens t0.5/a300 instance)
Outputs (outputs/research/eval_dual/):
  records.csv     per (method, record): V0 host counts + causes + thin + notebook overlap diagnostics
  images_v1.csv   per (method, image): V1 counts (all annotation sets of the image concatenated, one-to-one)
  loao_*.csv      leave-one-annotator-out records (beyond_human protocol) for the LOAO raters
  summary_<main>_vs_<base>.json   V0 + V1 comparisons, diagnostics, LOAO aspects, the registered rule
  CUDA_VISIBLE_DEVICES="" taskset -c 0-3 .venv/bin/python scripts/eval_dual.py table|loao|analyse [--base R2nb0 --main R2nb0_e300]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import nested_r2b0_eval as NE  # noqa: E402
import refine_b0exist_checks as RC  # noqa: E402
from filseg import refine as RF  # noqa: E402
from filseg import scorers as SC  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.metrics import evaluate_image, single_thread_worker  # noqa: E402

OUT = ROOT / "outputs" / "research" / "eval_dual"
CFG = dict(ens_exp="b0n_ens", r2_exp="R2n_context", b0_exp="b0_r101_s1024",
           cache_prefix=str(ROOT / "outputs" / "cache" / "refine_nested_o"))
METHODS = ("b0_single", "B_ens", "B_ens_a300", "R2nb0", "R2nb0_e300", "R2nfull")
LOAO_RATERS = ("B_ens", "R2nb0", "R2nb0_e300")
SEED, N_BOOT = 2026, 2000


def instances(stem, fold):
    M, info = NE.instances(stem, fold, CFG)
    ens = NE._load(("ens", fold), lambda: np.load(NE.EXPS / CFG["ens_exp"] / f"fold{fold}" / "val_probs_last.npz"))
    idx = NE._load(("idx", fold), lambda: json.load(open(Path(CFG["cache_prefix"] + str(fold)) / "index.json")))
    cand = NE._load(("cand", fold), lambda: np.load(Path(CFG["cache_prefix"] + str(fold)) / "cand.npy", mmap_mode="r"))
    cmap = np.asarray(cand[idx["files"].index(stem + ".jpeg")])
    z, by = RC._npz(NE.EXPS / CFG["r2_exp"] / f"fold{fold}" / "val_cand_probs.npz")
    cp = {j: (y0, x0, z[k].astype(np.float32)) for j, y0, x0, k in by.get(stem, [])}
    B300 = seg.prob_to_instances(ens[stem].astype(np.float32), seg.PostCfg(0.5, 300))
    inb = np.bincount(cmap[B300.union()], minlength=int(cmap.max()) + 1)
    keep = {j for j in range(1, len(inb)) if inb[j] > 0}
    M["R2nb0_e300"] = RF.assemble({j: v for j, v in cp.items() if j in keep}, thr=0.5, min_area=300)
    return M


def table_job(args):
    stem, fold, gts = args
    M = instances(stem, fold)
    rec, img = [], []
    sets = [gts[r] for r in sorted(gts)]
    for name in METHODS:
        P = M[name]
        for rid, G in gts.items():
            r, _, _ = evaluate_image(G, P, RC.CFG, image_key=rid, return_instances=False)
            rec.append(dict(method=name, rid=rid, stem=stem, fold=fold, **{k: r.get(k, 0) for k in RC.REC_KEYS},
                            **SC.notebook_diag(G, P)))
        tp, fp, fn, si = SC.v1_counts(sets, P)
        img.append(dict(method=name, stem=stem, fold=fold, n_sets=len(sets), v1_tp=tp, v1_fp=fp, v1_fn=fn, v1_si=si))
    return rec, img


def loao_job(args):
    import beyond_human as BH
    stem, set_ids = args[0], args[1]
    fold = args[2]
    S = BH.load_sets(set_ids)
    M_all = instances(stem, fold)
    recs, gtl, prl = [], [], []

    def add(rater, A, R, res):
        row, pg, pp = res
        base = dict(rater=rater, stem=stem, A=A, R=R, rec=f"{A}|{R}")
        recs.append({**base, **{c: row.get(c, np.nan) for c in BH.REC_COLS}})
        gtl.extend({**base, **{c: d[c] for c in BH.GT_COLS}} for d in pg)
        prl.extend({**base, **{c: d[c] for c in BH.PR_COLS}} for d in pp)

    ev = {(k, R): evaluate_image(S[R], M_all[k], BH.CFG, image_key=R) for k in LOAO_RATERS for R in set_ids}
    for A in set_ids:
        for R in set_ids:
            if R != A:
                add("human", A, R, evaluate_image(S[R], S[A], BH.CFG, image_key=f"{A}|{R}"))
                for k in LOAO_RATERS:
                    add(k, A, R, ev[(k, R)])
    return recs, gtl, prl


def cmd_table(a):
    import multiprocessing as mp
    single_thread_worker()
    jobs = []
    for k in range(5):
        files = (NE.EXPS / "b0n_ens" / f"fold{k}" / "val_files.txt").read_text().split()
        gts = seg.gt_records(files)
        by = {}
        for rid, g in gts.items():
            by.setdefault(rid.split("-", 1)[1], {})[rid] = g
        jobs += [(s, k, by[s]) for s in sorted(by)]
    t0, R, I = time.time(), [], []
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for rec, img in pool.imap_unordered(table_job, jobs, chunksize=2):
            R += rec; I += img
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(R)
    sp = RC.split_table().set_index("stem")
    df["group_id"] = df.stem.map(sp.group_id); df["n_sets"] = df.stem.map(sp.n_sets)
    df.to_csv(OUT / "records.csv", index=False)
    pd.DataFrame(I).to_csv(OUT / "images_v1.csv", index=False)
    print(f"table: {len(df)} record rows, {df.stem.nunique()} files, {time.time() - t0:.0f}s")


def cmd_loao(a):
    import multiprocessing as mp
    import beyond_human as BH
    single_thread_worker()
    sp = RC.split_table()
    multi = sp[sp.n_sets >= 2]
    jobs = [(r.stem, sorted(r.image_ids.split(";")), int(r.fold)) for r in multi.itertuples()]
    R, G, P = [], [], []
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for r, g, p in pool.imap_unordered(loao_job, jobs, chunksize=1):
            R += r; G += g; P += p
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
    gt["c_best_iou"] = [float(hum.loc[([x for x in sets_of[s] if x not in (A, Rr)][0], Rr, k)]) if len(sets_of[s]) == 3
                        else np.nan for s, A, Rr, k in zip(gt.stem, gt.A, gt.R, gt.gt_idx)]
    OUT.mkdir(parents=True, exist_ok=True)
    rec.to_csv(OUT / "loao_records.csv", index=False)
    gt.to_csv(OUT / "loao_ref_instances.csv", index=False)
    pr.to_csv(OUT / "loao_rater_instances.csv", index=False)
    print(f"loao: {len(rec)} rows, {rec.stem.nunique()} files")


def v1_boot(I, base, main, unit="stem"):
    A = I[I.method == base].set_index(unit).sort_index()
    B = I[I.method == main].set_index(unit).sort_index()
    cols = ["v1_tp", "v1_fp", "v1_fn", "v1_si"]
    XA, XB = A[cols].values.astype(float), B[cols].values.astype(float)

    def pq(X):
        return X[..., 3].sum(-1) / (X[..., 0].sum(-1) + .5 * X[..., 1].sum(-1) + .5 * X[..., 2].sum(-1))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(XA), size=(N_BOOT, len(XA)))
    d = pq(XB[idx]) - pq(XA[idx])
    folds = {int(k): float(pq(B[B.fold == k][cols].values.astype(float)) - pq(A[A.fold == k][cols].values.astype(float)))
             for k in sorted(A.fold.unique())}
    per = (XB[:, 3] - XA[:, 3])  # not used for the sign test (PQ is not additive); per-file PQ below
    return dict(A=float(pq(XA)), B=float(pq(XB)), diff=float(pq(XB) - pq(XA)),
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], per_fold=folds,
                folds_positive=int(sum(v > 0 for v in folds.values())))


def diag_sums(d):
    s = d.sum(numeric_only=True)
    keys = ["nb_gt_1to0", "nb_gt_1to1", "nb_gt_1toN", "nb_pred_0to1", "nb_pred_1to1", "nb_pred_Nto1", "over_frag",
            "under_frag", "fn_split", "fn_merged", "fp_merge", "fp_fragment", "fp_spurious", "fn_missed"]
    out = {k: int(s[k]) for k in keys}
    out.update(pair_iou_mean=float(s.pair_iou_sum / max(s.pair_n, 1)), pair_dice_mean=float(s.pair_dice_sum / max(s.pair_n, 1)),
               semantic_dice_pooled=float(2 * s.sem_inter / max(s.sem_gt + s.sem_pred, 1)),
               thin_recall=float(s.thin_gt_covered_px / max(s.thin_gt_px, 1)),
               thin_precision=float(s.thin_pred_in_gt_px / max(s.thin_pred_px, 1)))
    return out


def loao_aspects():
    import beyond_human as BH
    dt = {"prefix_A": str}
    rec = pd.read_csv(OUT / "loao_records.csv", dtype=dt)
    gt = pd.read_csv(OUT / "loao_ref_instances.csv", dtype=dt)
    pr = pd.read_csv(OUT / "loao_rater_instances.csv", dtype=dt)
    faint = json.load(open(ROOT / "outputs/beyond_human/summary.json"))["checks"]["faint_threshold_contrast_snr_q20"]
    return {k: BH.all_aspects(rec[rec.rater == k], gt[gt.rater == k], pr[pr.rater == k], faint)
            for k in ("human",) + LOAO_RATERS}, BH


def cmd_analyse(a):
    df = pd.read_csv(OUT / "records.csv")
    I = pd.read_csv(OUT / "images_v1.csv")
    sp = RC.split_table()
    RC.FOLDS = (0, 1, 2, 3, 4)
    rc = pd.read_csv(ROOT / "outputs/audit/review_cases.csv")
    review = set(rc.file_name.str[:-5])
    v0, _ = RC.compare(df, a.base, a.main, sp, review)
    v1 = v1_boot(I, a.base, a.main)
    rv1 = v1_boot(I[I.stem.isin(review)], a.base, a.main)
    res = dict(base=a.base, main=a.main, n_files=int(df.stem.nunique()),
               V0=dict(A=v0["A"]["PQ"], B=v0["B"]["PQ"], diff=v0["file_boot"]["PQ"]["diff"], ci95=v0["file_boot"]["PQ"]["ci95"],
                       per_fold={k: v["diff"] for k, v in v0["per_fold"].items()}, sign=v0["per_file_sign"],
                       review=v0["review_set_f01"]),
               V1=v1, V1_review=dict(A=rv1["A"], B=rv1["B"]),
               V1_all_methods={m: v1_boot(I, "B_ens", m)["B"] for m in METHODS},
               V0_all_methods={m: RC.summary_counts(df[df.method == m])["PQ"] for m in METHODS},
               diagnostics={m: diag_sums(df[df.method == m]) for m in METHODS})
    asp, BH = loao_aspects()
    keyset = ["PQ", "thin_recall", "thin_precision", "thin_precision_colocated", "merged_gt_rate", "fp_merge_rate",
              "split_gt_rate", "fp_fragment_rate", "fp_spurious_rate", "small_detected_rate", "faint_detected_rate"]
    res["loao"] = {k: {r: asp[r].get(k) for r in ("human",) + LOAO_RATERS} for k in keyset}
    newly = []
    for s, d in BH.ASPECTS.items():
        def v(r):
            x = asp[r].get(s)
            return np.nan if x is None else float(x)
        if BH.not_worse(v(a.base), v("human"), d) is True and BH.not_worse(v(a.main), v("human"), d) is False:
            newly.append(s)
    res["loao_newly_below_human"] = newly
    h = asp["human"]; m = asp[a.main]
    rule = dict(
        R1_v1_ci_gt0_and_4of5=bool(v1["ci95"][0] > 0 and v1["folds_positive"] >= 4),
        R2_v0_not_lower=bool(res["V0"]["diff"] >= 0 or (res["V0"]["ci95"][0] <= 0 <= res["V0"]["ci95"][1] and res["V0"]["ci95"][0] > -0.003)),
        R3_merges_not_above_human=bool(m["merged_gt_rate"] <= h["merged_gt_rate"] and m["fp_merge_rate"] <= h["fp_merge_rate"]),
        R4_no_loao_aspect_newly_below_human=not newly,
        R5_review_not_lower_v0_and_v1=bool(res["V0"]["review"]["B"] >= res["V0"]["review"]["A"] and rv1["B"] >= rv1["A"]))
    rule["PASS"] = all(rule.values())
    res["rule"] = rule
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"summary_{a.main}_vs_{a.base}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(dict(V0={k: res["V0"][k] for k in ("A", "B", "diff", "ci95", "per_fold")}, V1=v1, rule=rule,
                          newly=newly), indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "loao", "analyse"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base", default="R2nb0")
    ap.add_argument("--main", default="R2nb0_e300")
    a = ap.parse_args()
    {"table": cmd_table, "loao": cmd_loao, "analyse": cmd_analyse}[a.cmd](a)
