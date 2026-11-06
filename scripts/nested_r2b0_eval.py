"""PRE-REGISTERED evaluation of 'b0 existence + R2 refinement' under the nested design (D-035). CPU only.

Every prediction for outer fold k comes from models that never saw fold k and whose TRAINING INPUTS were produced
by models that never saw fold k (scripts/nested_stage1.py verify). The stage 1 of fold k is the mean of 4 inner b0
models, mirroring the 5-model mean used for the submission.

Methods (outer-validation files of fold k only):
  B_ens       : 4-inner-model mean (b0n_ens), t0.5, 8-conn, min_area 500         MATCHED BASELINE (submission recipe)
  R2nb0       : R2n_k refined masks of the stage-1 candidates (p > 0.3, >= 20 px of the SAME mean) that overlap a
                B_ens instance by >= 1 px, assembled (merge IoU > 0.5, conflicts by max prob), min_area 300    PRIMARY
  B_ens_a300  : B_ens with min_area 300 (information: isolates the min_area change)
  R2nfull     : all candidates refined, min_area 300 (information)
  b0_single   : existing b0 OOF (one fold model), t0.5/a500 (reference: the current submission's OOF 0.4099)

Adoption rule (fixed in experiments/registry.csv row R2b0_nested_prereg BEFORE any nested result exists; identical
conditions to R2b0_confirm_prereg, now against the matched baseline B_ens):
  over the 5 outer folds, R2nb0 vs B_ens: pooled PQ diff >= 0 AND thin-region recall and precision both higher AND
  per-file sign test more better than worse with p < 0.05 AND the 32-case review set PQ not lower AND no LOAO aspect
  where B_ens is at or above the pooled human level falls below it under R2nb0.
Validity gates (an analysis that fails them cannot ADOPT): all 5 post-R2n verify files (scripts/nested_stage1.py
verify, incl. check E) pass; the table covers every validation file of the 5 folds; LOAO covers every multi-set file
of those folds; no LOAO aspect is undetermined (NaN) under R2nb0 where B_ens is at or above the human level.
Also reported (not part of the rule): file / 14-day-group bootstrap CIs, per-fold diffs, folds 2-4 alone (not seen
when the R2b0 configuration was conceived), merges, FP classes.

  CUDA_VISIBLE_DEVICES="" taskset -c 0-3 .venv/bin/python scripts/nested_r2b0_eval.py table --workers 4
  ... loao --workers 4 ; ... analyse
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import refine_b0exist_checks as RC  # noqa: E402  (statistics + FP classes reused unchanged)
from filseg import refine as RF  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.metrics import evaluate_image, single_thread_worker  # noqa: E402

EXPS = Path(os.environ.get("FILSEG_EXPS", ROOT / "outputs" / "exps"))   # redirected only by wiring tests
METHODS = ("B_ens", "R2nb0", "B_ens_a300", "R2nfull", "b0_single")
BASE, MAIN = "B_ens", "R2nb0"
LOAO_RATERS = ("B_ens", "R2nb0", "R2nfull", "b0_single")
CONFIRM_FOLDS = (2, 3, 4)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["table", "loao", "analyse"])
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--ens_exp", default="b0n_ens")
    p.add_argument("--r2_exp", default="R2n_context")
    p.add_argument("--b0_exp", default="b0_r101_s1024")
    p.add_argument("--cache_prefix", default=str(ROOT / "outputs" / "cache" / "refine_nested_o"))
    p.add_argument("--files_per_fold", type=int, default=0, help="wiring tests only")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n_boot", type=int, default=2000, help="LOAO bootstrap (paired comparisons use RC.N_BOOT=2000)")
    p.add_argument("--out", default=str(ROOT / "outputs" / "nested" / "eval"))
    p.add_argument("--verify_dir", default=str(ROOT / "outputs" / "nested"))
    return p.parse_args()


# ============================================================================ instances (worker side)
_W: dict = {}


def _load(key, fn):
    if key not in _W:
        _W[key] = fn()
    return _W[key]


def instances(stem, fold, cfg):
    ens = _load(("ens", fold), lambda: np.load(EXPS / cfg["ens_exp"] / f"fold{fold}" / "val_probs_last.npz"))
    b0 = _load(("b0", fold), lambda: np.load(EXPS / cfg["b0_exp"] / f"fold{fold}" / "val_probs_last.npz"))
    cache = Path(cfg["cache_prefix"] + str(fold))
    idx = _load(("idx", fold), lambda: json.load(open(cache / "index.json")))
    fi = _load(("fi", fold), lambda: {f: i for i, f in enumerate(idx["files"])})
    cand = _load(("cand", fold), lambda: np.load(cache / "cand.npy", mmap_mode="r"))
    z, by = RC._npz(EXPS / cfg["r2_exp"] / f"fold{fold}" / "val_cand_probs.npz")
    p = ens[stem].astype(np.float32)
    B = seg.prob_to_instances(p, seg.PostCfg(0.5, 500))
    cmap = np.asarray(cand[fi[stem + ".jpeg"]])
    n = int(cmap.max())
    inb = np.bincount(cmap[B.union()], minlength=n + 1)
    keep = {j for j in range(1, n + 1) if inb[j] > 0}
    cp = {j: (y0, x0, z[k].astype(np.float32)) for j, y0, x0, k in by.get(stem, [])}
    M = {"B_ens": B,
         "R2nb0": RF.assemble({j: v for j, v in cp.items() if j in keep}, thr=0.5, min_area=300),
         "B_ens_a300": seg.prob_to_instances(p, seg.PostCfg(0.5, 300)),
         "R2nfull": RF.assemble(cp, thr=0.5, min_area=300),
         "b0_single": seg.prob_to_instances(b0[stem].astype(np.float32), seg.PostCfg(0.5, 500))}
    wins = idx["meta"][stem + ".jpeg"]["windows"]
    info = dict(stem=stem, fold=fold, n_cand=n, n_keep=len(keep), n_win=len(wins),
                n_win_keep=sum(1 for j, _, _ in wins if j in keep), n_B_ens=len(B))
    return M, info


def table_job(args):
    stem, fold, gts, cfg = args
    t0 = time.time()
    M, info = instances(stem, fold, cfg)
    rows = []
    for name, P in M.items():
        for rid, G in gts.items():
            r, _, _ = evaluate_image(G, P, RC.CFG, image_key=rid, return_instances=False)
            row = dict(method=name, rid=rid, stem=stem, fold=fold, **{k: r.get(k, 0) for k in RC.REC_KEYS})
            row.update(RC.fp_classes(rid, G, P, gts, M[BASE]))
            rows.append(row)
    info["sec"] = time.time() - t0
    return rows, info


def loao_job(args):
    import beyond_human as BH
    file_name, stem, fold, set_ids, cfg = args
    S = BH.load_sets(set_ids)
    M_all, _ = instances(stem, fold, cfg)
    M = {k: M_all[k] for k in LOAO_RATERS}
    recs, gts, prs = [], [], []

    def add(rater, A, R, res):
        row, pg, pp = res
        base = dict(rater=rater, stem=stem, A=A, R=R, rec=f"{A}|{R}")
        recs.append({**base, **{c: row.get(c, np.nan) for c in BH.REC_COLS}})
        gts.extend({**base, **{c: d[c] for c in BH.GT_COLS}} for d in pg)
        prs.extend({**base, **{c: d[c] for c in BH.PR_COLS}} for d in pp)

    model_eval = {(k, R): evaluate_image(S[R], P, BH.CFG, image_key=R) for k, P in M.items() for R in set_ids}
    for A in set_ids:
        for R in set_ids:
            if R != A:
                add("human", A, R, evaluate_image(S[R], S[A], BH.CFG, image_key=f"{A}|{R}"))
                for k in LOAO_RATERS:
                    add(k, A, R, model_eval[(k, R)])
    return recs, gts, prs


# ============================================================================ commands
def cfg_of(a):
    return dict(ens_exp=a.ens_exp, r2_exp=a.r2_exp, b0_exp=a.b0_exp, cache_prefix=a.cache_prefix)


def val_files(a, k):
    files = (EXPS / a.ens_exp / f"fold{k}" / "val_files.txt").read_text().split()
    assert files == (EXPS / a.r2_exp / f"fold{k}" / "val_files.txt").read_text().split(), "R2n/ens files differ"
    if a.files_per_fold:
        files = files[:a.files_per_fold]
    return files


def cmd_table(a):
    import multiprocessing as mp
    single_thread_worker()
    jobs = []
    for k in map(int, a.folds.split(",")):
        rc = json.load(open(EXPS / a.r2_exp / f"fold{k}" / "config.json"))
        assert Path(rc["cache_dir"]).resolve() == Path(a.cache_prefix + str(k)).resolve(), \
            f"R2n fold {k} was trained from {rc['cache_dir']}, not the nested cache (candidate ids would not match)"
        assert rc["exist_exp"] == a.ens_exp, f"R2n fold {k} existence source {rc['exist_exp']} != {a.ens_exp}"
        gts = seg.gt_records(val_files(a, k))
        by = {}
        for rid, g in gts.items():
            by.setdefault(rid.split("-", 1)[1], {})[rid] = g
        jobs += [(s, k, by[s], cfg_of(a)) for s in sorted(by)]
    t0 = time.time()
    R, I = [], []
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for i, (rows, info) in enumerate(pool.imap_unordered(table_job, jobs, chunksize=1)):
            R += rows
            I.append(info)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} files {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(R)
    sp = RC.split_table().set_index("stem")
    df["group_id"] = df.stem.map(sp.group_id)
    df["n_sets"] = df.stem.map(sp.n_sets)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "records.csv", index=False)
    pd.DataFrame(I).to_csv(out / "file_info.csv", index=False)
    print(f"table: {len(df)} rows, {df.stem.nunique()} files, {time.time() - t0:.0f}s")


def cmd_loao(a):
    import multiprocessing as mp
    import beyond_human as BH
    single_thread_worker()
    sp = RC.split_table()
    folds = list(map(int, a.folds.split(",")))
    allowed = {f for k in folds for f in val_files(a, k)}
    multi = sp[(sp.n_sets >= 2) & sp.fold.isin(folds) & sp.file_name.isin(allowed)]
    jobs = [(r.file_name, r.stem, int(r.fold), sorted(r.image_ids.split(";")), cfg_of(a)) for r in multi.itertuples()]
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
    # consensus reference (as in refine_b0exist_checks.cmd_loao): on 3-set files, the best IoU of the third annotator
    hum = gt[gt.rater == "human"].set_index(["A", "R", "gt_idx"]).best_iou
    sets_of = rec.groupby("stem").R.agg(lambda s: sorted(set(s))).to_dict()
    cb = []
    for stem, A, Rr, k in zip(gt.stem, gt.A, gt.R, gt.gt_idx):
        ss = sets_of[stem]
        cb.append(float(hum.loc[([x for x in ss if x not in (A, Rr)][0], Rr, k)]) if len(ss) == 3 else np.nan)
    gt["c_best_iou"] = cb
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec.to_csv(out / "loao_records.csv", index=False)
    gt.to_csv(out / "loao_ref_instances.csv", index=False)
    pr.to_csv(out / "loao_rater_instances.csv", index=False)
    print(f"loao: {len(rec)} record rows, {rec.stem.nunique()} files")


def loao_summary(out, n_boot):
    import beyond_human as BH
    dt = {"prefix_A": str}
    rec = pd.read_csv(out / "loao_records.csv", dtype=dt)
    gt = pd.read_csv(out / "loao_ref_instances.csv", dtype=dt)
    pr = pd.read_csv(out / "loao_rater_instances.csv", dtype=dt)
    faint = json.load(open(ROOT / "outputs/beyond_human/summary.json"))["checks"]["faint_threshold_contrast_snr_q20"]
    asp = {k: BH.all_aspects(rec[rec.rater == k], gt[gt.rater == k], pr[pr.rater == k], faint)
           for k in ("human",) + LOAO_RATERS}
    res = dict(n_files=int(rec.stem.nunique()), n_records=int((rec.rater == "human").sum()))
    res["vs_human"] = {k: BH.paired_boot(rec[rec.rater == "human"], rec[rec.rater == k], n_boot,
                                         stats=("PQ", "SQ", "RQ", "thin_recall", "thin_precision")) for k in LOAO_RATERS}
    res["vs_B_ens"] = {k: BH.paired_boot(rec[rec.rater == BASE], rec[rec.rater == k], n_boot,
                                         stats=("PQ", "SQ", "RQ", "thin_recall", "thin_precision"))
                       for k in LOAO_RATERS if k != BASE}
    res["aspects"], newly_below, undetermined = {}, [], []
    def val(k, s):
        v = asp[k].get(s)
        return np.nan if v is None else float(v)

    for s, d in BH.ASPECTS.items():
        h = val("human", s)
        row = dict(human=h, **{k: val(k, s) for k in LOAO_RATERS})
        nb_base = BH.not_worse(val(BASE, s), h, d)
        nb_main = BH.not_worse(val(MAIN, s), h, d)
        row.update(B_ens_not_worse_than_human=nb_base, R2nb0_not_worse_than_human=nb_main)
        if nb_base is True and nb_main is False:
            newly_below.append(s)
        if nb_base is True and nb_main is None:
            undetermined.append(s)
        res["aspects"][s] = row
    res["newly_below_human"] = newly_below
    res["undetermined_under_R2nb0"] = undetermined
    res["stems"] = sorted(rec.stem.unique())
    return res


def compare_block(df, a_name, b_name, sp, review_stems, folds):
    RC.FOLDS = tuple(folds)                     # RC.compare reports per_fold over RC.FOLDS
    r, d = RC.compare(df, a_name, b_name, sp, review_stems)
    r["review_set"] = r.pop("review_set_f01")
    return r, d


def cmd_analyse(a):
    out = Path(a.out)
    df = pd.read_csv(out / "records.csv")
    sp = RC.split_table()
    folds = sorted(df.fold.unique().tolist())
    rc = pd.read_csv(ROOT / "outputs/audit/review_cases.csv")
    review_stems = set(rc.file_name.str[:-5]) & set(df.stem)
    res = dict(folds=folds, n_files=int(df.stem.nunique()), n_records=int((df.method == BASE).sum()),
               n_time_groups=int(df.group_id.nunique()), n_review_files=len(review_stems), n_boot=RC.N_BOOT,
               design="nested (D-035): outer-train stage 1 = inner OOF, outer-val stage 1 = mean of 4 inner models")
    res["methods"] = {m: RC.summary_counts(df[df.method == m]) for m in METHODS}
    main, dmain = compare_block(df, BASE, MAIN, sp, review_stems, folds)
    res["R2nb0_vs_B_ens"] = main
    res["B_ens_vs_b0_single"] = compare_block(df, "b0_single", BASE, sp, review_stems, folds)[0]
    res["R2nb0_vs_B_ens_a300"] = compare_block(df, "B_ens_a300", MAIN, sp, review_stems, folds)[0]
    res["R2nb0_vs_R2nfull"] = compare_block(df, "R2nfull", MAIN, sp, review_stems, folds)[0]
    conf = [f for f in folds if f in CONFIRM_FOLDS]
    if conf and len(conf) < len(folds):
        dsub = df[df.fold.isin(conf)]
        res["R2nb0_vs_B_ens_folds234"] = compare_block(dsub, BASE, MAIN, sp, review_stems & set(dsub.stem), conf)[0]
    loao = None
    if (out / "loao_records.csv").exists():
        loao = loao_summary(out, a.n_boot)
        res["loao"] = loao
    A, B = main["A"], main["B"]
    rule = dict(pq_diff_ge_0=main["file_boot"]["PQ"]["diff"] >= 0,
                thin_recall_higher=B["thin_recall"] > A["thin_recall"],
                thin_precision_higher=B["thin_precision"] > A["thin_precision"],
                sign_test=main["per_file_sign"]["better"] > main["per_file_sign"]["worse"]
                and main["per_file_sign"]["sign_test_p"] < 0.05,
                review_set_not_lower=main["review_set"]["B"] >= main["review_set"]["A"],
                no_loao_aspect_newly_below_human=(not loao["newly_below_human"]) if loao else None)
    rule["all_5_folds_present"] = folds == [0, 1, 2, 3, 4]
    want = set()
    for k in folds:
        want |= {f[:-5] for f in (EXPS / a.ens_exp / f"fold{k}" / "val_files.txt").read_text().split()}
    rule["table_covers_all_val_files"] = set(df.stem) == want
    multi = set(sp[(sp.n_sets >= 2) & sp.stem.isin(want)].stem)
    rule["loao_covers_all_multiset_files"] = bool(loao) and set(loao["stems"]) == multi
    rule["no_loao_aspect_undetermined"] = bool(loao) and not loao["undetermined_under_R2nb0"]
    vf = [Path(a.verify_dir) / f"verify_o{k}.json" for k in range(5)]
    vs = [json.load(open(f)) if f.exists() else {} for f in vf]
    rule["verify_all_ok_incl_r2n"] = all(v.get("all_ok") is True and "E_r2_cache" in v for v in vs)
    rule["ADOPT"] = bool(all(v is True for v in rule.values()))
    res["rule"] = rule
    dmain.rename("dPQ_file").to_frame().join(sp.set_index("stem")[["fold", "group_id", "n_sets"]]).sort_values(
        "dPQ_file").to_csv(out / "per_file_diff_R2nb0_vs_B_ens.csv")
    json.dump(res, open(out / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(dict(methods={m: {k: res["methods"][m][k] for k in ("PQ", "TP", "FP", "FN", "thin_recall",
                                                                         "thin_precision")} for m in METHODS},
                          main={k: main[k] for k in ("file_boot", "per_fold", "per_file_sign", "review_set")},
                          rule=rule), indent=1, default=str))


def main():
    a = parse()
    assert a.workers <= 16
    {"table": cmd_table, "loao": cmd_loao, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
