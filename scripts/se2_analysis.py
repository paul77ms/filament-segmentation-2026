"""SE2 follow-up analysis (D-088): what the added predictions are, and whether a stricter selection keeps a gain.
CPU only. Existing SE2 models and outputs are not modified.

  classify  every instance SE2 adds in the final pipeline (score > m_k x tau_k, a_eval.SE2_MULT): matched (IoU > 0.5 with some
            annotation set) or, if unmatched, one exclusive category (precedence top-down):
              adjacent    the added mask, dilated by 3 px, touches an existing base prediction (duplicate/extension)
              fragment    overlaps a GT filament that another prediction already matches (IoU > 0.5): piece of an
                          existing object
              shape       best IoU with an annotated filament in [0.1, 0.5]: real structure, mask shape insufficient
              ambiguous   overlaps an annotated filament with IoU < 0.1, or lies within 10 px of one
              background  no annotated filament within 10 px
            -> outputs/research/se2_analysis/added.csv
  sweep     threshold multipliers m x tau_k on every outer fold -> V0, V1, recovered GT, added, unmatched
            -> sweep_records.csv, sweep_images.csv, sweep.csv
  select    the threshold multiplier chosen by V1 (annotation sets of an image pooled, the leaderboard-type scorer)
            from the sweep: per outer fold k, m_k = argmax over V1_GRID of V1 on the other 4 folds; the nested
            result (m_k on fold k) is compared with the pipeline without a small-fragment stage and with the SE2
            rule's reference; test time: m = argmax of V1 over all 5 folds -> threshold_v1.json,
            se2_vs_no_small_stage.json (needs scripts/a_eval.py loao for the LOAO items)
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs" / "research" / "se2_analysis"
MULTS = (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
V1_GRID = (0.6, 0.8, 1.0, 1.25, 1.5)       # registered grid of the V1 choice
TAU_ALL = 0.2124                           # PQ/2 of the base over all training folds (records/se2/tau_k.json)
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))      # 3 px dilation
K10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))   # 10 px


def _ctx(stem, fold):
    import se2 as S2
    import tune_nested as TN
    S2._lazy_cand(fold)
    ctx = TN.FileCtx(stem, fold)
    base = ctx.instances(dict(TN.DEFAULT, se_mult=1e9))
    R, _ = S2.fragments(ctx.p, ctx.cmap, ctx.cp)
    return ctx, base, R


_P = {}


def _scores(stem, n):
    if not _P:
        P = pd.read_csv(ROOT / "outputs" / "research" / "se2" / "preds_outer.csv")
        _P.update({(s, int(i)): float(v) for s, i, v in zip(P.stem, P.idx, P.score)})
    return np.array([_P[(stem, q)] for q in range(n)])


def classify_job(args):
    import a_eval as AE
    import small_exist as SE
    from filseg.scorers import iou_matrix
    stem, fold, gts = args
    ctx, base, R = _ctx(stem, fold)
    if R is None:
        return []
    sc = _scores(stem, len(R))
    Q, added = SE.add_only(base, R, np.flatnonzero(sc > AE.SE2_MULT[fold] * ctx.tau))
    if not added:
        return []
    sets = [gts[r] for r in sorted(gts)]
    nb = len(base)
    bunion = base.union() if nb else np.zeros(Q.shape, bool)
    rows = []
    # GT instances matched (IoU > 0.5) by a prediction other than the added one
    for qi, ridx in enumerate(added):
        j = nb + qi                                    # index of the added instance in Q (appended after the base)
        y0, x0, y1, x1 = (int(v) for v in Q.boxes[j])
        m = np.zeros(Q.shape, np.uint8); m[y0:y1, x0:x1] = Q.crops[j]
        best, best_any, frag, near = 0.0, 0.0, False, False
        for G in sets:
            if not len(G):
                continue
            inter, iou = iou_matrix(G, Q)
            best = max(best, float(iou[:, j].max()))
            for g in np.flatnonzero(inter[:, j] > 0):
                others = np.delete(iou[g], j)
                if len(others) and others.max() > 0.5:
                    frag = True
            gu = G.union()
            if (cv2.dilate(m, K10) > 0)[gu].any():
                near = True
        if best > 0.5:
            cat = "matched"
        elif nb and (cv2.dilate(m, K3) > 0)[bunion].any():
            cat = "adjacent"
        elif frag:
            cat = "fragment"
        elif best >= 0.1:
            cat = "shape"
        elif best > 0 or near:
            cat = "ambiguous"
        else:
            cat = "background"
        rows.append(dict(stem=stem, fold=fold, idx=int(ridx), area=int(Q.areas[j]), score=float(sc[ridx]), tau=float(ctx.tau),
                         best_iou=best, category=cat, n_sets=len(sets)))
    return rows


def sweep_job(args):
    import a_eval as AE
    import small_exist as SE
    stem, fold, gts = args
    ctx, base, R = _ctx(stem, fold)
    sets = [gts[r] for r in sorted(gts)]
    sc = _scores(stem, len(R)) if R is not None else np.zeros(0)
    recs, imgs = [], []
    variants = [("base", base)] + [(f"m{m}", SE.add_only(base, R, np.flatnonzero(sc > m * ctx.tau))[0] if R is not None else base)
                                   for m in MULTS]
    for name, Q in variants:
        rec, img, _ = AE._eval(name, Q, gts, stem, fold)
        img["no_set_fp"] = SE.no_set_fp(Q, sets); img["n_pred"] = len(Q)
        recs += rec; imgs.append(img)
    return recs, imgs


def _pool(fn, jobs, workers):
    import multiprocessing as mp
    from filseg.metrics import single_thread_worker
    single_thread_worker()
    with mp.get_context("forkserver").Pool(workers, initializer=single_thread_worker) as pool:
        return list(pool.imap_unordered(fn, jobs, chunksize=1))


def cmd_classify(a):
    import a_eval as AE
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rows = [r for out in _pool(classify_job, AE.all_jobs()[:a.limit or None], a.workers) for r in out]
    d = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); d.to_csv(OUT / "added.csv", index=False)
    print(d.category.value_counts().to_string())


def _summ(df, I, name, base_missed):
    import tune_nested as TN
    r, i = df[df.method == name], I[I.method == name]
    return dict(V0=TN.v0(r), V1=TN.v1(i), recovered=int(base_missed - r.fn_missed.sum()), n_pred=int(i.n_pred.sum()),
                no_set_fp=int(i.no_set_fp.sum()), fp=int(r.fp_host.sum()), tp=int(r.tp_host.sum()),
                n_to_1=int(r.nb_pred_Nto1.sum()), one_to_n=int(r.nb_gt_1toN.sum()))


def cmd_sweep(a):
    import a_eval as AE
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    out = _pool(sweep_job, AE.all_jobs()[:a.limit or None], a.workers)
    df = pd.DataFrame([r for o in out for r in o[0]]); I = pd.DataFrame([r for o in out for r in o[1]])
    df.to_csv(OUT / "sweep_records.csv", index=False); I.to_csv(OUT / "sweep_images.csv", index=False)
    bm = int(df[df.method == "base"].fn_missed.sum())
    rows = [dict(variant=n, **_summ(df, I, n, bm)) for n in ["base"] + [f"m{m}" for m in MULTS]]
    t = pd.DataFrame(rows); t.to_csv(OUT / "sweep.csv", index=False)
    print(t.round(4).to_string(index=False))


def _site_era(stem):
    y = int(stem[:4])
    return stem[14], ("2011-13" if y <= 2013 else "2014-16" if y <= 2016 else "2017-19" if y <= 2019 else "2020-22")


def cmd_select(a):
    import eval_dual as ED
    import refine_b0exist_checks as RC
    import tune_nested as TN
    df = pd.read_csv(OUT / "sweep_records.csv"); I = pd.read_csv(OUT / "sweep_images.csv")
    name = {m: f"m{m}" for m in V1_GRID}
    best = lambda X: max(V1_GRID, key=lambda m: (TN.v1(X[X.method == name[m]]), -abs(m - 1.0)))  # noqa: E731
    chosen = {k: best(I[I.fold != k]) for k in range(5)}
    m_test = best(I)
    keep = lambda D: pd.concat([D[(D.fold == k) & (D.method == name[m])] for k, m in chosen.items()]).assign(method="final")  # noqa: E731
    sp = RC.split_table(); RC.FOLDS = (0, 1, 2, 3, 4)
    rc = pd.read_csv(ROOT / "outputs/audit/review_cases.csv"); review = set(rc.file_name.str[:-5])
    cur = ROOT / "outputs" / "research" / "se2"
    c = pd.read_csv(cur / "records.csv"); ci = pd.read_csv(cur / "images_v1.csv")
    df2 = pd.concat([df[df.method == "base"], keep(df), c[c.method == "current"]])
    I2 = pd.concat([I[I.method == "base"], keep(I), ci[ci.method == "current"]])
    df2["group_id"] = df2.stem.map(sp.set_index("stem").group_id); df2["n_sets"] = df2.stem.map(sp.set_index("stem").n_sets)
    bm = int(df[df.method == "base"].fn_missed.sum())
    grid = [dict(variant=n, **_summ(df, I, n, bm)) for n in ["base"] + [name[m] for m in V1_GRID]]
    out = dict(chosen_by_fold=chosen, m_test=m_test, threshold_test=round(m_test * TAU_ALL, 5),
               grid_all_folds=grid, nested={n: _summ(df2, I2, n, bm) for n in ("base", "final", "current")})
    rv = {n: TN.v0(df2[(df2.method == n) & df2.stem.isin(review)]) for n in ("base", "final", "current")}
    for ref in ("base", "current"):
        v0c, _ = RC.compare(df2, ref, "final", sp, review)
        v1c = ED.v1_boot(I2, ref, "final")
        out[f"vs_{ref}"] = dict(V0=dict(diff=v0c["file_boot"]["PQ"]["diff"], ci95=v0c["file_boot"]["PQ"]["ci95"],
                                        group_ci95=v0c["group_boot"]["PQ"]["ci95"],
                                        per_fold={k: v["diff"] for k, v in v0c["per_fold"].items()}),
                                V1={k: v1c[k] for k in ("diff", "ci95", "per_fold", "folds_positive")},
                                review_V0=dict(ref=rv[ref], final=rv["final"]))
    # robustness against the pipeline without a small-fragment stage: by GONG site and by period
    se = {s: _site_era(s) for s in I2.stem.unique()}
    for key, j in (("by_site", 0), ("by_era", 1)):
        out[key] = {}
        for g in sorted({v[j] for v in se.values()}):
            st = {s for s, v in se.items() if v[j] == g}
            r, i = df2[df2.stem.isin(st)], I2[I2.stem.isin(st)]
            out[key][g] = dict(n=len(st), V0=round(TN.v0(r[r.method == "final"]) - TN.v0(r[r.method == "base"]), 4),
                               V1=round(TN.v1(i[i.method == "final"]) - TN.v1(i[i.method == "base"]), 4))
    # the registered SE2 rule (scripts/se2.py analyse) for the nested result vs its reference 'current'
    newly, small = None, None
    if (ROOT / "outputs" / "research" / "final" / "loao" / "loao_records.csv").exists():
        import beyond_human as BH
        ED.OUT, ED.LOAO_RATERS = cur / "loao", ("current",)
        a_cur, _ = ED.loao_aspects()
        ED.OUT, ED.LOAO_RATERS = ROOT / "outputs" / "research" / "final" / "loao", ("final",)
        a_fin, _ = ED.loao_aspects()
        f = lambda z: np.nan if z is None else float(z)  # noqa: E731
        newly = [s for s, d in BH.ASPECTS.items() if BH.not_worse(f(a_cur["current"].get(s)), f(a_cur["human"].get(s)), d) is True
                 and BH.not_worse(f(a_fin["final"].get(s)), f(a_fin["human"].get(s)), d) is False]
        small = dict(human=a_fin["human"].get("small_detected_rate"), current=a_cur["current"].get("small_detected_rate"),
                     final=a_fin["final"].get("small_detected_rate"))
    n, cu = out["nested"]["final"], out["nested"]["current"]
    v0c = out["vs_current"]["V0"]
    out["loao_small_detection"] = small; out["loao_newly_below_human"] = newly
    out["se2_rule_vs_current"] = dict(
        E1=bool(v0c["ci95"][0] > -0.002 and sum(v > 0 for v in v0c["per_fold"].values()) >= 3),
        E2=bool(out["vs_current"]["V1"]["diff"] >= 0),
        E3=bool((n["no_set_fp"] - cu["no_set_fp"]) <= 0.10 * cu["no_set_fp"]),
        E4=None if newly is None else bool(not newly and small["final"] > small["current"]),
        E5=bool(rv["final"] >= rv["current"]),
        E6=bool((n["n_to_1"] - cu["n_to_1"]) <= 0.10 * cu["n_to_1"]))
    dflt = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
    json.dump(out, open(OUT / "threshold_v1.json", "w"), indent=1, default=dflt)
    b = out["nested"]["base"]
    json.dump(dict(what="final pipeline (base + SE2 at m_k x tau_k) vs the same pipeline without any small-fragment stage "
                        "(base = R2 + assembly + cleanup), nested OOF, 707 files; file bootstrap 2000",
                   V0=dict(A=b["V0"], B=n["V0"], **out["vs_base"]["V0"]), V1=dict(A=b["V1"], B=n["V1"], **out["vs_base"]["V1"]),
                   counts=dict(base={k: b[k] for k in ("n_pred", "no_set_fp", "n_to_1", "one_to_n")} | dict(review_V0=rv["base"]),
                               final={k: n[k] for k in ("n_pred", "no_set_fp", "n_to_1", "one_to_n", "recovered")} | dict(review_V0=rv["final"])),
                   loao_small_detection=small, by_site=out["by_site"], by_era=out["by_era"]),
              open(OUT / "se2_vs_no_small_stage.json", "w"), indent=1, default=dflt)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["classify", "sweep", "select"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"classify": cmd_classify, "sweep": cmd_sweep, "select": cmd_select}[a.cmd](a)
