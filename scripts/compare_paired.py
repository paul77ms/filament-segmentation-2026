"""Paired comparison of two prediction methods on the same OOF records (same split, same GT).

Each method = (exp, grouping, thr, min_area). Per record counts (TP, FP, FN, sum IoU) come from
filseg.metrics.evaluate_image; the pooled PQ difference is bootstrapped over FILES (all annotation sets of
a file move together; seed 2026). Files in the same 14-day group are correlated, so the CI is somewhat
optimistic; a per-fold breakdown is also printed.

Usage:
  .venv/bin/python scripts/compare_paired.py --folds 1 \
      --a b0_r101_s1024:cc:0.5:300 --b b0_perset:cc:0.5:500 [--probs val_probs_last.npz]
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from filseg import seg  # noqa: E402
from filseg.metrics import EvalConfig, evaluate_image, single_thread_worker  # noqa: E402

CFG = EvalConfig(compute_boundary=False, compute_pair_boundary=False, compute_thin=True)


def parse_method(s):
    """exp:group:thr:min_area[:seed_thr[:probs_file]]  (seed_thr 'none' = no hysteresis)."""
    f = s.split(":")
    exp, group, thr, ma = f[:4]
    seed = float(f[4]) if len(f) > 4 and f[4].lower() != "none" else None
    probs = f[5] if len(f) > 5 else None
    return dict(exp=exp, group=group, thr=float(thr), min_area=int(ma), seed=seed, probs=probs, name=s)


_RZ = {}


def _refine_instances(meth, stem, fold):
    """R1/R2: assemble per-candidate refined probabilities (val_cand_probs.npz) into instances."""
    from filseg import refine as RF
    key = (meth["exp"], fold)
    if key not in _RZ:
        z = np.load(ROOT / "outputs" / "exps" / meth["exp"] / f"fold{fold}" / "val_cand_probs.npz")
        by = {}
        for k in z.files:
            st, j, y0, x0 = k.split("|")
            by.setdefault(st, []).append((int(j), int(y0), int(x0), k))
        _RZ.clear()
        _RZ[key] = (z, by)
    z, by = _RZ[key]
    cp = {j: (y0, x0, z[k].astype(np.float32)) for j, y0, x0, k in by.get(stem, [])}
    if meth.get("b0_exist"):  # keep only candidates overlapping a stage-1 b0 instance (b0 decides existence)
        # candidate cache and existence source of this refine run (refine_v1 + b0 OOF for runs before D-035;
        # nested runs: refine_nested_o<k> + b0n_ens = mean of the 4 inner models)
        rc = json.load(open(ROOT / "outputs" / "exps" / meth["exp"] / f"fold{fold}" / "config.json"))
        cache = Path(rc.get("cache_dir", RF.CACHE))
        bz = np.load(ROOT / "outputs" / "exps" / rc.get("exist_exp", "b0_r101_s1024") / f"fold{fold}" / "val_probs_last.npz")
        b0m = seg.prob_to_instances(bz[stem].astype(np.float32), seg.PostCfg(0.5, 500)).union()
        cm = np.load(cache / "cand.npy", mmap_mode="r")
        idx = json.load(open(cache / "index.json"))
        cmap = np.asarray(cm[idx["files"].index(stem + ".jpeg")])
        keep = set(np.unique(cmap[b0m & (cmap > 0)]).tolist())
        cp = {j: v for j, v in cp.items() if j in keep}
    return RF.assemble(cp, thr=meth["thr"], min_area=meth["min_area"])


def job(args):
    stem, fold, meth, probs_name, gts = args
    if meth["group"] in ("refine", "refineb0"):
        meth = dict(meth, b0_exist=meth["group"] == "refineb0")
        P = _refine_instances(meth, stem, fold)
        return _score(stem, fold, P, gts)
    z = np.load(ROOT / "outputs" / "exps" / meth["exp"] / f"fold{fold}" / (meth.get("probs") or probs_name))
    p = z[stem].astype(np.float32)
    P = seg.prob_to_instances(p, seg.PostCfg(meth["thr"], meth["min_area"], seed_thr=meth.get("seed")))
    return _score(stem, fold, P, gts)


def _score(stem, fold, P, gts):
    rows = []
    for rid, G in gts.items():
        r, _, _ = evaluate_image(G, P, CFG, image_key=rid, return_instances=False)
        rows.append(dict(rid=rid, stem=stem, fold=fold, tp=r["tp_host"], fp=r["fp_host"], fn=r["fn_host"],
                         si=r["sum_iou_host"], fn_split=r["fn_split"], fp_fragment=r["fp_fragment"],
                         fn_merged=r["fn_merged"], fp_merge=r["fp_merge"], fn_missed=r["fn_missed"],
                         fp_spurious=r["fp_spurious"], thin_gt=r.get("thin_gt_px", 0),
                         thin_cov=r.get("thin_gt_covered_px", 0), thin_pred=r.get("thin_pred_px", 0),
                         thin_pin=r.get("thin_pred_in_gt_px", 0)))
    return rows


def pooled(df):
    s = df[["tp", "fp", "fn", "si"]].sum()
    den = s.tp + 0.5 * s.fp + 0.5 * s.fn
    return s.si / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="1")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--probs", default="val_probs_last.npz")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    import multiprocessing as mp
    single_thread_worker()
    ma, mb = parse_method(a.a), parse_method(a.b)
    jobs = {"a": [], "b": []}
    for f in map(int, a.folds.split(",")):
        files = (ROOT / "outputs" / "exps" / ma["exp"] / f"fold{f}" / "val_files.txt").read_text().split()
        files_b = (ROOT / "outputs" / "exps" / mb["exp"] / f"fold{f}" / "val_files.txt").read_text().split()
        assert files == files_b, "different validation files"
        gts = seg.gt_records(files)
        by = {}
        for rid, g in gts.items():
            by.setdefault(rid.split("-", 1)[1], {})[rid] = g
        for key, m in (("a", ma), ("b", mb)):
            jobs[key] += [(s, f, m, a.probs, by[s]) for s in sorted(by)]
    with mp.get_context("forkserver").Pool(12, initializer=single_thread_worker) as pool:
        dfa = pd.DataFrame([r for rs in pool.map(job, jobs["a"], chunksize=2) for r in rs])
        dfb = pd.DataFrame([r for rs in pool.map(job, jobs["b"], chunksize=2) for r in rs])
    dfa, dfb = dfa.set_index("rid").sort_index(), dfb.set_index("rid").sort_index()
    assert (dfa.index == dfb.index).all()
    stems = dfa.stem.unique()
    rng = np.random.default_rng(2026)
    grp_a = {s: d for s, d in dfa.groupby("stem")}
    grp_b = {s: d for s, d in dfb.groupby("stem")}
    sums_a = np.array([grp_a[s][["tp", "fp", "fn", "si"]].sum().values for s in stems])
    sums_b = np.array([grp_b[s][["tp", "fp", "fn", "si"]].sum().values for s in stems])

    def pq(m):
        den = m[0] + 0.5 * m[1] + 0.5 * m[2]
        return m[3] / den

    diffs = []
    for _ in range(a.n_boot):
        idx = rng.integers(len(stems), size=len(stems))
        diffs.append(pq(sums_b[idx].sum(0)) - pq(sums_a[idx].sum(0)))
    diffs = np.array(diffs)

    def summary(df):
        s = df.sum(numeric_only=True)
        return dict(PQ=pooled(df), TP=int(s.tp), FP=int(s.fp), FN=int(s.fn),
                    fn_split=int(s.fn_split), fp_fragment=int(s.fp_fragment), fn_merged=int(s.fn_merged),
                    fp_merge=int(s.fp_merge), fn_missed=int(s.fn_missed), fp_spurious=int(s.fp_spurious),
                    thin_region_recall=float(s.thin_cov / s.thin_gt) if s.thin_gt else None,
                    thin_region_precision=float(s.thin_pin / s.thin_pred) if s.thin_pred else None)

    out = dict(a=ma["name"], b=mb["name"], folds=a.folds, probs=a.probs, n_files=int(len(stems)), n_records=int(len(dfa)),
               A=summary(dfa), B=summary(dfb), diff_B_minus_A=float(pooled(dfb) - pooled(dfa)),
               boot_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
               per_fold={str(f): dict(A=pooled(dfa[dfa.fold == f]), B=pooled(dfb[dfb.fold == f]))
                         for f in sorted(dfa.fold.unique())})
    dst = ROOT / "outputs" / "comparisons"
    dst.mkdir(parents=True, exist_ok=True)
    name = (f"{ma['exp']}-{ma['group']}_vs_{mb['exp']}-{mb['group']}_f{a.folds.replace(',', '')}_t{mb['thr']}_a{mb['min_area']}"
            f"_s{mb.get('seed')}_{(mb.get('probs') or a.probs).replace('.npz', '')}.json")
    json.dump(out, open(dst / name, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
