"""Candidate-detection recall of stage 1 (b0 OOF probabilities): which GT instances could a crop-refinement
stage even see? A refinement model that works on windows around candidates cannot recover a GT instance that no
candidate touches.

For candidate settings (thr_c, min_area_c) and every GT instance of every annotation set (all 5 folds):
  best_iou   = max IoU with a candidate component
  coverage   = |GT ∩ union(candidates)| / |GT|
  pmax       = max stage-1 probability inside the GT
  class:  hit (best IoU > 0.5) | partial (coverage >= 0.2) | weak (0 < coverage < 0.2) | missed (coverage == 0)
Strata: GT area bins and contrast quintiles (outputs/audit/instances.csv, joined by record id + instance index).
Usage: .venv/bin/python scripts/candidate_recall.py [--exp b0_r101_s1024] [--probs val_probs_last.npz]
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from filseg import seg  # noqa: E402
from filseg.metrics import InstanceSet, intersection_matrix, single_thread_worker  # noqa: E402

SETTINGS = [(0.5, 500), (0.5, 0), (0.3, 50), (0.3, 0), (0.2, 50), (0.1, 50), (0.1, 0)]


def job(args):
    stem, prob, gts = args
    prob = prob.astype(np.float32)
    rows = []
    cand = {}
    for thr, ma in SETTINGS:
        cand[(thr, ma)] = seg.prob_to_instances(prob, seg.PostCfg(thr, ma))
    for rid, G in gts.items():
        for i in range(len(G)):
            y0, x0, y1, x1 = G.boxes[i]
            c = G.crops[i]
            pmax = float(prob[y0:y1, x0:x1][c].max()) if c.any() else 0.0
            row = dict(rid=rid, stem=stem, idx=i, area=int(G.areas[i]), pmax=pmax)
            for (thr, ma), C in cand.items():
                if len(C) == 0:
                    bi, cov, nt = 0.0, 0.0, 0
                else:
                    gi = InstanceSet._build(G.shape, [G.boxes[i]], [c])
                    inter = intersection_matrix(gi, C).astype(np.float64)[0]
                    iou = inter / np.maximum(G.areas[i] + C.areas - inter, 1)
                    bi = float(iou.max())
                    nt = int((inter > 0).sum())
                    # union coverage (candidates are disjoint, so sum of intersections)
                    cov = float(inter.sum() / max(G.areas[i], 1))
                k = f"t{thr}_a{ma}"
                row[f"{k}_best_iou"] = bi
                row[f"{k}_cov"] = cov
                row[f"{k}_ntouch"] = nt
            rows.append(row)
    ncand = {f"t{t}_a{m}": len(C) for (t, m), C in cand.items()}
    return rows, dict(stem=stem, **ncand)


def classify(bi, cov):
    return np.select([bi > 0.5, cov >= 0.2, cov > 0], ["hit", "partial", "weak"], "missed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="b0_r101_s1024")
    ap.add_argument("--probs", default="val_probs_last.npz")
    a = ap.parse_args()
    import multiprocessing as mp
    single_thread_worker()
    base = ROOT / "outputs" / "exps" / a.exp
    jobs = []
    for f in range(5):
        files = (base / f"fold{f}" / "val_files.txt").read_text().split()
        gts = seg.gt_records(files)
        by = {}
        for rid, g in gts.items():
            by.setdefault(rid.split("-", 1)[1], {})[rid] = g
        z = np.load(base / f"fold{f}" / a.probs)
        jobs += [(s, z[s], by[s]) for s in sorted(by)]
    with mp.get_context("forkserver").Pool(12, initializer=single_thread_worker) as pool:
        res = pool.map(job, jobs, chunksize=2)
    df = pd.DataFrame([r for rs, _ in res for r in rs])
    nc = pd.DataFrame([c for _, c in res])
    inst = pd.read_csv(ROOT / "outputs" / "audit" / "instances.csv")
    key_cols = [c for c in ("image_id", "ann_idx") if c in inst.columns]
    if len(key_cols) == 2 and "contrast" in inst.columns:
        m = inst[key_cols + ["contrast"]].rename(columns={"image_id": "rid", "ann_idx": "idx"})
        df = df.merge(m, on=["rid", "idx"], how="left")
    df["area_bin"] = pd.cut(df.area, [0, 250, 500, 1000, 2000, 1e9], right=False,
                            labels=["<250", "250-500", "500-1k", "1k-2k", ">=2k"])
    if "contrast" in df:
        df["contrast_q"] = pd.qcut(df.contrast, 5, labels=["q1 faint", "q2", "q3", "q4", "q5 strong"])
    out = dict(n_gt=int(len(df)), settings={}, n_candidates_per_image={})
    for thr, ma in SETTINGS:
        k = f"t{thr}_a{ma}"
        df[f"{k}_class"] = classify(df[f"{k}_best_iou"].values, df[f"{k}_cov"].values)
        vc = df[f"{k}_class"].value_counts(normalize=True)
        out["settings"][k] = {c: round(float(vc.get(c, 0.0)), 4) for c in ("hit", "partial", "weak", "missed")}
        out["n_candidates_per_image"][k] = dict(mean=float(nc[k].mean()), median=float(nc[k].median()))
    out["stage1_silent_pmax_lt_0.1"] = float((df.pmax < 0.1).mean())
    out["stage1_pmax_lt_0.3"] = float((df.pmax < 0.3).mean())
    strata = {}
    for col in ("area_bin", "contrast_q"):
        if col not in df:
            continue
        g = df.groupby(col, observed=True)
        strata[col] = {str(k): {"n": int(len(d)),
                                "missed@t0.3a0": round(float((d["t0.3_a0_class"] == "missed").mean()), 3),
                                "missed@t0.1a0": round(float((d["t0.1_a0_class"] == "missed").mean()), 3),
                                "hit@t0.5a500": round(float((d["t0.5_a500_class"] == "hit").mean()), 3),
                                "hit@t0.3a0": round(float((d["t0.3_a0_class"] == "hit").mean()), 3),
                                "pmax<0.1": round(float((d.pmax < 0.1).mean()), 3)} for k, d in g}
    out["strata"] = strata
    dst = ROOT / "outputs" / "candidate_recall"
    dst.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst / "gt_instances.csv", index=False)
    json.dump(out, open(dst / "summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
