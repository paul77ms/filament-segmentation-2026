"""Post-processing selection on OOF probabilities with NESTED cross-validation.

For each held-out fold f, the post-processing config (extent thr, hysteresis seed_thr, min_area)
is chosen by pooled PQ_host on the OTHER folds' OOF records and then applied to fold f. The pooled
result over all held-out folds is the honest estimate ("nested"). The single config that is best on
all folds jointly is also reported, labelled OPTIMISTIC (selected and evaluated on the same data).

Speed: per (image, thr) connected components + per-record intersection matrix are computed once;
each (seed_thr, min_area) filter is a column subset. Predictions are disjoint components, so
unique and all-pairs (host) matching coincide except where GT overlaps itself (5/1154 records);
host-style counts are used here and the chosen config is re-scored with the full metric module.

Usage: .venv/bin/python scripts/tune_post.py --exp b0_r101_s1024 --folds 0,1,2,3,4 [--probs_tag best]
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

from filseg import seg  # noqa: E402
from filseg.metrics import InstanceSet, intersection_matrix, single_thread_worker  # noqa: E402

THRS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
SEEDS = (None, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_AREAS = (0, 50, 100, 200, 300, 500, 800)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--probs", default="val_probs_best.npz", help="npz file name inside each fold dir")
    p.add_argument("--workers", type=int, default=12)
    return p.parse_args()


def _image_job(args):
    """For one image: for each thr, components (area, max prob) and per-record intersection/IoU data."""
    stem, prob, gts, pterr, t_terr, guard = args
    prob = prob.astype(np.float32)
    out = {}
    for thr in THRS:
        n, lab, stats = cv2.connectedComponentsWithStats((prob > thr).astype(np.uint8), connectivity=8)[:3]
        boxes, crops, areas, pmax = [], [], [], []
        for k in range(1, n):
            x, y, w, h, a = stats[k]
            c = lab[y:y + h, x:x + w] == k
            boxes.append((y, x, y + h, x + w)); crops.append(c); areas.append(int(a))
            pmax.append(float(prob[y:y + h, x:x + w][c].max()))
        P = InstanceSet._build(prob.shape, boxes, crops)
        recs = {}
        for rid, G in gts.items():
            inter = intersection_matrix(G, P).astype(np.float64)
            uni = G.areas[:, None] + P.areas[None, :] - inter
            iou = np.where(uni > 0, inter / np.maximum(uni, 1), 0.0)
            recs[rid] = iou.astype(np.float32)
        out[thr] = dict(area=np.array(areas), pmax=np.array(pmax), iou=recs)
    return stem, out


def score(data, rids_by_stem, stems, thr, seed, ma):
    tp = fp = fn = 0
    si = 0.0
    for s in stems:
        d = data[s][thr]
        keep = d["area"] >= max(ma, 1)
        if seed is not None:
            keep &= d["pmax"] > seed
        for rid in rids_by_stem[s]:
            iou = d["iou"][rid][:, keep]
            cand = iou > 0.5
            tp += int(cand.sum()); si += float(iou[cand].sum())
            fp += int((~cand.any(0)).sum()); fn += int((~cand.any(1)).sum())
    den = tp + 0.5 * fp + 0.5 * fn
    return dict(PQ=si / den if den else float("nan"), SQ=si / tp if tp else float("nan"),
                RQ=tp / den if den else float("nan"), TP=tp, FP=fp, FN=fn)


def main():
    a = parse()
    import multiprocessing as mp
    folds = [int(f) for f in a.folds.split(",")]
    base = ROOT / "outputs" / "exps" / a.exp
    stems_by_fold, jobs = {}, []
    all_files = []
    for f in folds:
        files = (base / f"fold{f}" / "val_files.txt").read_text().split()
        all_files += files
        stems_by_fold[f] = [Path(x).stem for x in files]
    gts = seg.gt_records(all_files)
    rids_by_stem = {}
    for rid in gts:
        rids_by_stem.setdefault(rid.split("-", 1)[1], []).append(rid)
    for f in folds:
        z = np.load(base / f"fold{f}" / a.probs)
        for s in stems_by_fold[f]:
            jobs.append((s, z[s], {r: gts[r] for r in rids_by_stem[s]}, None, 0.5, 0))
    data = {}
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for s, d in pool.imap_unordered(_image_job, jobs, chunksize=2):
            data[s] = d
    configs = [(t, sd, ma) for t in THRS for sd in SEEDS if sd is None or sd > t for ma in MIN_AREAS]
    per_fold = {f: {c: score(data, rids_by_stem, stems_by_fold[f], *c) for c in configs} for f in folds}

    def pooled(stems, c):
        return score(data, rids_by_stem, stems, *c)

    nested, chosen = {"TP": 0, "FP": 0, "FN": 0, "si": 0.0}, {}
    for f in folds:
        others = [s for g in folds if g != f for s in stems_by_fold[g]]
        if others:
            best = max(configs, key=lambda c: pooled(others, c)["PQ"])
        else:
            best = max(configs, key=lambda c: per_fold[f][c]["PQ"])
        chosen[f] = best
        r = per_fold[f][best]
        nested["TP"] += r["TP"]; nested["FP"] += r["FP"]; nested["FN"] += r["FN"]; nested["si"] += r["SQ"] * r["TP"]
    den = nested["TP"] + 0.5 * nested["FP"] + 0.5 * nested["FN"]
    all_stems = [s for f in folds for s in stems_by_fold[f]]
    glob = {c: pooled(all_stems, c) for c in configs}
    gbest = max(configs, key=lambda c: glob[c]["PQ"])
    default = (0.5, None, 0)
    out = dict(
        exp=a.exp, folds=folds, probs=a.probs, n_images=len(all_stems), n_records=len(gts),
        default_t05_noseed_a0=glob[default],
        nested=dict(PQ=nested["si"] / den, RQ=nested["TP"] / den, SQ=nested["si"] / max(nested["TP"], 1),
                    TP=nested["TP"], FP=nested["FP"], FN=nested["FN"],
                    chosen={str(f): list(map(str, c)) for f, c in chosen.items()}),
        per_fold_default={str(f): per_fold[f][default] for f in folds},
        per_fold_nested={str(f): per_fold[f][chosen[f]] for f in folds},
        optimistic_global_best=dict(config=list(map(str, gbest)), **glob[gbest]),
        top10_global=[dict(config=list(map(str, c)), **glob[c])
                      for c in sorted(configs, key=lambda c: -glob[c]["PQ"])[:10]],
    )
    dst = base / f"tune_post_cc_tt0.5_{'-'.join(map(str, folds))}_{Path(a.probs).stem}.json"
    out["group"] = "cc"
    json.dump(out, open(dst, "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("default_t05_noseed_a0", "nested", "optimistic_global_best")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
