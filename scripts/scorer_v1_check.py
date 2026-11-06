"""Independent check of scorer hypothesis V1 (organiser-perspective analysis, 2026-10-29).

V0 = official self-evaluation notebook: one record per (annotation set, image); every IoU>0.5 pair is a TP; pooled PQ.
V1 = hypothesis for the hidden Kaggle scorer: one record per IMAGE whose GT is the concatenation of the instances of
     ALL its annotation sets (overlapping copies kept), matched ONE-TO-ONE to the predictions (maximum-IoU assignment
     restricted to IoU > 0.5); FP = unmatched predictions, FN = unmatched GT copies; PQ pooled over images.
Methods: b0 OOF (thr 0.5, 8-conn, min_area 500), B_ens and R2nb0 (nested evaluation instances), on all 707 files.
  CUDA_VISIBLE_DEVICES="" taskset -c 0-3 .venv/bin/python scripts/scorer_v1_check.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from filseg import seg  # noqa: E402
from filseg.metrics import intersection_matrix, single_thread_worker  # noqa: E402

CFG = dict(ens_exp="b0n_ens", r2_exp="R2n_context", b0_exp="b0_r101_s1024",
           cache_prefix=str(ROOT / "outputs" / "cache" / "refine_nested_o"))


def iou(G, P):
    if not len(G) or not len(P):
        return np.zeros((len(G), len(P)))
    inter = intersection_matrix(G, P).astype(np.float64)
    return inter / np.maximum(G.areas[:, None] + P.areas[None, :] - inter, 1)


def score_v1(sets, P):
    """sets: list of InstanceSet (one per annotation set); P: predictions -> (tp, fp, fn, sum_iou)."""
    M = np.concatenate([iou(G, P) for G in sets], 0) if sets else np.zeros((0, len(P)))
    n_gt, n_p = M.shape
    if not n_gt or not n_p:
        return 0, n_p, n_gt, 0.0
    W = np.where(M > 0.5, M, 0.0)
    r, c = linear_sum_assignment(-W)
    ok = W[r, c] > 0.5
    tp = int(ok.sum())
    return tp, n_p - tp, n_gt - tp, float(W[r, c][ok].sum())


def score_v0(sets, P):
    tp = fp = fn = 0; s = 0.0
    for G in sets:
        M = iou(G, P)
        hit = M > 0.5
        tp += int(hit.sum()); s += float(M[hit].sum())
        fp += int(len(P) - (hit.any(0).sum() if len(G) else 0)); fn += int(len(G) - (hit.any(1).sum() if len(P) else 0))
    return tp, fp, fn, s


def job(args):
    import nested_r2b0_eval as NE
    stem, fold, set_ids = args
    gts = seg.gt_records([stem + ".jpeg"])
    sets = [gts[r] for r in sorted(gts)]
    assert len(sets) == len(set_ids), (stem, set_ids, sorted(gts))
    M, _ = NE.instances(stem, fold, CFG)
    from filseg import refine as RF
    pb0 = NE._load(("b0", fold), lambda: np.load(NE.EXPS / CFG["b0_exp"] / f"fold{fold}" / "val_probs_last.npz"))
    M["b0_a300"] = seg.prob_to_instances(pb0[stem].astype(np.float32), seg.PostCfg(0.5, 300))
    ens = NE._load(("ens", fold), lambda: np.load(NE.EXPS / CFG["ens_exp"] / f"fold{fold}" / "val_probs_last.npz"))
    idx = NE._load(("idx", fold), lambda: json.load(open(Path(CFG["cache_prefix"] + str(fold)) / "index.json")))
    cand = NE._load(("cand", fold), lambda: np.load(Path(CFG["cache_prefix"] + str(fold)) / "cand.npy", mmap_mode="r"))
    cmap = np.asarray(cand[idx["files"].index(stem + ".jpeg")])
    z, by = NE.RC._npz(NE.EXPS / CFG["r2_exp"] / f"fold{fold}" / "val_cand_probs.npz")
    cp = {j: (y0, x0, z[k].astype(np.float32)) for j, y0, x0, k in by.get(stem, [])}
    B300 = seg.prob_to_instances(ens[stem].astype(np.float32), seg.PostCfg(0.5, 300))
    inb = np.bincount(cmap[B300.union()], minlength=int(cmap.max()) + 1)
    keep = {j for j in range(1, len(inb)) if inb[j] > 0}
    M["R2nb0_exist300"] = RF.assemble({j: v for j, v in cp.items() if j in keep}, thr=0.5, min_area=300)
    out = []
    for name in ("b0_single", "b0_a300", "B_ens", "B_ens_a300", "R2nb0", "R2nb0_exist300", "R2nfull"):
        P = M[name]
        out.append(dict(stem=stem, fold=fold, n_sets=len(sets), method=name,
                        **{f"v1_{k}": v for k, v in zip(("tp", "fp", "fn", "si"), score_v1(sets, P))},
                        **{f"v0_{k}": v for k, v in zip(("tp", "fp", "fn", "si"), score_v0(sets, P))}))
    return out


def pq(d, v):
    s = d[[f"{v}_tp", f"{v}_fp", f"{v}_fn", f"{v}_si"]].sum().values.astype(float)
    return s[3] / (s[0] + .5 * s[1] + .5 * s[2])


def main():
    import multiprocessing as mp
    single_thread_worker()
    sp = pd.read_csv(ROOT / "outputs" / "splits" / "folds_v1.csv")
    jobs = [(r.file_name[:-5], int(r.fold), sorted(r.image_ids.split(";"))) for r in sp.itertuples()]
    with mp.get_context("forkserver").Pool(4, initializer=single_thread_worker) as pool:
        df = pd.DataFrame([x for xs in pool.imap_unordered(job, jobs, chunksize=4) for x in xs])
    out = ROOT / "outputs" / "research" / "organiser"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scorer_v1_check_records.csv", index=False)
    res = {}
    for m, d in df.groupby("method"):
        res[m] = dict(V0=round(pq(d, "v0"), 4), V1=round(pq(d, "v1"), 4),
                      V1_by_n_sets={int(k): round(pq(g, "v1"), 4) for k, g in d.groupby("n_sets")},
                      V0_by_n_sets={int(k): round(pq(g, "v0"), 4) for k, g in d.groupby("n_sets")})
    print(json.dumps(res, indent=1))
    json.dump(res, open(out / "scorer_v1_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
