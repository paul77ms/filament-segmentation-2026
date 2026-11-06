"""Filaments one annotator missed but another drew: found by the final pipeline? by a third annotator? (nested OOF; D-093).
  CUDA_VISIBLE_DEVICES="" taskset -c 0-3 .venv/bin/python scripts/human_missed.py run       (uses scripts/a_eval.final_instances)
  .venv/bin/python scripts/human_missed.py summary   -> records/se2/analysis/human_missed_summary.json
"""
import os, sys, json; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from itertools import permutations
from filseg.scorers import iou_matrix
from filseg.metrics import single_thread_worker

def matched(G, P):            # per instance of G: matched (IoU > 0.5) by some instance of P
    if not len(G): return np.zeros(0, bool)
    if not len(P): return np.zeros(len(G), bool)
    return iou_matrix(G, P)[1].max(1) > 0.5

def job(args):
    import a_eval as AE
    stem, fold, gts = args
    if len(gts) < 2: return []
    final = AE.final_instances(stem, fold)
    sets = {r: gts[r] for r in sorted(gts)}
    rows = []
    for A, R in permutations(sets, 2):
        missA = ~matched(sets[R], sets[A])            # R's filaments that annotator A did not draw
        if not missA.any(): continue
        ms = matched(sets[R], final)
        others = [C for C in sets if C not in (A, R)]
        mC = matched(sets[R], sets[others[0]]) if others else None
        area = sets[R].areas
        for i in np.flatnonzero(missA):
            rows.append(dict(stem=stem, n_sets=len(sets), area=int(area[i]), model_se2=bool(ms[i]),
                             third_human=None if mC is None else bool(mC[i])))
    return rows

CSV = "outputs/research/se2_analysis/human_missed.csv"


def summary():
    d = pd.read_csv(CSV)
    d["sz"] = pd.cut(d.area, [0, 500, 1000, 1e7], labels=["<500", "500-1k", ">1k"])   # not "size": DataFrame.size
    out = {}
    for name, g in [("all_multi_set", d), ("three_set_images", d[d.n_sets == 3])] + \
            [(f"size_{s}", d[d.sz == s]) for s in ["<500", "500-1k", ">1k"]]:
        t = g[g.third_human.notna()]      # rows with a third annotator (3-set images)
        out[name] = dict(n=int(len(g)), model_final_se2=round(float(g.model_se2.mean()), 4), n_three_set=int(len(t)),
                         third_human=round(float(t.third_human.astype(bool).mean()), 4),
                         model_final_se2_same_cases=round(float(t.model_se2.mean()), 4))
    out["_note"] = "model = the final pipeline, nested OOF; produced by scripts/human_missed.py"
    json.dump(out, open("records/se2/analysis/human_missed_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        import multiprocessing as mp, a_eval as AE
        single_thread_worker()
        jobs = [j for j in AE.all_jobs() if len(j[2]) >= 2]
        with mp.get_context("forkserver").Pool(4, initializer=single_thread_worker) as pool:
            rows = [r for out in pool.imap_unordered(job, jobs, chunksize=1) for r in out]
        pd.DataFrame(rows).to_csv(CSV, index=False)
    summary()
