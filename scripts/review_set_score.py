"""Pooled PQ over the FIXED review set (outputs/audit/review_cases.csv), all annotation sets of each review file,
for one or more methods (exp:group:thr:min_area). Used by adoption rule D-019."""
import os
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd
import compare_paired as C
from filseg import seg
from filseg.metrics import single_thread_worker

if __name__ == "__main__":
    import multiprocessing as mp
    single_thread_worker()
    rc = pd.read_csv(ROOT / "outputs/audit/review_cases.csv")
    methods = [C.parse_method(m) for m in sys.argv[1:]]
    res = {}
    for m in methods:
        jobs = []
        for _, r in rc.iterrows():
            stem = Path(r.file_name).stem
            gts = {k: v for k, v in seg.gt_records([r.file_name]).items()}
            jobs.append((stem, int(r.fold_v1), m, m.get("probs") or "val_probs_last.npz", gts))
        with mp.get_context("forkserver").Pool(12, initializer=single_thread_worker) as pool:
            df = pd.DataFrame([x for rs in pool.map(C.job, jobs) for x in rs])
        df = df.merge(rc[["file_name", "case_id", "category"]].assign(stem=rc.file_name.str[:-5]), on="stem")
        res[m["name"]] = df
        print(m["name"], "pooled PQ over review set (", len(df), "records,", df.stem.nunique(), "files ):", round(C.pooled(df), 4))
    if len(methods) == 2:
        a, b = [res[m["name"]] for m in methods]
        per = []
        for cat in sorted(a.category.unique()):
            per.append((cat, round(C.pooled(a[a.category == cat]), 3), round(C.pooled(b[b.category == cat]), 3)))
        print("per category (A, B):", per)
        fa = a.groupby("case_id").apply(C.pooled); fb = b.groupby("case_id").apply(C.pooled)
        print("cases better/worse/equal for B:", int((fb > fa + 1e-9).sum()), int((fb < fa - 1e-9).sum()), int((abs(fb - fa) <= 1e-9).sum()))
