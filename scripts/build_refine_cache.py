"""Build outputs/cache/refine_v1 (see src/filseg/refine.py)."""
import os
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg import refine
from filseg.metrics import single_thread_worker
if __name__ == "__main__":
    single_thread_worker()
    t = time.time()
    d = refine.build_cache()
    idx = json.load(open(d / "index.json"))
    nw = {k: 0 for k in range(5)}; nc = {k: 0 for k in range(5)}
    for f, m in idx["meta"].items():
        nw[idx["fold"][f]] += len(m["windows"]); nc[idx["fold"][f]] += len(m["boxes"])
    print("built", d, round(time.time() - t, 1), "s; windows per fold", nw, "candidates per fold", nc)
