import sys, json, os
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_compare as sc
import functools
print = functools.partial(print, flush=True)
from collections import Counter
tot = Counter(); mism = 0; aggm = []; nbd = []
for seed in (1, 2, 3):
    sc.make_cases.__defaults__ = (seed,)
    r = sc.main(2000)
    for k, v in r["stats"].items():
        if k != "max_iou_err": tot[k] += v
    mism += r["n_record_mismatches"]; aggm += r["aggregate_mismatches"]; nbd.append(r["notebook_minus_bf_host"])
    if r["first_mismatches"]: print(seed, r["first_mismatches"])
print("small shapes, seeds 1-3:", dict(tot), "record_mismatches", mism, "agg_mismatches", aggm, "nb-bf", nbd)
# large shapes: filament-like, many instances
sc.SHAPES[:] = [(2048, 2048), (517, 300), (1023, 2047)]
orig = sc.make_cases
import cv2, numpy as np
def filament(rng, shape):
    """filament-like: thick polyline / thin ellipse / small rectangle, some touching borders"""
    h, w = shape
    m = np.zeros(shape, np.uint8)
    k = rng.integers(0, 4)
    cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
    if k == 0:
        n = int(rng.integers(2, 6))
        pts = np.stack([cx + np.cumsum(rng.integers(-60, 61, n)), cy + np.cumsum(rng.integers(-60, 61, n))], 1).astype(np.int32)
        cv2.polylines(m, [pts], False, 1, int(rng.integers(1, 12)))
    elif k == 1:
        cv2.ellipse(m, (cx, cy), (int(rng.integers(5, 150)), int(rng.integers(1, 12))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
    elif k == 2:
        y0 = 0 if rng.random() < 0.5 else cy; x0 = w - 40 if rng.random() < 0.5 else cx
        m[y0:y0 + int(rng.integers(1, 60)), max(0, x0):x0 + int(rng.integers(1, 60))] = 1
    else:
        m[cy:cy + 3, :] = 1 if rng.random() < 0.2 else 0     # full-width thin band (touches both borders)
        m[cy, cx] = 1
    if m.sum() == 0: m[cy, cx] = 1
    return m.astype(bool)
_orig_gen = sc.gen_instance
def big_cases(n, seed=10):
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n):
        shape = sc.SHAPES[int(rng.integers(0, 3))]
        ng = int(rng.integers(5, 14))
        gts = [filament(rng, shape) for _ in range(ng)]
        preds = sc.derive_preds(rng, gts, shape)
        while len(preds) < 30:
            preds += sc.derive_preds(rng, gts, shape)
        out.append((f"b{c:03d}", shape, gts, preds[:30]))
    return out
sc.make_cases = big_cases
sc.gen_instance = filament
r = sc.main(60)
print("large shapes:", json.dumps({k: r[k] for k in ("stats", "n_record_mismatches", "aggregate_mismatches", "parallel_equals_serial", "notebook_minus_bf_host", "seconds")}, default=str))
print("large agg", json.dumps(r["bf_aggregate"]))
