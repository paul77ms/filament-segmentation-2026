"""RLE adversarial checks + mutation tests of tests/test_rle.py (C-order bugs must be caught)."""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repository root (path-independent)
import importlib
import sys
import traceback

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, _REPO + "/src")
sys.path.insert(0, _REPO + "/tests")
from filseg import rle  # noqa: E402


def ref_counts(m):
    return mask_util.encode(np.asfortranarray((np.asarray(m) != 0).astype(np.uint8)))["counts"].decode("ascii")


def cases():
    rng = np.random.default_rng(123)
    out = []
    for h, w in [(1, 1), (1, 2), (2, 1), (3, 5), (5, 3), (7, 2048), (2048, 7), (1, 4096), (4096, 1),
                 (31, 17), (2048, 2048), (2047, 2049), (1000, 3)]:
        z = np.zeros((h, w), bool)
        out.append(("zeros", z))
        out.append(("ones", np.ones((h, w), bool)))
        b = z.copy(); b[0, :] = True; out.append(("top_row", b))
        b = z.copy(); b[-1, :] = True; out.append(("bottom_row", b))
        b = z.copy(); b[:, 0] = True; out.append(("left_col", b))
        b = z.copy(); b[:, -1] = True; out.append(("right_col", b))
        b = z.copy(); b[0, 0] = b[-1, -1] = True; out.append(("corners_tl_br", b))
        b = z.copy(); b[-1, 0] = b[0, -1] = True; out.append(("corners_bl_tr", b))
        b = z.copy(); b[-1, -1] = True; out.append(("last_pixel_only", b))
        b = np.ones((h, w), bool); b[0, 0] = False; out.append(("all_but_first", b))
        b = np.ones((h, w), bool); b[-1, -1] = False; out.append(("all_but_last", b))
        b = z.copy(); b[::2, ::3] = True; out.append(("lattice", b))
        if h * w <= 300_000:
            out.append(("random50", rng.random((h, w)) < 0.5))
        else:
            out.append(("random_sparse", rng.random((h, w)) < 0.002))
    return out


def check_all():
    n, bad = 0, []
    for name, m in cases():
        h, w = m.shape
        s = rle.encode(m)
        n += 1
        if s != ref_counts(m):
            bad.append((name, m.shape, "encode != pycocotools(asfortranarray)"))
        d = rle.decode(s, h, w)
        if d.shape != (h, w) or not np.array_equal(d.astype(bool), m):
            bad.append((name, m.shape, "roundtrip"))
        if rle.counts_to_string(rle.runs_from_mask(m)) != s or rle.string_to_counts(s) != rle.runs_from_mask(m):
            bad.append((name, m.shape, "reference codec"))
        if rle.rle_area(s, h, w) != int(m.sum()):
            bad.append((name, m.shape, "area"))
        (y0, x0, y1, x1), c = rle.rle_to_crop(s, h, w)
        if m.any():
            rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            tb = (rows[0], cols[0], rows[-1] + 1, cols[-1] + 1)
            if (y0, x0, y1, x1) != tuple(int(v) for v in tb) or not np.array_equal(c, m[tb[0]:tb[2], tb[1]:tb[3]]):
                bad.append((name, m.shape, "rle_to_crop"))
        elif (y0, x0, y1, x1) != (0, 0, 0, 0) or c.size:
            bad.append((name, m.shape, "rle_to_crop empty"))
        # non-contiguous / transposed / sliced views and other dtypes must encode identically
        for v in (np.asfortranarray(m), m.astype(np.uint8) * 255, m.astype(np.float32), m.astype(np.int64) * -3):
            if rle.encode(v) != s:
                bad.append((name, m.shape, f"dtype/layout {v.dtype} {v.flags['F_CONTIGUOUS']}"))
        big = np.zeros((h + 2, w + 3), bool); big[1:-1, 2:-1] = m
        if rle.encode(big[1:-1, 2:-1]) != s:
            bad.append((name, m.shape, "strided view"))
    return n, bad


# ------------------------------------------------------------------ mutation: wrong-order codecs
def c_decode(counts, h=2048, w=2048):
    if isinstance(counts, dict):
        h, w = counts["size"]; counts = counts["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode()
    runs = rle.string_to_counts(counts)
    flat = np.zeros(h * w, np.uint8); pos = 0; val = 0
    for r in runs:
        if val: flat[pos:pos + r] = 1
        pos += r; val ^= 1
    return flat.reshape((h, w), order="C")


def c_encode(mask):
    m = np.asarray(mask) != 0
    flat = m.ravel(order="C").astype(np.int8)
    ch = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate([[0], ch, [flat.size]])).tolist()
    if flat.size and flat[0] == 1:
        runs = [0] + runs
    return rle.counts_to_string(runs)


def run_m1_tests_under(patch):
    import test_rle
    importlib.reload(test_rle)
    orig = {k: getattr(rle, k) for k in patch}
    for k, f in patch.items():
        setattr(rle, k, f)
    res = {}
    try:
        for t in ("test_random_roundtrip_and_pycocotools_identity", "test_orientation_column_major"):
            try:
                getattr(test_rle, t)()
                res[t] = "PASSED (mutation NOT detected)"
            except AssertionError:
                res[t] = "failed (mutation detected)"
            except Exception as e:  # noqa: BLE001
                res[t] = f"error {type(e).__name__} (mutation detected)"
    finally:
        for k, f in orig.items():
            setattr(rle, k, f)
    return res


if __name__ == "__main__":
    n, bad = check_all()
    print(f"RLE adversarial cases: {n}, failures: {len(bad)}", bad[:10])
    print("baseline (no mutation):", run_m1_tests_under({}))
    print("C-order decode only:", run_m1_tests_under({"decode": c_decode}))
    print("C-order encode only:", run_m1_tests_under({"encode": c_encode}))
    print("C-order encode+decode (self-consistent):", run_m1_tests_under({"encode": c_encode, "decode": c_decode}))
    # roundtrip_check alone with a self-consistent C-order codec
    o_e, o_d = rle.encode, rle.decode
    rle.encode, rle.decode = c_encode, c_decode
    m = np.zeros((40, 30), bool); m[3:5, 10:28] = True
    print("roundtrip_check(check_reference=False) under self-consistent C-order codec:", rle.roundtrip_check(m),
          "| check_reference=True:", rle.roundtrip_check(m, check_reference=True))
    rle.encode, rle.decode = o_e, o_d
