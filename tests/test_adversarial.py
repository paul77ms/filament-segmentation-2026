"""Adversarial verification of filseg.metrics / filseg.rle (task MV, docs/metrics_verification.md).

Run:  PYTHONPATH=src .venv/bin/python tests/test_adversarial.py          (CPU only, ~10-40 s)
Also pytest-compatible. Real-data checks live in scripts/mv_verification/real_compare.py.

Contents
  1. independent brute-force PQ (python pixel sets, exhaustive matching) vs filseg.metrics on
     randomised synthetic cases (exact IoU = 0.5, duplicates, overlaps, empty sets, odd shapes)
  2. host-style metric vs the VERBATIM third-party copy of the organiser's self-evaluation notebook
     (skipped if torch or the copy is unavailable)
  3. RLE edge shapes + mutation tests proving tests/test_rle.py catches C-order codecs
  4. split / merge behaviour pinned on crafted cases, incl. the documented blind spots
  5. regression tests for every change MV made (one per bug)
"""
from __future__ import annotations

import importlib
import math
import os
import sys

import numpy as np
from pycocotools import mask as mask_util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import cv2  # noqa: E402

from filseg import rle  # noqa: E402
from filseg.metrics import (EvalConfig, InstanceSet, evaluate_dataset, evaluate_image,  # noqa: E402
                            match_instances, thin_region)

cv2.setNumThreads(1)
NB_PATH = os.path.join(ROOT, "docs", "thirdparty_metric", "ooLemon0-0_self-evaluation-notebook_cells.py")
S = (128, 160)


def close(a, b, tol=1e-9):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol


def R(y0, x0, y1, x1, shape=(256, 256)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


# =============================================================================== brute force
def _pix(m):
    return frozenset(np.flatnonzero(np.asarray(m).ravel()).tolist())


def bf_record(gts, preds, thr=0.5):
    """Deliberately simple: pixel sets, explicit loops, exhaustive one-to-one matching."""
    G, P = [_pix(m) for m in gts], [_pix(m) for m in preds]
    T = [[(len(g & p) / (len(g) + len(p) - len(g & p)) if (len(g) + len(p)) else 0.0) for p in P] for g in G]
    edges = [(i, j) for i in range(len(G)) for j in range(len(P)) if T[i][j] > thr]
    best = [-1.0, 0]

    def rec(k, ug, up, s, n):
        if k == len(edges):
            if s > best[0] + 1e-12:
                best[0], best[1] = s, n
            return
        rec(k + 1, ug, up, s, n)
        i, j = edges[k]
        if i not in ug and j not in up:
            rec(k + 1, ug | {i}, up | {j}, s + T[i][j], n + 1)

    if len(edges) <= 16:
        rec(0, frozenset(), frozenset(), 0.0, 0)
    else:                                   # too many edges for plain enumeration: skip case
        return None
    s, tp = max(best[0], 0.0), best[1]
    fp, fn = len(P) - tp, len(G) - tp
    den = tp + 0.5 * fp + 0.5 * fn
    tph = len(edges)
    sh = sum(T[i][j] for i, j in edges)
    fph = sum(1 for j in range(len(P)) if not any(T[i][j] > thr for i in range(len(G))))
    fnh = sum(1 for i in range(len(G)) if not any(T[i][j] > thr for j in range(len(P))))
    denh = tph + 0.5 * fph + 0.5 * fnh
    nan = float("nan")
    return dict(tp=tp, fp=fp, fn=fn, sum_iou=s, pq=(s / den if den else nan), pq_empty1=(s / den if den else 1.0),
                sq=(s / tp if tp else nan), rq=(tp / den if den else nan),
                tp_host=tph, fp_host=fph, fn_host=fnh, sum_iou_host=sh, pq_host=(sh / denh if denh else nan),
                n_gt=len(G), n_pred=len(P), iou=T)


def _gen(rng, shape):
    h, w = shape
    m = np.zeros(shape, np.uint8)
    k = rng.integers(0, 7)
    if k == 0:
        y0, y1 = sorted(rng.integers(0, h + 1, 2)); x0, x1 = sorted(rng.integers(0, w + 1, 2))
        m[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = 1
    elif k == 1:
        pts = np.stack([rng.integers(-2, w + 2, 4), rng.integers(-2, h + 2, 4)], 1).astype(np.int32)
        cv2.polylines(m, [pts], False, 1, int(rng.integers(1, 4)))
    elif k == 2:
        cv2.ellipse(m, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                    (int(rng.integers(1, max(2, w // 2))), int(rng.integers(1, max(2, h // 2)))),
                    float(rng.uniform(0, 180)), 0, 360, 1, -1)
    elif k == 3:
        m[int(rng.choice([0, h - 1])), int(rng.integers(0, w))] = 1            # border pixel
    elif k == 4:
        m[:] = 1
    elif k == 5:
        pass                                                                  # empty instance
    else:
        m[rng.random(shape) < 0.2] = 1
    return m.astype(bool)


def _derive(rng, gts, shape):
    h, w = shape
    out = []
    for g in gts:
        r = rng.random()
        px = np.flatnonzero(g.ravel())
        if r < 0.2:
            out.append(g.copy())
        elif r < 0.3:
            out += [g.copy(), g.copy()]                                       # duplicates
        elif r < 0.45 and px.size >= 2 and px.size % 2 == 0:                  # IoU exactly 0.5
            q = np.zeros(h * w, bool); q[px[: px.size // 2]] = True; out.append(q.reshape(shape))
        elif r < 0.55 and px.size >= 2:                                       # split
            cut = int(rng.integers(1, px.size))
            for piece in (px[:cut], px[cut:]):
                q = np.zeros(h * w, bool); q[piece] = True; out.append(q.reshape(shape))
        elif r < 0.65:
            out.append(cv2.dilate(g.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
        elif r < 0.7:
            out.append(np.zeros(shape, bool))
    if len(gts) >= 2 and rng.random() < 0.3:
        out.append(gts[0] | gts[1])                                           # merge
    out += [_gen(rng, shape) for _ in range(int(rng.integers(0, 3)))]
    rng.shuffle(out)
    return out[:9]


def _cases(n, seed):
    rng = np.random.default_rng(seed)
    shapes = [(1, 1), (1, 7), (7, 1), (16, 16), (23, 41), (64, 64), (57, 130)]
    out = []
    for c in range(n):
        shape = shapes[int(rng.integers(0, len(shapes)))]
        gts = [_gen(rng, shape) for _ in range(int(rng.integers(0, 6)))]
        preds = [] if rng.random() < 0.1 else _derive(rng, gts, shape)
        if rng.random() < 0.05:
            gts = []
        out.append((f"c{c}", shape, gts, preds))
    return out


def test_bruteforce_pq_random(n: int = 800, seed: int = 20261027):
    keys_i = ("tp", "fp", "fn", "tp_host", "fp_host", "fn_host")
    keys_f = ("sum_iou", "pq", "sq", "rq", "pq_empty1", "sum_iou_host", "pq_host")
    recs, items, n_half, n_nonunique, checked = [], [], 0, 0, 0
    for key, shape, gts, preds in _cases(n, seed):
        b = bf_record(gts, preds)
        if b is None:
            continue
        checked += 1
        row, _, _ = evaluate_image(gts, preds, EvalConfig(), image_key=key, shape=shape)
        mr = match_instances(gts, preds, shape=shape)
        if gts and preds:
            assert np.array_equal(np.asarray(b["iou"], float), mr.iou), key       # bit-identical IoU
        for k in keys_i:
            assert row[k] == b[k], (key, k, row[k], b[k])
        for k in keys_f:
            assert close(row[k], b[k]), (key, k, row[k], b[k])
        n_half += sum(v == 0.5 for r_ in b["iou"] for v in r_)
        n_nonunique += not row["unique_by_construction"]
        recs.append(b)
        items.append((key, InstanceSet.from_masks(gts, shape=shape), InstanceSet.from_masks(preds, shape=shape)))
    assert checked >= 500 and n_half >= 30 and n_nonunique >= 50, (checked, n_half, n_nonunique)
    # dataset aggregation: pooled / mean / empty conventions
    s, _ = evaluate_dataset(items)
    Sm = lambda k: sum(r[k] for r in recs)  # noqa: E731
    den = Sm("tp") + 0.5 * Sm("fp") + 0.5 * Sm("fn")
    denh = Sm("tp_host") + 0.5 * Sm("fp_host") + 0.5 * Sm("fn_host")
    pqs = [r["pq"] for r in recs if not math.isnan(r["pq"])]
    pqh = [r["pq_host"] for r in recs if not math.isnan(r["pq_host"])]
    assert close(s["PQ_pooled"], Sm("sum_iou") / den)
    assert close(s["SQ_pooled"], Sm("sum_iou") / Sm("tp")) and close(s["RQ_pooled"], Sm("tp") / den)
    assert close(s["PQ_pooled_host"], Sm("sum_iou_host") / denh)
    assert close(s["PQ_img_mean"], sum(pqs) / len(pqs))
    assert close(s["PQ_img_mean_empty1"], sum(r["pq_empty1"] for r in recs) / len(recs))
    assert close(s["PQ_img_mean_host"], sum(pqh) / len(pqh))
    assert s["n_records_both_empty"] == sum(1 for r in recs if r["n_gt"] == 0 and r["n_pred"] == 0)
    # standard PQ with one-to-one matching: denominator is always (n_gt + n_pred) / 2
    assert close(s["PQ_pooled"], Sm("sum_iou") / (0.5 * (Sm("n_gt") + Sm("n_pred"))))
    return checked, n_half, n_nonunique


# =============================================================================== host notebook
def _load_notebook():
    try:
        import torch  # noqa: F401
    except Exception:
        return None
    if not os.path.exists(NB_PATH):
        return None
    os.environ.setdefault("MPLBACKEND", "Agg")
    ns: dict = {}
    exec(compile(open(NB_PATH).read(), NB_PATH, "exec"), ns)
    return ns


def test_host_matches_verbatim_notebook(n: int = 300, seed: int = 7):
    nb = _load_notebook()
    if nb is None:
        print("SKIP notebook comparison (torch or notebook copy missing)")
        return
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    rows, items = [], []
    for key, shape, gts, preds in _cases(n, seed):
        H, W = shape
        g = torch.from_numpy(np.stack(gts).astype(np.float32)) if gts else torch.zeros((0, H, W))
        p = torch.from_numpy(np.stack(preds).astype(np.float32)) if preds else torch.zeros((0, H, W))
        iou, _ = nb["get_overlap_matrices"](g, p)
        rows.append(dict(iou_matrix=iou, n_gt=len(gts), n_pred=len(preds)))
        r, _, _ = evaluate_image(gts, preds, EvalConfig(compute_thin=False, compute_boundary=False), shape=shape)
        if gts and preds:
            hit = iou > 0.5                                  # float32 IoU; threshold decisions identical
            assert (int(hit.sum()), nb["fp_count_hit"](hit), nb["fn_count_hit"](hit)) == \
                (r["tp_host"], r["fp_host"], r["fn_host"]), key
        items.append((key, InstanceSet.from_masks(gts, shape=shape), InstanceSet.from_masks(preds, shape=shape)))
    s, _ = evaluate_dataset(items, EvalConfig(compute_thin=False, compute_boundary=False))
    pq_nb = nb["get_pq_score"](pd.DataFrame(rows))
    assert abs(pq_nb - s["PQ_pooled_host"]) < 1e-6, (pq_nb, s["PQ_pooled_host"])   # float32 vs float64
    # documented difference: nothing to score -> notebook returns 0.0, filseg returns NaN
    assert nb["get_pq_score"](pd.DataFrame([dict(iou_matrix=None, n_gt=0, n_pred=0)])) == 0.0
    s0, _ = evaluate_dataset([("e", [], [])], shape=S)
    assert math.isnan(s0["PQ_pooled_host"])
    # filament_id parsing: notebook splits at the FIRST '_' (filseg.rle mirrors it since MV)
    df = pd.DataFrame({"filament_id": ["20140609195854Bh_3", "a_b_7"]})
    nb_stems = df["filament_id"].str.split("_", n=1).str[0].tolist()
    assert nb_stems == [rle.stem_from_filament_id(f) for f in df["filament_id"]]


# =============================================================================== RLE
def _ref(m):
    return mask_util.encode(np.asfortranarray((np.asarray(m) != 0).astype(np.uint8)))["counts"].decode("ascii")


def test_rle_edge_shapes():
    rng = np.random.default_rng(3)
    n = 0
    for h, w in [(1, 1), (1, 2), (2, 1), (3, 5), (5, 3), (7, 2048), (2048, 7), (1, 4096), (4096, 1),
                 (31, 17), (2047, 2049), (2048, 2048)]:
        z = np.zeros((h, w), bool)
        variants = [z, ~z, rng.random((h, w)) < (0.4 if h * w < 100_000 else 0.002)]
        for sl in [(0, slice(None)), (-1, slice(None)), (slice(None), 0), (slice(None), -1)]:
            b = z.copy(); b[sl] = True; variants.append(b)                     # border rows / cols
        b = z.copy(); b[-1, -1] = True; variants.append(b)                     # last pixel only
        b = ~z; b[0, 0] = False; variants.append(b)                            # all but first
        for m in variants:
            s = rle.encode(m)
            assert s == _ref(m), (h, w)
            assert np.array_equal(rle.decode(s, h, w).astype(bool), m), (h, w)
            assert rle.counts_to_string(rle.runs_from_mask(m)) == s
            assert rle.rle_area(s, h, w) == int(m.sum())
            (y0, x0, y1, x1), c = rle.rle_to_crop(s, h, w)
            if m.any():
                rr, cc = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
                assert (y0, x0, y1, x1) == (rr[0], cc[0], rr[-1] + 1, cc[-1] + 1)
                assert np.array_equal(c, m[y0:y1, x0:x1])
            else:
                assert (y0, x0, y1, x1) == (0, 0, 0, 0) and c.size == 0
            # layouts / dtypes / strided views encode identically
            big = np.zeros((h + 2, w + 3), bool); big[1:-1, 2:-1] = m
            for v in (np.asfortranarray(m), m.astype(np.uint8) * 255, m.astype(np.float32), big[1:-1, 2:-1]):
                assert rle.encode(v) == s
            n += 1
    return n


def _c_decode(counts, h=2048, w=2048):
    if isinstance(counts, dict):
        h, w = counts["size"]; counts = counts["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode()
    flat = np.zeros(h * w, np.uint8); pos = val = 0
    for r in rle.string_to_counts(counts):
        if val:
            flat[pos:pos + r] = 1
        pos += r; val ^= 1
    return flat.reshape((h, w), order="C")


def _c_encode(mask):
    flat = (np.asarray(mask) != 0).ravel(order="C").astype(np.int8)
    ch = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate([[0], ch, [flat.size]])).tolist()
    if flat.size and flat[0] == 1:
        runs = [0] + runs
    return rle.counts_to_string(runs)


def test_rle_tests_are_orientation_sensitive():
    """tests/test_rle.py must FAIL if decode and/or encode used C (row-major) order."""
    import test_rle
    importlib.reload(test_rle)
    targets = ("test_random_roundtrip_and_pycocotools_identity", "test_orientation_column_major")
    for patch in ({"decode": _c_decode}, {"encode": _c_encode}, {"encode": _c_encode, "decode": _c_decode}):
        orig = {k: getattr(rle, k) for k in patch}
        try:
            for k, f in patch.items():
                setattr(rle, k, f)
            for t in targets:
                try:
                    getattr(test_rle, t)()
                except (AssertionError, ValueError):
                    continue
                raise AssertionError(f"{t} did not detect mutation {sorted(patch)}")
        finally:
            for k, f in orig.items():
                setattr(rle, k, f)
    # roundtrip_check alone is NOT orientation-sensitive unless check_reference=True (documented)
    o_e, o_d = rle.encode, rle.decode
    try:
        rle.encode, rle.decode = _c_encode, _c_decode
        m = np.zeros((40, 30), bool); m[3:5, 10:28] = True
        assert rle.roundtrip_check(m) is True
        assert rle.roundtrip_check(m, check_reference=True) is False
    finally:
        rle.encode, rle.decode = o_e, o_d


# =============================================================================== split / merge
def _sm(g, p):
    r, pg, pp = evaluate_image(g, p, EvalConfig(), shape=(256, 256))
    return r, pg, pp


def test_split_merge_crafted():
    g = R(100, 20, 103, 220)                                                    # thin GT 3 x 200
    r, _, _ = _sm([g], [R(100, 20, 103, 120), R(100, 120, 103, 220)])          # exact halves
    assert (r["tp"], r["n_split_gt"], r["n_split_fragments"], r["fn_split"], r["fp_fragment"]) == (0, 1, 2, 1, 2)
    r, _, _ = _sm([g], [R(100, 20, 103, 80), R(100, 80, 103, 150), R(100, 150, 103, 220)])
    assert (r["n_split_gt"], r["n_split_fragments"]) == (1, 3)
    r, _, _ = _sm([g], [R(98, 20, 104, 120), R(98, 120, 104, 220)])            # 2x wider pieces: still split
    assert r["n_split_gt"] == 1
    g1, g2 = R(100, 20, 103, 120), R(100, 124, 103, 220)
    r, _, _ = _sm([g1, g2], [R(97, 20, 106, 220)])                             # merge (even if fat)
    assert (r["n_merge_pred"], r["n_merged_gt"], r["fn_merged"], r["fp_merge"]) == (1, 2, 2, 1)
    r, _, _ = _sm([g1, R(100, 124, 103, 224)], [R(100, 20, 103, 173)])          # 49 % of 2nd GT: no merge
    assert r["n_merge_pred"] == 0 and r["tp"] == 1
    # ---- documented blind spots (pinned so that a definition change is noticed)
    r, _, _ = _sm([g], [R(97, 20, 106, 120), R(97, 120, 106, 220)])            # pieces 3x wider than GT
    assert r["n_split_gt"] == 0 and r["fn_low_iou"] == 1                         # not seen as split
    r, _, _ = _sm([g], [R(100, 20 + 3 * k, 103, 23 + 3 * k) for k in range(66)])  # 9-px shards < min_px
    assert r["n_split_gt"] == 0 and r["fn_missed"] == 1 and r["fp_spurious"] == 66
    r, _, _ = _sm([g], [g.copy(), g.copy()])                                    # duplicates count as 'split'
    assert r["n_split_gt"] == 1 and r["fp_duplicate"] == 1 and r["fp_fragment"] == 0
    r, _, _ = _sm([R(100, 20, 103, 120), R(100, 130, 103, 230)],                 # crossing split+merge
                  [R(100, 20, 103, 70), R(100, 70, 103, 180), R(100, 180, 103, 230)])
    assert r["n_merge_pred"] == 1 and r["n_split_gt"] == 0


# =============================================================================== MV regressions
def test_regression_stem_parsing_matches_host():
    """Bug 1: stem_from_filament_id used rsplit (LAST '_'); the host notebook splits at the FIRST '_'."""
    assert rle.stem_from_filament_id("20140609195854Bh_3") == "20140609195854Bh"
    assert rle.stem_from_filament_id("abc_def_3") == "abc"                      # host: 'abc'; old code: 'abc_def'
    try:
        rle.make_filament_id("abc_def", 3)
        raise AssertionError("stem containing the separator not rejected")
    except ValueError:
        pass
    fmt = rle.SubmissionFormat(id_template="{stem}-{k:03d}", id_sep="-")
    assert rle.stem_from_filament_id(rle.make_filament_id("abc_def", 7, fmt), fmt) == "abc_def"


def test_regression_evaluate_dataset_shape():
    """Bug 2: evaluate_dataset had no `shape`; an empty GT list with non-2048 pred masks raised."""
    p = [R(1, 1, 5, 5, S)]
    s, df = evaluate_dataset([("a", [], p), ("b", p, [])], shape=S)
    assert (s["TP"], s["FP"], s["FN"]) == (0, 1, 1)
    s2, _ = evaluate_dataset([("a", [], p)], shape=S, n_workers=2)
    assert s2["FP"] == 1


def test_regression_best_index_without_overlap():
    """Bug 3: per-instance best_pred / best_gt was 0 (argmax of zeros) when nothing overlaps."""
    _, pg, pp = evaluate_image([R(0, 0, 5, 5, S), R(50, 50, 60, 60, S)], [R(100, 100, 110, 110, S), R(51, 51, 60, 60, S)],
                               shape=S)
    assert pg[0]["best_pred"] == -1 and pg[0]["best_iou"] == 0.0
    assert pg[1]["best_pred"] == 1
    assert pp[0]["best_gt"] == -1 and pp[1]["best_gt"] == 1


def test_thin_region_structuring_element_is_cv2_ellipse():
    """Doc fix: thin_region opens with cv2 MORPH_ELLIPSE (2r+1), not the Euclidean disk(r)."""
    rng = np.random.default_rng(0)
    m = cv2.dilate((rng.random((80, 90)) < 0.03).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    for r in (2, 3):
        pad = r + 1
        mm = np.pad(m.astype(np.uint8), pad)
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (yy * yy + xx * xx <= r * r).astype(np.uint8)
        assert int(el.sum()) != int(disk.sum())
        exp = (mm.astype(bool) & ~cv2.morphologyEx(mm, cv2.MORPH_OPEN, el).astype(bool))[pad:-pad, pad:-pad]
        assert np.array_equal(thin_region(m, r), exp)


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            out = fn()
            print(f"ok  {name}  ({time.time() - t:.1f}s){'  ' + str(out) if out is not None else ''}")
