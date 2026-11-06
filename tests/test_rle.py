"""Tests for filseg.rle. Run:  PYTHONPATH=src .venv/bin/python tests/test_rle.py [--skip-real]
(also pytest-compatible). The real-data test decodes ALL train polygons (CPU, <=16 workers) and
writes outputs/audit/m1_rle_roundtrip_train.csv + m1_gt_overlap_by_record.csv."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import numpy as np
from pycocotools import mask as mask_util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from filseg import rle  # noqa: E402

ANN = os.path.join(ROOT, "MAGFiLO_1.0_Kaggle_2026", "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
AUDIT = os.path.join(ROOT, "outputs", "audit")


def _crop_equal(bc, m) -> bool:
    (y0, x0, y1, x1), c = bc
    m = np.asarray(m) != 0
    if not m.any():
        return (y0, x0, y1, x1) == (0, 0, 0, 0) and c.size == 0
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    tight = (int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1)
    return (y0, x0, y1, x1) == tight and np.array_equal(c, m[y0:y1, x0:x1])


def _random_masks(rng):
    shapes = [(1, 1), (1, 7), (7, 1), (5, 9), (64, 64), (100, 37), (2048, 2048)]
    out = []
    for h, w in shapes:
        # dense random 2048^2 masks have ~1.7M runs -> slow pure-python reference; keep them sparse
        probs = (0.0, 0.001, 0.01, 1.0) if h * w > 1_000_000 else (0.0, 0.01, 0.3, 0.7, 1.0)
        for p in probs:
            out.append(rng.random((h, w)) < p)
        # blobby structure (long runs) + a mask whose first pixel is foreground
        m = np.zeros((h, w), bool)
        m[: max(1, h // 3), : max(1, w // 2)] = True
        out.append(m)
    return out


def test_random_roundtrip_and_pycocotools_identity():
    rng = np.random.default_rng(0)
    for m in _random_masks(rng):
        h, w = m.shape
        s = rle.encode(m)
        assert isinstance(s, str)
        # 1) roundtrip
        d = rle.decode(s, h, w)
        assert d.dtype == np.uint8 and d.shape == (h, w)
        assert np.array_equal(d.astype(bool), m)
        # 2) byte-for-byte identical to pycocotools' own encoder on Fortran-ordered uint8
        ref = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"]
        assert s.encode("ascii") == ref
        # 3) independent pure-python reference (numpy column-major runs + ported compressor)
        runs = rle.runs_from_mask(m)
        assert sum(runs) == h * w
        assert rle.counts_to_string(runs) == s
        assert rle.string_to_counts(s) == runs
        assert np.array_equal(rle.mask_from_runs(runs, h, w).astype(bool), m)
        # 4) pycocotools decode of its own RLE (Fortran) == our decode
        assert np.array_equal(mask_util.decode({"size": [h, w], "counts": ref}), d)
        assert rle.roundtrip_check(m, strict=True, check_reference=True)
        assert rle.rle_area(s, h, w) == int(m.sum())
        # 5) band-only crop decoder == tight crop of the full decode
        assert _crop_equal(rle.rle_to_crop(s, h, w), m)
    # uint8 masks with value 255 and bool masks give the same string
    m = rng.random((50, 60)) < 0.2
    assert rle.encode(m) == rle.encode(m.astype(np.uint8) * 255) == rle.encode(m.astype(np.int32))


def test_orientation_column_major():
    h, w = 4, 6
    m = np.zeros((h, w), bool)
    m[0, 1] = True                      # row 0, column 1 -> column-major index 1*h + 0 = 4
    runs = rle.runs_from_mask(m)
    assert runs == [4, 1, h * w - 5], runs
    s = rle.encode(m)
    assert mask_util.decode({"size": [h, w], "counts": s.encode()})[0, 1] == 1
    # a C-order (row-major) interpretation would put the pixel at flat index 4 -> (0, 4) instead
    assert rle.decode(s, h, w)[0, 4] == 0
    # non-square, non-symmetric mask is NOT equal to its transposed decode
    m2 = np.zeros((2048, 2048), bool); m2[100:110, 1500:1900] = True
    d2 = rle.decode(rle.encode(m2))
    assert np.array_equal(d2.astype(bool), m2) and not np.array_equal(d2.T.astype(bool), m2)


def test_polygon_rasterisation_matches_pycocotools():
    poly = [[10.5, 10.2, 60.1, 12.0, 58.3, 30.7, 12.0, 28.0, 10.5, 10.2]]
    m = rle.polygons_to_mask(poly, 100, 120)
    r = mask_util.merge(mask_util.frPyObjects(poly, 100, 120))
    assert np.array_equal(m, mask_util.decode(r))
    assert int(m.sum()) == int(mask_util.area(r))
    ann = {"segmentation": poly}
    assert np.array_equal(rle.ann_to_mask(ann, 100, 120), m)
    # RLE-typed annotations (compressed and uncompressed) dispatch on the actual type
    enc = rle.encode(m)
    assert np.array_equal(rle.ann_to_mask({"segmentation": {"size": [100, 120], "counts": enc}}, 100, 120), m)
    unc = {"size": [100, 120], "counts": rle.runs_from_mask(m)}
    assert np.array_equal(rle.ann_to_mask({"segmentation": unc}, 100, 120), m)


def test_submission_io_roundtrip():
    rng = np.random.default_rng(1)
    masks = {f"2014060919585{i}Bh": [rng.random((64, 48)) < 0.1 for _ in range(3)] for i in range(3)}
    rows = []
    for stem, ms in masks.items():
        for k, m in enumerate(ms):
            rows.append((rle.make_filament_id(stem, k), rle.encode(m)))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sub.csv")
        assert rle.write_submission(rows, p) == 9
        with open(p) as f:
            assert f.readline().strip() == "filament_id,segmentation_rle"
        df = rle.read_submission(p)
        assert list(df.columns[:2]) == ["filament_id", "segmentation_rle"]
        g = rle.group_submission_by_stem(df)
        for stem, ms in masks.items():
            dec = [rle.decode(c, 64, 48).astype(bool) for c in g[stem]]
            assert all(np.array_equal(a, b) for a, b in zip(dec, ms))
        # configurable column names / id template
        fmt = rle.SubmissionFormat(id_col="id", rle_col="rle", id_template="{stem}-{k:03d}", id_sep="-")
        p2 = os.path.join(td, "sub2.csv")
        rle.write_submission([{"id": rle.make_filament_id("abc_def", 7, fmt), "rle": rows[0][1]}], p2, fmt)
        df2 = rle.read_submission(p2, fmt)
        assert df2.loc[0, "id"] == "abc_def-007" and df2.loc[0, "stem"] == "abc_def"
        # duplicates and invalid strings are rejected
        try:
            rle.write_submission([rows[0], rows[0]], p)
            raise AssertionError("duplicate id not rejected")
        except ValueError:
            pass
        try:
            rle.write_submission([("x_0", "bad,string")], p)
            raise AssertionError("invalid counts not rejected")
        except ValueError:
            pass


# ----------------------------------------------------------------------------- real data
def _check_record(args):
    """All annotations of one annotation-set record: polygon -> mask -> our RLE -> mask."""
    image_id, anns = args
    out = []
    H = W = 2048
    cover = np.zeros((H, W), np.uint8)
    sum_area = 0
    for a in anns:
        seg = a["segmentation"]
        assert isinstance(seg, list), type(seg)
        r = rle.ann_to_rle(a, H, W)                          # pycocotools polygon RLE
        m = rle.decode(r, H, W)
        s = rle.encode(m)
        ok_rt = bool(np.array_equal(rle.decode(s, H, W), m))
        ok_bytes = s.encode("ascii") == r["counts"]
        ok_ref = rle.counts_to_string(rle.runs_from_mask(m)) == s
        ok_crop = _crop_equal(rle.rle_to_crop(r, H, W), m)
        area = int(m.sum())
        cover += m
        sum_area += area
        out.append(dict(ann_id=a["id"], image_id=image_id, n_poly=len(seg), n_vertices=len(seg[0]) // 2,
                        area_json=float(a["area"]), area_mask=area, area_equal=(area == float(a["area"])),
                        roundtrip_ok=ok_rt, pycocotools_bytes_equal=ok_bytes, reference_encoder_equal=ok_ref,
                        band_crop_equal=ok_crop))
    union = int((cover > 0).sum())
    rec = dict(image_id=image_id, n_ann=len(anns), sum_area=sum_area, union_area=union,
               overlap_px=sum_area - union, max_coverage=int(cover.max()) if anns else 0)
    return out, rec


def _single_thread():
    import cv2
    cv2.setNumThreads(1)


def test_real_train_roundtrip(n_workers: int = 16):
    if not os.path.exists(ANN):
        print("SKIP real-data test: annotation file not found")
        return
    import multiprocessing as mp
    import pandas as pd

    t0 = time.time()
    d = json.load(open(ANN))
    by_img: dict = {im["id"]: [] for im in d["images"]}
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    jobs = list(by_img.items())
    rows, recs = [], []
    with mp.get_context("fork").Pool(min(n_workers, 16), initializer=_single_thread) as pool:
        for out, rec in pool.imap_unordered(_check_record, jobs, chunksize=4):
            rows.extend(out)
            recs.append(rec)
    dt = time.time() - t0
    df = pd.DataFrame(rows)
    rec = pd.DataFrame(recs).sort_values("image_id")
    os.makedirs(AUDIT, exist_ok=True)
    df.to_csv(os.path.join(AUDIT, "m1_rle_roundtrip_train.csv"), index=False)
    rec.to_csv(os.path.join(AUDIT, "m1_gt_overlap_by_record.csv"), index=False)
    n = len(df)
    print(f"[real] records={len(jobs)} annotations={n} wall={dt:.1f}s workers={min(n_workers, 16)}")
    print(f"[real] roundtrip_ok={int(df.roundtrip_ok.sum())}/{n} pycocotools_bytes_equal="
          f"{int(df.pycocotools_bytes_equal.sum())}/{n} reference_encoder_equal={int(df.reference_encoder_equal.sum())}/{n}"
          f" area_equal_json={int(df.area_equal.sum())}/{n} band_crop_equal={int(df.band_crop_equal.sum())}/{n}")
    print(f"[real] records with overlapping GT instances: {int((rec.overlap_px > 0).sum())}/{len(rec)};"
          f" total overlap px={int(rec.overlap_px.sum())}; max coverage={int(rec.max_coverage.max())}")
    assert n == len(d["annotations"])
    assert df.roundtrip_ok.all()
    assert df.pycocotools_bytes_equal.all()
    assert df.reference_encoder_equal.all()
    assert df.band_crop_equal.all()
    if not df.area_equal.all():
        bad = df[~df.area_equal]
        print(f"[real] WARNING area mismatch vs JSON for {len(bad)} annotations, e.g.\n{bad.head()}")


if __name__ == "__main__":
    test_random_roundtrip_and_pycocotools_identity()
    test_orientation_column_major()
    test_polygon_rasterisation_matches_pycocotools()
    test_submission_io_roundtrip()
    print("synthetic RLE tests: OK")
    if "--skip-real" not in sys.argv:
        test_real_train_roundtrip()
        print("real-data RLE test: OK")
