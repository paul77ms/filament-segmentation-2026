"""Stand-alone validator for a submission CSV (helper; does not import filseg for the core checks).

Usage:
  NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/check_submission_csv.py \
      outputs/submissions/b0_r101_s1024_5fold_last_t05_a500.csv \
      [--test_dir MAGFiLO_1.0_Kaggle_2026/test/test_images] [--official_nb self-evaluation-notebook.ipynb] \
      [--json out.json]

Checks (plain `csv` module + numpy + pycocotools only, as the official overview recommends):
  1. raw bytes: no BOM, ASCII only, no quote characters (' or ") anywhere, consistent line endings,
     header line exactly `filament_id,segmentation_rle`;
  2. every data row has exactly 2 fields; filament_id unique; image id = text before the FIRST '_'
     (as the official self-evaluation notebook: str.split("_", n=1).str[0]) is a test-image stem;
     tail after the first '_' non-empty; ids per image unique;
  3. segmentation_rle uses only the COCO compressed alphabet (chr 48..111); its runs (independent
     pure-python LEB128 decoder) sum to exactly 2048*2048 (pycocotools' C decoder does not check this);
  4. pycocotools.mask.decode({"size": [2048, 2048], "counts": <str>}) -> (2048, 2048) uint8, non-empty;
  5. canonical form: pycocotools.mask.encode(np.asfortranarray(mask))["counts"] == the CSV string
     (i.e. identical to encodeMask on the Fortran-order mask);
  6. pycocotools.coco.COCO.annToMask on an annotation dict with that RLE gives the same mask;
  7. no two masks of one image overlap (pixel coverage <= 1);
  8. images of the test directory with no row are listed (allowed: no rows for an empty image);
  9. optional: the OFFICIAL self-evaluation notebook's own code cells (imports/helpers/PQ) are executed and the
     submission is scored against itself re-labelled as one GT annotation set per image
     ("000000-<stem>_<tail>"): this exercises the organiser's id parsing + RLE decoding path; PQ must be 1.0.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

H = W = 2048
HEADER = "filament_id,segmentation_rle"
VALID = set(chr(c) for c in range(48, 112))


def leb128_runs(s: str) -> list[int]:
    """Pure-python port of pycocotools rleFrString (independent of the C decoder)."""
    cnts, p, b = [], 0, s.encode("ascii")
    while p < len(b):
        x, k, more = 0, 0, True
        while more:
            c = b[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--test_dir", default="MAGFiLO_1.0_Kaggle_2026/test/test_images")
    ap.add_argument("--official_nb", default="self-evaluation-notebook.ipynb")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rep: dict = {"csv": a.csv, "errors": []}
    err = rep["errors"]

    raw = Path(a.csv).read_bytes()
    rep["bytes"] = len(raw)
    rep["bom"] = raw.startswith(b"\xef\xbb\xbf")
    rep["ascii_only"] = all(b < 128 for b in raw)
    rep["n_double_quote"] = raw.count(b'"')
    rep["n_single_quote"] = raw.count(b"'")
    rep["n_crlf"] = raw.count(b"\r\n")
    rep["n_lf"] = raw.count(b"\n")
    rep["ends_with_newline"] = raw.endswith(b"\n")
    first = raw.split(b"\n", 1)[0].rstrip(b"\r").decode("ascii", "replace")
    rep["header_line"] = first
    if rep["bom"]:
        err.append("BOM present")
    if not rep["ascii_only"]:
        err.append("non-ASCII bytes")
    if rep["n_double_quote"] or rep["n_single_quote"]:
        err.append("quote characters present")
    if first != HEADER:
        err.append(f"header {first!r} != {HEADER!r}")

    rows = list(csv.reader(io.StringIO(raw.decode("ascii", "replace"), newline="")))
    hdr, body = rows[0], rows[1:]
    rep["n_rows"] = len(body)
    bad_fields = [i for i, r in enumerate(body) if len(r) != 2]
    if bad_fields:
        err.append(f"{len(bad_fields)} rows without exactly 2 fields, e.g. line {bad_fields[0] + 2}")

    test_stems = sorted(p.stem for p in Path(a.test_dir).glob("*.jpeg")) if Path(a.test_dir).is_dir() else []
    rep["n_test_images"] = len(test_stems)
    stem_ok = re.compile(r"^\d{14}[A-Z][a-z]$")
    ids = [r[0] for r in body]
    rep["ids_unique"] = len(set(ids)) == len(ids)
    if not rep["ids_unique"]:
        err.append("duplicate filament_id")
    by_img: dict[str, list] = defaultdict(list)
    tails = []
    for fid, s in (r[:2] for r in body if len(r) == 2):
        if "_" not in fid:
            err.append(f"id without '_': {fid}")
            continue
        stem, tail = fid.split("_", 1)
        tails.append(tail)
        if not tail:
            err.append(f"empty tail: {fid}")
        if not stem_ok.match(stem):
            err.append(f"stem pattern: {fid}")
        if test_stems and stem not in set(test_stems):
            err.append(f"stem not a test image: {fid}")
        by_img[stem].append((fid, s))
    rep["tail_pattern"] = ("all integer tails, min %d max %d" % (min(map(int, tails)), max(map(int, tails)))
                           if tails and all(t.isdigit() for t in tails) else "non-integer tails present")
    tails_contig = all(sorted(int(f.split("_", 1)[1]) for f, _ in v) == list(range(len(v))) for v in by_img.values())
    rep["tails_0_to_n-1_per_image"] = bool(tails_contig)
    rep["n_images_with_rows"] = len(by_img)
    rep["images_without_rows"] = [s for s in test_stems if s not in by_img]
    rep["max_rows_per_image"] = max((len(v) for v in by_img.values()), default=0)

    coco = COCO()
    coco.imgs = {s: {"id": s, "height": H, "width": W} for s in by_img}
    n_dec = n_canon = n_coco = n_sum = 0
    areas, n_overlap_img, overlap_px = [], 0, 0
    on_disk_frac = []
    yy, xx = np.mgrid[0:H, 0:W]
    disk = (yy - 1023.0) ** 2 + (xx - 1024.0) ** 2 <= 905.0 ** 2
    for stem, items in by_img.items():
        cover = np.zeros((H, W), np.uint8)
        for fid, s in items:
            if not s or set(s) - VALID:
                err.append(f"invalid RLE alphabet: {fid}")
                continue
            runs = leb128_runs(s)
            if sum(runs) == H * W and all(r >= 0 for r in runs):
                n_sum += 1
            else:
                err.append(f"runs sum {sum(runs)} != {H * W} or negative run: {fid}")
                continue
            m = mask_util.decode({"size": [H, W], "counts": s})
            if m.shape != (H, W) or m.dtype != np.uint8 or m.max() > 1:
                err.append(f"decode shape/dtype: {fid} {m.shape} {m.dtype}")
                continue
            n_dec += 1
            a_ = int(m.sum())
            areas.append(a_)
            if a_ == 0:
                err.append(f"empty mask: {fid}")
            enc = mask_util.encode(np.asfortranarray(m))["counts"].decode("ascii")
            n_canon += enc == s
            if enc != s:
                err.append(f"not canonical encodeMask(Fortran) string: {fid}")
            mc = coco.annToMask({"image_id": stem, "segmentation": {"size": [H, W], "counts": s}})
            n_coco += bool(np.array_equal(mc, m))
            on_disk_frac.append(float(m[disk].sum()) / max(a_, 1))
            if np.any(cover & m):
                overlap_px += int(np.count_nonzero(cover & m))
            cover += m
        if cover.max(initial=0) > 1:
            n_overlap_img += 1
    rep.update(n_runs_sum_ok=n_sum, n_decoded_2048=n_dec, n_canonical_encodeMask=n_canon, n_annToMask_equal=n_coco,
               n_images_with_overlap=n_overlap_img, overlap_px=overlap_px,
               area_min=int(min(areas)) if areas else None, area_median=float(np.median(areas)) if areas else None,
               area_max=int(max(areas)) if areas else None,
               min_fraction_inside_disk_r905=min(on_disk_frac) if on_disk_frac else None)
    if n_overlap_img:
        err.append(f"overlapping masks in {n_overlap_img} images")

    # 9. official notebook code path (self-consistency, PQ must be exactly 1)
    nbp = Path(a.official_nb)
    if nbp.exists():
        try:
            import hashlib

            import pandas as pd
            import torch
            torch.set_num_threads(min(8, torch.get_num_threads()))
            os.environ.setdefault("MPLBACKEND", "Agg")
            nb = json.loads(nbp.read_text())
            code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
            g: dict = {}
            exec(code[0], g)          # imports
            exec(code[2], g)          # helper functions (get_overlap_matrices ... get_overlap_df)
            exec(code[3], g)          # get_pq_score
            pred_df = pd.DataFrame([r[:2] for r in body], columns=["filament_id", "segmentation_rle"])
            gt_df = pred_df.copy()
            gt_df["filament_id"] = "000000-" + gt_df["filament_id"]
            od = g["get_overlap_df"](gt_df, pred_df)
            pq = g["get_pq_score"](od)
            rep["official_nb_sha256"] = hashlib.sha256(nbp.read_bytes()).hexdigest()
            rep["official_nb_self_pq"] = pq
            rep["official_nb_n_entries"] = int(len(od))
            rep["official_nb_n_pred_total"] = int(od["n_pred"].sum())
            if not (abs(pq - 1.0) < 1e-6 and int(od["n_pred"].sum()) == len(body)):
                err.append(f"official notebook self-PQ {pq} / n_pred {int(od['n_pred'].sum())}")
        except Exception as e:  # report, do not hide
            rep["official_nb_error"] = repr(e)
            err.append(f"official notebook path failed: {e!r}")

    rep["ok"] = not err
    out = json.dumps(rep, indent=1, default=str)
    print(out)
    if a.json:
        Path(a.json).write_text(out)
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
