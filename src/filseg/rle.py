"""COCO-compressed RLE encode/decode, polygon rasterisation and submission CSV I/O.

Conventions (verified by tests/test_rle.py against pycocotools):
  * A mask is an (H, W) array, nonzero = foreground.
  * RLE runs are taken in COLUMN-MAJOR (Fortran) order, starting with a run of zeros
    (possibly of length 0) -- this is the COCO / pycocotools convention.
  * `encode` returns only the compressed ``counts`` string (ASCII). The image size is NOT
    part of the string; `decode` needs (h, w) (default 2048 x 2048).

Submission format: third-party sources (not the official Kaggle page, which we could not
fetch) state the CSV is ``filament_id,segmentation_rle`` with one row per predicted filament,
``filament_id = "<image stem>_<k>"`` and ``segmentation_rle`` = COCO compressed counts only.
Everything about the format is therefore parametrised in `SubmissionFormat` so it can be
changed once the official sample submission is confirmed.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_util

DEFAULT_H = 2048
DEFAULT_W = 2048

__all__ = [
    "encode", "decode", "encode_rle_dict", "rle_area", "rle_bbox",
    "polygons_to_rle", "polygons_to_mask", "ann_to_rle", "ann_to_mask",
    "roundtrip_check", "rle_to_crop", "runs_from_mask", "mask_from_runs",
    "counts_to_string", "string_to_counts",
    "SubmissionFormat", "make_filament_id", "stem_from_filament_id",
    "write_submission", "read_submission", "group_submission_by_stem",
]


# --------------------------------------------------------------------------- core RLE
def _as_fortran_u8(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got shape {m.shape}")
    if m.dtype != np.uint8:
        m = (m != 0).astype(np.uint8)
    elif m.max(initial=0) > 1:
        m = (m != 0).astype(np.uint8)
    return np.asfortranarray(m)


def encode(mask: np.ndarray) -> str:
    """(H, W) bool/uint8 mask -> COCO compressed RLE ``counts`` string (size not included)."""
    rle = mask_util.encode(_as_fortran_u8(mask))
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, (bytes, bytearray)) else str(counts)


def encode_rle_dict(mask: np.ndarray) -> dict:
    """(H, W) mask -> pycocotools RLE dict ``{"size": [H, W], "counts": bytes}``."""
    return mask_util.encode(_as_fortran_u8(mask))


def _rle_dict(counts: str | bytes | dict, h: int, w: int) -> dict:
    if isinstance(counts, dict):
        rle = dict(counts)
        if isinstance(rle.get("counts"), list):  # uncompressed RLE
            rle = mask_util.frPyObjects(rle, rle["size"][0], rle["size"][1])
        return rle
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {"size": [int(h), int(w)], "counts": counts}


def decode(counts: str | bytes | dict, h: int = DEFAULT_H, w: int = DEFAULT_W) -> np.ndarray:
    """COCO compressed counts string (or RLE dict) -> (H, W) uint8 mask in {0, 1}, C-contiguous."""
    m = mask_util.decode(_rle_dict(counts, h, w))
    return np.ascontiguousarray(m)


def rle_area(counts: str | bytes | dict, h: int = DEFAULT_H, w: int = DEFAULT_W) -> int:
    return int(mask_util.area(_rle_dict(counts, h, w)))


def rle_bbox(counts: str | bytes | dict, h: int = DEFAULT_H, w: int = DEFAULT_W) -> np.ndarray:
    """COCO bbox [x, y, w, h] of the RLE (float)."""
    return mask_util.toBbox(_rle_dict(counts, h, w))


# --------------------------------------------------------------------------- polygons
def polygons_to_rle(polygons: Sequence[Sequence[float]], h: int = DEFAULT_H, w: int = DEFAULT_W) -> dict:
    """COCO polygon list ``[[x0, y0, x1, y1, ...], ...]`` -> merged RLE dict (pycocotools rasteriser).

    This is the same path pycocotools uses to compute ``annotation["area"]`` (frPyObjects -> merge).
    """
    if len(polygons) == 0:
        return mask_util.encode(np.zeros((h, w), np.uint8, order="F"))
    polys = [list(map(float, p)) for p in polygons]
    rles = mask_util.frPyObjects(polys, h, w)
    return mask_util.merge(rles) if len(rles) > 1 else rles[0]


def polygons_to_mask(polygons: Sequence[Sequence[float]], h: int = DEFAULT_H, w: int = DEFAULT_W) -> np.ndarray:
    return decode(polygons_to_rle(polygons, h, w), h, w)


def ann_to_rle(ann: Mapping, h: int = DEFAULT_H, w: int = DEFAULT_W) -> dict:
    """COCO annotation -> RLE dict, dispatching on the ACTUAL type of ``segmentation``.

    list -> polygons; dict with list counts -> uncompressed RLE; dict with str/bytes counts -> compressed.
    """
    seg = ann["segmentation"]
    if isinstance(seg, list):
        return polygons_to_rle(seg, h, w)
    if isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):
            return mask_util.frPyObjects(seg, seg["size"][0], seg["size"][1])
        c = seg["counts"]
        return {"size": list(seg["size"]), "counts": c.encode("ascii") if isinstance(c, str) else c}
    raise TypeError(f"unsupported segmentation type {type(seg)!r}")


def ann_to_mask(ann: Mapping, h: int = DEFAULT_H, w: int = DEFAULT_W) -> np.ndarray:
    return decode(ann_to_rle(ann, h, w), h, w)


# --------------------------------------------------------------------------- pure-python reference
def runs_from_mask(mask: np.ndarray) -> list[int]:
    """Uncompressed COCO runs computed with numpy only (column-major, first run = zeros)."""
    flat = (np.asarray(mask) != 0).ravel(order="F").astype(np.int8)
    n = flat.size
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [n]])
    runs = np.diff(bounds).tolist()
    if n and flat[0] == 1:
        runs = [0] + runs
    return [int(r) for r in runs]


def mask_from_runs(runs: Sequence[int], h: int, w: int) -> np.ndarray:
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for r in runs:
        if val:
            flat[pos:pos + r] = 1
        pos += r
        val ^= 1
    if pos != h * w:
        raise ValueError(f"runs sum to {pos}, expected {h * w}")
    return flat.reshape((h, w), order="F")


def counts_to_string(runs: Sequence[int]) -> str:
    """Pure-python port of pycocotools ``rleToString`` (LEB128-like, delta vs run i-2)."""
    out = []
    for i, x in enumerate(runs):
        x = int(x)
        if i > 2:
            x -= int(runs[i - 2])
        more = True
        while more:
            c = x & 0x1F
            x >>= 5  # arithmetic shift, like C on signed long
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def string_to_counts(s: str) -> list[int]:
    """Pure-python port of pycocotools ``rleFrString``."""
    cnts: list[int] = []
    p = 0
    b = s.encode("ascii") if isinstance(s, str) else s
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


def rle_to_crop(counts: str | bytes | dict, h: int = DEFAULT_H, w: int = DEFAULT_W):
    """Decode only the column band spanned by the mask (runs are column-major), without
    allocating the full (h, w) image. Returns ((y0, x0, y1, x1) half-open tight bbox, bool crop);
    an empty mask gives ((0, 0, 0, 0), zeros((0, 0)))."""
    r = _rle_dict(counts, h, w)
    h, w = int(r["size"][0]), int(r["size"][1])
    runs = np.asarray(string_to_counts(r["counts"]), np.int64)
    pos = np.concatenate([[0], np.cumsum(runs)])
    if pos[-1] != h * w:
        raise ValueError(f"RLE runs sum to {pos[-1]}, expected {h * w}")
    m = len(runs)
    starts, ends = pos[1:m:2], pos[2:m + 1:2]          # foreground = odd-indexed runs
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]
    if starts.size == 0:
        return (0, 0, 0, 0), np.zeros((0, 0), bool)
    x0, x1 = int(starts.min() // h), int((ends.max() - 1) // h + 1)
    off, L = x0 * h, (x1 - x0) * h
    d = np.zeros(L + 1, np.int32)
    np.add.at(d, starts - off, 1)
    np.add.at(d, ends - off, -1)
    band = (np.cumsum(d[:-1]) > 0).reshape(x1 - x0, h).T   # (h, x1-x0)
    rows = np.flatnonzero(band.any(1))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    return (y0, x0, y1, x1), np.ascontiguousarray(band[y0:y1])


def roundtrip_check(mask: np.ndarray, *, strict: bool = False, check_reference: bool = False) -> bool:
    """True iff decode(encode(mask)) == (mask != 0) exactly (shape + every pixel).

    check_reference=True additionally checks that the string equals the pure-python reference
    encoder applied to numpy-computed column-major runs (independent of pycocotools).
    """
    m = np.asarray(mask) != 0
    h, w = m.shape
    s = encode(m)
    ok = bool(np.array_equal(decode(s, h, w).astype(bool), m))
    if ok and check_reference:
        ok = s == counts_to_string(runs_from_mask(m))
    if strict and not ok:
        raise AssertionError("RLE roundtrip mismatch")
    return ok


# --------------------------------------------------------------------------- submission I/O
@dataclass(frozen=True)
class SubmissionFormat:
    """Submission CSV layout. Defaults follow third-party descriptions; NOT officially confirmed."""
    id_col: str = "filament_id"
    rle_col: str = "segmentation_rle"
    id_template: str = "{stem}_{k}"   # k = 0-based instance index within the image
    id_sep: str = "_"                 # separator used to recover the stem: fid.split(sep, 1)[0] (first sep, as the host notebook)
    extra_cols: tuple = ()            # e.g. ("height", "width") if the official format needs size


DEFAULT_FORMAT = SubmissionFormat()


def make_filament_id(stem: str, k: int, fmt: SubmissionFormat = DEFAULT_FORMAT) -> str:
    """Raises ValueError if `stem` contains fmt.id_sep: the host notebook recovers the image id
    with ``filament_id.str.split("_", n=1).str[0]`` (FIRST separator), so such an id would be
    attributed to the wrong image by the scorer. (MV verification, docs/metrics_verification.md.)"""
    if fmt.id_sep and fmt.id_sep in stem:
        raise ValueError(f"stem {stem!r} contains the id separator {fmt.id_sep!r}; the scorer would mis-assign it")
    return fmt.id_template.format(stem=stem, k=k)


def stem_from_filament_id(fid: str, fmt: SubmissionFormat = DEFAULT_FORMAT) -> str:
    """Image stem of a filament id, split at the FIRST separator, mirroring the host self-evaluation
    notebook (``pred_df["filament_id"].str.split("_", n=1).str[0]``; third-party copy in
    docs/thirdparty_metric/). Was ``rsplit`` (last separator) before MV; identical for all train/test
    stems, which contain no '_'."""
    return fid.split(fmt.id_sep, 1)[0]


_VALID_CHARS = frozenset(chr(c) for c in range(48, 48 + 64))


def _check_counts_string(s: str) -> None:
    if not isinstance(s, str) or not s:
        raise ValueError("RLE counts must be a non-empty str")
    bad = set(s) - _VALID_CHARS
    if bad:
        raise ValueError(f"RLE counts contain invalid characters {sorted(bad)[:5]}")


def write_submission(rows: Iterable, path: str | os.PathLike, fmt: SubmissionFormat = DEFAULT_FORMAT,
                     *, check_unique: bool = True, validate_counts: bool = True) -> int:
    """Write rows to CSV. ``rows``: iterable of (filament_id, counts) tuples or dicts keyed by
    fmt.id_col / fmt.rle_col (+ fmt.extra_cols). Returns the number of rows written.

    Images with zero predictions simply contribute no rows (whether the official scorer wants a
    placeholder row is UNCONFIRMED).
    """
    cols = [fmt.id_col, fmt.rle_col, *fmt.extra_cols]
    seen: set[str] = set()
    n = 0
    tmp = f"{os.fspath(path)}.tmp"
    with open(tmp, "w", newline="") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        wr.writerow(cols)
        for r in rows:
            if isinstance(r, Mapping):
                vals = [r[c] for c in cols]
            else:
                vals = list(r)
                if len(vals) != len(cols):
                    raise ValueError(f"row has {len(vals)} fields, expected {cols}")
            fid, counts = str(vals[0]), vals[1]
            if isinstance(counts, (bytes, bytearray)):
                counts = counts.decode("ascii")
            if validate_counts:
                _check_counts_string(counts)
            if check_unique:
                if fid in seen:
                    raise ValueError(f"duplicate {fmt.id_col}: {fid}")
                seen.add(fid)
            wr.writerow([fid, counts, *vals[2:]])
            n += 1
    os.replace(tmp, path)
    return n


def read_submission(path: str | os.PathLike, fmt: SubmissionFormat = DEFAULT_FORMAT):
    """Read a submission CSV into a pandas DataFrame (all columns as str; no NA conversion)
    with an added ``stem`` column."""
    import pandas as pd

    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    for c in (fmt.id_col, fmt.rle_col):
        if c not in df.columns:
            raise ValueError(f"column {c!r} missing; found {list(df.columns)}")
    df["stem"] = df[fmt.id_col].map(lambda s: stem_from_filament_id(s, fmt))
    return df


def group_submission_by_stem(df, fmt: SubmissionFormat = DEFAULT_FORMAT) -> dict[str, list[str]]:
    """stem -> list of counts strings (row order preserved)."""
    out: dict[str, list[str]] = {}
    for stem, c in zip(df["stem"], df[fmt.rle_col]):
        out.setdefault(stem, []).append(c)
    return out
