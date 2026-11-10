"""Helpers for notebooks/reproduce_submission.ipynb.

- locate the competition data, the test images or any folder of new images,
- export predicted masks and outline overlays (export_masks),
- rebuild the stored model weights from their < 50 MB parts and verify every checksum,
- compare a freshly produced submission CSV with the reference submission (instance-level, overlap-based),
- draw a small figure of the pipeline stages for one image (r2b0_figure).

Only numpy, opencv and the standard library are used (plus filseg_infer / weight_parts from this folder).
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import filseg_infer as K
import weight_parts

IMAGE_SUFFIXES = (".jpeg", ".jpg")


# ----------------------------------------------------------------------------- data
def find_test_dir(data) -> Path:
    """The folder of images to predict: the competition's test_images folder (given itself, as the
    MAGFiLO_1.0_Kaggle_2026 folder or as the folder that contains it), or any folder of .jpeg/.jpg images (new data)."""
    data = Path(data)
    for cand in (data, data / "test" / "test_images", data / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"):
        if cand.name == "test_images" and cand.is_dir():
            return cand
    if data.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in data.iterdir()):
        return data
    raise FileNotFoundError(
        f"No images under {data}. Give a folder of 2048x2048 GONG H-alpha JPEGs, or the competition data: unzip "
        "filament-segmentation-2026.zip in the repository root, or set FILSEG_DATA to the folder that contains "
        "MAGFiLO_1.0_Kaggle_2026/.")


def data_root(test_dir) -> Path:
    """The MAGFiLO_1.0_Kaggle_2026 folder for the competition layout, otherwise the image folder itself."""
    test_dir = Path(test_dir)
    return test_dir.parent.parent if test_dir.name == "test_images" else test_dir


def locate_data(repo) -> Path:
    """The competition data folder (MAGFiLO_1.0_Kaggle_2026) at the usual places: the repository root, next to the
    repository, the current folder, or a Kaggle notebook's input. Set FILSEG_DATA to use any other folder."""
    repo = Path(repo)
    tried = [repo / "MAGFiLO_1.0_Kaggle_2026", repo.parent / "MAGFiLO_1.0_Kaggle_2026", Path.cwd() / "MAGFiLO_1.0_Kaggle_2026",
             Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
             Path("/kaggle/input/filament-segmentation-2026")]
    for cand in tried:
        try:
            t = find_test_dir(cand)
        except FileNotFoundError:
            continue
        if t.name == "test_images":
            return t.parent.parent                          # .../MAGFiLO_1.0_Kaggle_2026
    raise FileNotFoundError(
        "Competition data not found. Looked in:\n  " + "\n  ".join(map(str, tried)) + "\nUnzip "
        "filament-segmentation-2026.zip in the repository root, or set FILSEG_DATA (or DATA in the notebook's first cell) "
        "to the folder MAGFiLO_1.0_Kaggle_2026, its test/test_images folder, or the folder that contains it.")


def list_test_images(test_dir, limit: int = 0) -> list[Path]:
    """Same file selection and order as filseg_infer.run()."""
    files = sorted(p for p in Path(test_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return files[:limit] if limit else files


# ----------------------------------------------------------------------------- weights
def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_sha256sums(path) -> dict[str, str]:
    """Parse a `sha256sum` file -> {file name: hex digest}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.lstrip("*").strip()] = digest
    return out


def prepare_weights(weights_dir, log=print) -> list[Path]:
    """Return the whole checkpoint files, rebuilding each one from its parts when needed.

    For every <name>.manifest.json: if <name> already exists with the expected sha256 it is reused; otherwise the
    parts are checked (size + sha256 against the manifest and SHA256SUMS), concatenated and written to <name>.
    Finally the sha256 of every whole file is recomputed from disk and compared with SHA256SUMS.
    """
    weights_dir = Path(weights_dir)
    sums = read_sha256sums(weights_dir / "SHA256SUMS")
    manifests = sorted(weights_dir.glob("*.manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no *.manifest.json in {weights_dir}")
    paths = []
    for man_path in manifests:
        man = json.loads(man_path.read_text())
        for part in man["parts"]:
            if sums.get(part["path"]) != part["sha256"]:
                raise RuntimeError(f"SHA256SUMS and {man_path.name} disagree on {part['path']}")
        target = weights_dir / man["file"]
        if target.is_file() and target.stat().st_size == man["bytes"] and sha256_file(target) == man["sha256"]:
            how = "already present"
        else:
            weight_parts.join(man_path, target)          # checks every part and the whole file before writing
            how = f"rebuilt from {len(man['parts'])} parts"
        digest = sha256_file(target)
        if not (digest == man["sha256"] == sums.get(man["file"])):
            raise RuntimeError(f"{target.name}: sha256 {digest} does not match SHA256SUMS")
        log(f"{target.name}: {man['bytes']:,} bytes, sha256 {digest[:16]}... OK ({how})")
        paths.append(target)
    return paths


# ----------------------------------------------------------------------------- submissions
def image_of(filament_id: str) -> str:
    return filament_id.rsplit("_", 1)[0]


def read_submission(path) -> dict[str, list[tuple[str, str]]]:
    """{image stem: [(filament_id, rle counts), ...]} in file order; checks the header."""
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        if header != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"{path}: unexpected header {header}")
        out = defaultdict(list)
        for fid, rle in rd:
            out[image_of(fid)].append((fid, rle))
    return dict(out)


def export_masks(csv_path, image_dir, masks_dir=None, overlays_dir=None, limit: int = 0) -> int:
    """For each predicted image: a 16-bit PNG label map (0 = background, k = the k-th filament of the image in the CSV,
    i.e. filament_id <stem>_<k-1>) and/or a JPEG of the image with the filament outlines. Returns the images written."""
    sub = read_submission(csv_path)
    for d in (masks_dir, overlays_dir):
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
    files = list_test_images(image_dir, limit)
    for f in files:
        rows = sub.get(f.stem, [])
        lab = np.zeros((K.H, K.W), np.uint16)
        for k, (_, rle) in enumerate(rows, start=1):
            lab[K.mask_from_counts(rle).astype(bool)] = k
        if masks_dir:
            cv2.imwrite(str(Path(masks_dir) / f"{f.stem}.png"), lab)
        if overlays_dir:
            img = cv2.cvtColor(K.read_image(f), cv2.COLOR_GRAY2BGR)
            for k in range(1, len(rows) + 1):
                cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, (0, 200, 255), 2)
            cv2.imwrite(str(Path(overlays_dir) / f"{f.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return len(files)


def _label_map(rows):
    """Label image (0 = background, k = k-th filament) and areas; raises if two filaments overlap."""
    lab = np.zeros(K.H * K.W, np.int32)
    areas = []
    for k, (_, rle) in enumerate(rows, start=1):
        m = K.mask_from_counts(rle).ravel()
        idx = np.flatnonzero(m)
        if lab[idx].any():
            raise ValueError("overlapping filaments in one image")
        lab[idx] = k
        areas.append(idx.size)
    return lab, np.asarray(areas, np.int64)


def compare_submissions(new_csv, ref_csv, images=None, match_iou: float = 0.5) -> dict:
    """Instance-level comparison of two submissions, matched by overlap (never by filament_id).

    images: image stems to compare (default: union of both files). Pairs with IoU > match_iou are matches (they are
    unique, as in Panoptic Quality). Returned: image-set and per-image row-count agreement, the IoU distribution of
    matched pairs, bit-identical masks, filaments that disappear (reference only) or appear (new only), pixel
    disagreement, and an agreement PQ that treats the reference as ground truth.
    """
    new, ref = read_submission(new_csv), read_submission(ref_csv)
    if images is None:
        images = sorted(set(new) | set(ref))
    images = list(images)
    ious, identical, disappeared, appeared = [], 0, [], []
    rows_differ, px_xor, px_ref = [], 0, 0
    for img in images:
        a, b = new.get(img, []), ref.get(img, [])
        if len(a) != len(b):
            rows_differ.append((img, len(a), len(b)))
        lab_a, area_a = _label_map(a)
        lab_b, area_b = _label_map(b)
        px_xor += int(np.count_nonzero((lab_a > 0) != (lab_b > 0)))
        px_ref += int(area_b.sum())
        inter = np.bincount(lab_a * (len(b) + 1) + lab_b, minlength=(len(a) + 1) * (len(b) + 1))
        inter = inter.reshape(len(a) + 1, len(b) + 1)[1:, 1:].astype(np.float64)
        union = area_a[:, None] + area_b[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        hit_a, hit_b = np.zeros(len(a), bool), np.zeros(len(b), bool)
        for i, j in zip(*np.nonzero(iou > match_iou)):
            hit_a[i] = hit_b[j] = True
            ious.append(iou[i, j])
            identical += bool(inter[i, j] == area_a[i] == area_b[j])
        for j in np.flatnonzero(~hit_b):
            best = float(iou[:, j].max()) if len(a) else 0.0
            disappeared.append(dict(filament_id=b[j][0], area=int(area_b[j]), best_iou=round(best, 4)))
        for i in np.flatnonzero(~hit_a):
            best = float(iou[i].max()) if len(b) else 0.0
            appeared.append(dict(filament_id=a[i][0], area=int(area_a[i]), best_iou=round(best, 4)))
    ious = np.asarray(ious)
    tp, fp, fn = len(ious), len(appeared), len(disappeared)
    q = (lambda p: float(np.percentile(ious, p))) if tp else (lambda p: float("nan"))
    return dict(
        images_compared=len(images),
        images_with_rows_new=sum(bool(new.get(i)) for i in images),
        images_with_rows_ref=sum(bool(ref.get(i)) for i in images),
        same_images_with_predictions={i for i in images if new.get(i)} == {i for i in images if ref.get(i)},
        rows_new=sum(len(new.get(i, [])) for i in images),
        rows_ref=sum(len(ref.get(i, [])) for i in images),
        images_with_different_row_count=rows_differ,
        matched=tp, bit_identical_masks=identical,
        iou_min=q(0), iou_p01=q(1), iou_p05=q(5), iou_p25=q(25), iou_median=q(50),
        iou_mean=float(ious.mean()) if tp else float("nan"),
        disappeared=disappeared, appeared=appeared,
        pixel_disagreement_fraction=px_xor / max(px_ref, 1),
        agreement_pq=float(ious.sum() / (tp + 0.5 * fp + 0.5 * fn)) if (tp + fp + fn) else 1.0,
    )


# Tolerances of verdict(): minimum fraction of reference filaments matched, maximum fraction that appear or disappear,
# minimum median IoU of the matched filaments.
VERDICT_TOLERANCE = dict(matched=0.97, changed=0.03, median_iou=0.95)
# Fixed on 2026-10-29 before the first CPU-vs-GPU comparison. Several steps react to fp32/bf16 rounding: the stage-1
# map is thresholded at 0.3 (candidates, >= 20 px) and 0.5 (existence, >= 500 px), and the refined masks at 0.5,
# merged at IoU 0.5 and kept from 300 px.


def verdict(cmp: dict) -> str:
    """Pre-declared reading of compare_submissions(); thresholds in VERDICT_TOLERANCE (fixed before the CPU runs)."""
    tol = VERDICT_TOLERANCE
    n = max(cmp["rows_ref"], 1)
    if cmp["matched"] == cmp["rows_ref"] == cmp["rows_new"] == cmp["bit_identical_masks"]:
        return "BIT-IDENTICAL to the reference submission"
    if (cmp["same_images_with_predictions"] and cmp["matched"] / n >= tol["matched"]
            and (len(cmp["appeared"]) + len(cmp["disappeared"])) / n <= tol["changed"]
            and cmp["iou_median"] >= tol["median_iou"]):
        return "REPRODUCED within numerical tolerance (fp32/bf16 rounding)"
    return "DIFFERENT from the reference beyond the expected numerical tolerance - investigate"


# ----------------------------------------------------------------------------- figure
def _panel(img_u8, title, size):
    p = cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_AREA)
    if p.ndim == 2:
        p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
    (w, h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(p, (0, 0), (w + 12, h + 12), (0, 0, 0), -1)
    cv2.putText(p, title, (6, h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def _draw_instances(canvas_bgr, comps, seed=0):
    """Colour fill + thick outline of each (bbox, crop) instance, in place."""
    rng = np.random.default_rng(seed)
    for (y0, x0, y1, x1), m in comps:
        color = rng.integers(80, 256, 3).tolist()
        roi = canvas_bgr[y0:y1, x0:x1]
        roi[m] = (0.15 * roi[m] + 0.85 * np.asarray(color)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(roi, cnts, -1, color, 6)
    return canvas_bgr


def r2b0_figure(img, res, size: int = 448) -> bytes:
    """Pipeline-stages PNG: (1) input + disk window, (2) stage-1 mean probability, (3) candidates (kept = overlap a stage-1
    instance of >= 500 px: orange; dropped: grey), (4) final instances after R2_final refinement and assembly.
    res = filseg_infer_r2b0.predict_image(...)."""
    full = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(full, (K.CROP0, K.CROP0), (K.CROP1, K.CROP1), (0, 200, 255), 8)
    heat = cv2.applyColorMap(res["prob_u8"], cv2.COLORMAP_VIRIDIS)
    cand = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cmap = res["cand"]
    kept = np.isin(cmap, np.asarray(res["kept"], cmap.dtype)) & (cmap > 0)
    grow = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0   # stay visible when downscaled
    cand[grow((cmap > 0) & ~kept)] = (150, 150, 150)
    cand[grow(kept)] = (0, 140, 255)
    inst = _draw_instances(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), res["instances"])
    n_c, n_k = len(res["boxes"]), len(res["kept"])
    panels = [_panel(full, "1 input 2048x2048, disk window", size),
              _panel(heat, "2 stage-1 mean probability (5 folds)", size),
              _panel(cand, f"3 candidates p>0.3: {n_k} of {n_c} kept", size),
              _panel(inst, f"4 R2_final, assembled: {len(res['instances'])} filaments", size)]
    ok, png = cv2.imencode(".png", np.hstack(panels))
    assert ok
    return png.tobytes()
