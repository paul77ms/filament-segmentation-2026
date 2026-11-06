"""Baseline: full-disk semantic U-Net -> full-resolution probability -> connected-component instances.

Geometry (verified in docs/data_audit.md): every image has the solar disk centred at ~(1023, 1024)
with R ~ 903 px; the window [CROP0:CROP1] = [120:1928] contains the whole disk and every annotated
filament. The model sees this 1808x1808 window resized to S x S (S=1024 by default, factor 1.766).
Predictions are mapped back with the same pixel-centre convention (cv2 INTER_AREA down,
bilinear/align_corners=False up) and pasted into a zero 2048x2048 canvas.

Targets: union of the instances of ONE annotation set, rasterised exactly with pycocotools at full
resolution, then area-resized to S (soft target in [0,1], stored as uint8 0..255). Multiple
annotation sets of a file are kept separate; training samples one set at random per visit.
Only `segmentation` is used -- never `spine` or `category_id`.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

from . import data as D
from .metrics import InstanceSet

CROP0, CROP1 = 120, 1928
CROP = CROP1 - CROP0  # 1808
H = W = 2048
CACHE_DIR = D.PROJECT_ROOT / "outputs" / "cache"


# ----------------------------------------------------------------------------- geometry
def to_model_space(img_full: np.ndarray, size: int, interp=cv2.INTER_AREA) -> np.ndarray:
    crop = img_full[CROP0:CROP1, CROP0:CROP1]
    return cv2.resize(crop, (size, size), interpolation=interp)


def set_union_mask(anns: list[dict]) -> np.ndarray:
    """Exact full-resolution union of the polygons of one annotation set (uint8 0/1)."""
    polys = [p for a in anns for p in a["segmentation"]]
    rles = mask_util.frPyObjects(polys, H, W)
    return mask_util.decode(mask_util.merge(rles))


# ----------------------------------------------------------------------------- cache
def _cache_one(args):
    file_name, sets, size = args
    cv2.setNumThreads(1)
    img = D.read_image(D.train_image_path(file_name))
    x = to_model_space(img, size)
    ts = []
    for _, anns in sets:
        m = set_union_mask(anns).astype(np.float32)
        t = to_model_space(m, size)  # area-average = soft coverage
        ts.append(np.clip(np.rint(t * 255), 0, 255).astype(np.uint8))
    return file_name, x, ts


def build_cache(size: int = 1024, n_workers: int = 16, force: bool = False) -> Path:
    import fcntl
    import multiprocessing as mp
    import shutil

    final = CACHE_DIR / f"train_s{size}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Exclusive lock: concurrent callers wait instead of rebuilding the same files.
    with open(CACHE_DIR / f".train_s{size}.lock", "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        if (final / "index.json").exists() and not force:
            return final
        tmp = CACHE_DIR / f".train_s{size}.building"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        _build_cache_into(tmp, size, n_workers, mp)
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)  # atomic publish: index.json exists only in a complete cache
    return final


def _build_cache_into(out: Path, size: int, n_workers: int, mp) -> None:
    coco = D.load_coco()
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    files = sorted({im["file_name"] for im in coco["images"]})
    sets_by_file = {f: [] for f in files}
    for im in sorted(coco["images"], key=lambda r: r["id"]):
        sets_by_file[im["file_name"]].append((im["id"], by_img.get(im["id"], [])))
    jobs = [(f, sets_by_file[f], size) for f in files]
    n_sets = sum(len(v) for v in sets_by_file.values())
    X = np.lib.format.open_memmap(out / "images.npy", "w+", np.uint8, (len(files), size, size))
    T = np.lib.format.open_memmap(out / "targets.npy", "w+", np.uint8, (n_sets, size, size))
    index = {"size": size, "crop": [CROP0, CROP1], "files": [], "set_ids": [], "file_sets": {}}
    srow = 0
    with mp.get_context("fork").Pool(n_workers) as pool:
        for i, (f, x, ts) in enumerate(pool.imap(_cache_one, jobs, chunksize=2)):
            assert f == files[i]
            X[i] = x
            rows = []
            for (sid, _), t in zip(sets_by_file[f], ts):
                T[srow] = t
                index["set_ids"].append(sid)
                rows.append(srow)
                srow += 1
            index["files"].append(f)
            index["file_sets"][f] = rows
    X.flush(); T.flush()
    del X, T
    with open(out / "index.json", "w") as fh:
        json.dump(index, fh)


# ----------------------------------------------------------------------------- GT for evaluation
def gt_records(file_names=None) -> dict:
    """{annotation-set id: InstanceSet} for the given files (all train files if None)."""
    coco = D.load_coco()
    keep = None if file_names is None else set(file_names)
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    out = {}
    for im in coco["images"]:
        if keep is not None and im["file_name"] not in keep:
            continue
        out[im["id"]] = InstanceSet.from_coco_anns(by_img.get(im["id"], []))
    return out


# ----------------------------------------------------------------------------- model
_SAFE_GLOBALS = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"),
                 ("torch._utils", "_rebuild_tensor"), ("torch._utils", "_rebuild_parameter"),
                 ("torch.nn.parameter", "Parameter")}


def load_legacy_state_dict(path):
    """Load a legacy (.tar-format) torchvision checkpoint without executing arbitrary pickle code.

    torch 2.6 refuses weights_only=True for the legacy format, so we pass a restricted pickle
    module whose Unpickler only resolves tensor/storage rebuild helpers and OrderedDict.
    """
    import pickle
    import types

    import torch

    class _Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in _SAFE_GLOBALS or (module == "torch" and name.endswith("Storage")):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    mod = types.ModuleType("restricted_pickle")
    mod.Unpickler = _Restricted
    mod.load = pickle.load
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=mod)


def build_model(encoder: str = "resnet101", pretrained: str | None = "local", decoder_channels=(256, 128, 64, 32, 16),
                classes: int = 1):
    import segmentation_models_pytorch as smp
    import torch

    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=classes,
                     decoder_channels=decoder_channels)
    if pretrained == "local":
        path = {"resnet101": Path.home() / ".cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth"}[encoder]
        sd = load_legacy_state_dict(path)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        sd["conv1.weight"] = sd["conv1.weight"].sum(1, keepdim=True)  # RGB -> 1 channel
        own = {k: v for k, v in model.encoder.state_dict().items() if not k.endswith("num_batches_tracked")}
        assert set(own) == set(sd), (set(own) ^ set(sd))  # legacy file has no num_batches_tracked
        assert all(own[k].shape == sd[k].shape for k in own)
        model.encoder.load_state_dict(sd, strict=False)
    return model


# ----------------------------------------------------------------------------- inference
def normalize(x_uint8):
    """uint8 tensor/array -> float, fixed affine (dataset-level mean/std of the model-space image)."""
    return (x_uint8.float() / 255.0 - 0.30) / 0.22


def predict_logits(model, x_uint8_batch, tta: bool = False):
    """x_uint8_batch: torch uint8 (B, S, S) on device -> logits (B, S, S) float32."""
    import torch

    x = normalize(x_uint8_batch)[:, None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(x).float()
        if tta:
            y = y + torch.flip(model(torch.flip(x, [3])).float(), [3])
            y = y + torch.flip(model(torch.flip(x, [2])).float(), [2])
            y = y + torch.flip(model(torch.flip(x, [2, 3])).float(), [2, 3])
            y = y / 4
    return y[:, 0]


def logits_to_full_prob(logits_s):
    """(B, S, S) logits (torch, on device) -> (B, 2048, 2048) float32 probability (numpy).

    Logits are upsampled bilinearly (align_corners=False, same pixel-centre convention as
    cv2.resize) BEFORE the sigmoid/threshold (docs/data_audit.md: thresholding first loses IoU).
    """
    import torch
    import torch.nn.functional as F

    up = F.interpolate(logits_s[:, None], size=(CROP, CROP), mode="bilinear", align_corners=False)[:, 0]
    p = torch.sigmoid(up)
    full = torch.zeros((p.shape[0], H, W), dtype=torch.float32, device=p.device)
    full[:, CROP0:CROP1, CROP0:CROP1] = p
    return full.cpu().numpy()


@dataclass(frozen=True)
class PostCfg:
    thr: float = 0.5           # extent threshold (component = connected region with p > thr)
    min_area: int = 0          # drop components smaller than this (px, full resolution)
    connectivity: int = 8
    seed_thr: float | None = None  # hysteresis: keep a component only if max p within it > seed_thr


def prob_to_instances(prob: np.ndarray, cfg: PostCfg = PostCfg()) -> InstanceSet:
    """Threshold -> connected components -> (optional hysteresis filter) -> InstanceSet (disjoint)."""
    fg = (prob > cfg.thr).astype(np.uint8)
    n, lab, stats = cv2.connectedComponentsWithStats(fg, connectivity=cfg.connectivity)[:3]
    return labels_to_instances(n, lab, stats, min_area=cfg.min_area, shape=prob.shape,
                               prob=prob if cfg.seed_thr is not None else None, seed_thr=cfg.seed_thr)


def labels_to_instances(n, lab, stats, min_area: int = 0, shape=(H, W), prob=None, seed_thr=None) -> InstanceSet:
    """cv2.connectedComponentsWithStats output -> InstanceSet, using the stats bboxes (fast).

    With prob/seed_thr: hysteresis -- a component is kept only if its maximum probability exceeds
    seed_thr (existence decided at a high threshold, extent at the lower one). No pixels are added,
    so separate components are never joined.
    """
    boxes, crops = [], []
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if a < max(min_area, 1):
            continue
        crop = lab[y:y + h, x:x + w] == k
        if prob is not None and seed_thr is not None and not (prob[y:y + h, x:x + w][crop].max() > seed_thr):
            continue
        boxes.append((y, x, y + h, x + w))
        crops.append(crop)
    return InstanceSet._build(shape, boxes, crops)
