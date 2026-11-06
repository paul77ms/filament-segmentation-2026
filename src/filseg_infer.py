"""Stage 1 of the final pipeline: the 5-fold U-Net (ResNet-101 encoder), image I/O, the stage-1 probability map,
connected components and the COCO RLE writer. Self-contained.

Dependencies: torch, torchvision, numpy, opencv-python. No segmentation_models_pytorch and no pycocotools needed.
The model class reproduces segmentation_models_pytorch 0.6.0 Unet(resnet101, in_channels=1, classes=1) exactly
(identical state_dict keys). `in_channels` (default 1) only sets the input channels of the first convolution: the
refinement model R2_final uses the same U-Net with a 10-channel first convolution (src/filseg_infer_r2b0.py).

Stage 1 (used by filseg_infer_r2b0.stage1):
  JPEG (2048x2048, 1 channel) -> crop [120:1928]^2 -> resize to 1024 (INTER_AREA) -> (x/255 - 0.30)/0.22
  -> each fold model -> logits upsampled bilinearly to 1808^2 (before the sigmoid) -> sigmoid -> pasted into a zero
  2048^2 canvas -> mean over the 5 models (stored as float16).
Masks are written as COCO compressed RLE counts (column-major, no size); filament_id = '<image stem>_<k>'.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CROP0, CROP1 = 120, 1928
CROP = CROP1 - CROP0
H = W = 2048
SIZE = 1024
THR, MIN_AREA = 0.5, 500


# ----------------------------------------------------------------------------- model (smp 0.6.0 Unet equivalent)
class ResNetEncoder(torchvision.models.resnet.ResNet):
    def __init__(self, in_channels=1):
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])  # resnet101
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        del self.fc
        del self.avgpool

    def forward(self, x):
        feats = [x]
        x = self.relu(self.bn1(self.conv1(x)))
        feats.append(x)
        x = self.layer1(self.maxpool(x))
        feats.append(x)
        x = self.layer2(x); feats.append(x)
        x = self.layer3(x); feats.append(x)
        x = self.layer4(x); feats.append(x)
        return feats


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv1 = conv_bn_relu(cin + cskip, cout)
        self.conv2 = conv_bn_relu(cout, cout)

    def forward(self, x, h, w, skip=None):
        x = F.interpolate(x, size=(h, w), mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels=(1, 64, 256, 512, 1024, 2048), decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        enc = list(encoder_channels[1:])[::-1]
        in_ch = [enc[0]] + list(decoder_channels[:-1])
        skip_ch = list(enc[1:]) + [0]
        self.blocks = nn.ModuleList([DecoderBlock(i, s, o) for i, s, o in zip(in_ch, skip_ch, decoder_channels)])

    def forward(self, feats):
        shapes = [f.shape[2:] for f in feats][::-1]
        feats = feats[1:][::-1]
        x, skips = feats[0], feats[1:]
        for i, blk in enumerate(self.blocks):
            h, w = shapes[i + 1]
            x = blk(x, h, w, skips[i] if i < len(skips) else None)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels)
        self.decoder = UnetDecoder()
        self.segmentation_head = nn.Sequential(nn.Conv2d(16, 1, kernel_size=3, padding=1))

    def forward(self, x):
        return self.segmentation_head(self.decoder(self.encoder(x)))


def load_models(weight_paths, device, in_channels=1):
    models = []
    for p in weight_paths:
        m = Unet(in_channels)
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True)
        models.append(m.to(device).eval())
    return models


# ----------------------------------------------------------------------------- pre/post-processing
def read_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise IOError(f"cannot decode {path}")
    if img.ndim == 3:
        img = img[..., 0]
    if img.shape != (H, W):
        raise ValueError(f"{path}: expected {H}x{W}, got {img.shape}")
    return img


def to_model_space(img):
    return cv2.resize(img[CROP0:CROP1, CROP0:CROP1], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def normalize(xb_uint8):
    return (xb_uint8.float() / 255.0 - 0.30) / 0.22


def logits_to_full_prob(logits):
    up = F.interpolate(logits[:, None], size=(CROP, CROP), mode="bilinear", align_corners=False)[:, 0]
    p = torch.sigmoid(up)
    full = torch.zeros((p.shape[0], H, W), dtype=torch.float32, device=p.device)
    full[:, CROP0:CROP1, CROP0:CROP1] = p
    return full.cpu().numpy()


def components(prob, thr=THR, min_area=MIN_AREA):
    """8-connected components of prob > thr with >= min_area px, in label order -> list of (bbox, crop mask)."""
    n, lab, stats = cv2.connectedComponentsWithStats((prob > thr).astype(np.uint8), connectivity=8)[:3]
    out = []
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if a < max(min_area, 1):
            continue
        out.append(((y, x, y + h, x + w), lab[y:y + h, x:x + w] == k))
    return out


# ----------------------------------------------------------------------------- RLE (pure numpy/python COCO format)
def runs_from_mask(mask):
    flat = (np.asarray(mask) != 0).ravel(order="F").astype(np.int8)
    change = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate([[0], change, [flat.size]])).tolist()
    if flat.size and flat[0] == 1:
        runs = [0] + runs
    return [int(r) for r in runs]


def counts_to_string(runs):
    out = []
    for i, x in enumerate(runs):
        x = int(x)
        if i > 2:
            x -= int(runs[i - 2])
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def string_to_counts(s):
    cnts, p, b = [], 0, s.encode("ascii")
    while p < len(b):
        x, k, more = 0, 0, True
        while more:
            c = b[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1; k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def mask_from_counts(s, h=H, w=W):
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for r in string_to_counts(s):
        if val:
            flat[pos:pos + r] = 1
        pos += r; val ^= 1
    assert pos == h * w, "RLE length mismatch"
    return flat.reshape((h, w), order="F")


def encode(mask):
    return counts_to_string(runs_from_mask(mask))


# ----------------------------------------------------------------------------- end to end
