"""SE2 stage of the final pipeline: existence scoring of the small fragments that the 500 px existence gate
discards, after R2_final has redrawn them at native resolution (add-only). Self-contained.

Dependencies: torch, torchvision, numpy, opencv-python (filseg_infer_r2b0 supplies the stage-1 map, the candidates and
the R2 refinement).
  fragments  candidates (p > 0.3, >= 20 px) that overlap a small component of p > 0.5 (100-499 px) and do not touch the
             gate (components of p > 0.5 with >= 500 px); redrawn by R2_final and assembled (>= 100 px)
  scorer     4-channel 96x96 native crop around the fragment's box centre (H-alpha image, rint(p16 * 255), redrawn
             fragment, gate mask) -> ResNet-101 stem + layer1 + layer2 -> global average pool -> linear -> sigmoid;
             trained to predict the fragment's PQ gain averaged over the annotation sets
  decision   add iff score > SE2_TAU and the fragment has zero pixel overlap with the image's existing masks
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision

H = W = 2048
THR, A_LO, A_HI, CROP = 0.5, 100, 500, 96
# SE2_TAU = 0.6 x 0.2124. 0.2124 = PQ/2 of the pipeline before this stage (out-of-fold, training data): the exact
# break-even of the official per-annotation-set metric. m = 0.6 maximises the leaderboard-type PQ (annotation sets of
# an image pooled) over the nested out-of-fold predictions of all 707 training images and is the leave-one-outer-fold-
# out choice in 4 of 5 folds (scripts/se2_analysis.py select, records/se2/analysis/threshold_v1.json). No test data.
SE2_TAU = 0.12744


def build_scorer() -> torch.nn.Module:
    m = torchvision.models.resnet101(weights=None)
    conv = torch.nn.Conv2d(4, 64, 7, 2, 3, bias=False)
    return torch.nn.Sequential(conv, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2,
                               torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(512, 1))


def load_scorer(path, device) -> torch.nn.Module:
    net = build_scorer()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.to(device).eval()


def _crop_at(arr, cy, cx, r=CROP // 2):
    out = np.zeros((2 * r, 2 * r), arr.dtype)
    y0, x0 = cy - r, cx - r
    sy0, sx0, sy1, sx1 = max(y0, 0), max(x0, 0), min(y0 + 2 * r, H), min(x0 + 2 * r, W)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def to_input(X):
    x = torch.from_numpy(X.astype(np.float32) / 255.0)
    x[:, 0] = (x[:, 0] - 0.30) / 0.22
    return x


@torch.no_grad()
def scores(net, X, device):
    if not len(X):
        return np.zeros(0)
    return torch.sigmoid(net(to_input(X).to(device)).squeeze(1)).float().cpu().numpy()


# ----------------------------------------------------------------------------- SE2 (refined fragments)
def fragment_candidates(p16, cmap, kept):
    """SE2: candidate ids (p > 0.3, >= 20 px, from filseg_infer_r2b0.candidates) that overlap a small component
    (p16 > 0.5, 100-499 px) and do not touch the >= 500 px gate (`kept`), and the gate mask. Port of
    scripts/se2.py fragments() (validated by the nested evaluation SE2_refined_existence)."""
    p = p16.astype(np.float32)
    n, lab, st, _ = cv2.connectedComponentsWithStats((p > THR).astype(np.uint8), connectivity=8)
    area = st[:, cv2.CC_STAT_AREA]
    small = np.zeros(n, bool)
    small[1:] = (area[1:] >= A_LO) & (area[1:] < A_HI)
    big = np.zeros(n, bool)
    big[1:] = area[1:] >= A_HI
    js = set(np.unique(cmap[small[lab]]).tolist()) - {0} - set(kept)
    return sorted(js), big[lab]


def crops_refined(frags, img_u8, p16, gate):
    """4-channel 96x96 crops (H-alpha, rint(p16*255), refined mask, gate mask) at each refined instance's box centre."""
    pu8 = np.clip(np.rint(p16.astype(np.float32) * 255), 0, 255).astype(np.uint8)
    g8 = gate.astype(np.uint8) * 255
    X = []
    for (y0, x0, y1, x1), c in frags:
        m = np.zeros((H, W), np.uint8)
        m[y0:y1, x0:x1] = c.astype(np.uint8) * 255
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        X.append(np.stack([_crop_at(img_u8, cy, cx), _crop_at(pu8, cy, cx), _crop_at(m, cy, cx), _crop_at(g8, cy, cx)]))
    return np.stack(X) if X else np.zeros((0, 4, CROP, CROP), np.uint8)


def additions_refined(net, frags, img_u8, p16, gate, cover, tau, device):
    """SE2 decision on refined fragments (list of (bbox, crop), disjoint): add iff score > tau and zero overlap with
    `cover` (updated in place). Returns the added full-resolution masks and all scores."""
    sc = scores(net, crops_refined(frags, img_u8, p16, gate), device)
    out = []
    for q in np.flatnonzero(sc > tau):
        (y0, x0, y1, x1), c = frags[int(q)]
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = c
        if (cover & m).any():
            continue
        out.append(m)
        cover |= m
    return out, sc
