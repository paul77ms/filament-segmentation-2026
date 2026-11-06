"""Randomised adversarial comparison: brute force (bf_pq) vs filseg.metrics vs the verbatim
host self-evaluation notebook functions (third-party copy, torch CPU)."""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repository root (path-independent)

import math
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import cv2
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, _REPO + "/src")
torch.set_num_threads(1)
cv2.setNumThreads(1)

import bf_pq  # noqa: E402
from filseg.metrics import EvalConfig, InstanceSet, evaluate_dataset, evaluate_image, match_instances  # noqa: E402

NB_PATH = _REPO + "/docs/thirdparty_metric/ooLemon0-0_self-evaluation-notebook_cells.py"


def load_notebook():
    ns: dict = {}
    exec(compile(open(NB_PATH).read(), NB_PATH, "exec"), ns)
    return ns


SHAPES = [(1, 1), (1, 9), (9, 1), (16, 16), (23, 41), (64, 64), (57, 130), (128, 128), (200, 77)]


def gen_instance(rng, shape):
    h, w = shape
    k = rng.integers(0, 9)
    m = np.zeros(shape, np.uint8)
    if k == 0:                                  # rectangle, sometimes clamped to borders
        y0, y1 = sorted(rng.integers(0, h + 1, 2)); x0, x1 = sorted(rng.integers(0, w + 1, 2))
        if rng.random() < 0.3: y0 = 0
        if rng.random() < 0.3: x1 = w
        m[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = 1
    elif k == 1:                                # thin polyline (filament-like)
        pts = np.stack([rng.integers(-2, w + 2, 4), rng.integers(-2, h + 2, 4)], 1).astype(np.int32)
        cv2.polylines(m, [pts], False, 1, int(rng.integers(1, 4)))
    elif k == 2:                                # ellipse
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        ax = (int(rng.integers(1, max(2, w // 2))), int(rng.integers(1, max(2, h // 2))))
        cv2.ellipse(m, c, ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
    elif k == 3:                                # noisy blob
        y0, y1 = sorted(rng.integers(0, h + 1, 2)); x0, x1 = sorted(rng.integers(0, w + 1, 2))
        m[y0:y1 + 1, x0:x1 + 1] = 1
        m &= (rng.random(shape) < 0.7).astype(np.uint8)
    elif k == 4:                                # single pixel (incl. corners)
        m[int(rng.choice([0, h - 1, rng.integers(0, h)])), int(rng.choice([0, w - 1, rng.integers(0, w)]))] = 1
    elif k == 5:                                # full image
        m[:] = 1
    elif k == 6:                                # empty
        pass
    elif k == 7:                                # column-wrapping run: bottom of col c, top of col c+1
        c = int(rng.integers(0, max(1, w - 1)))
        m[h - min(h, 2):, c] = 1
        if c + 1 < w: m[:min(h, 2), c + 1] = 1
    else:                                       # random sparse
        m[rng.random(shape) < rng.uniform(0.01, 0.3)] = 1
    return m.astype(bool)


def flat_pixels(m):
    return np.flatnonzero(m.ravel())


def derive_preds(rng, gts, shape):
    h, w = shape
    preds = []
    for g in gts:
        r = rng.random()
        if r < 0.15:
            preds.append(g.copy())
        elif r < 0.25:                          # duplicate
            preds += [g.copy(), g.copy()]
        elif r < 0.35:                          # shift
            dy, dx = rng.integers(-3, 4, 2)
            s = np.zeros_like(g)
            ys, xs = np.nonzero(g)
            ys, xs = ys + dy, xs + dx
            ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            s[ys[ok], xs[ok]] = True
            preds.append(s)
        elif r < 0.45:                          # dilate / erode
            k = np.ones((3, 3), np.uint8)
            op = cv2.dilate if rng.random() < 0.5 else cv2.erode
            preds.append(op(g.astype(np.uint8), k).astype(bool))
        elif r < 0.55:                          # split into 2-3 pieces along a random order
            px = flat_pixels(g)
            if px.size >= 2:
                cuts = np.sort(rng.integers(1, px.size, int(rng.integers(1, 3))))
                for piece in np.split(px, cuts):
                    q = np.zeros(h * w, bool); q[piece] = True; preds.append(q.reshape(shape))
        elif r < 0.65:                          # exact IoU = 0.5 constructs and +-1 px
            px = flat_pixels(g)
            a = px.size
            if a >= 2:
                mode = rng.integers(0, 4)
                q = np.zeros(h * w, bool)
                if mode == 0 and a % 2 == 0:    # subset with half the pixels -> IoU exactly 0.5
                    q[px[: a // 2]] = True
                elif mode == 1:                 # subset with ceil(a/2)+... -> just above / at 0.5
                    q[px[: a // 2 + 1]] = True
                else:                           # superset with |g| (or |g|-1) extra pixels outside
                    out = np.setdiff1d(np.arange(h * w), px)
                    extra = a if mode == 2 else a - 1
                    if out.size >= extra:
                        q[px] = True; q[out[:extra]] = True
                    else:
                        q[px] = True
                preds.append(q.reshape(shape))
        elif r < 0.7:
            preds.append(np.zeros(shape, bool))  # empty predicted mask
        # else: missed (no pred)
    # merges of two gts
    if len(gts) >= 2 and rng.random() < 0.3:
        i, j = rng.choice(len(gts), 2, replace=False)
        preds.append(gts[i] | gts[j])
    # pure false positives
    for _ in range(int(rng.integers(0, 3))):
        preds.append(gen_instance(rng, shape))
    rng.shuffle(preds)
    return preds


def make_cases(n, seed=0):
    rng = np.random.default_rng(seed)
    cases = []
    for c in range(n):
        shape = SHAPES[int(rng.integers(0, len(SHAPES)))]
        ng = int(rng.integers(0, 6))
        gts = [gen_instance(rng, shape) for _ in range(ng)]
        if rng.random() < 0.1:
            preds = []
        else:
            preds = derive_preds(rng, gts, shape)[:10]
        if rng.random() < 0.05:
            gts = []
        cases.append((f"c{c:04d}", shape, gts, preds))
    return cases


def host_record(nb, gts, preds, shape):
    H, W = shape
    g = torch.from_numpy(np.stack(gts).astype(np.float32)) if gts else torch.zeros((0, H, W))
    p = torch.from_numpy(np.stack(preds).astype(np.float32)) if preds else torch.zeros((0, H, W))
    iou, dice = nb["get_overlap_matrices"](g, p)
    return iou, len(gts), len(preds)


def main(n=1500):
    t0 = time.time()
    nb = load_notebook()
    cases = make_cases(n)
    cfg_full = EvalConfig()
    keys_int = ["tp", "fp", "fn", "tp_host", "fp_host", "fn_host"]
    keys_flt = ["sum_iou", "pq", "sq", "rq", "pq_empty1", "sum_iou_host", "pq_host"]
    mism, recs_bf, items, host_rows = [], [], [], []
    stats = dict(n_cases=len(cases), n_iou_exact_half=0, cases_with_half=0, cases_nonunique=0,
                 card_ties=0, empty_gt=0, empty_pred=0, both_empty=0, max_iou_err=0.0, n_pairs=0)
    for key, shape, gts, preds in cases:
        G = [bf_pq.pixset(m) for m in gts]
        P = [bf_pq.pixset(m) for m in preds]
        b = bf_pq.bf_record(G, P)
        recs_bf.append(b)
        row, pg, pp = evaluate_image(gts, preds, cfg_full, image_key=key, shape=shape)
        mr = match_instances(gts, preds, shape=shape)
        # IoU matrix: must be bit-identical (same integer inter / union division)
        if b["n_gt"] and b["n_pred"]:
            T = np.asarray(b["iou"], float)
            err = float(np.abs(T - mr.iou).max())
            stats["max_iou_err"] = max(stats["max_iou_err"], err)
            stats["n_pairs"] += T.size
            if not np.array_equal(T, mr.iou):
                mism.append((key, "iou_matrix", err))
        for k in keys_int:
            if row[k] != b[k]:
                mism.append((key, k, row[k], b[k]))
        for k in keys_flt:
            x, y = row[k], b[k]
            if not ((math.isnan(x) and math.isnan(y)) or abs(x - y) <= 1e-9):
                mism.append((key, k, x, y))
        stats["n_iou_exact_half"] += b["n_iou_exact_half"]
        stats["cases_with_half"] += b["n_iou_exact_half"] > 0
        stats["cases_nonunique"] += not row["unique_by_construction"]
        stats["card_ties"] += b["card_tie"]
        stats["empty_gt"] += b["n_gt"] == 0
        stats["empty_pred"] += b["n_pred"] == 0
        stats["both_empty"] += b["n_gt"] == 0 and b["n_pred"] == 0
        items.append((key, InstanceSet.from_masks(gts, shape=shape), InstanceSet.from_masks(preds, shape=shape)))
        # verbatim host notebook (float32 torch)
        iou_h, ng, npr = host_record(nb, gts, preds, shape)
        host_rows.append(dict(annotator_image=key, iou_matrix=iou_h, dice_matrix=None, n_gt=ng, n_pred=npr))
        hit = (iou_h > 0.5) if (ng and npr) else None
        if ng and npr:
            tp_nb = int(hit.sum()); fp_nb = nb["fp_count_hit"](hit); fn_nb = nb["fn_count_hit"](hit)
        else:
            tp_nb, fp_nb, fn_nb = 0, (npr if ng == 0 else 0), (ng if npr == 0 else 0)
        if (tp_nb, fp_nb, fn_nb) != (b["tp_host"], b["fp_host"], b["fn_host"]):
            mism.append((key, "notebook_counts", (tp_nb, fp_nb, fn_nb), (b["tp_host"], b["fp_host"], b["fn_host"])))
    # ---- dataset aggregation
    s_m1, df = evaluate_dataset(items, cfg_full)
    s_bf = bf_pq.bf_aggregate(recs_bf)
    agg_mism = []
    for k, v in s_bf.items():
        x = s_m1[k]
        if isinstance(v, float):
            if not ((math.isnan(x) and math.isnan(v)) or abs(x - v) <= 1e-9):
                agg_mism.append((k, x, v))
        elif x != v:
            agg_mism.append((k, x, v))
    # parallel path equals serial
    s_par, df_par = evaluate_dataset(items, cfg_full, n_workers=4)
    par_equal = all((isinstance(v, float) and ((math.isnan(v) and math.isnan(s_par[k])) or abs(v - s_par[k]) <= 1e-12))
                    or (not isinstance(v, float) and v == s_par[k]) for k, v in s_m1.items() if k != "config")
    nb_pq = nb["get_pq_score"](pd.DataFrame(host_rows))
    out = dict(stats=stats, n_record_mismatches=len(mism), first_mismatches=mism[:10],
               aggregate_mismatches=agg_mism, parallel_equals_serial=par_equal,
               bf_aggregate=s_bf, m1_aggregate={k: s_m1[k] for k in s_bf},
               notebook_PQ=nb_pq, notebook_minus_bf_host=nb_pq - s_bf["PQ_pooled_host"],
               seconds=time.time() - t0)
    return out


if __name__ == "__main__":
    import json
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    res = main(n)
    print(json.dumps(res, indent=1, default=str))
