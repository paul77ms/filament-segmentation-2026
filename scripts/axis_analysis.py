"""Central axis of the predicted filaments vs the annotators' spines (report analysis; CPU only, no pipeline change).

Axis of a mask = the longest geodesic path of its 8-connected skeleton (skimage skeletonize; two breadth-first
passes on the skeleton graph), as in the "Filament2D" recipe of MosaicSuite (re-implemented, no code copied).
For every GT filament matched by the final pipeline (nested OOF, IoU > 0.5, one-to-one per record):
  d_mean  symmetric mean distance (px, 2048 grid) between the predicted axis and the annotator's spine polyline
  cover   fraction of the spine within 3 px of the predicted axis; len_ratio = axis length / spine length
Human reference: the same numbers between two annotators' spines of the same filament (matched masks, IoU > 0.5,
every ordered pair of sets of a multi-set image). Also: the annotator's spine vs the axis computed from the
annotator's OWN mask (how well "skeleton longest path" reproduces a hand-drawn spine).
  CUDA_VISIBLE_DEVICES="" taskset -c 0 .venv/bin/python scripts/axis_analysis.py run
  ... figure -> outputs/report/axis/{summary.json, rows.csv, fig_axis.png}
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs" / "report" / "axis"


def axis_of(crop):
    """Longest geodesic path of the 8-connected skeleton of a bool crop -> (N, 2) array of (y, x) or None."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, shortest_path
    from skimage.morphology import skeletonize
    sk = skeletonize(crop)
    ys, xs = np.nonzero(sk)
    if len(ys) < 2:
        return np.stack([ys, xs], 1).astype(np.float32) if len(ys) else None
    idx = -np.ones(sk.shape, np.int64); idx[ys, xs] = np.arange(len(ys))
    rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < sk.shape[0]) & (x2 >= 0) & (x2 < sk.shape[1])
        j = np.full(len(ys), -1); j[ok] = idx[y2[ok], x2[ok]]
        m = j >= 0
        rows += list(np.flatnonzero(m)); cols += list(j[m]); w += [np.hypot(dy, dx)] * int(m.sum())
    n = len(ys)
    G = coo_matrix((w + w, (rows + cols, cols + rows)), shape=(n, n)).tocsr()
    d0 = shortest_path(G, indices=0, directed=False)
    d0[~np.isfinite(d0)] = -1
    a = int(np.argmax(d0))
    da, pred = shortest_path(G, indices=a, directed=False, return_predecessors=True)
    da[~np.isfinite(da)] = -1
    b = int(np.argmax(da))
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return np.stack([ys[path], xs[path]], 1).astype(np.float32)


def _dense(poly, step=1.0):
    """Resample a polyline (N, 2) at ~1 px spacing."""
    if poly is None or len(poly) == 0:
        return None
    if len(poly) == 1:
        return poly
    seg = np.diff(poly, axis=0); L = np.hypot(seg[:, 0], seg[:, 1])
    out = [poly[:1]]
    for p, s, l in zip(poly[:-1], seg, L):
        k = max(int(np.ceil(l / step)), 1)
        out.append(p + s * (np.arange(1, k + 1)[:, None] / k))
    return np.concatenate(out)


def compare(a, b):
    """a, b: dense polylines -> (symmetric mean distance, fraction of b within 3 px of a, length(a)/length(b))."""
    from scipy.spatial import cKDTree
    ta, tb = cKDTree(a), cKDTree(b)
    dab = ta.query(b)[0]; dba = tb.query(a)[0]
    length = lambda p: float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0  # noqa: E731
    return float((dab.mean() + dba.mean()) / 2), float((dab <= 3).mean()), length(a) / max(length(b), 1e-6)


_COCO = {}


def coco_spines():
    if not _COCO:
        from filseg import data as D
        c = D.load_coco()
        by = {}
        for a in c["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        _COCO["by"] = by
    return _COCO["by"]


def job(args):
    import a_eval as AE
    from filseg.scorers import iou_matrix
    stem, fold, gts = args
    by = coco_spines()
    P = AE.final_instances(stem, fold)
    pax = {}
    rows = []
    sp = {}
    for rid, G in gts.items():
        anns = by[rid]                              # same order as InstanceSet.from_coco_anns (checked by area)
        assert len(anns) == len(G)
        sp[rid] = [_dense(np.array(a["spine"], np.float32).reshape(-1, 2)[:, ::-1]) for a in anns]   # (y, x)
        inter, iou = iou_matrix(G, P)
        for i in range(len(G)):
            gy0, gx0 = G.boxes[i][:2]
            own = axis_of(G.crops[i])
            row = dict(rid=rid, stem=stem, gt_idx=i, area=int(G.areas[i]), kind="model")
            if own is not None and sp[rid][i] is not None:
                d, c, l = compare(_dense(own + [gy0, gx0]), sp[rid][i])
                row.update(own_d=d, own_cover=c, own_len=l)
            if len(P) and iou[i].max() > 0.5:
                j = int(iou[i].argmax())
                if j not in pax:
                    y0, x0 = P.boxes[j][:2]
                    ax = axis_of(P.crops[j]); pax[j] = None if ax is None else _dense(ax + [y0, x0])
                if pax[j] is not None and sp[rid][i] is not None:
                    d, c, l = compare(pax[j], sp[rid][i])
                    row.update(pred_d=d, pred_cover=c, pred_len=l, iou=float(iou[i, j]))
            rows.append(row)
    rids = sorted(gts)
    for A in rids:                                   # human reference: annotator A's spine vs annotator R's spine
        for Rr in rids:
            if A == Rr:
                continue
            _, iou = iou_matrix(gts[Rr], gts[A])
            for i in range(len(gts[Rr])):
                if len(gts[A]) and iou[i].max() > 0.5:
                    j = int(iou[i].argmax())
                    if sp[A][j] is not None and sp[Rr][i] is not None:
                        d, c, l = compare(sp[A][j], sp[Rr][i])
                        rows.append(dict(rid=Rr, stem=stem, gt_idx=i, kind="human", pred_d=d, pred_cover=c, pred_len=l,
                                         iou=float(iou[i, j])))
    return rows


def cmd_run(a):
    import a_eval as AE
    from filseg.metrics import single_thread_worker
    single_thread_worker()
    jobs = AE.all_jobs()[:a.limit or None]
    t0, rows = time.time(), []
    for k, j in enumerate(jobs):
        rows += job(j)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / ("rows_test.csv" if a.limit else "rows.csv"), index=False)
    m, h = df[(df.kind == "model") & df.pred_d.notna()], df[df.kind == "human"]
    q = lambda s: dict(median=float(s.median()), p25=float(s.quantile(.25)), p75=float(s.quantile(.75)))  # noqa: E731
    summ = dict(n_model=int(len(m)), n_human=int(len(h)),
                model_vs_spine=dict(d=q(m.pred_d), cover3=q(m.pred_cover), len_ratio=q(m.pred_len)),
                human_vs_human=dict(d=q(h.pred_d), cover3=q(h.pred_cover), len_ratio=q(h.pred_len)),
                own_mask_axis_vs_spine=dict(d=q(df.own_d.dropna()), cover3=q(df.own_cover.dropna()), len_ratio=q(df.own_len.dropna())))
    json.dump(summ, open(OUT / ("summary_test.json" if a.limit else "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"run": cmd_run}[a.cmd](a)
