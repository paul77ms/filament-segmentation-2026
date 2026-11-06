"""Real-data adversarial check (CPU only).

Per stem (all 707 train files):
  * RLE: polygon -> pycocotools dense mask (independent of filseg) -> filseg.rle.encode must equal
    pycocotools.encode(np.asfortranarray(mask)) byte-for-byte; filseg.rle.decode must invert it;
    a C-order (row-major) decode of the same string must FAIL to reproduce the mask (sensitivity);
    rle_to_crop and InstanceSet.from_coco_anns crops must equal dense tight crops; area == JSON area.
  * PQ: every annotation set vs itself and every ordered pair (A as GT, B as pred) of different sets
    of the same file: brute force (python pixel sets) vs filseg.metrics.evaluate_image vs the verbatim
    host notebook functions (torch CPU, float32).
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repository root (path-independent)

import json
import math
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, _REPO + "/src")
import bf_pq  # noqa: E402

ANN = _REPO + "/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
NB_PATH = _REPO + "/docs/thirdparty_metric/ooLemon0-0_self-evaluation-notebook_cells.py"
H = W = 2048
NB = None


def init():
    global NB
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    NB = {}
    exec(compile(open(NB_PATH).read(), NB_PATH, "exec"), NB)


def c_order_decode(s, h, w):
    """Deliberately WRONG decoder (row-major) used only to prove the checks are orientation-sensitive."""
    from filseg.rle import string_to_counts
    runs = string_to_counts(s)
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for r in runs:
        if val:
            flat[pos:pos + r] = 1
        pos += r
        val ^= 1
    return flat.reshape((h, w), order="C")


def tight(m):
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if rows.size == 0:
        return (0, 0, 0, 0), np.zeros((0, 0), bool)
    b = (int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1)
    return b, m[b[0]:b[2], b[1]:b[3]].astype(bool)


def nb_host(gt_rid, gt_strs, pred_stem, pred_strs):
    """Run the verbatim notebook path (get_overlap_df on submission-format DataFrames)."""
    gt_df = pd.DataFrame({"filament_id": [f"{gt_rid}_{k}" for k in range(len(gt_strs))],
                          "segmentation_rle": gt_strs})
    pred_df = pd.DataFrame({"filament_id": [f"{pred_stem}_{k}" for k in range(len(pred_strs))],
                            "segmentation_rle": pred_strs}) if pred_strs else \
        pd.DataFrame({"filament_id": pd.Series([], dtype=str), "segmentation_rle": pd.Series([], dtype=str)})
    od = NB["get_overlap_df"](gt_df, pred_df)
    assert len(od) == 1
    r = od.iloc[0]
    iou = r["iou_matrix"]
    ng, npr = int(r["n_gt"]), int(r["n_pred"])
    if ng and npr:
        hit = iou > 0.5
        tp = int(hit.sum()); fp = NB["fp_count_hit"](hit); fn = NB["fn_count_hit"](hit)
        s = float(sum(iou[hit].tolist()))
    else:
        tp, fp, fn, s = 0, (npr if ng == 0 else 0), (ng if npr == 0 else 0), 0.0
    return dict(iou_matrix=iou, n_gt=ng, n_pred=npr), (tp, fp, fn, s)


def job(args):
    from filseg import rle
    from filseg.metrics import EvalConfig, InstanceSet, evaluate_image
    stem, recs = args                       # recs: {record_id: [anns]}
    t0 = time.time()
    rle_rows, pq_rows, nb_rows = [], [], []
    pix, strs, isets = {}, {}, {}
    for rid, anns in recs.items():
        P, S = [], []
        for a in anns:
            r = mask_util.merge(mask_util.frPyObjects(a["segmentation"], H, W))
            m = mask_util.decode(r)                                 # independent dense mask
            ref = mask_util.encode(np.asfortranarray(m))["counts"].decode("ascii")
            s = rle.encode(m)
            dec = rle.decode(s, H, W)
            cdec = c_order_decode(s, H, W)
            b, c = tight(m)
            cb, cc = rle.rle_to_crop(s, H, W)
            rle_rows.append(dict(ann_id=a["id"], record=rid, area=int(m.sum()), area_json=float(a["area"]),
                                 enc_eq_pycoco_fortran=(s == ref), enc_eq_polygon_rle=(s == r["counts"].decode("ascii")),
                                 roundtrip=bool(np.array_equal(dec, m)),
                                 c_order_decode_differs=not np.array_equal(cdec, m),
                                 transpose_differs=not np.array_equal(m.T, m),
                                 rle_to_crop_eq=(cb == b and np.array_equal(cc, c)),
                                 touches_border=bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())))
            P.append(bf_pq.pixset(m))
            S.append(s)
            del m, dec, cdec
        pix[rid], strs[rid] = P, S
        iset = InstanceSet.from_coco_anns(anns)
        # M1 crop path == dense tight crops
        ok = True
        for k, a in enumerate(anns):
            m = mask_util.decode(mask_util.merge(mask_util.frPyObjects(a["segmentation"], H, W)))
            b, c = tight(m)
            ok &= tuple(int(v) for v in iset.boxes[k]) == b and np.array_equal(iset.crops[k], c)
        rle_rows.append(dict(record=rid, iset_crops_eq_dense=ok))
        isets[rid] = iset
    cfg = EvalConfig()
    ids = sorted(recs)
    for a in ids:
        for b in ids:
            kind = "self" if a == b else "pair"
            br = bf_pq.bf_record(pix[a], pix[b])
            bd = bf_pq.bf_diag(pix[a], pix[b])
            row, _, _ = evaluate_image(isets[a], isets[b], cfg, image_key=f"{a}|{b}")
            nbr, (tp, fp, fn, s) = nb_host(a, strs[a], stem, strs[b])
            out = dict(key=f"{a}|{b}", kind=kind, stem=stem, n_gt=br["n_gt"], n_pred=br["n_pred"],
                       nb_tp=tp, nb_fp=fp, nb_fn=fn, nb_sum_iou=s)
            for k in ("tp", "fp", "fn", "sum_iou", "pq", "sq", "rq", "pq_empty1",
                      "tp_host", "fp_host", "fn_host", "sum_iou_host", "pq_host", "card_tie", "n_iou_exact_half"):
                out[f"bf_{k}"] = br[k]
            for k, v in bd.items():
                out[f"bfd_{k}"] = v
            for k in ("tp", "fp", "fn", "sum_iou", "pq", "sq", "rq", "pq_empty1", "tp_host", "fp_host", "fn_host",
                      "sum_iou_host", "pq_host", "unique_by_construction") + tuple(bd):
                out[f"m1_{k}"] = row[k]
            pq_rows.append(out)
            nbr["kind"] = kind
            nb_rows.append(nbr)
    return stem, rle_rows, pq_rows, nb_rows, time.time() - t0


def main(n_workers=12):
    import multiprocessing as mp
    t0 = time.time()
    d = json.load(open(ANN))
    by_rec = {im["id"]: [] for im in d["images"]}
    for a in d["annotations"]:
        by_rec[a["image_id"]].append(a)
    by_stem: dict = {}
    for rid, anns in by_rec.items():
        by_stem.setdefault(rid.split("-", 1)[1], {})[rid] = anns
    jobs = sorted(by_stem.items(), key=lambda kv: -sum(len(v) for v in kv[1].values()) * len(kv[1]))
    rle_rows, pq_rows, nb_rows, times = [], [], [], []
    with mp.get_context("fork").Pool(n_workers, initializer=init) as pool:
        for i, (stem, a, b, c, dt) in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            rle_rows += a; pq_rows += b; nb_rows += c; times.append(dt)
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} stems, {time.time() - t0:.0f}s", flush=True)
    init()
    out_dir = sys.argv[1] if len(sys.argv) > 1 else HERE
    R = pd.DataFrame(rle_rows)
    Q = pd.DataFrame(pq_rows)
    R.to_csv(os.path.join(out_dir, "mv_real_rle_checks.csv"), index=False)
    Q.to_csv(os.path.join(out_dir, "mv_real_pq_records.csv"), index=False)
    res = dict(wall_s=time.time() - t0, n_stems=len(jobs), job_s_median=float(np.median(times)), job_s_max=float(max(times)))
    ann = R[R.ann_id.notna()]
    res["rle"] = {k: f"{int(ann[k].sum())}/{len(ann)}" for k in
                  ("enc_eq_pycoco_fortran", "enc_eq_polygon_rle", "roundtrip", "c_order_decode_differs",
                   "transpose_differs", "rle_to_crop_eq", "touches_border")}
    res["rle"]["area_eq_json"] = f"{int((ann.area == ann.area_json).sum())}/{len(ann)}"
    recs = R[R.iset_crops_eq_dense.notna()]
    res["rle"]["iset_crops_eq_dense_records"] = f"{int(recs.iset_crops_eq_dense.astype(bool).sum())}/{len(recs)}"
    for kind in ("self", "pair"):
        q = Q[Q.kind == kind]
        mism = {}
        for k in ("tp", "fp", "fn", "tp_host", "fp_host", "fn_host") + tuple(c[4:] for c in Q.columns if c.startswith("bfd_")):
            src = f"bfd_{k}" if f"bfd_{k}" in q else f"bf_{k}"
            mism[k] = int((q[src] != q[f"m1_{k}"]).sum())
        for k in ("sum_iou", "pq", "sq", "rq", "pq_empty1", "sum_iou_host", "pq_host"):
            a_, b_ = q[f"bf_{k}"].astype(float), q[f"m1_{k}"].astype(float)
            bad = ~((a_.isna() & b_.isna()) | ((a_ - b_).abs() <= 1e-9))
            mism[k] = int(bad.sum())
        mism["nb_counts_vs_bf_host"] = int(((q.nb_tp != q.bf_tp_host) | (q.nb_fp != q.bf_fp_host) | (q.nb_fn != q.bf_fn_host)).sum())
        mism["nb_sum_iou_absdiff_max"] = float((q.nb_sum_iou - q.bf_sum_iou_host).abs().max())
        S = lambda c: float(q[c].sum())
        agg = dict(
            n_records=len(q),
            bf_PQ_pooled=S("bf_sum_iou") / (S("bf_tp") + 0.5 * S("bf_fp") + 0.5 * S("bf_fn")),
            m1_PQ_pooled=S("m1_sum_iou") / (S("m1_tp") + 0.5 * S("m1_fp") + 0.5 * S("m1_fn")),
            bf_PQ_pooled_host=S("bf_sum_iou_host") / (S("bf_tp_host") + 0.5 * S("bf_fp_host") + 0.5 * S("bf_fn_host")),
            m1_PQ_pooled_host=S("m1_sum_iou_host") / (S("m1_tp_host") + 0.5 * S("m1_fp_host") + 0.5 * S("m1_fn_host")),
            nb_PQ_get_pq_score=NB["get_pq_score"](pd.DataFrame([r for r in nb_rows if r["kind"] == kind])),
            bf_PQ_img_mean=float(q.bf_pq.mean()), m1_PQ_img_mean=float(q.m1_pq.mean()),
            TP=int(S("bf_tp")), FP=int(S("bf_fp")), FN=int(S("bf_fn")),
            TP_host=int(S("bf_tp_host")), FP_host=int(S("bf_fp_host")), FN_host=int(S("bf_fn_host")),
            n_records_nonunique=int((~q.m1_unique_by_construction.astype(bool)).sum()),
            n_card_tie=int(q.bf_card_tie.sum()), n_iou_exact_half=int(q.bf_n_iou_exact_half.sum()),
            diag_totals={k[4:]: int(q[k].sum()) for k in q.columns if k.startswith("bfd_")})
        # cross-check filseg.metrics.aggregate on the m1 rows
        res[kind] = dict(mismatch_counts=mism, aggregate=agg)
    print(json.dumps(res, indent=1, default=float))
    with open(os.path.join(out_dir, "mv_real_summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main(int(os.environ.get("NW", "12")))
