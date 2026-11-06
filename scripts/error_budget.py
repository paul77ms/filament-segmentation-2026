"""D1 - error budget of the frozen baseline b0 (5-fold OOF, thr 0.5 / 8-connected / min_area 500), host-style PQ.

Host-style PQ (official notebook reading): every (gt, pred) pair with IoU > 0.5 is a TP, FP = predictions with no
such pair, FN = GT instances with no such pair; PQ = sum IoU(TP) / (TP + FP/2 + FN/2), pooled over annotation-set
RECORDS (the same prediction is scored against every annotation set of a file).

Error causes are the exclusive labels of filseg.metrics.evaluate_image (computed there for the one-to-one matching;
host FN is a subset of those FNs, host FP == unique FP minus 'duplicate', so every host error carries a cause):
  FN: merged > split > missed (best IoU < 0.1) > low_iou      FP: duplicate > merge > fragment > spurious > low_iou
Extra labels computed here:
  other_set_match (FP): the prediction matches (IoU > 0.5) an instance of ANOTHER annotation set of the same file.
  fn_drawn_other_iou (FN, multi-set files): some other set has an instance with IoU > 0.5 with this GT.
  fn_drawn_other_cov (FN, multi-set files): >= 50% of the GT is covered by the union of some other set.
LOW-IoU PAIRS: for every host FN with cause low_iou, its best-IoU prediction is taken; if that prediction is a host FP
with cause low_iou, (gt, pred) is a candidate pair; candidates are made one-to-one greedily by descending IoU within
the record (a pred / gt is used once).
Realistic membership counterfactuals (recounted with the actual pixels, per record, oracle grouping):
  join   : for every GT with >= 2 fragments (fragment = >= 50% of the pred inside the GT and >= 10 px), its fragment
           predictions are joined into one instance (union of the existing pixels; nothing is drawn).
  part   : every merge prediction (>= 2 GT with >= 50% of their area inside it) is partitioned among its contained GTs
           by nearest GT pixel (Euclidean); nothing is added or removed.
Extent diagnostic (DIAGNOSTIC ONLY): per TP pair and per low-IoU pair, area ratio, missed / extra pixel fractions and
their distance to the other mask, probability in missed pixels, IoU after instance-wise dilation / erosion of the
prediction by an exact Euclidean disk of radius 1, 2, 3 px, centroid offset (radial / tangential to the disk centre),
IoU under integer shifts in [-2, 2]^2, skeleton-based length and width (skimage skeletonize).

Usage (CPU only):
  CUDA_VISIBLE_DEVICES="" NUMPY_MADVISE_HUGEPAGE=0 taskset -c 0-3 .venv/bin/python scripts/error_budget.py --workers 4
  (--summarize_only rebuilds summary.json and the summary CSVs from the saved per-instance CSVs)
Outputs: outputs/error_budget/{records,gt_instances,pred_instances,pairs_extent,counterfactual,class_profiles,
         strata,extent_summary}.csv and summary.json; report: docs/error_budget.md
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from filseg import data as D  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.metrics import (EvalConfig, InstanceSet, evaluate_image, intersection_matrix,  # noqa: E402
                            single_thread_worker)

EXP = ROOT / "outputs" / "exps" / "b0_r101_s1024"
OUT = ROOT / "outputs" / "error_budget"
CFG = EvalConfig(compute_boundary=False, compute_pair_boundary=False, compute_thin=False)
POST = seg.PostCfg(0.5, 500, connectivity=8, seed_thr=None)
POST_A0 = seg.PostCfg(0.5, 0, connectivity=8, seed_thr=None)
H = W = 2048
RING_PX = 5  # same ring as scripts/data_audit.py (GT contrast in outputs/audit/instances.csv)
RING_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))
SHIFTS = (-2, -1, 0, 1, 2)
PAD = 6
REF = dict(PQ=0.4099, TP=4800, FP=2758, FN=3399)


def euclid_disk(k):
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return (yy * yy + xx * xx <= k * k).astype(np.uint8)


KER = {k: euclid_disk(k) for k in (1, 2, 3)}


# ----------------------------------------------------------------------------- helpers
def iou_mat(inter, ga, pa):
    u = ga[:, None] + pa[None, :] - inter
    return np.where(u > 0, inter / np.maximum(u, 1), 0.0)


def host_counts(inter, ga, pa):
    """Host-style counts from an intersection matrix (preds disjoint or not, does not matter)."""
    ng, npr = inter.shape
    if ng == 0 or npr == 0:
        return dict(tp=0, fp=int(npr), fn=int(ng), si=0.0)
    iou = iou_mat(inter.astype(np.float64), ga.astype(np.float64), pa.astype(np.float64))
    c = iou > 0.5
    return dict(tp=int(c.sum()), fp=int((~c.any(0)).sum()), fn=int((~c.any(1)).sum()), si=float(iou[c].sum()))


def frag_cont(inter, ga, pa, tau=0.5, min_px=10):
    big = inter >= min_px
    frag = big & (inter >= tau * pa[None, :])
    cont = big & (inter >= tau * ga[:, None])
    return frag, cont


def join_groups(inter, ga, pa):
    """Union-find over predictions: join the fragments of every GT with >= 2 fragments. Returns (M, n_joined_gt)."""
    ng, npr = inter.shape
    parent = list(range(npr))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    n = 0
    if ng and npr:
        frag, _ = frag_cont(inter, ga, pa)
        for g in np.flatnonzero(frag.sum(1) >= 2):
            js = np.flatnonzero(frag[g])
            n += 1
            for j in js[1:]:
                ra, rb = find(js[0]), find(j)
                if ra != rb:
                    parent[rb] = ra
    roots = sorted({find(j) for j in range(npr)})
    col = {r: k for k, r in enumerate(roots)}
    M = np.zeros((npr, len(roots)), np.int64)
    for j in range(npr):
        M[j, col[find(j)]] = 1
    return M, n


def partition_merges(G: InstanceSet, P: InstanceSet, inter):
    """Split every merge prediction among its contained GTs (nearest GT pixel). Returns new (inter, pa, n_merge_preds)."""
    ng, npr = inter.shape
    if ng == 0 or npr == 0:
        return inter, P.areas.copy(), 0
    _, cont = frag_cont(inter, G.areas, P.areas)
    merge = np.flatnonzero(cont.sum(0) >= 2)
    if merge.size == 0:
        return inter, P.areas.copy(), 0
    keep = [j for j in range(npr) if j not in set(merge.tolist())]
    boxes, crops = [], []
    for j in merge:
        gs = np.flatnonzero(cont[:, j])
        bb = np.vstack([P.boxes[j]] + [G.boxes[g] for g in gs])
        y0, x0 = max(bb[:, 0].min() - 1, 0), max(bb[:, 1].min() - 1, 0)
        y1, x1 = min(bb[:, 2].max() + 1, H), min(bb[:, 3].max() + 1, W)
        pm = np.zeros((y1 - y0, x1 - x0), bool)
        py0, px0, py1, px1 = P.boxes[j]
        pm[py0 - y0:py1 - y0, px0 - x0:px1 - x0] = P.crops[j]
        dist = []
        for g in gs:
            gm = np.ones((y1 - y0, x1 - x0), np.uint8)
            gy0, gx0, gy1, gx1 = G.boxes[g]
            gm[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0][G.crops[g]] = 0
            dist.append(cv2.distanceTransform(gm, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        lab = np.argmin(np.stack(dist), 0)
        for k in range(len(gs)):
            piece = pm & (lab == k)
            b, c = InstanceSet._crop(piece, y0, x0)
            if c.size and c.any():
                boxes.append(b)
                crops.append(c)
    pieces = InstanceSet._build((H, W), boxes, crops)
    inter_new = np.concatenate([inter[:, keep], intersection_matrix(G, pieces)], 1)
    pa_new = np.concatenate([P.areas[keep], pieces.areas])
    return inter_new, pa_new, int(merge.size)


def skel_len(m):
    from skimage.morphology import skeletonize
    if not m.any():
        return 0
    return int(skeletonize(m).sum())


def pair_extent(G, gi, P, pj, lab, prob, disk):
    gy0, gx0, gy1, gx1 = (int(v) for v in G.boxes[gi])
    py0, px0, py1, px1 = (int(v) for v in P.boxes[pj])
    y0, x0 = max(min(gy0, py0) - PAD, 0), max(min(gx0, px0) - PAD, 0)
    y1, x1 = min(max(gy1, py1) + PAD, H), min(max(gx1, px1) + PAD, W)
    gm = np.zeros((y1 - y0, x1 - x0), bool)
    pm = np.zeros_like(gm)
    gm[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0] = G.crops[gi]
    pm[py0 - y0:py1 - y0, px0 - x0:px1 - x0] = P.crops[pj]
    ga, pa = int(gm.sum()), int(pm.sum())
    inter = int((gm & pm).sum())
    r = dict(gt_area=ga, pred_area=pa, inter=inter, iou=inter / max(ga + pa - inter, 1))
    pm8 = pm.astype(np.uint8)
    for k in (1, 2, 3):
        d = cv2.dilate(pm8, KER[k], borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        e = cv2.erode(pm8, KER[k], borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        r[f"iou_dil{k}"] = (gm & d).sum() / max((gm | d).sum(), 1)
        r[f"iou_ero{k}"] = (gm & e).sum() / max((gm | e).sum(), 1)
        r[f"pred_area_ero{k}"] = int(e.sum())
    dP = cv2.distanceTransform((~pm).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dG = cv2.distanceTransform((~gm).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    miss, extra = gm & ~pm, pm & ~gm
    dm, de = dP[miss], dG[extra]
    r["miss_px"], r["extra_px"] = int(miss.sum()), int(extra.sum())
    for k in (1, 2, 3):
        r[f"miss_le{k}"] = int((dm <= k + 1e-6).sum())
        r[f"extra_le{k}"] = int((de <= k + 1e-6).sum())
    labc = lab[y0:y1, x0:x1]
    r["miss_in_other_pred"] = int((miss & (labc > 0) & (labc != pj + 1)).sum())
    pc = prob[y0:y1, x0:x1]
    pmiss = pc[miss]
    for t in (0.2, 0.3, 0.4):
        r[f"miss_p_gt{int(t * 10)}"] = int((pmiss > t).sum())
    r["miss_p_mean"] = float(pmiss.mean()) if pmiss.size else np.nan
    pex = pc[extra]
    r["extra_p_mean"] = float(pex.mean()) if pex.size else np.nan
    yg, xg = np.nonzero(gm)
    yp, xp = np.nonzero(pm)
    cgy, cgx = yg.mean() + y0, xg.mean() + x0
    cpy, cpx = yp.mean() + y0, xp.mean() + x0
    dy, dx = cpy - cgy, cpx - cgx
    uy, ux = cgy - disk[1], cgx - disk[0]
    nrm = max(np.hypot(uy, ux), 1e-6)
    uy, ux = uy / nrm, ux / nrm
    r.update(off_dy=dy, off_dx=dx, off_radial=dy * uy + dx * ux, off_tangential=-dy * ux + dx * uy,
             gt_r_frac=float(np.hypot(cgy - disk[1], cgx - disk[0]) / disk[2]))
    best, bdy, bdx = r["iou"], 0, 0
    for sy in SHIFTS:
        for sx in SHIFTS:
            if sy == 0 and sx == 0:
                r["iou_s0_0"] = r["iou"]
                continue
            ps = np.roll(np.roll(pm, sy, 0), sx, 1)
            v = (gm & ps).sum() / max((gm | ps).sum(), 1)
            r[f"iou_s{sy}_{sx}"] = v
            if v > best:
                best, bdy, bdx = v, sy, sx
    r.update(iou_shift_best=best, shift_best_dy=bdy, shift_best_dx=bdx)
    lg, lp = skel_len(gm), skel_len(pm)
    r.update(gt_skel=lg, pred_skel=lp, gt_width=ga / max(lg, 1), pred_width=pa / max(lp, 1))
    return r


# ----------------------------------------------------------------------------- per file job
def job(args):
    stem, fold, fname, recs, disk = args
    t0 = time.time()
    with np.load(EXP / f"fold{fold}" / "val_probs_last.npz") as z:
        prob = z[stem].astype(np.float32)
    P = seg.prob_to_instances(prob, POST)
    P0 = seg.prob_to_instances(prob, POST_A0)
    img = D.read_image(D.train_image_path(fname)).astype(np.float32)
    npr = len(P)
    lab = np.zeros((H, W), np.int32)
    for j in range(npr):
        y0, x0, y1, x1 = P.boxes[j]
        lab[y0:y1, x0:x1][P.crops[j]] = j + 1
    # union of every annotation set of the file (for 'touches any annotation')
    all_union = np.zeros((H, W), bool)
    unions = {}
    for rid, G in recs.items():
        unions[rid] = G.union()
        all_union |= unions[rid]
    # ---- record-independent prediction features
    pf = []
    for j in range(npr):
        y0, x0, y1, x1 = (int(v) for v in P.boxes[j])
        c = P.crops[j]
        pv = prob[y0:y1, x0:x1][c]
        Y0, X0 = max(y0 - PAD, 0), max(x0 - PAD, 0)
        Y1, X1 = min(y1 + PAD, H), min(x1 + PAD, W)
        own = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        own[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = c
        ring = (cv2.dilate(own, RING_K) > 0) & (own == 0)
        r2 = ring & (lab[Y0:Y1, X0:X1] == 0)
        if r2.sum() >= 10:
            ring = r2
        ic = img[Y0:Y1, X0:X1]
        ins, rv = ic[own > 0], ic[ring]
        rm, im_ = float(rv.mean()), float(ins.mean())
        yy, xx = np.nonzero(c)
        cy, cx = yy.mean() + y0, xx.mean() + x0
        rr = float(np.hypot(cy - disk[1], cx - disk[0]))
        pf.append(dict(pmax=float(pv.max()), pmean=float(pv.mean()), contrast=rm - im_,
                       contrast_snr=(rm - im_) / max(float(rv.std()), 1e-6), centroid_r_frac=rr / disk[2],
                       dist_to_limb_px=disk[2] - rr,
                       touch_any_gt_frac=float(all_union[y0:y1, x0:x1][c].mean())))
    rec_rows, gt_rows, pred_rows, pair_rows = [], [], [], []
    n_sets = len(recs)
    for rid, G in recs.items():
        ng = len(G)
        row, pg, pp = evaluate_image(G, P, CFG, image_key=rid, return_instances=True)
        inter = intersection_matrix(G, P)
        iou = iou_mat(inter.astype(np.float64), G.areas.astype(np.float64), P.areas.astype(np.float64))
        cand = iou > 0.5
        gt_tp = cand.any(1) if npr else np.zeros(ng, bool)
        pr_tp = cand.any(0) if ng else np.zeros(npr, bool)
        hc = host_counts(inter, G.areas, P.areas)
        assert hc["tp"] == row["tp_host"] and hc["fp"] == row["fp_host"] and hc["fn"] == row["fn_host"], rid
        fn_cause = np.array([("" if gt_tp[i] else pg[i]["fn_cause"]) for i in range(ng)], dtype=object)
        fp_cause = np.array([("" if pr_tp[j] else pp[j]["fp_cause"]) for j in range(npr)], dtype=object)
        assert all(c != "" for c in fn_cause[~gt_tp]) and all(c not in ("", "duplicate") for c in fp_cause[~pr_tp])
        # ---- other annotation sets of the same file
        other = {k: g for k, g in recs.items() if k != rid}
        om = np.zeros(npr, bool)
        drawn_iou = np.zeros(ng, np.int64)
        drawn_cov = np.zeros(ng, np.int64)
        for k, G2 in other.items():
            if len(G2) and npr:
                i2 = intersection_matrix(G2, P)
                om |= (iou_mat(i2.astype(float), G2.areas.astype(float), P.areas.astype(float)) > 0.5).any(0)
            if len(G2) and ng:
                igg = intersection_matrix(G, G2)
                drawn_iou += (iou_mat(igg.astype(float), G.areas.astype(float), G2.areas.astype(float)) > 0.5).any(1)
                u2 = unions[k]
                for i in range(ng):
                    y0, x0, y1, x1 = G.boxes[i]
                    drawn_cov[i] += int(u2[y0:y1, x0:x1][G.crops[i]].sum() >= 0.5 * G.areas[i])
        # ---- min_area-0 alternative (was the GT detected but dropped by min_area?)
        if ng and len(P0):
            i0 = intersection_matrix(G, P0)
            iou0 = iou_mat(i0.astype(float), G.areas.astype(float), P0.areas.astype(float))
            gt_best_a0 = iou0.max(1)
        else:
            gt_best_a0 = np.zeros(ng)
        # ---- low-IoU pairs (greedy one-to-one by IoU within the record)
        cands = []
        best_pred = iou.argmax(1) if npr else np.full(ng, -1)
        best_status = np.array([""] * ng, dtype=object)
        for i in range(ng):
            if fn_cause[i] != "low_iou":
                continue
            j = int(best_pred[i])
            best_status[i] = "tp_other_gt" if pr_tp[j] else f"fp_{fp_cause[j]}"
            if (not pr_tp[j]) and fp_cause[j] == "low_iou":
                cands.append((iou[i, j], i, j))
        cands.sort(key=lambda t: -t[0])
        ug, up, pairs = set(), set(), []
        for v, i, j in cands:
            if i in ug or j in up:
                continue
            ug.add(i); up.add(j); pairs.append((i, j, v))
        paired_gt = {i: j for i, j, _ in pairs}
        paired_pr = {j: i for i, j, _ in pairs}
        pr_best_gt = iou.argmax(0) if ng else np.full(npr, -1)
        mutual = sum(int(pr_best_gt[j] == i) for i, j, _ in pairs)
        # ---- fragments: to which GT (TP or FN) do fragment FPs belong?
        frag, cont = frag_cont(inter, G.areas, P.areas) if (ng and npr) else (np.zeros((ng, npr), bool),) * 2
        split_gt = frag.sum(1) >= 2
        frag_of_fn_split = np.zeros(npr, bool)
        frag_of_tp_gt = np.zeros(npr, bool)
        for j in np.flatnonzero(fp_cause == "fragment"):
            gs = np.flatnonzero(frag[:, j] & split_gt)
            frag_of_fn_split[j] = bool((fn_cause[gs] == "split").any())
            frag_of_tp_gt[j] = bool(gt_tp[gs].any())
        # ---- realistic membership counterfactuals (recount)
        ga, pa = G.areas, P.areas
        Mj, n_join = join_groups(inter, ga, pa)
        hc_join = host_counts(inter @ Mj, ga, pa @ Mj)
        inter_p, pa_p, n_part = partition_merges(G, P, inter)
        hc_part = host_counts(inter_p, ga, pa_p)
        Mb, _ = join_groups(inter_p, ga, pa_p)
        hc_both = host_counts(inter_p @ Mb, ga, pa_p @ Mb)
        # split-FN union IoU (fragments joined, no extent change) - closed form (preds disjoint)
        split_union_iou = np.full(ng, np.nan)
        for i in np.flatnonzero(fn_cause == "split"):
            js = np.flatnonzero(frag[i])
            it = inter[i, js].sum()
            split_union_iou[i] = it / max(ga[i] + pa[js].sum() - it, 1)
        rec = dict(rid=rid, stem=stem, fold=fold, n_sets=n_sets, n_gt=ng, n_pred=npr, tp=hc["tp"], fp=hc["fp"],
                   fn=hc["fn"], si=hc["si"], tp_unique=row["tp"], n_lowiou_pairs=len(pairs), n_lowiou_mutual=mutual,
                   join_tp=hc_join["tp"], join_fp=hc_join["fp"], join_fn=hc_join["fn"], join_si=hc_join["si"],
                   n_join_gt=n_join, part_tp=hc_part["tp"], part_fp=hc_part["fp"], part_fn=hc_part["fn"],
                   part_si=hc_part["si"], n_part_pred=n_part, both_tp=hc_both["tp"], both_fp=hc_both["fp"],
                   both_fn=hc_both["fn"], both_si=hc_both["si"])
        for c in ("missed", "low_iou", "split", "merged"):
            rec[f"fn_{c}"] = int((fn_cause == c).sum())
        for c in ("spurious", "low_iou", "fragment", "merge"):
            rec[f"fp_{c}"] = int((fp_cause == c).sum())
        rec["fp_other_set_match"] = int((om & ~pr_tp).sum())
        rec_rows.append(rec)
        # ---- per GT
        for i in range(ng):
            y0, x0, y1, x1 = (int(v) for v in G.boxes[i])
            pv = prob[y0:y1, x0:x1][G.crops[i]]
            tp_iou = float(iou[i][cand[i]].max()) if (npr and cand[i].any()) else np.nan
            gt_rows.append(dict(rid=rid, stem=stem, fold=fold, n_sets=n_sets, gt_idx=i, area=int(G.areas[i]),
                                status="tp" if gt_tp[i] else f"fn_{fn_cause[i]}", tp_iou=tp_iou,
                                best_iou=float(iou[i].max()) if npr else 0.0, best_iou_a0=float(gt_best_a0[i]),
                                pmax=float(pv.max()) if pv.size else np.nan, pmean=float(pv.mean()) if pv.size else np.nan,
                                pcov05=float((pv > 0.5).mean()) if pv.size else np.nan,
                                pcov03=float((pv > 0.3).mean()) if pv.size else np.nan,
                                lowiou_best_status=best_status[i], lowiou_paired=i in paired_gt,
                                n_fragments=int(frag[i].sum()) if npr else 0,
                                split_union_iou=split_union_iou[i],
                                n_other_sets=n_sets - 1, n_other_sets_drawn_iou=int(drawn_iou[i]),
                                n_other_sets_drawn_cov=int(drawn_cov[i])))
        # ---- per pred (record-dependent)
        for j in range(npr):
            d = dict(rid=rid, stem=stem, fold=fold, n_sets=n_sets, pred_idx=j, area=int(P.areas[j]),
                     status="tp" if pr_tp[j] else f"fp_{fp_cause[j]}", best_iou=float(iou[:, j].max()) if ng else 0.0,
                     other_set_match=bool(om[j]), lowiou_paired=j in paired_pr,
                     best_gt_status=(("tp" if gt_tp[pr_best_gt[j]] else f"fn_{fn_cause[pr_best_gt[j]]}")
                                     if (ng and iou[:, j].max() > 0) else "none"),
                     frag_of_fn_split=bool(frag_of_fn_split[j]), frag_of_tp_gt=bool(frag_of_tp_gt[j]),
                     n_contained_gt=int(cont[:, j].sum()) if ng else 0)
            d.update(pf[j])
            pred_rows.append(d)
        # ---- extent diagnostic pairs: TP pairs + low-IoU pairs
        for i, j in zip(*np.nonzero(cand)):
            r = pair_extent(G, int(i), P, int(j), lab, prob, disk)
            r.update(rid=rid, stem=stem, fold=fold, n_sets=n_sets, gt_idx=int(i), pred_idx=int(j), kind="tp")
            pair_rows.append(r)
        for i, j, _ in pairs:
            r = pair_extent(G, int(i), P, int(j), lab, prob, disk)
            r.update(rid=rid, stem=stem, fold=fold, n_sets=n_sets, gt_idx=int(i), pred_idx=int(j), kind="lowiou")
            pair_rows.append(r)
    return dict(rec=rec_rows, gt=gt_rows, pred=pred_rows, pair=pair_rows, t=time.time() - t0, stem=stem)


# ----------------------------------------------------------------------------- aggregation helpers
def pq(N, D):
    return N / D if D > 0 else float("nan")


def q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = pd.Series(x).dropna()
    return {f"q{int(v * 100)}": (float(x.quantile(v)) if len(x) else np.nan) for v in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N files only")
    ap.add_argument("--summarize_only", action="store_true", help="recompute summary.json from the saved CSVs")
    a = ap.parse_args()
    if a.summarize_only:
        R, GT, PR, PA = (pd.read_csv(OUT / f"{n}.csv") for n in ("records", "gt_instances", "pred_instances", "pairs_extent"))
        summarize(R, GT, PR, PA, int((GT.area_audit.round() != GT.area).sum()))
        return
    import multiprocessing as mp
    single_thread_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fold_of, fname_of = {}, {}
    for f in range(5):
        for fn in (EXP / f"fold{f}" / "val_files.txt").read_text().split():
            fold_of[Path(fn).stem] = f
            fname_of[Path(fn).stem] = fn
    files = [fname_of[s] for s in sorted(fname_of)]
    if a.limit:
        files = files[:a.limit]
    gts = seg.gt_records(files)
    by_stem = {}
    for rid, g in gts.items():
        by_stem.setdefault(rid.split("-", 1)[1], {})[rid] = g
    img = pd.read_csv(ROOT / "outputs" / "audit" / "images.csv",
                      usecols=["file_name", "img_disk_cx", "img_disk_cy", "img_disk_r"]).drop_duplicates("file_name")
    disk = {Path(r.file_name).stem: (float(r.img_disk_cx), float(r.img_disk_cy), float(r.img_disk_r))
            for r in img.itertuples()}
    jobs = [(s, fold_of[s], fname_of[s], {k: by_stem[s][k] for k in sorted(by_stem[s])}, disk[s])
            for s in sorted(by_stem)]
    print(f"{len(jobs)} files, {len(gts)} records; loading took {time.time() - t0:.0f}s", flush=True)
    res = []
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            res.append(r)
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame([x for r in res for x in r["rec"]]).sort_values("rid").reset_index(drop=True)
    GT = pd.DataFrame([x for r in res for x in r["gt"]]).sort_values(["rid", "gt_idx"]).reset_index(drop=True)
    PR = pd.DataFrame([x for r in res for x in r["pred"]]).sort_values(["rid", "pred_idx"]).reset_index(drop=True)
    PA = pd.DataFrame([x for r in res for x in r["pair"]]).sort_values(["kind", "rid", "gt_idx"]).reset_index(drop=True)
    # ---- join GT audit features (outputs/audit/instances.csv: image_id == rid, ann_idx == gt_idx)
    ins = pd.read_csv(ROOT / "outputs" / "audit" / "instances.csv",
                      usecols=["image_id", "ann_idx", "area", "contrast", "contrast_snr", "mean_width",
                               "centroid_r_frac", "dist_to_limb_px", "width_dt_med"])
    ins = ins.rename(columns={"image_id": "rid", "ann_idx": "gt_idx", "area": "area_audit"})
    GT = GT.merge(ins, on=["rid", "gt_idx"], how="left")
    n_area_mismatch = int((GT.area_audit.round() != GT.area).sum())
    for df, name in ((R, "records"), (GT, "gt_instances"), (PR, "pred_instances"), (PA, "pairs_extent")):
        df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"jobs done {time.time() - t0:.0f}s; GT audit-area mismatches: {n_area_mismatch}", flush=True)
    summarize(R, GT, PR, PA, n_area_mismatch, time.time() - t0, a.limit)


def summarize(R, GT, PR, PA, n_area_mismatch=0, elapsed=0.0, limit=0):
    S = {}
    TP, FP, FN, N = int(R.tp.sum()), int(R.fp.sum()), int(R.fn.sum()), float(R.si.sum())
    Dn = TP + FP / 2 + FN / 2
    PQ0, SQm, RQ0 = N / Dn, N / TP, TP / Dn
    S["reference"] = dict(PQ=PQ0, SQ=SQm, RQ=RQ0, TP=TP, FP=FP, FN=FN, N_sum_iou=N, D=Dn, n_records=len(R),
                          n_files=int(R.stem.nunique()), n_gt=int(len(GT)), n_pred_rows=int(len(PR)),
                          tp_unique=int(R.tp_unique.sum()), gt_audit_area_mismatch=n_area_mismatch)
    ok = (round(PQ0, 4) == REF["PQ"] and TP == REF["TP"] and FP == REF["FP"] and FN == REF["FN"])
    S["reference"]["matches_frozen_reference"] = bool(ok)
    print("REFERENCE", json.dumps(S["reference"]), flush=True)
    if not ok and not limit:
        print("!!! reference mismatch - STOP", flush=True)
        json.dump(S, open(OUT / "summary.json", "w"), indent=1, default=float)
        sys.exit(2)
    # ---- 1. counts by cause
    fn_c = GT.status[GT.status != "tp"].value_counts()
    fp_c = PR.status[PR.status != "tp"].value_counts()
    S["fn_by_cause"] = {k: int(v) for k, v in fn_c.items()}
    S["fp_by_cause"] = {k: int(v) for k, v in fp_c.items()}
    multi_gt = GT[GT.n_sets > 1]
    multi_pr = PR[PR.n_sets > 1]
    fnm = multi_gt[multi_gt.status != "tp"]
    fpm = multi_pr[multi_pr.status != "tp"]
    S["multi_set"] = dict(
        n_records=int((R.n_sets > 1).sum()), n_files=int(R[R.n_sets > 1].stem.nunique()),
        TP=int(R[R.n_sets > 1].tp.sum()), FP=int(R[R.n_sets > 1].fp.sum()), FN=int(R[R.n_sets > 1].fn.sum()),
        fp_other_set_match=int(fpm.other_set_match.sum()),
        fp_other_set_match_by_cause={k: int(v) for k, v in fpm[fpm.other_set_match].status.value_counts().items()},
        fp_by_cause={k: int(v) for k, v in fpm.status.value_counts().items()},
        fn_not_drawn_iou=int((fnm.n_other_sets_drawn_iou == 0).sum()),
        fn_not_drawn_cov=int((fnm.n_other_sets_drawn_cov == 0).sum()),
        fn_drawn_by_all_others_iou=int((fnm.n_other_sets_drawn_iou == fnm.n_other_sets).sum()),
        fn_not_drawn_iou_by_cause={k: int(v) for k, v in fnm[fnm.n_other_sets_drawn_iou == 0].status.value_counts().items()},
        fn_not_drawn_cov_by_cause={k: int(v) for k, v in fnm[fnm.n_other_sets_drawn_cov == 0].status.value_counts().items()},
        fn_by_cause={k: int(v) for k, v in fnm.status.value_counts().items()},
        tp_gt_not_drawn_iou=int(((multi_gt.status == "tp") & (multi_gt.n_other_sets_drawn_iou == 0)).sum()),
        tp_gt=int((multi_gt.status == "tp").sum()))
    # low-IoU pairing
    lo = GT[GT.status == "fn_low_iou"]
    S["lowiou"] = dict(fn_low_iou=int(len(lo)), fp_low_iou=int((PR.status == "fp_low_iou").sum()),
                       pairs=int(R.n_lowiou_pairs.sum()), pairs_mutual_best=int(R.n_lowiou_mutual.sum()),
                       fn_best_pred_status={k: int(v) for k, v in lo.lowiou_best_status.value_counts().items()},
                       fn_fp_lowiou_lost_tie=int(((lo.lowiou_best_status == "fp_low_iou") & ~lo.lowiou_paired).sum()))
    fl = PR[PR.status == "fp_low_iou"]
    S["lowiou"]["fp_low_iou_unpaired"] = int((~fl.lowiou_paired).sum())
    S["lowiou"]["fp_low_iou_unpaired_best_gt_status"] = {
        k: int(v) for k, v in fl[~fl.lowiou_paired].best_gt_status.value_counts().items()}
    fr = PR[PR.status == "fp_fragment"]
    S["fragment_fp"] = dict(n=int(len(fr)), of_fn_split_gt=int(fr.frag_of_fn_split.sum()),
                            of_tp_gt_only=int((fr.frag_of_tp_gt & ~fr.frag_of_fn_split).sum()),
                            other=int((~fr.frag_of_tp_gt & ~fr.frag_of_fn_split).sum()))
    # min_area effect on missed
    mi = GT[GT.status == "fn_missed"]
    S["missed_detail"] = dict(
        n=int(len(mi)), tp_if_min_area0=int((mi.best_iou_a0 > 0.5).sum()),
        pmax_lt01=int((mi.pmax < 0.1).sum()), pmax_01_03=int(((mi.pmax >= 0.1) & (mi.pmax < 0.3)).sum()),
        pmax_03_05=int(((mi.pmax >= 0.3) & (mi.pmax <= 0.5)).sum()), pmax_gt05=int((mi.pmax > 0.5).sum()),
        area_lt500=int((mi.area < 500).sum()))
    sp = PR[PR.status == "fp_spurious"]
    S["spurious_detail"] = dict(n=int(len(sp)), overlaps_any_annotation_ge10pct=int((sp.touch_any_gt_frac >= 0.1).sum()),
                                other_set_match=int(sp.other_set_match.sum()), on_multi_set=int((sp.n_sets > 1).sum()),
                                not_darker_than_ring=int((sp.contrast <= 0).sum()))
    S["prob_saturation"] = {st: dict(n=int(len(d)), frac_pmax_lt_0_99=float((d.pmax < 0.99).mean()),
                                     frac_pmax_lt_0_999=float((d.pmax < 0.999).mean()),
                                     pmean_q10=float(d.pmean.quantile(.1)), pmean_q50=float(d.pmean.median()))
                            for st, d in PR.groupby("status")}
    st_rows = []
    G2 = GT.copy()
    G2["area_bin"] = pd.cut(G2.area, [0, 250, 500, 1000, 2000, 4000, 1e9]).astype(str)
    G2["contrast_q5"] = pd.qcut(G2.contrast, 5).astype(str)
    G2["r_bin"] = pd.cut(G2.centroid_r_frac, [0, .5, .7, .8, .9, 1.1]).astype(str)
    for col in ("area_bin", "contrast_q5", "r_bin", "n_sets"):
        for b, d in G2.groupby(col):
            r = dict(side="gt", stratum=col, bin=str(b), n=len(d))
            r.update({k: float(v) for k, v in d.status.value_counts(normalize=True).items()})
            st_rows.append(r)
    P2 = PR.copy()
    P2["area_bin"] = pd.cut(P2.area, [0, 1000, 2000, 4000, 1e9]).astype(str)
    P2["r_bin"] = pd.cut(P2.centroid_r_frac, [0, .5, .7, .8, .9, 1.1]).astype(str)
    for col in ("area_bin", "r_bin", "n_sets"):
        for b, d in P2.groupby(col):
            r = dict(side="pred", stratum=col, bin=str(b), n=len(d))
            r.update({k: float(v) for k, v in d.status.value_counts(normalize=True).items()})
            st_rows.append(r)
    pd.DataFrame(st_rows).to_csv(OUT / "strata.csv", index=False)
    all_fn = GT[GT.status != "tp"]
    S["fn_tp_if_min_area0"] = {k: int(v) for k, v in all_fn[all_fn.best_iou_a0 > 0.5].status.value_counts().items()}
    # ---- 2. counterfactuals
    n = {c: int((GT.status == f"fn_{c}").sum()) for c in ("missed", "low_iou", "split", "merged")}
    m = {c: int((PR.status == f"fp_{c}").sum()) for c in ("spurious", "low_iou", "fragment", "merge")}
    npair = int(R.n_lowiou_pairs.sum())
    osm = int(((PR.status != "tp") & PR.other_set_match).sum())
    rows = []

    def add(name, Np, Dp, note=""):
        rows.append(dict(scenario=name, N=Np, D=Dp, PQ=Np / Dp, dPQ=Np / Dp - PQ0, note=note))

    add("baseline b0", N, Dn)
    for c, v in m.items():
        add(f"remove all FP {c} ({v})", N, Dn - v / 2, "D' = D - n/2")
    add(f"remove FP other_set_match ({osm}) [ambiguity, not removable without TP loss on other sets]", N, Dn - osm / 2)
    add(f"remove ALL FP ({FP})", N, Dn - FP / 2)
    add(f"recover FN missed ({n['missed']}) as TP @SQ_m", N + n["missed"] * SQm, Dn + n["missed"] / 2,
        "N' = N + n SQ_m, D' = D + n/2")
    add(f"fix low-IoU pairs ({npair}) -> TP @SQ_m", N + npair * SQm, Dn, "FN+FP -> TP: D unchanged")
    unp = n["low_iou"] - npair
    add(f"low-IoU pairs + recover unpaired FN low_iou ({unp}) @SQ_m", N + n["low_iou"] * SQm, Dn + unp / 2)
    add(f"split: FN split ({n['split']}) -> TP @SQ_m, FP fragment ({m['fragment']}) removed",
        N + n["split"] * SQm, Dn + n["split"] / 2 - m["fragment"] / 2)
    add(f"merged: FN merged ({n['merged']}) -> TP @SQ_m, FP merge ({m['merge']}) removed",
        N + n["merged"] * SQm, Dn + n["merged"] / 2 - m["merge"] / 2)
    gN = N + (n["split"] + n["merged"]) * SQm
    gD = Dn + (n["split"] + n["merged"]) / 2 - (m["fragment"] + m["merge"]) / 2
    add("all grouping (split+merged+fragment+merge) @SQ_m", gN, gD)
    for k, lab in (("join", "realistic join of fragments (recount, actual pixels)"),
                   ("part", "realistic partition of merge preds (recount)"),
                   ("both", "realistic join + partition (recount)")):
        tp_, fp_, fn_, si_ = (int(R[f"{k}_tp"].sum()), int(R[f"{k}_fp"].sum()), int(R[f"{k}_fn"].sum()),
                              float(R[f"{k}_si"].sum()))
        add(lab, si_, tp_ + fp_ / 2 + fn_ / 2, f"TP/FP/FN {tp_}/{fp_}/{fn_}, SQ {si_ / max(tp_, 1):.4f}")
    for sq in (0.70, 0.75):
        add(f"SQ -> {sq:.2f} at fixed RQ", sq * TP, Dn, "PQ = SQ x RQ")
    dN = N + n["missed"] * SQm
    dD = Dn + n["missed"] / 2 - m["spurious"] / 2
    add("all detection (missed recovered + spurious removed)", dN, dD)
    add("perfect: every FN -> TP @SQ_m, every FP removed", N + FN * SQm, TP + FN)
    # realistic partial-fix scenarios
    def scen(name, f_miss=0.0, sq_new=None, f_spur=0.0, f_lowfp=0.0, f_pair=0.0, f_split=0.0, f_merge=0.0,
             sq_rec=None, note=""):
        sqr = SQm if sq_rec is None else sq_rec
        Np, Dp = N, Dn
        k = f_miss * n["missed"]; Np += k * sqr; Dp += k / 2
        Dp -= f_spur * m["spurious"] / 2
        k = f_pair * npair; Np += k * sqr
        Dp -= f_lowfp * (m["low_iou"] - npair) / 2
        k = f_split * n["split"]; Np += k * sqr; Dp += k / 2 - f_split * m["fragment"] / 2
        k = f_merge * n["merged"]; Np += k * sqr; Dp += k / 2 - f_merge * m["merge"] / 2
        if sq_new is not None:  # raise the IoU of the EXISTING TPs to sq_new
            Np += (sq_new - SQm) * TP
        add(name, Np, Dp, note)

    scen("S1: 25% missed (@IoU .60), 30% spurious removed, 30% low-IoU pairs, 30% split/merge",
         f_miss=.25, sq_rec=.60, f_spur=.30, f_pair=.30, f_split=.30, f_merge=.30)
    scen("S2: S1 + SQ of existing TPs +0.03", f_miss=.25, sq_rec=.60, f_spur=.30, f_pair=.30, f_split=.30, f_merge=.30,
         sq_new=SQm + 0.03)
    scen("S3: 40% missed (@IoU .60), 40% spurious removed, 50% low-IoU pairs, 50% split/merge",
         f_miss=.40, sq_rec=.60, f_spur=.40, f_pair=.50, f_split=.50, f_merge=.50)
    scen("S4: S3 + SQ of existing TPs +0.03", f_miss=.40, sq_rec=.60, f_spur=.40, f_pair=.50, f_split=.50, f_merge=.50,
         sq_new=SQm + 0.03)
    # uniform fraction needed for 0.49 and 0.55 (all classes fixed by the same fraction f, recovered at SQ_m)
    def uni(f, sq_rec=None):
        sqr = SQm if sq_rec is None else sq_rec
        Np = N + f * (n["missed"] + npair + n["split"] + n["merged"]) * sqr
        Dp = Dn + f * (n["missed"] / 2 - m["spurious"] / 2 - (m["low_iou"] - npair) / 2
                       + n["split"] / 2 - m["fragment"] / 2 + n["merged"] / 2 - m["merge"] / 2)
        return Np / Dp
    need = {}
    for tgt in (0.49, 0.55):
        for lab, sqr in (("SQ_m", None), ("0.60", 0.60)):
            fs = np.linspace(0, 1, 10001)
            v = np.array([uni(f, sqr) for f in fs])
            ok = np.flatnonzero(v >= tgt)
            need[f"{tgt}_{lab}"] = float(fs[ok[0]]) if ok.size else None
    S["uniform_fraction_needed"] = need
    # composed scenarios: realistic grouping recount (join + partition, actual pixels) + partial formula fixes of
    # the disjoint remaining classes (missed GT, spurious FP, low-IoU pairs), new TPs at IoU `sqr`
    bt, bf, bn, bs = (int(R.both_tp.sum()), int(R.both_fp.sum()), int(R.both_fn.sum()), float(R.both_si.sum()))
    def comp(name, f_pair=0.0, f_miss=0.0, f_spur=0.0, sqr=0.60, d_sq=0.0):
        Np = bs + f_pair * npair * sqr + f_miss * n["missed"] * sqr + d_sq * bt
        Dp = bt + bf / 2 + bn / 2 + f_miss * n["missed"] / 2 - f_spur * m["spurious"] / 2
        add(name, Np, Dp, "composed on the realistic join+partition recount; new TPs at IoU %.2f" % sqr)
    comp("C1: realistic grouping + 50% low-IoU pairs", f_pair=.5)
    comp("C2: C1 + 25% missed recovered", f_pair=.5, f_miss=.25)
    comp("C3: C2 + 30% spurious removed", f_pair=.5, f_miss=.25, f_spur=.3)
    comp("C4: realistic grouping + 30% low-IoU pairs + 30% missed + 30% spurious", f_pair=.3, f_miss=.3, f_spur=.3)
    comp("C5: realistic grouping + 60% low-IoU pairs + 50% missed + 50% spurious", f_pair=.6, f_miss=.5, f_spur=.5)
    for f in (0.25, 0.5):
        v = uni(f)
        rows.append(dict(scenario=f"U{int(f * 100)}: every error class fixed by {int(f * 100)}% (recovered @SQ_m)",
                         N=np.nan, D=np.nan, PQ=v, dPQ=v - PQ0, note="uniform fraction of every class"))
    CF = pd.DataFrame(rows)
    CF.to_csv(OUT / "counterfactual.csv", index=False)
    S["counterfactual"] = rows
    S["n_classes"] = dict(fn=n, fp=m, lowiou_pairs=npair, other_set_match_fp=osm, SQ_m=SQm)
    # ---- 3. class profiles
    prof = []
    for st, d in GT.groupby("status"):
        r = dict(side="gt", cls=st, n=len(d), share_of_side=len(d) / len(GT),
                 share_of_errors=(len(d) / FN if st != "tp" else np.nan))
        for col in ("area", "contrast", "contrast_snr", "mean_width", "centroid_r_frac", "pmax", "pmean", "pcov03"):
            qq = q(d[col])
            r.update({f"{col}_{k}": v for k, v in qq.items() if k in ("q10", "q25", "q50", "q75", "q90")})
        for s in (1, 2, 3):
            r[f"nsets{s}_share"] = float((d.n_sets == s).mean())
        r["frac_r_gt_0.8"] = float((d.centroid_r_frac > 0.8).mean())
        r["frac_area_lt_500"] = float((d.area < 500).mean())
        prof.append(r)
    PRx = PR.copy()
    PRx["cls"] = PRx.status
    osm_rows = PRx[(PRx.status != "tp") & PRx.other_set_match].copy()
    osm_rows["cls"] = "fp_other_set_match(any cause)"
    for st, d in pd.concat([PRx, osm_rows]).groupby("cls"):
        r = dict(side="pred", cls=st, n=len(d), share_of_side=len(d) / len(PR),
                 share_of_errors=(len(d) / FP if st != "tp" else np.nan))
        for col in ("area", "contrast", "contrast_snr", "centroid_r_frac", "pmax", "pmean", "touch_any_gt_frac"):
            qq = q(d[col])
            r.update({f"{col}_{k}": v for k, v in qq.items() if k in ("q10", "q25", "q50", "q75", "q90")})
        for s in (1, 2, 3):
            r[f"nsets{s}_share"] = float((d.n_sets == s).mean())
        r["frac_r_gt_0.8"] = float((d.centroid_r_frac > 0.8).mean())
        r["frac_pmax_lt_0.9"] = float((d.pmax < 0.9).mean())
        r["frac_other_set_match"] = float(d.other_set_match.mean())
        prof.append(r)
    PF = pd.DataFrame(prof)
    PF.to_csv(OUT / "class_profiles.csv", index=False)
    S["class_profiles"] = prof
    # ---- 4. extent diagnostic
    ext = []
    for kind, d in PA.groupby("kind"):
        r = dict(kind=kind, n=len(d), mean_iou=d.iou.mean(), median_area_ratio=(d.pred_area / d.gt_area).median(),
                 pooled_area_ratio=d.pred_area.sum() / d.gt_area.sum(),
                 frac_pairs_pred_smaller=float((d.pred_area < d.gt_area).mean()),
                 gt_px_missed_frac=d.miss_px.sum() / d.gt_area.sum(), pred_px_extra_frac=d.extra_px.sum() / d.pred_area.sum(),
                 miss_within1=d.miss_le1.sum() / max(d.miss_px.sum(), 1), miss_within2=d.miss_le2.sum() / max(d.miss_px.sum(), 1),
                 miss_within3=d.miss_le3.sum() / max(d.miss_px.sum(), 1),
                 extra_within1=d.extra_le1.sum() / max(d.extra_px.sum(), 1),
                 extra_within2=d.extra_le2.sum() / max(d.extra_px.sum(), 1),
                 extra_within3=d.extra_le3.sum() / max(d.extra_px.sum(), 1),
                 miss_in_other_pred=d.miss_in_other_pred.sum() / max(d.miss_px.sum(), 1),
                 miss_p_gt02=d.miss_p_gt2.sum() / max(d.miss_px.sum(), 1),
                 miss_p_gt03=d.miss_p_gt3.sum() / max(d.miss_px.sum(), 1),
                 miss_p_gt04=d.miss_p_gt4.sum() / max(d.miss_px.sum(), 1),
                 median_width_ratio=(d.pred_width / d.gt_width).median(),
                 median_length_ratio=(d.pred_skel / d.gt_skel.clip(lower=1)).median(),
                 mean_off_radial=d.off_radial.mean(), mean_off_tangential=d.off_tangential.mean(),
                 mean_off_dy=d.off_dy.mean(), mean_off_dx=d.off_dx.mean(),
                 mean_abs_offset=np.hypot(d.off_dy, d.off_dx).mean(),
                 mean_iou_best_shift=d.iou_shift_best.mean(),
                 frac_best_shift_zero=float(((d.shift_best_dy == 0) & (d.shift_best_dx == 0)).mean()),
                 se_off_dy=d.off_dy.std() / np.sqrt(len(d)), se_off_dx=d.off_dx.std() / np.sqrt(len(d)),
                 se_off_radial=d.off_radial.std() / np.sqrt(len(d)),
                 corr_log_gtwidth_log_widthratio=float(np.corrcoef(np.log(d.gt_width.clip(lower=1e-3)),
                                                                   np.log(d.pred_width / d.gt_width.clip(lower=1e-3)))[0, 1]),
                 median_gt_width=d.gt_width.median(), median_pred_width=d.pred_width.median(),
                 gt_area_missed_beyond3px=(d.miss_px - d.miss_le3).sum() / d.gt_area.sum(),
                 pred_area_extra_beyond3px=(d.extra_px - d.extra_le3).sum() / d.pred_area.sum())
        u = d.gt_area + d.pred_area - d.inter
        for t in (2, 3, 4):  # ORACLE: add only the missed GT pixels with p > 0.t (upper bound of a lower threshold)
            io = (d.inter + d[f"miss_p_gt{t}"]) / u
            r[f"oracle_addGTp{t}_mean_iou"], r[f"oracle_addGTp{t}_n_iou_gt05"] = io.mean(), int((io > 0.5).sum())
        for nm, io in (("oracle_trim_extra_le2", d.inter / (u - d.extra_le2)),
                       ("oracle_fill_miss_le2", (d.inter + d.miss_le2) / u),
                       ("oracle_band2_fill_trim", (d.inter + d.miss_le2) / (u - d.extra_le2))):
            r[f"{nm}_mean_iou"], r[f"{nm}_n_iou_gt05"] = io.mean(), int((io > 0.5).sum())
        gc, pi = d.inter / d.gt_area, d.inter / d.pred_area
        typ = np.select([(gc >= 0.7) & (pi < 0.6), (pi >= 0.7) & (gc < 0.6), (gc < 0.6) & (pi < 0.6)],
                        ["pred_larger_spills_out", "pred_partial_inside_gt", "offset_or_both_partial"], "mixed")
        for k, v in pd.Series(typ).value_counts().items():
            r[f"type_{k}"] = int(v)
        r.update({f"iou_q{int(v * 100)}": d.iou.quantile(v) for v in (0.1, 0.25, 0.5, 0.75, 0.9)})
        r["n_iou_gt04"] = int((d.iou > 0.4).sum())
        for k in (1, 2, 3):
            r[f"mean_iou_dil{k}"] = d[f"iou_dil{k}"].mean()
            r[f"mean_iou_ero{k}"] = d[f"iou_ero{k}"].mean()
            r[f"frac_improved_dil{k}"] = float((d[f"iou_dil{k}"] > d.iou).mean())
            r[f"frac_improved_ero{k}"] = float((d[f"iou_ero{k}"] > d.iou).mean())
            r[f"n_cross05_dil{k}"] = int(((d[f"iou_dil{k}"] > 0.5) != (d.iou > 0.5)).sum())
            r[f"n_cross05_ero{k}"] = int(((d[f"iou_ero{k}"] > 0.5) != (d.iou > 0.5)).sum())
            r[f"n_vanish_ero{k}"] = int((d[f"pred_area_ero{k}"] == 0).sum())
        best = max(((sy, sx) for sy in SHIFTS for sx in SHIFTS), key=lambda t: d[f"iou_s{t[0]}_{t[1]}"].mean())
        r["best_global_shift"] = f"{best[0]},{best[1]}"
        r["mean_iou_best_global_shift"] = d[f"iou_s{best[0]}_{best[1]}"].mean()
        ext.append(r)
    # TP pairs stratified by GT width (thin vs thick) and by area-ratio class
    tp = PA[PA.kind == "tp"].copy()
    tp["wbin"] = pd.cut(tp.gt_width, [0, 6, 9, 13, 1e9], labels=["w<6", "6-9", "9-13", ">=13"])
    for wb, d in tp.groupby("wbin", observed=True):
        r = dict(kind=f"tp gt_width {wb}", n=len(d), mean_iou=d.iou.mean(),
                 median_area_ratio=(d.pred_area / d.gt_area).median(),
                 median_width_ratio=(d.pred_width / d.gt_width).median(),
                 median_length_ratio=(d.pred_skel / d.gt_skel.clip(lower=1)).median(),
                 gt_px_missed_frac=d.miss_px.sum() / d.gt_area.sum(), pred_px_extra_frac=d.extra_px.sum() / d.pred_area.sum(),
                 miss_within2=d.miss_le2.sum() / max(d.miss_px.sum(), 1),
                 extra_within2=d.extra_le2.sum() / max(d.extra_px.sum(), 1))
        for k in (1, 2, 3):
            r[f"mean_iou_dil{k}"] = d[f"iou_dil{k}"].mean()
            r[f"mean_iou_ero{k}"] = d[f"iou_ero{k}"].mean()
        ext.append(r)
    EX = pd.DataFrame(ext)
    EX.to_csv(OUT / "extent_summary.csv", index=False)
    S["extent"] = ext
    tpi = GT.tp_iou.dropna()
    S["tp_iou_dist"] = dict(n=int(len(tpi)), mean=float(tpi.mean()), q10=float(tpi.quantile(.1)),
                            q25=float(tpi.quantile(.25)), q50=float(tpi.median()), q75=float(tpi.quantile(.75)),
                            frac_lt_0_6=float((tpi < 0.6).mean()), frac_ge_0_8=float((tpi >= 0.8).mean()))
    S["elapsed_s"] = elapsed
    json.dump(S, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(CF[["scenario", "PQ", "dPQ"]].to_string(), flush=True)
    print("-> ", OUT, flush=True)


if __name__ == "__main__":
    main()
