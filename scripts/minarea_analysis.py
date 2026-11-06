"""What does the min_area size filter of the b0 baseline actually do?  (MEASUREMENT ONLY)

No new rescoring, no classifier, no pipeline change. The b0 pipeline is reproduced exactly:
OOF probabilities -> p > 0.5 -> 8-connected components (seg.prob_to_instances / labels_to_instances)
-> drop components with area < min_area. Metric = filseg.metrics.evaluate_image (EvalConfig defaults,
boundary off), pooled host-style PQ over all 1154 annotation-set records.

Stages (all CPU, forkserver pool, single-threaded workers):
  compute  : per file -> per-record counts for every min_area, per-GT-instance overlap facts,
             every component < 500 px with its class (a)-(e), fragmented-GT facts  -> OUT/raw/*.csv
  analyze  : tables, strata, decomposition checks, bootstrap CIs              -> OUT/*.csv, summary.json
  figures  : native-resolution overlays of random examples (seed 2026)        -> OUT/figures/*.png

Component classes (removed components, area in [m, 500); precedence a > b > b_largest > c > d > e):
  (a) TP-worthy     : IoU > 0.5 with some GT instance of at least one annotation set of the file.
  (b) fragment      : >= 30 % of the component lies inside some GT instance g (any set), and g is also
                      overlapped (>= 10 px) by ANOTHER component that is LARGER than this one.
  (b_largest)       : as (b) but every other component overlapping g is smaller (this is the largest of
                      several pieces). Not in the original spec (gap between (b) and (c)); reported apart.
  (c) part alone    : >= 30 % inside some GT instance g, no other component overlaps any such g (>= 10 px),
                      IoU <= 0.5.
  (d) unannotated dark: zero pixel overlap with every GT instance of every set AND
                      ring_mean - inside_mean > ring_std (5-px ring, ring excludes other p>0.5 pixels when
                      >= 10 ring px remain; raw uint8 intensities).
  (e) everything else: no GT overlap and not darker by > ring std (e_noise), or 0 < GT overlap but
                      < 30 % inside any single GT instance (e_weak_gt, flagged).
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

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
from filseg.metrics import (EvalConfig, evaluate_image, intersection_matrix,  # noqa: E402
                            single_thread_worker)

EXP = ROOT / "outputs" / "exps" / "b0_r101_s1024"
PROBS = "val_probs_last.npz"
OUT = ROOT / "outputs" / "minarea_analysis"
RAW = OUT / "raw"
FIG = OUT / "figures"
THR = 0.5
MINAREAS = [0, 50, 100, 200, 250, 300, 500, 800]   # 250 added to the requested grid (item-3 step)
BASE = 500
STEPS = [250, 100, 0]
CFG = EvalConfig(compute_boundary=False, compute_pair_boundary=False, compute_thin=True)
CX, CY, RSUN = 1023.0, 1024.0, 902.0                # docs/pipeline_processing.md (fixed disk geometry)
RING_PX = 5
RING_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))  # as data_audit
INSIDE_FRAC = 0.30     # class (b)/(c): >= 30 % of the component inside one GT instance
OV_PX = 10             # "another component overlaps g": >= 10 px intersection (filseg split_min_px)
TOUCH_FRAC = 0.10      # GT "touched": one component covers >= 10 % of the GT
FRAG_TAU, FRAG_PX = 0.5, 10   # filseg containment rule for fragments (component >= 50 % inside g, >= 10 px)
SEED = 2026
CLASS_ORDER = ["a", "b", "b_largest", "c", "d", "e"]


# ============================================================================ compute stage
def comp_contrast(img, fg, lab, k, box):
    """inside mean, ring mean, ring std, contrast (ring - inside), snr; ring = 5 px, excludes other fg."""
    y0, x0, y1, x1 = box
    pad = RING_PX + 1
    Y0, X0 = max(y0 - pad, 0), max(x0 - pad, 0)
    Y1, X1 = min(y1 + pad, img.shape[0]), min(x1 + pad, img.shape[1])
    own = (lab[Y0:Y1, X0:X1] == k).astype(np.uint8)
    ring = (cv2.dilate(own, RING_K) > 0) & (own == 0)
    r2 = ring & (fg[Y0:Y1, X0:X1] == 0)
    if r2.sum() >= 10:
        ring = r2
    ic = img[Y0:Y1, X0:X1]
    ins, rv = ic[own > 0], ic[ring]
    im, rm, rs = float(ins.mean()), float(rv.mean()), float(rv.std())
    return im, rm, rs, rm - im, (rm - im) / max(rs, 1e-6)


def shape_desc(crop):
    """moment elongation sqrt(l1/l2) (1 = round) and bbox fill ratio of a boolean crop (information only)."""
    m = cv2.moments(crop.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return np.nan, np.nan
    a, b, c = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    t = np.sqrt(max((a - b) ** 2 / 4 + c * c, 0.0))
    l1, l2 = (a + b) / 2 + t, (a + b) / 2 - t
    return float(np.sqrt(l1 / max(l2, 1e-6))), float(crop.sum() / crop.size)


def components(p):
    fg = (p > THR).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
    C = seg.labels_to_instances(n, lab, stats, min_area=0)   # all components; index i <-> label i + 1
    assert len(C) == n - 1 and (C.areas == stats[1:, cv2.CC_STAT_AREA]).all()
    return fg, n, lab, stats, cents, C


def status_of(best_iou, nfrag):
    if best_iou > 0.5:
        return "matched"
    if nfrag >= 2:
        return "split"
    if best_iou < 0.1:
        return "missed"
    return "low_iou"


def file_job(args):
    stem, fold, gts = args
    t0 = time.time()
    with np.load(EXP / f"fold{fold}" / PROBS) as z:
        p = z[stem].astype(np.float32)
    img = D.read_image(D.train_image_path(stem + ".jpeg")).astype(np.float32)
    fg, n, lab, stats, cents, C = components(p)
    ac = C.areas.astype(np.int64)
    rids = sorted(gts)

    # ---------------- 1. record-level metric for every min_area (exact pipeline function)
    rec_rows = []
    for m in MINAREAS:
        P = seg.labels_to_instances(n, lab, stats, min_area=m)
        assert len(P) == int((ac >= max(m, 1)).sum())
        for rid in rids:
            r, _, _ = evaluate_image(gts[rid], P, CFG, image_key=rid, return_instances=False)
            rec_rows.append(dict(
                rid=rid, stem=stem, fold=fold, min_area=m, n_gt=r["n_gt"], n_pred=r["n_pred"],
                tp=r["tp_host"], fp=r["fp_host"], fn=r["fn_host"], si=r["sum_iou_host"],
                tp_u=r["tp"], fp_u=r["fp"], fn_u=r["fn"], si_u=r["sum_iou"],
                fn_missed=r["fn_missed"], fn_merged=r["fn_merged"], fn_split=r["fn_split"],
                fn_low_iou=r["fn_low_iou"], fp_spurious=r["fp_spurious"], fp_duplicate=r["fp_duplicate"],
                fp_merge=r["fp_merge"], fp_fragment=r["fp_fragment"], fp_low_iou=r["fp_low_iou"],
                n_split_gt=r["n_split_gt"], n_split_fragments=r["n_split_fragments"],
                n_merge_pred=r["n_merge_pred"], n_merged_gt=r["n_merged_gt"],
                thin_gt=r["thin_gt_px"], thin_cov=r["thin_gt_covered_px"],
                thin_pred=r["thin_pred_px"], thin_pin=r["thin_pred_in_gt_px"]))

    # ---------------- intersections of every GT instance with EVERY component (min_area 0)
    inters = {rid: intersection_matrix(gts[rid], C) for rid in rids}
    ious = {}
    for rid in rids:
        ag = gts[rid].areas.astype(np.int64)
        I = inters[rid]
        U = ag[:, None] + ac[None, :] - I
        ious[rid] = np.where(U > 0, I / np.maximum(U, 1), 0.0)

    # ---------------- 2. per-GT facts per min_area
    gt_rows = []
    for rid in rids:
        G = gts[rid]
        I, IoU = inters[rid], ious[rid]
        for g in range(len(G)):
            ag = int(G.areas[g])
            el, fill = shape_desc(G.crops[g])
            row = dict(rid=rid, stem=stem, fold=fold, gt_idx=g, area=ag,
                       box_y0=int(G.boxes[g, 0]), box_x0=int(G.boxes[g, 1]), elong=el, fill=fill)
            for m in MINAREAS:
                cols = ac >= max(m, 1)
                Ig, Jg = I[g, cols], IoU[g, cols]
                acm = ac[cols]
                row[f"biou_{m}"] = float(Jg.max()) if Jg.size else 0.0
                row[f"cov_{m}"] = float(Ig.max() / ag) if (Ig.size and ag) else 0.0
                row[f"nfrag_{m}"] = int(((Ig >= FRAG_PX) & (Ig >= FRAG_TAU * acm)).sum())
                row[f"nov_{m}"] = int((Ig >= OV_PX).sum())
            gt_rows.append(row)

    # ---------------- 4. fragmented GT (containment rule on ALL components)
    frag_rows = []
    for rid in rids:
        G = gts[rid]
        I, IoU = inters[rid], ious[rid]
        for g in range(len(G)):
            pieces = np.flatnonzero((I[g] >= FRAG_PX) & (I[g] >= FRAG_TAU * ac))
            if len(pieces) < 2:
                continue
            pa = ac[pieces]
            row = dict(rid=rid, stem=stem, fold=fold, gt_idx=g, gt_area=int(G.areas[g]),
                       n_pieces=len(pieces), n_small=int((pa < BASE).sum()), n_big=int((pa >= BASE).sum()),
                       piece_areas=";".join(str(int(a)) for a in sorted(pa, reverse=True)),
                       piece_labels=";".join(str(int(j) + 1) for j in pieces[np.argsort(-pa)]),
                       piece_best_iou=float(IoU[g, pieces].max()),
                       small_piece_px_inside=int(I[g, pieces[pa < BASE]].sum()),
                       big_piece_px_inside=int(I[g, pieces[pa >= BASE]].sum()),
                       small_piece_tp=int(((IoU[g, pieces] > 0.5) & (pa < BASE)).sum()))
            for m in (0, 100, 250, 500):
                cols = ac >= max(m, 1)
                b = float(IoU[g, cols].max()) if cols.any() else 0.0
                nf = int(((I[g, cols] >= FRAG_PX) & (I[g, cols] >= FRAG_TAU * ac[cols])).sum())
                row[f"biou_{m}"] = b
                row[f"nfrag_{m}"] = nf
                row[f"status_{m}"] = status_of(b, nf)
            frag_rows.append(row)

    # ---------------- 3. classify every component < 500 px
    comp_rows = []
    small = np.flatnonzero(ac < BASE)
    kept_lab = np.flatnonzero(ac >= BASE) + 1
    if len(kept_lab):
        not_kept = (~np.isin(lab, kept_lab)).astype(np.uint8)
        dist_kept = cv2.distanceTransform(not_kept, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    else:
        dist_kept = None
    for i in small:
        k = i + 1
        y0, x0, y1, x1 = (int(v) for v in C.boxes[i])
        crop = C.crops[i]
        pv = p[y0:y1, x0:x1][crop]
        im, rm, rs, con, snr = comp_contrast(img, fg, lab, k, (y0, x0, y1, x1))
        el, fill = shape_desc(crop)
        p05 = float(np.percentile(img[y0:y1, x0:x1][crop], 5))
        dk = float(dist_kept[y0:y1, x0:x1][crop].min()) if dist_kept is not None else np.inf
        cx, cy = float(cents[k, 0]), float(cents[k, 1])
        per_set_iou, per_set_tp, n_pairs, sum_iou, n_rec_fp = [], 0, 0, 0.0, 0
        any_gt_px, max_inside, cand = 0, 0.0, []
        for rid in rids:
            I, IoU = inters[rid], ious[rid]
            col_i, col_j = I[:, i], IoU[:, i]
            b = float(col_j.max()) if col_j.size else 0.0
            per_set_iou.append(b)
            pairs = col_j > 0.5
            if pairs.any():
                per_set_tp += 1
                n_pairs += int(pairs.sum())
                sum_iou += float(col_j[pairs].sum())
            else:
                n_rec_fp += 1
            any_gt_px += int(col_i.sum())
            if col_i.size:
                max_inside = max(max_inside, float(col_i.max() / ac[i]))
                for g in np.flatnonzero(col_i >= INSIDE_FRAC * ac[i]):
                    others = np.flatnonzero(I[g] >= OV_PX)
                    others = others[others != i]
                    cand.append(dict(rid=rid, g=int(g), inter=int(col_i[g]), others=others,
                                     g_matched_500=bool((IoU[g, ac >= BASE] > 0.5).any()) if (ac >= BASE).any() else False,
                                     g_area=int(gts[rid].areas[g])))
        best_iou = max(per_set_iou) if per_set_iou else 0.0
        info = dict(other_max_area=0, other_max_kept=False, cand_g_matched_500=False, cand_g_area=0, cand_rid="")
        sub = ""
        if best_iou > 0.5:
            cls = "a"
        elif cand:
            cand.sort(key=lambda d: -d["inter"])
            larger = [d for d in cand if (ac[d["others"]] > ac[i]).any()]
            anyother = [d for d in cand if len(d["others"])]
            if larger:
                cls, d0 = "b", larger[0]
            elif anyother:
                cls, d0 = "b_largest", anyother[0]
            else:
                cls, d0 = "c", cand[0]
            oa = ac[d0["others"]]
            info = dict(other_max_area=int(oa.max()) if oa.size else 0,
                        other_max_kept=bool(oa.size and oa.max() >= BASE),
                        cand_g_matched_500=d0["g_matched_500"], cand_g_area=d0["g_area"], cand_rid=d0["rid"])
        elif any_gt_px == 0:
            cls = "d" if con > rs else "e"
            sub = "" if cls == "d" else "e_noise"
        else:
            cls, sub = "e", "e_weak_gt"
        comp_rows.append(dict(
            stem=stem, fold=fold, label=k, area=int(ac[i]), y0=y0, x0=x0, y1=y1, x1=x1, cx=cx, cy=cy,
            r_frac=float(np.hypot(cx - CX, cy - CY) / RSUN), ew_frac=float((cx - CX) / RSUN),
            max_p=float(pv.max()), mean_p=float(pv.mean()),
            mean_inside=im, mean_ring=rm, ring_std=rs, contrast=con, contrast_snr=snr,
            elong=el, fill=fill, p05_inside=p05, ring_minus_p05=rm - p05, dist_to_kept=dk,
            n_sets=len(rids), best_iou=best_iou, per_set_best_iou=";".join(f"{v:.3f}" for v in per_set_iou),
            n_sets_tp=per_set_tp, n_pairs=n_pairs, sum_iou=sum_iou, n_rec_fp=n_rec_fp,
            gt_overlap_px=any_gt_px, max_inside_frac=max_inside, cls=cls, sub=sub or cls, **info))
    return dict(rec=rec_rows, gt=gt_rows, frag=frag_rows, comp=comp_rows,
                info=dict(stem=stem, fold=fold, n_comp=int(n - 1), n_small=int(len(small)), sec=time.time() - t0))


def stage_compute(workers):
    import multiprocessing as mp
    single_thread_worker()
    jobs = []
    for f in range(5):
        files = (EXP / f"fold{f}" / "val_files.txt").read_text().split()
        gts = seg.gt_records(files)
        by = {}
        for rid, g in gts.items():
            by.setdefault(rid.split("-", 1)[1], {})[rid] = g
        assert len(by) == len(files)
        jobs += [(s, f, by[s]) for s in sorted(by)]
    print(f"{len(jobs)} files, {sum(len(j[2]) for j in jobs)} records", flush=True)
    t0 = time.time()
    acc = {k: [] for k in ("rec", "gt", "frag", "comp", "info")}
    with mp.get_context("forkserver").Pool(workers, initializer=single_thread_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(file_job, jobs, chunksize=1)):
            for k in acc:
                acc[k] += res[k] if k != "info" else [res[k]]
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    for k, rows in acc.items():
        pd.DataFrame(rows).to_csv(RAW / f"{k}.csv", index=False)
    print(f"compute done {time.time() - t0:.0f}s", flush=True)


# ============================================================================ analyze stage
def pq_of(s):
    den = s["tp"] + 0.5 * s["fp"] + 0.5 * s["fn"]
    return dict(PQ=s["si"] / den, SQ=s["si"] / s["tp"] if s["tp"] else np.nan, RQ=s["tp"] / den)


def stage_analyze():
    rec = pd.read_csv(RAW / "rec.csv")
    gt = pd.read_csv(RAW / "gt.csv")
    frag = pd.read_csv(RAW / "frag.csv")
    comp = pd.read_csv(RAW / "comp.csv")
    info = pd.read_csv(RAW / "info.csv")
    S = {}
    n_files, n_rec = rec.stem.nunique(), rec[rec.min_area == BASE].rid.nunique()
    S["n_files"], S["n_records"] = int(n_files), int(n_rec)
    S["n_gt_instances"] = int(len(gt))
    S["n_components_total_thr05"] = int(info.n_comp.sum())
    S["n_components_lt500"] = int(info.n_small.sum())

    # ---------------- item 1: by_minarea
    num = [c for c in rec.columns if c not in ("rid", "stem", "fold", "min_area")]
    rows = []
    for m, d in rec.groupby("min_area"):
        s = d[num].sum()
        r = dict(min_area=int(m), **{k: float(v) for k, v in pq_of(s).items()})
        r.update(TP=int(s.tp), FP=int(s.fp), FN=int(s.fn), n_pred=int(s.n_pred), n_gt=int(s.n_gt),
                 PQ_unique=float(s.si_u / (s.tp_u + 0.5 * s.fp_u + 0.5 * s.fn_u)))
        for c in ("fn_missed", "fn_merged", "fn_split", "fn_low_iou", "fp_spurious", "fp_duplicate", "fp_merge",
                  "fp_fragment", "fp_low_iou", "n_split_gt", "n_split_fragments", "n_merge_pred", "n_merged_gt"):
            r[c] = int(s[c])
        r["thin_region_recall"] = float(s.thin_cov / s.thin_gt)
        r["thin_region_precision"] = float(s.thin_pin / s.thin_pred)
        for f, df in d.groupby("fold"):
            r[f"PQ_fold{f}"] = pq_of(df[num].sum())["PQ"]
        rows.append(r)
    bym = pd.DataFrame(rows).sort_values("min_area")
    base = bym.set_index("min_area").loc[BASE]
    for c in ("PQ", "TP", "FP", "FN"):
        bym[f"d{c}_vs500"] = bym[c] - base[c]
    # check against the stored b0 evaluation
    ref = json.load(open(EXP / "eval_oof_cc_f01234_t0.5_a500.json"))["all"]
    # the stored TP/FP/FN are UNIQUE-matching counts (filseg.aggregate 'TP'); host counts can differ by
    # non-unique candidate pairs (overlapping GT instances inside one set)
    s5 = rec[rec.min_area == BASE][["tp_u", "fp_u", "fn_u", "si_u"]].sum()
    S["check_vs_eval_oof_a500"] = dict(PQ_host_here=float(base.PQ), PQ_host_stored=ref["PQ_pooled_host"],
                                       PQ_unique_here=float(base.PQ_unique), PQ_unique_stored=ref["PQ_pooled"],
                                       TP_unique=[int(s5.tp_u), ref["TP"]], FP_unique=[int(s5.fp_u), ref["FP"]],
                                       FN_unique=[int(s5.fn_u), ref["FN"]],
                                       host_counts_here=[int(base.TP), int(base.FP), int(base.FN)])
    assert abs(base.PQ - ref["PQ_pooled_host"]) < 1e-9 and abs(base.PQ_unique - ref["PQ_pooled"]) < 1e-9
    assert int(s5.tp_u) == ref["TP"] and int(s5.fp_u) == ref["FP"] and int(s5.fn_u) == ref["FN"]
    # file-bootstrap CI of PQ(m) - PQ(500)
    rng = np.random.default_rng(SEED)
    stems = np.array(sorted(rec.stem.unique()))
    sums = {m: d.groupby("stem")[["tp", "fp", "fn", "si"]].sum().loc[stems].values for m, d in rec.groupby("min_area")}
    idxs = [rng.integers(len(stems), size=len(stems)) for _ in range(2000)]

    def pqv(a):
        return a[3] / (a[0] + 0.5 * a[1] + 0.5 * a[2])
    ci = {}
    for m in MINAREAS:
        diffs = np.array([pqv(sums[m][ix].sum(0)) - pqv(sums[BASE][ix].sum(0)) for ix in idxs])
        ci[m] = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    bym["dPQ_vs500_ci_lo"] = bym.min_area.map(lambda m: ci[m][0])
    bym["dPQ_vs500_ci_hi"] = bym.min_area.map(lambda m: ci[m][1])
    # marginal step table (adjacent grid points)
    bi = bym.set_index("min_area")
    step_rows = []
    for lo, hi in zip(MINAREAS[:-1], MINAREAS[1:]):
        step_rows.append(dict(step=f"{hi}->{lo}", dTP=int(bi.TP[lo] - bi.TP[hi]), dFP=int(bi.FP[lo] - bi.FP[hi]),
                              dPQ=float(bi.PQ[lo] - bi.PQ[hi]),
                              n_comp_added=int(((comp.area >= max(lo, 1)) & (comp.area < hi)).sum()) if hi <= BASE else None))
    S["marginal_steps"] = step_rows
    bym.to_csv(OUT / "by_minarea.csv", index=False)
    S["by_minarea"] = bym.to_dict(orient="records")

    # ---------------- item 2: strata (join with audit features, verified)
    feat = pd.read_csv(ROOT / "outputs" / "audit" / "instances.csv",
                       usecols=["image_id", "ann_idx", "ann_id", "area", "mask_bbox_x", "mask_bbox_y",
                                "contrast", "contrast_snr", "centroid_r_frac"])
    feat = feat.rename(columns={"image_id": "rid", "ann_idx": "gt_idx", "area": "area_audit"})
    gt = gt.merge(feat, on=["rid", "gt_idx"], how="left", validate="one_to_one")
    join = dict(n_gt=int(len(gt)), n_joined=int(gt.area_audit.notna().sum()),
                n_area_equal=int((gt.area_audit == gt.area).sum()),
                n_bbox_equal=int(((gt.mask_bbox_x == gt.box_x0) & (gt.mask_bbox_y == gt.box_y0)).sum()),
                n_ann_id_unique=int(gt.ann_id.nunique()), n_feat_rows=int(len(feat)))
    S["join_check_instances_csv"] = join
    assert join["n_joined"] == join["n_gt"] == join["n_area_equal"] == join["n_bbox_equal"] == join["n_ann_id_unique"]
    edges = [0, 250, 500, 1000, 2000, np.inf]
    names = ["<250", "250-500", "500-1000", "1000-2000", ">=2000"]
    gt["area_bin"] = pd.cut(gt.area, edges, right=False, labels=names)
    gt["contrast_q"], cq_edges = pd.qcut(gt.contrast, 5, labels=[f"Q{i}" for i in range(1, 6)], retbins=True)
    gt["snr_q"], sq_edges = pd.qcut(gt.contrast_snr, 5, labels=[f"Q{i}" for i in range(1, 6)], retbins=True)
    S["contrast_quintile_edges_greylevels"] = [float(v) for v in cq_edges]
    S["contrast_snr_quintile_edges"] = [float(v) for v in sq_edges]
    srows = []
    for strat in ("area_bin", "contrast_q", "snr_q", "all"):
        groups = [("all", gt)] if strat == "all" else list(gt.groupby(strat, observed=True))
        for name, d in groups:
            for m in MINAREAS:
                b = d[f"biou_{m}"]
                srows.append(dict(stratifier=strat, stratum=str(name), min_area=m, n=len(d),
                                  median_area=float(d.area.median()), median_contrast=float(d.contrast.median()),
                                  matched=float((b > 0.5).mean()), located=float((b >= 0.1).mean()),
                                  touched=float((d[f"cov_{m}"] >= TOUCH_FRAC).mean()),
                                  frag_only=float(((d[f"nfrag_{m}"] >= 2) & (b <= 0.5)).mean()),
                                  frag_only_loose=float(((d[f"nov_{m}"] >= 2) & (b <= 0.5)).mean()),
                                  untouched=float((d[f"cov_{m}"] < TOUCH_FRAC).mean())))
    strata = pd.DataFrame(srows)
    strata.to_csv(OUT / "strata.csv", index=False)
    # area x contrast cross-tab at 500 and 0 (information)
    xt = {}
    for m in (0, 100, 250, 500):
        xt[str(m)] = (gt.assign(mt=gt[f"biou_{m}"] > 0.5).pivot_table(index="area_bin", columns="contrast_q",
                                                                     values="mt", aggfunc="mean", observed=True)
                      .round(3).to_dict())
    S["matched_area_x_contrast"] = xt

    # ---------------- item 3: removed components
    comp["size_bin"] = pd.cut(comp.area, [1, 50, 100, 200, 250, 300, 400, 500], right=False,
                              labels=["1-49", "50-99", "100-199", "200-249", "250-299", "300-399", "400-499"])
    comp.to_csv(OUT / "removed_components.csv", index=False)
    # decomposition check: for host-style all-pairs counting, adding a component changes TP/FP/sum IoU
    # only through its own pairs (components are disjoint) -> exact per-component accounting.
    dec = {}
    for m in STEPS:
        c = comp[comp.area >= max(m, 1)]
        dTP, dFP, dSI = int(c.n_pairs.sum()), int(c.n_rec_fp.sum()), float(c.sum_iou.sum())
        recm = rec[rec.min_area == m][["tp", "fp", "fn", "si"]].sum()
        rec5 = rec[rec.min_area == BASE][["tp", "fp", "fn", "si"]].sum()
        dec[f"500->{m}"] = dict(dTP_components=dTP, dTP_records=int(recm.tp - rec5.tp), dFP_components=dFP,
                                dFP_records=int(recm.fp - rec5.fp), dFN_records=int(recm.fn - rec5.fn),
                                dSI_components=dSI, dSI_records=float(recm.si - rec5.si))
        assert dTP == recm.tp - rec5.tp and dFP == recm.fp - rec5.fp and abs(dSI - (recm.si - rec5.si)) < 1e-6
    S["decomposition_check"] = dec
    tot = rec[rec.min_area == BASE][["tp", "fp", "fn", "si"]].sum()
    cls_rows, cls_bin_rows = [], []
    for m in STEPS:
        c = comp[comp.area >= max(m, 1)]
        for cl in CLASS_ORDER + ["all"]:
            d = c if cl == "all" else c[c.cls == cl]
            # PQ if, starting from min_area 500, ONLY this class's removed components were added back
            # (oracle bookkeeping, not a realisable rule)
            t = dict(tp=tot.tp + d.n_pairs.sum(), fp=tot.fp + d.n_rec_fp.sum(), fn=tot.fn - d.n_pairs.sum(),
                     si=tot.si + d.sum_iou.sum())
            multi = d[d.n_sets >= 2]
            cls_rows.append(dict(step=f"500->{m}", cls=cl, n=len(d), share=len(d) / max(len(c), 1),
                                 median_area=float(d.area.median()) if len(d) else np.nan,
                                 median_max_p=float(d.max_p.median()) if len(d) else np.nan,
                                 median_contrast=float(d.contrast.median()) if len(d) else np.nan,
                                 median_snr=float(d.contrast_snr.median()) if len(d) else np.nan,
                                 frac_snr_gt1=float((d.contrast_snr > 1).mean()) if len(d) else np.nan,
                                 median_r_frac=float(d.r_frac.median()) if len(d) else np.nan,
                                 median_elong=float(d.elong.median()) if len(d) else np.nan,
                                 frac_elong_lt2=float((d.elong < 2).mean()) if len(d) else np.nan,
                                 frac_contrast_gt30=float((d.contrast > 30).mean()) if len(d) else np.nan,
                                 frac_compact_dark=float(((d.elong < 2) & (d.contrast > 25)).mean()) if len(d) else np.nan,
                                 median_dist_to_kept=float(d.dist_to_kept.median()) if len(d) else np.nan,
                                 frac_dist_to_kept_le20=float((d.dist_to_kept <= 20).mean()) if len(d) else np.nan,
                                 add_TP_pairs=int(d.n_pairs.sum()), add_FP_records=int(d.n_rec_fp.sum()),
                                 add_sum_iou=float(d.sum_iou.sum()),
                                 PQ_if_only_this_class_added=pq_of(t)["PQ"] - pq_of(tot)["PQ"],
                                 n_multi_set=len(multi),
                                 multi_matched_all_sets=int((multi.n_sets_tp == multi.n_sets).sum()),
                                 multi_matched_some_not_all=int(((multi.n_sets_tp > 0) & (multi.n_sets_tp < multi.n_sets)).sum()),
                                 multi_matched_one_set=int((multi.n_sets_tp == 1).sum()),
                                 e_weak_gt=int((d["sub"] == "e_weak_gt").sum()),
                                 other_kept=int(d.other_max_kept.sum()),
                                 cand_g_matched_500=int(d.cand_g_matched_500.sum())))
        for (sb, cl), d in c.groupby(["size_bin", "cls"], observed=True):
            cls_bin_rows.append(dict(step=f"500->{m}", size_bin=str(sb), cls=cl, n=len(d),
                                     add_TP_pairs=int(d.n_pairs.sum()), add_FP_records=int(d.n_rec_fp.sum())))
    cls_tab = pd.DataFrame(cls_rows)
    cls_bin = pd.DataFrame(cls_bin_rows)
    cls_tab.to_csv(OUT / "removed_class_summary.csv", index=False)
    cls_bin.to_csv(OUT / "removed_class_by_sizebin.csv", index=False)
    S["removed_class_summary"] = cls_tab.to_dict(orient="records")
    S["removed_class_by_sizebin_500to0"] = (cls_bin[cls_bin.step == "500->0"]
                                            .pivot_table(index="size_bin", columns="cls", values="n", fill_value=0)
                                            .astype(int).to_dict())
    # oracle bookkeeping per band (NOT realisable rules; upper/lower bounds of what keep/drop could do)
    def pq_add(d):
        return pq_of(dict(tp=tot.tp + d.n_pairs.sum(), fp=tot.fp + d.n_rec_fp.sum(), fn=tot.fn - d.n_pairs.sum(),
                          si=tot.si + d.sum_iou.sum()))["PQ"] - pq_of(tot)["PQ"]
    S["oracle_bookkeeping_dPQ_vs500"] = {
        f"[{lo},500)": dict(add_all=pq_add(comp[comp.area >= lo]),
                            add_only_a=pq_add(comp[(comp.area >= lo) & (comp.cls == "a")]),
                            add_all_but_b=pq_add(comp[(comp.area >= lo) & (comp.cls != "b")]),
                            add_a_and_d=pq_add(comp[(comp.area >= lo) & comp.cls.isin(["a", "d"])]),
                            add_dist_to_kept_gt20=pq_add(comp[(comp.area >= lo) & (comp.dist_to_kept > 20)]))
        for lo in (300, 250, 100)}
    dd = comp[(comp.cls == "d") & (comp.area >= 100)]
    S["class_d_500to100_by_n_sets"] = {str(k): int(v) for k, v in dd.n_sets.value_counts().sort_index().items()}
    S["class_d_500to100_frac_abs_ew_gt0.8"] = float((dd.ew_frac.abs() > 0.8).mean())
    # class (a): single-set vs multi-set
    a = comp[(comp.cls == "a") & (comp.area >= 100)]
    S["class_a_500to100_sets"] = dict(
        n=int(len(a)), single_set_file=int((a.n_sets == 1).sum()),
        multi_set_file=int((a.n_sets >= 2).sum()),
        multi_matched_all=int(((a.n_sets >= 2) & (a.n_sets_tp == a.n_sets)).sum()),
        multi_matched_one=int(((a.n_sets >= 2) & (a.n_sets_tp == 1)).sum()),
        three_set_matched_2=int(((a.n_sets == 3) & (a.n_sets_tp == 2)).sum()))
    # (d) contrast vs annotated small GT
    small_gt = gt[gt.area < 500]
    S["contrast_reference_annotated_lt500"] = dict(
        n=int(len(small_gt)), median_contrast=float(small_gt.contrast.median()),
        median_snr=float(small_gt.contrast_snr.median()), frac_snr_gt1=float((small_gt.contrast_snr > 1).mean()),
        median_elong=float(small_gt.elong.median()), frac_elong_lt2=float((small_gt.elong < 2).mean()),
        frac_contrast_gt30=float((small_gt.contrast > 30).mean()),
        frac_compact_dark=float(((small_gt.elong < 2) & (small_gt.contrast > 25)).mean()),
        note="audit contrast: ring excludes all instances of the set; component contrast: ring excludes p>0.5")
    S["contrast_reference_annotated_all"] = dict(
        n=int(len(gt)), median_contrast=float(gt.contrast.median()), median_elong=float(gt.elong.median()),
        frac_elong_lt2=float((gt.elong < 2).mean()),
        frac_compact_dark=float(((gt.elong < 2) & (gt.contrast > 25)).mean()))

    # ---------------- item 4: fragmentation
    F = frag
    pa_lists = F.piece_areas.astype(str).str.split(";").map(lambda v: [int(x) for x in v])
    all_p = np.concatenate(pa_lists.values) if len(F) else np.zeros(0)
    fr = dict(n_split_gt_at_0=int(len(F)), n_pieces=int(len(all_p)), n_pieces_lt500=int((all_p < BASE).sum()),
              n_pieces_lt100=int((all_p < 100).sum()), n_pieces_lt250=int((all_p < 250).sum()),
              frac_pieces_lt500=float((all_p < BASE).mean()),
              n_gt_all_pieces_small=int((F.n_big == 0).sum()), n_gt_one_big=int((F.n_big == 1).sum()),
              n_gt_ge2_big=int((F.n_big >= 2).sum()),
              n_gt_with_small_piece=int((F.n_small >= 1).sum()),
              transition_0_to_500=F.groupby(["status_0", "status_500"]).size().rename("n").reset_index()
              .to_dict(orient="records"),
              status_500_by_nbig=F.groupby(["n_big", "status_500"]).size().rename("n").reset_index()
              .to_dict(orient="records"),
              matched_at_0=int((F.status_0 == "matched").sum()), matched_at_500=int((F.status_500 == "matched").sum()),
              small_piece_tp=int(F.small_piece_tp.sum()),
              share_inside_px_small_pieces=float(F.small_piece_px_inside.sum() /
                                                 (F.small_piece_px_inside.sum() + F.big_piece_px_inside.sum())))
    # GT that are split at 500 (pieces all >= 500): from gt table
    fr["n_split_gt_containment_by_min_area"] = {str(m): int((gt[f"nfrag_{m}"] >= 2).sum()) for m in MINAREAS}
    fr["n_frag_only_by_min_area"] = {str(m): int(((gt[f"nfrag_{m}"] >= 2) & (gt[f"biou_{m}"] <= 0.5)).sum())
                                     for m in MINAREAS}
    # among split GTs, best IoU only depends on which pieces survive: never increases when pieces are removed
    fr["biou_increase_count"] = int((F.biou_500 > F.biou_0 + 1e-12).sum())
    fr["median_biou_0"] = float(F.biou_0.median())
    fr["median_biou_500"] = float(F.biou_500.median())
    # fp fragments: pieces of split GT that are < 500 and their record-level role
    S["fragmentation"] = fr
    F.to_csv(OUT / "fragmented_gt.csv", index=False)
    json.dump(S, open(OUT / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: S[k] for k in ("check_vs_eval_oof_a500", "join_check_instances_csv", "decomposition_check")},
                     indent=1, default=str))
    print(bym[["min_area", "PQ", "SQ", "RQ", "TP", "FP", "FN", "dPQ_vs500_ci_lo", "dPQ_vs500_ci_hi"]].to_string())
    print(cls_tab[["step", "cls", "n", "add_TP_pairs", "add_FP_records", "PQ_if_only_this_class_added"]].to_string())


# ============================================================================ figures stage
SET_COLS = [(0, 255, 0), (255, 160, 0), (255, 0, 255)]   # RGB, per annotation set (sorted id order)
KEPT_COL, TARGET_COL, OTHER_COL = (0, 255, 255), (255, 40, 40), (200, 200, 0)


def boundary(mask):
    m = mask.astype(np.uint8)
    er = cv2.erode(cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0),
                   cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))[1:-1, 1:-1]
    return (m > 0) & (er == 0)


def stretch(x):
    lo, hi = np.percentile(x, [0.5, 99.5])
    return np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8), float(lo), float(hi)


def put_lines(canvas, lines, y0=14, scale=0.42):
    for i, t in enumerate(lines):
        cv2.putText(canvas, t, (6, y0 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_job(args):
    kind, ex, out_path = args
    stem, fold = ex["stem"], int(ex["fold"])
    with np.load(EXP / f"fold{fold}" / PROBS) as z:
        p = z[stem].astype(np.float32)
    img = D.read_image(D.train_image_path(stem + ".jpeg")).astype(np.float32)
    fg, n, lab, stats, cents, C = components(p)
    gts = seg.gt_records([stem + ".jpeg"])
    rids = sorted(gts)
    if kind == "comp":
        y0, x0, y1, x1 = int(ex["y0"]), int(ex["x0"]), int(ex["y1"]), int(ex["x1"])
        target_labels = [int(ex["label"])]
    else:
        G = gts[ex["rid"]]
        g = int(ex["gt_idx"])
        y0, x0, y1, x1 = (int(v) for v in G.boxes[g])
        target_labels = [int(v) for v in str(ex["piece_labels"]).split(";")]
    S = max(256, int(np.ceil(max(y1 - y0, x1 - x0) + 64)))
    cy_, cx_ = (y0 + y1) // 2, (x0 + x1) // 2
    Y0 = int(np.clip(cy_ - S // 2, 0, 2048 - S)); X0 = int(np.clip(cx_ - S // 2, 0, 2048 - S))
    sl = (slice(Y0, Y0 + S), slice(X0, X0 + S))
    raw8, lo, hi = stretch(img[sl])
    base_rgb = np.stack([raw8] * 3, -1)
    dim = (base_rgb * 0.75).astype(np.uint8)
    # panel 2: components
    p2 = dim.copy()
    labc = lab[sl]
    areas = stats[:, cv2.CC_STAT_AREA]
    for k in np.unique(labc):
        if k == 0:
            continue
        if kind == "comp" and k in target_labels:
            continue
        if kind == "frag" and k in target_labels and areas[k] < BASE:
            continue
        col = KEPT_COL if areas[k] >= BASE else OTHER_COL
        p2[boundary(labc == k)] = col
    for k in target_labels:
        if kind == "frag" and areas[k] >= BASE:
            continue
        p2[boundary(labc == k)] = TARGET_COL
    # panel 3: GT outlines per set
    p3 = dim.copy()
    for si, rid in enumerate(rids):
        G = gts[rid]
        for gi in range(len(G)):
            gy0, gx0, gy1, gx1 = G.boxes[gi]
            if gy1 <= Y0 or gy0 >= Y0 + S or gx1 <= X0 or gx0 >= X0 + S:
                continue
            m = np.zeros((S, S), bool)
            ty0, tx0 = max(gy0, Y0), max(gx0, X0)
            ty1, tx1 = min(gy1, Y0 + S), min(gx1, X0 + S)
            m[ty0 - Y0:ty1 - Y0, tx0 - X0:tx1 - X0] = G.crops[gi][ty0 - gy0:ty1 - gy0, tx0 - gx0:tx1 - gx0]
            p3[boundary(m)] = SET_COLS[si % 3]
    # panel 4: probability
    pr = np.clip(p[sl] * 255, 0, 255).astype(np.uint8)
    p4 = np.stack([pr] * 3, -1)
    gap = 4
    W = 4 * S + 3 * gap
    hdr_h = 66
    lab_h = 16
    canvas = np.zeros((hdr_h + lab_h + S, W, 3), np.uint8)
    for j, pnl in enumerate((base_rgb, p2, p3, p4)):
        canvas[hdr_h + lab_h:, j * (S + gap):j * (S + gap) + S] = pnl
    set_txt = "  ".join(f"{rid.split('-')[0]}={nm}" for rid, nm in zip(rids, ("green", "orange", "magenta")))
    if kind == "comp":
        lines = [f"class ({ex['cls']}{'' if ex['sub'] == ex['cls'] else ' / ' + ex['sub']})  {stem} fold {fold}  "
                 f"comp label {int(ex['label'])}  area {int(ex['area'])} px  crop {S}x{S} at y{Y0} x{X0}",
                 f"max p {ex['max_p']:.2f}  mean p {ex['mean_p']:.2f}  best IoU {ex['best_iou']:.2f} "
                 f"(per set {ex['per_set_best_iou']})  max inside one GT {ex['max_inside_frac']:.2f}",
                 f"contrast ring-inside {ex['contrast']:.1f} grey (ring std {ex['ring_std']:.1f}, snr "
                 f"{ex['contrast_snr']:.2f})  r/R {ex['r_frac']:.2f}  sets: {set_txt}"]
    else:
        lines = [f"fragmented GT  {ex['rid']} gt#{int(ex['gt_idx'])} area {int(ex['gt_area'])}  {stem} fold {fold}  "
                 f"crop {S}x{S} at y{Y0} x{X0}",
                 f"pieces {int(ex['n_pieces'])} (areas {ex['piece_areas']}); <500: {int(ex['n_small'])}  "
                 f"best IoU a0 {ex['biou_0']:.2f} -> a500 {ex['biou_500']:.2f}",
                 f"status a0 {ex['status_0']} -> a500 {ex['status_500']}  sets: {set_txt}"]
    put_lines(canvas, lines)
    titles = [f"raw linear {lo:.0f}-{hi:.0f} (p0.5-p99.5)",
              "cyan kept>=500, red " + ("piece<500" if kind == "frag" else "target") + ", yellow <500",
              "GT outlines per set (colours in header)", "prob 0 black - 1 white"]
    for j, t in enumerate(titles):
        cv2.putText(canvas, t, (j * (S + gap) + 3, hdr_h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return str(out_path), dict(crop_y0=Y0, crop_x0=X0, crop=S)


def stage_figures(workers, n_per_class=6, n_frag=4):
    import multiprocessing as mp
    single_thread_worker()
    FIG.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(OUT / "removed_components.csv")
    frag = pd.read_csv(OUT / "fragmented_gt.csv")
    rng = np.random.default_rng(SEED)
    jobs, index = [], []
    pool_c = comp[(comp.area >= 100) & (comp.area < BASE)].sort_values(["stem", "label"]).reset_index(drop=True)
    picked = {}
    for cl in CLASS_ORDER:
        d = pool_c[pool_c.cls == cl]
        k = min(n_per_class, len(d))
        pick = np.sort(rng.choice(len(d), size=k, replace=False)) if k else []
        picked[cl] = set(int(v) for v in pick)
        for j, ix in enumerate(pick):
            ex = d.iloc[ix].to_dict()
            path = FIG / f"removed_{cl}_{j + 1}_{ex['stem']}_L{int(ex['label'])}.png"
            jobs.append(("comp", ex, path))
            index.append(dict(kind="removed_component", cls=cl, file=path.name, stem=ex["stem"], label=int(ex["label"]),
                              area=int(ex["area"]), best_iou=ex["best_iou"], contrast=ex["contrast"],
                              snr=ex["contrast_snr"], n_in_class=len(d)))
    fd = frag[frag.n_small >= 1].sort_values(["rid", "gt_idx"]).reset_index(drop=True)
    pick = np.sort(rng.choice(len(fd), size=min(n_frag, len(fd)), replace=False))
    for j, ix in enumerate(pick):
        ex = fd.iloc[ix].to_dict()
        path = FIG / f"fragmented_{j + 1}_{ex['rid']}_g{int(ex['gt_idx'])}.png"
        jobs.append(("frag", ex, path))
        index.append(dict(kind="fragmented_gt", cls="frag", file=path.name, stem=ex["stem"], rid=ex["rid"],
                          gt_idx=int(ex["gt_idx"]), n_pieces=int(ex["n_pieces"]), n_small=int(ex["n_small"]),
                          status_0=ex["status_0"], status_500=ex["status_500"], n_in_class=len(fd)))
    # second random batch for class (d) (largest FP source); drawn AFTER all draws above, so the first
    # batches are unchanged; excludes the first (d) batch
    d = pool_c[pool_c.cls == "d"]
    rest = np.array(sorted(set(range(len(d))) - picked["d"]))
    pick = np.sort(rng.choice(rest, size=min(n_per_class, len(rest)), replace=False))
    for j, ix in enumerate(pick):
        ex = d.iloc[ix].to_dict()
        path = FIG / f"removed_d_extra_{j + 7}_{ex['stem']}_L{int(ex['label'])}.png"
        jobs.append(("comp", ex, path))
        index.append(dict(kind="removed_component_extra", cls="d", file=path.name, stem=ex["stem"],
                          label=int(ex["label"]), area=int(ex["area"]), best_iou=ex["best_iou"],
                          contrast=ex["contrast"], snr=ex["contrast_snr"], n_in_class=len(d)))
    with mp.get_context("forkserver").Pool(workers, initializer=single_thread_worker) as pool:
        res = pool.map(render_job, jobs, chunksize=1)
    for row, (_, extra) in zip(index, res):
        row.update(extra)
    pd.DataFrame(index).to_csv(FIG / "figures_index.csv", index=False)
    print(pd.DataFrame(index)[["kind", "cls", "file"]].to_string())


def stage_plot():
    """Trade-off curve: (left) pooled PQ vs min_area with 95 % file-bootstrap CI of PQ(m) - PQ(500) drawn
    around PQ(500)+diff; (right) TP and FP added (or removed) relative to min_area 500. One y-axis per panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = pd.read_csv(OUT / "by_minarea.csv").sort_values("min_area")
    c1, c2, ink, ink2, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    x = b.min_area.values
    p500 = float(b.set_index("min_area").PQ[BASE])
    a0 = ax[0]
    lo, hi = p500 + b.dPQ_vs500_ci_lo.values, p500 + b.dPQ_vs500_ci_hi.values
    a0.fill_between(x, lo, hi, color=c1, alpha=0.15, linewidth=0)
    a0.plot(x, b.PQ.values, color=c1, lw=2, marker="o", ms=5, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    for xi, yi in zip(x, b.PQ.values):
        if xi in (0, 100, 250, 300, 500, 800):
            a0.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=8, color=ink2)
    a0.set_title("Pooled PQ (host) vs min_area", loc="left", fontsize=10, color=ink)
    a0.set_ylabel("PQ", color=ink2, fontsize=9)
    a1 = ax[1]
    a1.axhline(0, color=ink2, lw=0.8)
    a1.plot(x, b.dTP_vs500.values, color=c1, lw=2, marker="o", ms=5, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
            label="TP pairs added")
    a1.plot(x, b.dFP_vs500.values, color=c2, lw=2, marker="o", ms=5, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
            label="FP records added")
    a1.annotate("FP added", (x[0], b.dFP_vs500.values[0]), textcoords="offset points", xytext=(6, -2), fontsize=8,
                color=ink2)
    a1.annotate("TP added", (x[0], b.dTP_vs500.values[0]), textcoords="offset points", xytext=(6, 6), fontsize=8,
                color=ink2)
    a1.set_title("Change vs min_area 500 (count, all 1154 records)", loc="left", fontsize=10, color=ink)
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    for a_ in ax:
        a_.set_facecolor("#fcfcfb")
        a_.set_xlabel("min_area (px, full resolution)", color=ink2, fontsize=9)
        a_.set_xticks(x)
        a_.tick_params(colors=ink2, labelsize=8)
        a_.grid(axis="y", color=grid, lw=0.6)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(grid)
    fig.text(0.01, 0.01, "b0 OOF, thr 0.5, 8-conn. components; shaded = 95 % file-bootstrap CI of PQ(m) - PQ(500), "
             "2000 resamples, seed 2026", fontsize=7, color=ink2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "tradeoff_curve.png", facecolor=fig.get_facecolor())
    print("saved", FIG / "tradeoff_curve.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["compute", "analyze", "figures", "plot", "all"], default="all")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("compute", "all"):
        stage_compute(a.workers)
    if a.stage in ("analyze", "all"):
        stage_analyze()
    if a.stage in ("figures", "all"):
        stage_figures(a.workers)
    if a.stage in ("plot", "all"):
        stage_plot()


if __name__ == "__main__":
    main()
