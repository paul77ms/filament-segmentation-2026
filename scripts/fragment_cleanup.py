"""Fragment cleanup after R2 (registry row R2b0_fragment_cleanup; rule registered before any result). CPU only.

Base: R2nb0 (R2 refinement) instances from nested_r2b0_eval.instances with eval_dual's CFG (the code path of
eval_dual.instances / width_correction / merge_guard), outer-validation files of the 5 nested folds (707 files).

CLEANUP (fixed before running, not tuned), per R2nb0 instance, independently of the other instances:
  parts = 8-connected components of the instance mask (cv2, on the tight crop; equivalent to the full image because
          the crop contains every pixel of the instance).
  if the instance has >= 2 parts: drop every part with area < A px, except the largest part (largest = maximum area,
          ties -> lowest label), which is never dropped.  Nothing else changes: instance order, the other parts, the
          other instances; min_area is NOT re-applied (an instance may end below 300 px; counted).
  Only pixels are removed -> no new overlaps; every cleaned mask is a subset of its R2nb0 mask.
  A = 20 px  -> R2nb0_fc20  (PRIMARY, the registered method)
  A = 50 px  -> R2nb0_fc50  (SENSITIVITY row only; the rule is decided on A = 20)

RULE vs base R2nb0 (all must hold) -- copied from experiments/registry.csv R2b0_fragment_cleanup:
  F1 V0 diff >= 0, or CI includes 0 with lower bound > -0.003; F2 the same for V1; F3 notebook N:1 and 1:N counts not
  higher AND mean pair IoU not lower; F4 no LOAO aspect at/above human under R2nb0 that falls below human; F5 32-case
  review set (outputs/audit/review_cases.csv) PQ not lower under V0 and V1.
  CIs: file bootstrap, seed 2026, 2000 draws (refine_b0exist_checks.boot_diff / eval_dual.v1_boot); per-fold diffs.

Commands (outputs in outputs/research/fragment_cleanup/):
  table     per file: R2nb0, R2nb0_fc20, R2nb0_fc50 -> records.csv (V0 host counts + causes + thin + notebook diag + pair
            decomposition vs R2nb0), images_v1.csv, instances.csv (every R2nb0 instance: parts, parts/pixels dropped)
  loao      eval_dual's loao code path with R2nb0 and both cleaned methods -> loao/*.csv
  analyse   comparisons, diagnostics, LOAO aspects, rule -> summary.json, methods_table.csv
  testcsv   the same cleanup (A = 20) applied IN MEMORY to the test CSV before cleanup (read only) -> change counts
            (test_changes.json, test_instances.csv) and the would-be CSV test_r2b0_cleanup_a20.csv (same ids / order)
  CUDA_VISIBLE_DEVICES="" NUMPY_MADVISE_HUGEPAGE=0 taskset -c 0-3 .venv/bin/python scripts/fragment_cleanup.py <cmd>
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

import eval_dual as ED  # noqa: E402
import a_eval as AE  # noqa: E402
import nested_r2b0_eval as NE  # noqa: E402
import refine_b0exist_checks as RC  # noqa: E402
from filseg.metrics import InstanceSet, single_thread_worker  # noqa: E402

OUT = ROOT / "outputs" / "research" / "fragment_cleanup"
A_MAIN, A_SENS = 20, 50
AS = (A_MAIN, A_SENS)
BASE = "R2nb0"


def mname(A):
    return f"{BASE}_fc{A}"


MAIN, SENS = mname(A_MAIN), mname(A_SENS)
METHODS = (BASE, MAIN, SENS)
LOAO_RATERS = METHODS
TEST_CSV = ROOT / "outputs" / "submissions" / "track2" / "r2b0_5fold_t05_a500_r2a300.csv"   # read only
TEST_OUT = OUT / "test_r2b0_cleanup_a20.csv"
HW = 2048


# ============================================================================ cleanup
def parts(crop):
    """8-connected parts of one instance crop -> (label image, part areas (index j = label j + 1))."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(crop, np.uint8), connectivity=8)
    return lab, st[1:, cv2.CC_STAT_AREA].astype(np.int64)


def clean_crop(crop, A, lp=None):
    """-> (new tight crop with its offset (dy0, dx0) relative to the old crop, or None if unchanged, info dict)."""
    lab, ar = lp if lp is not None else parts(crop)
    info = dict(n_parts=int(len(ar)), n_drop=0, px_drop=0)
    if len(ar) < 2:
        return None, info
    big = int(np.argmax(ar))                        # never dropped (ties -> lowest label)
    drop = np.array([j for j in range(len(ar)) if j != big and ar[j] < A], np.int64)
    if not len(drop):
        return None, info
    info.update(n_drop=int(len(drop)), px_drop=int(ar[drop].sum()))
    keep = np.setdiff1d(np.arange(1, len(ar) + 1), drop + 1)
    (y0, x0, y1, x1), c = InstanceSet._crop(np.isin(lab, keep))
    return ((y0, x0), c), info


def inst_info(crop):
    lab, ar = parts(crop)
    srt = np.sort(ar)[::-1]
    d = dict(area=int(ar.sum()), n_parts=int(len(ar)), largest_px=int(srt[0]) if len(ar) else 0,
             minor_px=int(srt[1:].sum()), minor_lt20=int((srt[1:] < 20).sum()),
             minor_20_50=int(((srt[1:] >= 20) & (srt[1:] < 50)).sum()), minor_ge50=int((srt[1:] >= 50).sum()))
    res = {}
    for A in AS:
        new, ci = clean_crop(crop, A, (lab, ar))
        after = new[1] if new is not None else crop
        n_after = int(len(ar) - ci["n_drop"])
        d.update({f"drop_parts_a{A}": ci["n_drop"], f"drop_px_a{A}": ci["px_drop"],
                  f"area_after_a{A}": int(after.sum()), f"n_parts_after_a{A}": n_after})
        res[A] = new
    return d, res


def cleanup(P, A, news=None):
    """Apply the cleanup with threshold A to every instance of P -> (InstanceSet, changed indices)."""
    boxes = [tuple(int(v) for v in b) for b in P.boxes]
    crops = list(P.crops)
    changed = []
    for k in range(len(P)):
        new = news[k] if news is not None else clean_crop(P.crops[k], A)[0]
        if new is None:
            continue
        (dy0, dx0), c = new
        y0, x0 = boxes[k][0] + dy0, boxes[k][1] + dx0
        boxes[k] = (y0, x0, y0 + c.shape[0], x0 + c.shape[1])
        crops[k] = c
        changed.append(k)
    if not changed:
        return P, ()
    S = InstanceSet._build(P.shape, boxes, crops)
    assert (S.areas <= P.areas).all() and (S.areas > 0).all()
    return S, tuple(changed)


# ============================================================================ table
def table_job(args):
    stem, fold, gts = args
    t0 = time.time()
    M, _ = NE.instances(stem, fold, ED.CFG)
    P = M[BASE]
    gb = {rid: AE.gt_bins(G)[1] for rid, G in gts.items()}
    insts, news = [], {A: [] for A in AS}
    for k in range(len(P)):
        d, res = inst_info(P.crops[k])
        insts.append(dict(stem=stem, fold=fold, idx=k, box=";".join(str(int(v)) for v in P.boxes[k]), **d))
        for A in AS:
            news[A].append(res[A])
    rec, img, _ = AE._eval(BASE, P, gts, stem, fold)
    for x in rec:
        x.update(**AE.pair_stats(gts[x["rid"]], P, P, gb[x["rid"]]), n_changed=0)
    recs, imgs = list(rec), [img]
    for A in AS:
        nm = mname(A)
        S, ch = cleanup(P, A, news[A])
        if not ch:
            recs += [dict(x, method=nm) for x in rec]; imgs.append(dict(img, method=nm))
            continue
        r2, i2, _ = AE._eval(nm, S, gts, stem, fold)
        for x in r2:
            x.update(**AE.pair_stats(gts[x["rid"]], S, P, gb[x["rid"]]), n_changed=len(ch))
        recs += r2; imgs.append(i2)
    return recs, imgs, insts, time.time() - t0


def cmd_table(a):
    import multiprocessing as mp
    single_thread_worker()
    jobs = AE.all_jobs()
    if a.limit:
        jobs = jobs[:a.limit]
    t0 = time.time()
    R, I, N, secs = [], [], [], []
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for i, (rec, img, ins, s) in enumerate(pool.imap_unordered(table_job, jobs, chunksize=2)):
            R += rec; I += img; N += ins; secs.append(s)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} files {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(R)
    sp = RC.split_table().set_index("stem")
    df["group_id"] = df.stem.map(sp.group_id); df["n_sets"] = df.stem.map(sp.n_sets)
    sfx = "_test" if a.limit else ""
    df.to_csv(OUT / f"records{sfx}.csv", index=False)
    pd.DataFrame(I).to_csv(OUT / f"images_v1{sfx}.csv", index=False)
    pd.DataFrame(N).to_csv(OUT / f"instances{sfx}.csv", index=False)
    print(f"table: {len(df)} rows, {df.stem.nunique()} files, {time.time() - t0:.0f}s wall, "
          f"{sum(secs):.0f}s cpu-job, max file {max(secs):.1f}s")


# ============================================================================ LOAO (eval_dual code path)
def _loao_instances(stem, fold):
    M, _ = NE.instances(stem, fold, ED.CFG)
    for A in AS:
        M[mname(A)] = cleanup(M[BASE], A)[0]
    return M


def loao_wrap(args):
    ED.instances = _loao_instances
    ED.LOAO_RATERS = LOAO_RATERS
    return ED.loao_job(args)


def cmd_loao(a):
    ED.OUT = OUT / "loao"
    ED.LOAO_RATERS = LOAO_RATERS
    ED.loao_job = loao_wrap
    t0 = time.time()
    ED.cmd_loao(a)
    print(f"loao {time.time() - t0:.0f}s")


# ============================================================================ analyse
def ci_ok(x):
    return bool(x["diff"] >= 0 or (x["ci95"][0] <= 0 <= x["ci95"][1] and x["ci95"][0] > -0.003))


def ratio(n, d):
    return float(n / d) if d else float("nan")


def compare(df, I, sp, review, main, asp, BH):
    v0, _ = RC.compare(df, BASE, main, sp, review)
    A = df[df.method == BASE].sort_values("rid").reset_index(drop=True)
    B = df[df.method == main].sort_values("rid").reset_index(drop=True)
    A, B = A.assign(pair_n_d=A.pair_n), B.assign(pair_n_d=B.pair_n)   # boot_diff needs distinct column names
    pb = RC.boot_diff(A, B, "stem", extra={"pair_iou": ("pair_iou_sum", "pair_n"),
                                           "pair_dice": ("pair_dice_sum", "pair_n_d"),
                                           "thin_recall": ("thin_gt_covered_px", "thin_gt_px"),
                                           "thin_precision": ("thin_pred_in_gt_px", "thin_pred_px")})
    v1 = ED.v1_boot(I, BASE, main)
    rv1 = ED.v1_boot(I[I.stem.isin(review)], BASE, main)
    dA, dB = ED.diag_sums(A), ED.diag_sums(B)
    pf = {}
    for k in range(5):
        a_, b_ = A[A.fold == k].sum(numeric_only=True), B[B.fold == k].sum(numeric_only=True)
        pf[k] = dict(pair_iou=float(b_.pair_iou_sum / b_.pair_n - a_.pair_iou_sum / a_.pair_n),
                     pair_dice=float(b_.pair_dice_sum / b_.pair_n - a_.pair_dice_sum / a_.pair_n),
                     N1=int(b_.nb_pred_Nto1 - a_.nb_pred_Nto1), oneN=int(b_.nb_gt_1toN - a_.nb_gt_1toN))
    cA, cB = RC.summary_counts(A), RC.summary_counts(B)
    res = dict(base=BASE, main=main, n_files=int(B.stem.nunique()), n_records=int(len(B)),
               n_files_changed=int(B[B.n_changed > 0].stem.nunique()),
               n_records_v0_changed=int((A[RC.C4].values != B[RC.C4].values).any(1).sum()),
               V0=dict(A=v0["A"]["PQ"], B=v0["B"]["PQ"], diff=v0["file_boot"]["PQ"]["diff"],
                       ci95=v0["file_boot"]["PQ"]["ci95"], per_fold={k: v["diff"] for k, v in v0["per_fold"].items()},
                       group_boot=v0["group_boot"]["PQ"], sign=v0["per_file_sign"], review=v0["review_set_f01"],
                       SQ=[cA["SQ"], cB["SQ"]], RQ=[cA["RQ"], cB["RQ"]], TP=[cA["TP"], cB["TP"]],
                       FP=[cA["FP"], cB["FP"]], FN=[cA["FN"], cB["FN"]]),
               V1=dict(v1, SQ=[AE.v1_sq(I, BASE), AE.v1_sq(I, main)],
                       TP=[int(I[I.method == BASE].v1_tp.sum()), int(I[I.method == main].v1_tp.sum())],
                       FP=[int(I[I.method == BASE].v1_fp.sum()), int(I[I.method == main].v1_fp.sum())],
                       FN=[int(I[I.method == BASE].v1_fn.sum()), int(I[I.method == main].v1_fn.sum())]),
               V1_review=dict(A=rv1["A"], B=rv1["B"], diff=rv1["diff"], ci95=rv1["ci95"]),
               pair_boot={k: pb[k] for k in ("pair_iou", "pair_dice", "thin_recall", "thin_precision")},
               per_fold_diag=pf, diagnostics={BASE: dA, main: dB},
               causes={BASE: {k: cA.get(k) for k in CAUSES}, main: {k: cB.get(k) for k in CAUSES}},
               pair_decomposition=AE.pair_decomp(B))
    newly = None
    if asp is not None:
        res["loao_aspects"] = {s: {r: (None if asp[r].get(s) is None else float(asp[r].get(s)))
                                   for r in ("human", BASE, main)} for s in BH.ASPECTS}
        newly, undet, base_at = [], [], []
        for s, d in BH.ASPECTS.items():
            def v(r):
                x = asp[r].get(s)
                return np.nan if x is None else float(x)
            nb_b, nb_m = BH.not_worse(v(BASE), v("human"), d), BH.not_worse(v(main), v("human"), d)
            if nb_b is True:
                base_at.append(s)
            if nb_b is True and nb_m is False:
                newly.append(s)
            if nb_b is True and nb_m is None:
                undet.append(s)
        res["loao_base_at_or_above_human"] = base_at
        res["loao_newly_below_human"], res["loao_undetermined"] = newly, undet
    rule = dict(
        F1_v0=ci_ok(res["V0"]),
        F2_v1=ci_ok(v1),
        F3_n1_and_1n_not_higher_and_pair_iou_not_lower=bool(dB["nb_pred_Nto1"] <= dA["nb_pred_Nto1"] and
                                                          dB["nb_gt_1toN"] <= dA["nb_gt_1toN"] and
                                                          dB["pair_iou_mean"] >= dA["pair_iou_mean"]),
        F4_no_loao_aspect_newly_below_human=(None if newly is None else not newly),
        F5_review_not_lower_v0_and_v1=bool(res["V0"]["review"]["B"] >= res["V0"]["review"]["A"] and rv1["B"] >= rv1["A"]))
    rule["PASS"] = all(v is True for v in rule.values())
    res["rule_numbers"] = dict(N1=[dA["nb_pred_Nto1"], dB["nb_pred_Nto1"]], oneN=[dA["nb_gt_1toN"], dB["nb_gt_1toN"]],
                               pair_iou=[dA["pair_iou_mean"], dB["pair_iou_mean"]],
                               review_V0=[res["V0"]["review"]["A"], res["V0"]["review"]["B"]],
                               review_V1=[rv1["A"], rv1["B"]])
    res["rule"] = rule
    return res


CAUSES = ("fn_missed", "fn_low_iou", "fn_split", "fn_merged", "fp_spurious", "fp_low_iou", "fp_fragment", "fp_merge",
          "fp_duplicate")


def instance_stats(ins):
    out = dict(n_instances=int(len(ins)), n_files=int(ins.stem.nunique()),
               multipart_before=int((ins.n_parts >= 2).sum()),
               multipart_share_before=ratio((ins.n_parts >= 2).sum(), len(ins)),
               n_minor_parts=int((ins.n_parts - 1).clip(lower=0).sum()),
               minor_parts_lt20=int(ins.minor_lt20.sum()), minor_parts_20_50=int(ins.minor_20_50.sum()),
               minor_parts_ge50=int(ins.minor_ge50.sum()), minor_px=int(ins.minor_px.sum()),
               total_px=int(ins.area.sum()))
    for A in AS:
        ch = ins[ins[f"drop_parts_a{A}"] > 0]
        out[f"a{A}"] = dict(
            instances_changed=int(len(ch)), files_changed=int(ch.stem.nunique()),
            parts_removed=int(ins[f"drop_parts_a{A}"].sum()), px_removed=int(ins[f"drop_px_a{A}"].sum()),
            px_removed_share_of_all=ratio(ins[f"drop_px_a{A}"].sum(), ins.area.sum()),
            px_removed_share_of_changed=ratio(ch[f"drop_px_a{A}"].sum(), ch.area.sum()),
            multipart_after=int((ins[f"n_parts_after_a{A}"] >= 2).sum()),
            multipart_share_after=ratio((ins[f"n_parts_after_a{A}"] >= 2).sum(), len(ins)),
            became_single_part=int(((ins.n_parts >= 2) & (ins[f"n_parts_after_a{A}"] == 1)).sum()),
            area_after_lt300=int((ins[f"area_after_a{A}"] < 300).sum()),
            area_after_lt300_changed=int((ch[f"area_after_a{A}"] < 300).sum()),
            per_fold={int(k): dict(instances_changed=int((g[f"drop_parts_a{A}"] > 0).sum()),
                                   parts_removed=int(g[f"drop_parts_a{A}"].sum()), px_removed=int(g[f"drop_px_a{A}"].sum()))
                      for k, g in ins.groupby("fold")})
    return out


def cmd_analyse(a):
    df = pd.read_csv(OUT / "records.csv")
    I = pd.read_csv(OUT / "images_v1.csv")
    ins = pd.read_csv(OUT / "instances.csv")
    sp = RC.split_table()
    RC.FOLDS = (0, 1, 2, 3, 4)
    rc = pd.read_csv(ROOT / "outputs/audit/review_cases.csv")
    review = set(rc.file_name.str[:-5])
    lo = OUT / "loao" / "loao_records.csv"
    asp, BH = None, None
    if lo.exists():
        ED.OUT = OUT / "loao"
        ED.LOAO_RATERS = LOAO_RATERS
        asp, BH = ED.loao_aspects()
    res = dict(method="drop 8-connected parts < A px of multi-part R2nb0 instances; largest part kept",
               A_primary=A_MAIN, A_sensitivity=A_SENS, instances=instance_stats(ins), comparisons={})
    for m in (MAIN, SENS):
        res["comparisons"][m] = compare(df, I, sp, review, m, asp, BH)
        if asp is not None:
            res["comparisons"][m]["loao_n_files"] = int(pd.read_csv(lo).stem.nunique())
    # reproduction of the base against eval_dual (V0 host counts, notebook diag, V1 counts)
    ev = pd.read_csv(ROOT / "outputs/research/eval_dual/records.csv")
    evi = pd.read_csv(ROOT / "outputs/research/eval_dual/images_v1.csv")
    cols = RC.C4 + ["pair_n", "pair_iou_sum", "nb_pred_Nto1", "nb_gt_1toN"]
    c1 = ["v1_tp", "v1_fp", "v1_fn", "v1_si"]
    dbase = ED.diag_sums(df[df.method == BASE])
    res["base_check"] = dict(
        reproduces_eval_dual_v0=bool(np.allclose(ev[ev.method == BASE][cols].sum().values.astype(float),
                                                 df[df.method == BASE][cols].sum().values.astype(float))),
        reproduces_eval_dual_v1=bool(np.allclose(evi[evi.method == BASE][c1].sum().values.astype(float),
                                                 I[I.method == BASE][c1].sum().values.astype(float))),
        V0=RC.summary_counts(df[df.method == BASE])["PQ"], V1=ED.v1_boot(I, BASE, BASE)["A"],
        N1=dbase["nb_pred_Nto1"], oneN=dbase["nb_gt_1toN"], pair_iou=dbase["pair_iou_mean"],
        expected=dict(V0=.4243, V1=.3589, N1=364, oneN=451, pair_iou=.5624))
    rows = []
    for m in METHODS:
        d = df[df.method == m]
        sc = RC.summary_counts(d)
        dd = ED.diag_sums(d)
        pdc = AE.pair_decomp(d)
        rows.append(dict(method=m, role={BASE: "base", MAIN: "PRIMARY A=20", SENS: "sensitivity A=50"}[m],
                         V0=sc["PQ"], V0_SQ=sc["SQ"], V0_TP=sc["TP"], V0_FP=sc["FP"], V0_FN=sc["FN"],
                         V1=ED.v1_boot(I, BASE, m)["B"], V1_SQ=AE.v1_sq(I, m), N1=dd["nb_pred_Nto1"],
                         one_to_N=dd["nb_gt_1toN"], over_frag=dd["over_frag"], under_frag=dd["under_frag"],
                         pair_n=int(d.pair_n.sum()), pair_iou=dd["pair_iou_mean"], pair_dice=dd["pair_dice_mean"],
                         thin_recall=dd["thin_recall"], thin_precision=dd["thin_precision"],
                         tp_cross_up=pdc["tp_cross_up"], tp_cross_down=pdc["tp_cross_down"],
                         **{k: sc.get(k) for k in CAUSES}))
    pd.DataFrame(rows).to_csv(OUT / "methods_table.csv", index=False)
    res["methods_table"] = rows
    json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    brief = {m: dict(V0={k: r["V0"][k] for k in ("A", "B", "diff", "ci95", "per_fold")},
                     V1={k: r["V1"][k] for k in ("A", "B", "diff", "ci95", "per_fold")},
                     review=dict(V0=r["V0"]["review"], V1=r["V1_review"]), rule_numbers=r["rule_numbers"],
                     pair=r["pair_boot"], decomp=r["pair_decomposition"], causes=r["causes"],
                     newly=r.get("loao_newly_below_human"), undet=r.get("loao_undetermined"), rule=r["rule"],
                     files_changed=r["n_files_changed"], records_v0_changed=r["n_records_v0_changed"])
             for m, r in res["comparisons"].items()}
    print(json.dumps(dict(base_check=res["base_check"], instances=res["instances"], brief=brief), indent=1, default=str))


# ============================================================================ test CSV (in memory; read only)
def cmd_testcsv(a):
    import csv
    import io
    from pycocotools import mask as mask_util
    raw = TEST_CSV.read_bytes()                      # read only; nothing is written under outputs/submissions/
    rows = list(csv.reader(io.StringIO(raw.decode("ascii"), newline="")))
    hdr, body = rows[0], rows[1:]
    assert hdr == ["filament_id", "segmentation_rle"], hdr
    out_lines, per = [], []
    t0 = time.time()
    for fid, s in body:
        m = mask_util.decode({"size": [HW, HW], "counts": s.encode("ascii")}).astype(bool)
        (y0, x0, y1, x1), crop = InstanceSet._crop(m)
        d, res = inst_info(crop)
        new = res[A_MAIN]
        s_out = s
        if new is not None:
            (dy0, dx0), c = new
            m2 = np.zeros((HW, HW), np.uint8)
            m2[y0 + dy0:y0 + dy0 + c.shape[0], x0 + dx0:x0 + dx0 + c.shape[1]] = c
            assert m2.sum() > 0 and not (m2.astype(bool) & ~m).any()     # subset of the original, non-empty
            s_out = mask_util.encode(np.asfortranarray(m2))["counts"].decode("ascii")
            assert int(m2.sum()) == d[f"area_after_a{A_MAIN}"]
        else:
            enc = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")
            assert enc == s, f"unchanged row not canonical: {fid}"
        out_lines.append(f"{fid},{s_out}")
        per.append(dict(filament_id=fid, image=fid.split("_", 1)[0], changed_a20=new is not None, **d))
    OUT.mkdir(parents=True, exist_ok=True)
    TEST_OUT.write_bytes(("\n".join([",".join(hdr)] + out_lines) + "\n").encode("ascii"))
    ti = pd.DataFrame(per)
    ti.to_csv(OUT / "test_instances.csv", index=False)
    res = dict(source=str(TEST_CSV.relative_to(ROOT)), output=str(TEST_OUT.relative_to(ROOT)), n_rows=int(len(ti)),
               n_images=int(ti.image.nunique()), ids_same_order=[r[0] for r in body] == [x.split(",", 1)[0] for x in out_lines],
               multipart_before=int((ti.n_parts >= 2).sum()), multipart_share_before=ratio((ti.n_parts >= 2).sum(), len(ti)),
               n_minor_parts=int((ti.n_parts - 1).clip(lower=0).sum()), minor_parts_lt20=int(ti.minor_lt20.sum()),
               minor_parts_20_50=int(ti.minor_20_50.sum()), minor_parts_ge50=int(ti.minor_ge50.sum()),
               total_px=int(ti.area.sum()), seconds=round(time.time() - t0, 1))
    for A in AS:
        ch = ti[ti[f"drop_parts_a{A}"] > 0]
        res[f"a{A}"] = dict(instances_changed=int(len(ch)), images_changed=int(ch.image.nunique()),
                            parts_removed=int(ti[f"drop_parts_a{A}"].sum()), px_removed=int(ti[f"drop_px_a{A}"].sum()),
                            px_removed_share_of_all=ratio(ti[f"drop_px_a{A}"].sum(), ti.area.sum()),
                            multipart_after=int((ti[f"n_parts_after_a{A}"] >= 2).sum()),
                            multipart_share_after=ratio((ti[f"n_parts_after_a{A}"] >= 2).sum(), len(ti)),
                            min_area_after=int(ti[f"area_after_a{A}"].min()),
                            area_after_lt300=int((ti[f"area_after_a{A}"] < 300).sum()),
                            max_px_removed_one_instance=int(ti[f"drop_px_a{A}"].max()),
                            written=(A == A_MAIN))
    res["rows_changed_in_written_csv"] = int(sum(o.split(",", 1)[1] != r[1] for o, r in zip(out_lines, body)))
    json.dump(res, open(OUT / "test_changes.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "loao", "analyse", "testcsv"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="table on the first N files only (test, *_test.csv)")
    a = ap.parse_args()
    assert a.workers <= 4
    {"table": cmd_table, "loao": cmd_loao, "analyse": cmd_analyse, "testcsv": cmd_testcsv}[a.cmd](a)
