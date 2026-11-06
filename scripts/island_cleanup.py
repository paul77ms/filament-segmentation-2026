"""Island cleanup after R2 (registry row R2b0_island5; rule registered before any result). CPU only.

Base: R2nb0_fc20 = R2 + the 20 px cleanup: R2nb0 instances (nested_r2b0_eval.instances with eval_dual's CFG) after
the fragment cleanup of scripts/fragment_cleanup.py (A = 20 px). Outer-validation files of the 5 nested folds (707).
The nested caches outputs/cache/refine_nested_o* were deleted (D-056); their candidate maps are recomputed in memory
(_LazyCand, nothing written). analyse checks that the base records equal fragment_cleanup's R2nb0_fc20 records.

ISLAND CLEANUP (fixed before running, not tuned), per R2nb0_fc20 instance, independently of the other instances:
  parts = 8-connected components of the instance mask (fragment_cleanup.parts).
  if the instance has >= 2 parts: drop every part with area < max(20 px, 5% of the instance's total area), except the
  largest part (never dropped; ties -> lowest label).  This is fragment_cleanup.clean_crop with A = max(20, 0.05 * area);
  the instance area is that of the base (R2nb0_fc20) instance, i.e. the instance the rule is applied to (as for the
  test CSV, which is the fc20 CSV).  Nothing else changes; min_area is NOT re-applied; only pixels are removed.
  -> R2nb0_isl5.   (Reported only: the same rule applied to R2nb0 directly, with the R2nb0 area as denominator.)

RULE vs base R2nb0_fc20 (all must hold) -- experiments/registry.csv R2b0_island5:
  F1 V0 diff >= 0, or CI includes 0 with lower bound > -0.003; F2 the same for V1; F3 notebook N:1 and 1:N not higher
  AND mean pair IoU not lower; F4 no LOAO aspect at/above human under the base that falls below human; F5 32-case
  review set PQ not lower under V0 and V1.  CIs: file bootstrap seed 2026, 2000 draws; 14-day group bootstrap and
  per-fold diffs reported.  The evaluation logic is fragment_cleanup's (compare / LOAO / diagnostics), base switched.

Commands (outputs in outputs/research/island_cleanup/):
  table     per file: R2nb0_fc20, R2nb0_isl5 -> records.csv, images_v1.csv, instances.csv
  loao      eval_dual's loao code path with both methods -> loao/*.csv
  analyse   base reproduction check (stops on mismatch), comparison, diagnostics, LOAO, rule -> summary.json
  testcsv   the rule applied IN MEMORY to the fc20 test CSV (read only) -> test_changes.json,
            test_instances.csv, test_r2b0_fc20_isl5.csv (same ids / order)
  CUDA_VISIBLE_DEVICES="" NUMPY_MADVISE_HUGEPAGE=0 taskset -c 0-3 .venv/bin/python scripts/island_cleanup.py <cmd>
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
import fragment_cleanup as FC  # noqa: E402
import a_eval as AE  # noqa: E402
import nested_r2b0_eval as NE  # noqa: E402
import refine_b0exist_checks as RC  # noqa: E402
from filseg import refine as RF  # noqa: E402
from filseg.metrics import InstanceSet, single_thread_worker  # noqa: E402

OUT = ROOT / "outputs" / "research" / "island_cleanup"
A_MIN, FRAC = 20, 0.05
R0 = FC.BASE                      # "R2nb0"
BASE = FC.MAIN                    # "R2nb0_fc20"  (R2 + 20 px cleanup)
MAIN = f"{R0}_isl5"
METHODS = (BASE, MAIN)
LOAO_RATERS = METHODS
TEST_CSV = ROOT / "outputs" / "submissions" / "track2" / "r2b0_5fold_t05_a500_r2a300_fc20.csv"   # read only
TEST_OUT = OUT / "test_r2b0_fc20_isl5.csv"
HW = FC.HW
EXPECTED_BASE = dict(V0=.42452, V1=.35915, N1=322, oneN=405, pair_iou=.5670)


# ============================================================================ stage-1 candidate maps (no cache)
class _NoMeta(dict):
    def __missing__(self, k):          # nested_r2b0_eval.instances reads windows only for its info dict (unused here)
        return {"windows": []}


class _LazyCand:
    """Substitute for outputs/cache/refine_nested_o<k>/cand.npy (deleted in D-056) on the outer-validation files:
    the candidate label map recomputed in memory exactly as filseg.refine._build_one built it from the cached uint8
    stage-1 probability (u8 = rint(p * 255) of b0n_ens/fold<k>; u8 / 255 > CAND_THR; 8-conn; >= CAND_MIN px; labels in
    cv2 order) -- the same recomputation nested_stage1 verify check C passed.  Nothing is written to disk."""

    def __init__(self, fold, files):
        self.fold, self.files = fold, files

    def __getitem__(self, i):
        ens = NE._load(("ens", self.fold),
                       lambda: np.load(NE.EXPS / ED.CFG["ens_exp"] / f"fold{self.fold}" / "val_probs_last.npz"))
        u8 = np.clip(np.rint(ens[self.files[i][:-5]].astype(np.float32) * 255), 0, 255).astype(np.uint8)
        p = u8.astype(np.float32) / 255.0
        n, lab, stats = cv2.connectedComponentsWithStats((p > RF.CAND_THR).astype(np.uint8), connectivity=8)[:3]
        keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= RF.CAND_MIN]
        lut = np.zeros(n, np.uint16)
        for j, k in enumerate(keep, 1):
            lut[k] = j
        return lut[lab]


def ensure_stage1(fold):
    """Install the in-memory candidate maps into nested_r2b0_eval's worker cache when the nested cache is absent."""
    if ("cand", fold) in NE._W or (Path(ED.CFG["cache_prefix"] + str(fold)) / "index.json").exists():
        return
    files = (NE.EXPS / ED.CFG["ens_exp"] / f"fold{fold}" / "val_files.txt").read_text().split()
    NE._W[("idx", fold)] = dict(files=files, meta=_NoMeta())
    NE._W[("fi", fold)] = {f: i for i, f in enumerate(files)}
    NE._W[("cand", fold)] = _LazyCand(fold, files)


# ============================================================================ cleanup (fragment_cleanup.clean_crop)
def thr_of(ar):
    return max(float(A_MIN), FRAC * float(ar.sum()))


def isl_crop(crop, lp=None):
    lab, ar = lp if lp is not None else FC.parts(crop)
    return FC.clean_crop(crop, thr_of(ar), (lab, ar))


def inst_info(crop, crop_r0=None):
    """info for one base (fc20) instance; crop_r0 = its R2nb0 crop (the alternative-denominator variant)."""
    lab, ar = FC.parts(crop)
    srt = np.sort(ar)[::-1]
    thr = thr_of(ar)
    new, ci = isl_crop(crop, (lab, ar))
    after = new[1] if new is not None else crop
    dropped = np.sort(np.asarray([a for j, a in enumerate(ar) if j != int(np.argmax(ar)) and a < thr], np.int64)) \
        if len(ar) >= 2 else np.zeros(0, np.int64)
    minor = srt[1:]
    d = dict(area=int(ar.sum()), n_parts=int(len(ar)), largest_px=int(srt[0]) if len(ar) else 0,
             minor_px=int(minor.sum()), minor_lt20=int((minor < 20).sum()),
             minor_20_50=int(((minor >= 20) & (minor < 50)).sum()), minor_ge50=int((minor >= 50).sum()),
             thr=thr, drop_parts=ci["n_drop"], drop_px=ci["px_drop"],
             drop_parts_20_50=int(((dropped >= 20) & (dropped < 50)).sum()), drop_parts_ge50=int((dropped >= 50).sum()),
             max_part_dropped=int(dropped.max()) if len(dropped) else 0,
             area_after=int(after.sum()), n_parts_after=int(len(ar) - ci["n_drop"]))
    if crop_r0 is not None:   # the same rule applied to R2nb0 directly (denominator = R2nb0 area); reported only
        lab0, ar0 = FC.parts(crop_r0)
        new0, _ = FC.clean_crop(crop_r0, thr_of(ar0), (lab0, ar0))
        d.update(area_r0=int(ar0.sum()), n_parts_r0=int(len(ar0)),
                 area_after_alt_r0=int((new0[1] if new0 is not None else crop_r0).sum()))
    return d, new


# ============================================================================ table
def table_job(args):
    stem, fold, gts = args
    t0 = time.time()
    ensure_stage1(fold)
    M, _ = NE.instances(stem, fold, ED.CFG)
    P0 = M[R0]
    P, _ = FC.cleanup(P0, FC.A_MAIN)                 # base R2nb0_fc20 (fragment_cleanup code path)
    assert len(P) == len(P0)
    gb = {rid: AE.gt_bins(G)[1] for rid, G in gts.items()}
    insts, news = [], []
    for k in range(len(P)):
        d, new = inst_info(P.crops[k], P0.crops[k])
        insts.append(dict(stem=stem, fold=fold, idx=k, box=";".join(str(int(v)) for v in P.boxes[k]), **d))
        news.append(new)
    rec, img, _ = AE._eval(BASE, P, gts, stem, fold)
    for x in rec:
        x.update(**AE.pair_stats(gts[x["rid"]], P, P, gb[x["rid"]]), n_changed=0)
    recs, imgs = list(rec), [img]
    S, ch = FC.cleanup(P, None, news)
    if not ch:
        recs += [dict(x, method=MAIN) for x in rec]; imgs.append(dict(img, method=MAIN))
    else:
        r2, i2, _ = AE._eval(MAIN, S, gts, stem, fold)
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
    ensure_stage1(fold)
    M, _ = NE.instances(stem, fold, ED.CFG)
    M[BASE] = FC.cleanup(M[R0], FC.A_MAIN)[0]
    P = M[BASE]
    M[MAIN] = FC.cleanup(P, None, [isl_crop(P.crops[k])[0] for k in range(len(P))])[0]
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
def base_check(df, I):
    """R2nb0_fc20 here vs fragment_cleanup's R2nb0_fc20 records (exact) and the registered numbers."""
    fr = pd.read_csv(FC.OUT / "records.csv")
    fi = pd.read_csv(FC.OUT / "images_v1.csv")
    cols = RC.C4 + ["n_gt", "n_pred", "pair_n", "pair_iou_sum", "pair_dice_sum", "nb_pred_Nto1", "nb_gt_1toN",
                    "thin_gt_covered_px", "thin_gt_px", "thin_pred_in_gt_px", "thin_pred_px"]
    c1 = ["v1_tp", "v1_fp", "v1_fn", "v1_si"]
    a_ = df[df.method == BASE].sort_values("rid").reset_index(drop=True)
    b_ = fr[fr.method == BASE].sort_values("rid").reset_index(drop=True)
    ai = I[I.method == BASE].sort_values("stem").reset_index(drop=True)
    bi = fi[fi.method == BASE].sort_values("stem").reset_index(drop=True)
    d = ED.diag_sums(a_)
    got = dict(V0=RC.summary_counts(a_)["PQ"], V1=ED.v1_boot(I, BASE, BASE)["A"], N1=d["nb_pred_Nto1"],
               oneN=d["nb_gt_1toN"], pair_iou=d["pair_iou_mean"])
    tol = dict(V0=5e-6, V1=5e-6, N1=0, oneN=0, pair_iou=5e-5)
    match = {k: bool(abs(got[k] - EXPECTED_BASE[k]) <= tol[k]) for k in got}
    res = dict(expected=EXPECTED_BASE, got=got, match_registered=match,
               records_identical_to_fragment_cleanup=bool(
                   len(a_) == len(b_) and (a_.rid.values == b_.rid.values).all() and
                   np.allclose(a_[cols].values.astype(float), b_[cols].values.astype(float))),
               v1_identical_to_fragment_cleanup=bool(
                   len(ai) == len(bi) and (ai.stem.values == bi.stem.values).all() and
                   (ai[c1].values == bi[c1].values).all()))
    res["ok"] = bool(all(match.values()) and res["records_identical_to_fragment_cleanup"]
                     and res["v1_identical_to_fragment_cleanup"])
    return res


def v1_group_boot(I, sp):
    """14-day group bootstrap for V1: V1 counts summed per group, then eval_dual.v1_boot with unit = group_id."""
    g = sp.set_index("stem").group_id
    J = I.assign(group_id=I.stem.map(g))
    assert J.group_id.notna().all()
    J = J.groupby(["method", "group_id"], as_index=False).agg(
        v1_tp=("v1_tp", "sum"), v1_fp=("v1_fp", "sum"), v1_fn=("v1_fn", "sum"), v1_si=("v1_si", "sum"),
        fold=("fold", "first"), n_fold=("fold", "nunique"))
    assert (J.n_fold == 1).all()
    r = ED.v1_boot(J, BASE, MAIN, unit="group_id")
    return dict(diff=r["diff"], ci95=r["ci95"], n_units=int(J.group_id.nunique()))


def instance_stats(ins):
    ratio = FC.ratio
    ch = ins[ins.drop_parts > 0]
    mp_b, mp_a = (ins.n_parts >= 2), (ins.n_parts_after >= 2)
    out = dict(n_instances=int(len(ins)), n_files=int(ins.stem.nunique()),
               multipart_r2nb0=int((ins.n_parts_r0 >= 2).sum()),
               multipart_share_r2nb0=ratio((ins.n_parts_r0 >= 2).sum(), len(ins)),
               multipart_before=int(mp_b.sum()), multipart_share_before=ratio(mp_b.sum(), len(ins)),
               multipart_after=int(mp_a.sum()), multipart_share_after=ratio(mp_a.sum(), len(ins)),
               n_minor_parts_before=int((ins.n_parts - 1).clip(lower=0).sum()),
               n_minor_parts_after=int((ins.n_parts_after - 1).clip(lower=0).sum()),
               minor_parts_before_20_50=int(ins.minor_20_50.sum()), minor_parts_before_ge50=int(ins.minor_ge50.sum()),
               minor_px_before=int(ins.minor_px.sum()), total_px=int(ins.area.sum()),
               instances_changed=int(len(ch)), files_changed=int(ch.stem.nunique()),
               became_single_part=int((mp_b & ~mp_a).sum()),
               parts_removed=int(ins.drop_parts.sum()), parts_removed_20_50=int(ins.drop_parts_20_50.sum()),
               parts_removed_ge50=int(ins.drop_parts_ge50.sum()), px_removed=int(ins.drop_px.sum()),
               px_removed_share_of_all=ratio(ins.drop_px.sum(), ins.area.sum()),
               px_removed_share_of_changed=ratio(ch.drop_px.sum(), ch.area.sum()),
               max_part_removed_px=int(ins.max_part_dropped.max()),
               max_px_removed_one_instance=int(ins.drop_px.max()),
               max_share_removed_one_instance=float((ch.drop_px / ch.area).max()) if len(ch) else 0.0,
               median_thr_changed=float(ch.thr.median()) if len(ch) else None,
               thr_above_20_share=ratio((ins.thr > A_MIN).sum(), len(ins)),
               area_after_lt300=int((ins.area_after < 300).sum()),
               alt_denominator_r2nb0_instances_differ=int((ins.area_after_alt_r0 != ins.area_after).sum()),
               per_fold={int(k): dict(instances_changed=int((g.drop_parts > 0).sum()),
                                      parts_removed=int(g.drop_parts.sum()), px_removed=int(g.drop_px.sum()),
                                      multipart_before=int((g.n_parts >= 2).sum()),
                                      multipart_after=int((g.n_parts_after >= 2).sum()))
                         for k, g in ins.groupby("fold")})
    return out


def cmd_analyse(a):
    df = pd.read_csv(OUT / "records.csv")
    I = pd.read_csv(OUT / "images_v1.csv")
    ins = pd.read_csv(OUT / "instances.csv")
    bc = base_check(df, I)
    print(json.dumps(dict(base_check=bc), indent=1, default=str), flush=True)
    if not bc["ok"]:
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(dict(base_check=bc, stopped="base R2nb0_fc20 not reproduced"), open(OUT / "summary.json", "w"),
                  indent=1, default=str)
        sys.exit("STOP: base R2nb0_fc20 not reproduced")
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
    FC.BASE = BASE                  # fragment_cleanup.compare with the base switched to R2nb0_fc20
    cmp = FC.compare(df, I, sp, review, MAIN, asp, BH)
    cmp["V1_group_boot"] = v1_group_boot(I, sp)
    if asp is not None:
        cmp["loao_n_files"] = int(pd.read_csv(lo).stem.nunique())
        # LOAO base (fc20) records vs fragment_cleanup's fc20 LOAO records (reproduction)
        mine = pd.read_csv(lo); theirs = pd.read_csv(FC.OUT / "loao" / "loao_records.csv")
        k = ["stem", "rec"]
        num = [c for c in mine.columns if c in theirs.columns and c not in k and pd.api.types.is_numeric_dtype(mine[c])]
        x = mine[mine.rater == BASE].sort_values(k).reset_index(drop=True)
        y = theirs[theirs.rater == BASE].sort_values(k).reset_index(drop=True)
        cmp["loao_base_identical_to_fragment_cleanup"] = bool(
            len(x) == len(y) and (x[k].values == y[k].values).all() and
            np.allclose(x[num].values.astype(float), y[num].values.astype(float), equal_nan=True))
    res = dict(method="drop 8-connected parts < max(20 px, 5% of instance area) of multi-part R2nb0_fc20 instances; "
                      "largest part kept", base=BASE, main=MAIN, base_check=bc, instances=instance_stats(ins),
               comparison=cmp)
    rows = []
    for m in METHODS:
        d = df[df.method == m]
        sc = RC.summary_counts(d)
        dd = ED.diag_sums(d)
        pdc = AE.pair_decomp(d)
        rows.append(dict(method=m, role={BASE: "base (R2 + 20 px cleanup)", MAIN: "island cleanup 5%"}[m],
                         V0=sc["PQ"], V0_SQ=sc["SQ"], V0_TP=sc["TP"], V0_FP=sc["FP"], V0_FN=sc["FN"],
                         V1=ED.v1_boot(I, BASE, m)["B"], V1_SQ=AE.v1_sq(I, m), N1=dd["nb_pred_Nto1"],
                         one_to_N=dd["nb_gt_1toN"], over_frag=dd["over_frag"], under_frag=dd["under_frag"],
                         pair_n=int(d.pair_n.sum()), pair_iou=dd["pair_iou_mean"], pair_dice=dd["pair_dice_mean"],
                         thin_recall=dd["thin_recall"], thin_precision=dd["thin_precision"],
                         tp_cross_up=pdc["tp_cross_up"], tp_cross_down=pdc["tp_cross_down"],
                         **{k: sc.get(k) for k in FC.CAUSES}))
    pd.DataFrame(rows).to_csv(OUT / "methods_table.csv", index=False)
    res["methods_table"] = rows
    json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    r = cmp
    brief = dict(V0={k: r["V0"][k] for k in ("A", "B", "diff", "ci95", "per_fold", "group_boot", "sign")},
                 V1={k: r["V1"][k] for k in ("A", "B", "diff", "ci95", "per_fold")}, V1_group=r["V1_group_boot"],
                 review=dict(V0=r["V0"]["review"], V1=r["V1_review"]), rule_numbers=r["rule_numbers"],
                 pair=r["pair_boot"], per_fold_diag=r["per_fold_diag"], decomp=r["pair_decomposition"],
                 causes=r["causes"], loao_base_at=r.get("loao_base_at_or_above_human"),
                 newly=r.get("loao_newly_below_human"), undet=r.get("loao_undetermined"),
                 loao_repro=r.get("loao_base_identical_to_fragment_cleanup"), rule=r["rule"],
                 files_changed=r["n_files_changed"], records_v0_changed=r["n_records_v0_changed"])
    print(json.dumps(dict(instances=res["instances"], brief=brief), indent=1, default=str))


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
        d, new = inst_info(crop)
        fc20_noop = FC.clean_crop(crop, FC.A_MAIN)[0] is None     # the source CSV is already fc20-clean
        s_out = s
        if new is not None:
            (dy0, dx0), c = new
            m2 = np.zeros((HW, HW), np.uint8)
            m2[y0 + dy0:y0 + dy0 + c.shape[0], x0 + dx0:x0 + dx0 + c.shape[1]] = c
            assert m2.sum() > 0 and not (m2.astype(bool) & ~m).any()     # subset of the original, non-empty
            s_out = mask_util.encode(np.asfortranarray(m2))["counts"].decode("ascii")
            assert int(m2.sum()) == d["area_after"]
        else:
            enc = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")
            assert enc == s, f"unchanged row not canonical: {fid}"
        out_lines.append(f"{fid},{s_out}")
        per.append(dict(filament_id=fid, image=fid.split("_", 1)[0], changed=new is not None, fc20_noop=fc20_noop, **d))
    OUT.mkdir(parents=True, exist_ok=True)
    TEST_OUT.write_bytes(("\n".join([",".join(hdr)] + out_lines) + "\n").encode("ascii"))
    ti = pd.DataFrame(per)
    ti.to_csv(OUT / "test_instances.csv", index=False)
    ch = ti[ti.drop_parts > 0]
    ratio = FC.ratio
    res = dict(source=str(TEST_CSV.relative_to(ROOT)), output=str(TEST_OUT.relative_to(ROOT)), n_rows=int(len(ti)),
               n_images=int(ti.image.nunique()),
               ids_same_order=[r[0] for r in body] == [x.split(",", 1)[0] for x in out_lines],
               source_is_fc20_clean=bool(ti.fc20_noop.all()),
               multipart_before=int((ti.n_parts >= 2).sum()), multipart_share_before=ratio((ti.n_parts >= 2).sum(), len(ti)),
               multipart_after=int((ti.n_parts_after >= 2).sum()),
               multipart_share_after=ratio((ti.n_parts_after >= 2).sum(), len(ti)),
               n_minor_parts_before=int((ti.n_parts - 1).clip(lower=0).sum()),
               n_minor_parts_after=int((ti.n_parts_after - 1).clip(lower=0).sum()),
               total_px=int(ti.area.sum()), rows_changed=int(len(ch)), images_changed=int(ch.image.nunique()),
               became_single_part=int(((ti.n_parts >= 2) & (ti.n_parts_after == 1)).sum()),
               parts_removed=int(ti.drop_parts.sum()), parts_removed_20_50=int(ti.drop_parts_20_50.sum()),
               parts_removed_ge50=int(ti.drop_parts_ge50.sum()), px_removed=int(ti.drop_px.sum()),
               px_removed_share_of_all=ratio(ti.drop_px.sum(), ti.area.sum()),
               max_part_removed_px=int(ti.max_part_dropped.max()), max_px_removed_one_instance=int(ti.drop_px.max()),
               max_share_removed_one_instance=float((ch.drop_px / ch.area).max()) if len(ch) else 0.0,
               min_area_before=int(ti.area.min()), min_area_after=int(ti.area_after.min()),
               area_after_lt300=int((ti.area_after < 300).sum()), area_after_lt500=int((ti.area_after < 500).sum()),
               seconds=round(time.time() - t0, 1))
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
