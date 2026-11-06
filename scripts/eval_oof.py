"""Full OOF evaluation of one method over CV folds with a FIXED post-processing config.

Reports (all pooled over annotation-set records, filseg.metrics conventions):
  PQ_pooled_host (notebook reading) / PQ_pooled (standard) / SQ / RQ, per-fold PQ, per-instance IoU/Dice,
  FN/FP causes, split/merge counters, thin_region_recall/precision (mask-derived, NOT barb accuracy),
  boundary F (2 px), the notebook-style rubric diagnostics (filseg.diagnostics), and a like-for-like
  comparison with inter-annotator agreement on the multi-set records only.

Usage: .venv/bin/python scripts/eval_oof.py --exp b0_r101_s1024 --folds 0,1,2,3,4 --thr 0.5 --min_area 500
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from filseg import diagnostics as DG  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.metrics import EvalConfig, aggregate, evaluate_dataset, single_thread_worker  # noqa: E402

_ARGS = {}


def make_pred(args):
    stem, path, group, thr, ma = args
    z = np.load(path)
    p = z[stem].astype(np.float32)
    return stem, seg.prob_to_instances(p, seg.PostCfg(thr, ma))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--probs", default="val_probs_last.npz")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min_area", type=int, default=500)
    ap.add_argument("--group", choices=["cc"], default="cc")
    a = ap.parse_args()
    import multiprocessing as mp
    single_thread_worker()
    base = ROOT / "outputs" / "exps" / a.exp
    gts, jobs, fold_of = {}, [], {}
    for f in map(int, a.folds.split(",")):
        files = (base / f"fold{f}" / "val_files.txt").read_text().split()
        g = seg.gt_records(files)
        gts.update(g)
        for fn in files:
            fold_of[Path(fn).stem] = f
            jobs.append((Path(fn).stem, str(base / f"fold{f}" / a.probs), a.group, a.thr, a.min_area))
    with mp.get_context("forkserver").Pool(12, initializer=single_thread_worker) as pool:
        preds = dict(pool.map(make_pred, jobs, chunksize=2))
    items = [(rid, g, preds[rid.split("-", 1)[1]]) for rid, g in gts.items()]
    summ, df = evaluate_dataset(items, EvalConfig(), n_workers=12)
    df["fold"] = df.image_key.map(lambda k: fold_of[k.split("-", 1)[1]])
    df["stem"] = df.image_key.map(lambda k: k.split("-", 1)[1])
    nsets = df.groupby("stem").image_key.transform("count")
    per_fold = {str(f): aggregate(d)["PQ_pooled_host"] for f, d in df.groupby("fold")}
    multi = aggregate(df[nsets > 1])
    single = aggregate(df[nsets == 1])
    dsum, _ = DG.aggregate_diagnostics(items, n_workers=12)
    ia = json.load(open(ROOT / "outputs" / "audit" / "diagnostics_interannotator.json"))
    keys = ["PQ_pooled_host", "PQ_pooled", "SQ_pooled", "RQ_pooled", "TP", "FP", "FN", "n_gt", "n_pred",
            "gt_best_iou_mean", "gt_best_dice_mean", "pred_best_iou_mean", "semantic_dice_pooled_px",
            "semantic_dice_img_mean", "n_split_gt", "n_merge_pred", "fn_missed", "fn_split", "fn_merged",
            "fn_low_iou", "fp_spurious", "fp_fragment", "fp_merge", "fp_low_iou", "fp_duplicate",
            "thin_region_recall_pooled_px", "thin_region_precision_pooled_px", "boundary_f_pooled",
            "tp_boundary_f_mean", "PQ_img_mean"]
    out = dict(exp=a.exp, folds=a.folds, probs=a.probs, group=a.group, thr=a.thr, min_area=a.min_area,
               n_records=int(len(df)), all={k: summ.get(k) for k in keys}, per_fold_PQ_host=per_fold,
               per_fold_PQ_host_std=float(np.std(list(per_fold.values()), ddof=1)),
               multi_set_records={k: multi.get(k) for k in ("PQ_pooled_host", "SQ_pooled", "RQ_pooled", "TP", "FP", "FN")},
               single_set_records={k: single.get(k) for k in ("PQ_pooled_host", "SQ_pooled", "RQ_pooled", "TP", "FP", "FN")},
               interannotator_reference_multi_set=dict(PQ=0.3399, note="pooled over 1196 ordered human pairs, same 296 files' sets (docs/inter_annotator.md)"),
               rubric_diagnostics=DG.to_jsonable(dsum),
               rubric_interannotator_reference={k: ia.get(k) for k in ("dice_variants", "notebook_pq")} if isinstance(ia, dict) else None)
    dst = base / f"eval_oof_{a.group}_f{a.folds.replace(',', '')}_t{a.thr}_a{a.min_area}.json"
    json.dump(out, open(dst, "w"), indent=1, default=float)
    df.to_csv(str(dst).replace(".json", "_per_record.csv"), index=False)
    show = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["all"].items()}
    print(json.dumps(dict(all=show, per_fold=per_fold, multi=out["multi_set_records"], single=out["single_set_records"]),
                     indent=1, default=float))
    print("->", dst)


if __name__ == "__main__":
    main()
