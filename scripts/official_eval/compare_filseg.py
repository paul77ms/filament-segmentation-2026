"""Compare the official-notebook run (run_official_notebook.py output) with filseg on EXACTLY the same records.

filseg side is computed from the SAME prediction CSV (filseg.rle.read_submission -> InstanceSet.from_rles) and from
the train COCO json directly (seg.gt_records), i.e. independent of the GT CSV fed to the notebook; the GT CSV is
additionally decoded and checked pixel-identical to the json GT. Serial (no pool).

Usage: .venv/bin/python scripts/official_eval/compare_filseg.py --nb_json outputs/official_eval/official_nb_b0_oof_full.json
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filseg import diagnostics as DG  # noqa: E402
from filseg import rle as R  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.metrics import EvalConfig, InstanceSet, aggregate, evaluate_dataset, single_thread_worker  # noqa: E402


def same_set(a: InstanceSet, b: InstanceSet) -> bool:
    return len(a) == len(b) and np.array_equal(a.boxes, b.boxes) and np.array_equal(a.areas, b.areas) and \
        all(np.array_equal(x, y) for x, y in zip(a.crops, b.crops))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb_json", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    single_thread_worker()
    nb = json.load(open(a.nb_json))
    nb_rec = pd.read_csv(a.nb_json.replace(".json", "_per_record.csv"))
    records = list(nb_rec["annotator_image"])
    meta = json.load(open(Path(nb["pred_csv"]).parent / "build_meta.json"))

    sub = R.read_submission(nb["pred_csv"])
    by_stem = R.group_submission_by_stem(sub)
    preds = {s: InstanceSet.from_rles(v) for s, v in by_stem.items()}
    gts_json = seg.gt_records()   # all train annotation sets, from the COCO json
    gt_csv = R.read_submission(nb["gt_csv"])
    gt_csv["rec"] = gt_csv["filament_id"].str.split("_", n=1).str[0]
    gt_from_csv = {r: InstanceSet.from_rles(list(g["segmentation_rle"])) for r, g in gt_csv.groupby("rec", sort=False)}
    n_gt_csv_equal = sum(same_set(gts_json[r], gt_from_csv[r]) for r in records)

    items = [(r, gts_json[r], preds.get(r.split("-", 1)[1], InstanceSet.empty())) for r in records]
    summ, df = evaluate_dataset(items, EvalConfig(), n_workers=1)
    dsum, frames = DG.aggregate_diagnostics(items, n_workers=1)

    fs = df[["image_key", "n_gt", "n_pred", "tp", "fp", "fn", "tp_host", "fp_host", "fn_host", "sum_iou_host"]].rename(
        columns={c: f"{c}_fs" for c in ("n_gt", "n_pred", "tp", "fp", "fn")})
    m = nb_rec.rename(columns={c: f"{c}_nb" for c in ("n_gt", "n_pred", "tp", "fp", "fn")}).merge(
        fs, left_on="annotator_image", right_on="image_key")
    assert len(m) == len(records)
    mism = {k: int((m[f"{k}_nb"] != m[f"{k}_host"]).sum()) for k in ("tp", "fp", "fn")}
    mism["n_gt"] = int((m["n_gt_nb"] != m["n_gt_fs"]).sum())
    mism["n_pred"] = int((m["n_pred_nb"] != m["n_pred_fs"]).sum())
    d_sum = (m["sum_iou_f32"] - m["sum_iou_host"]).abs()
    unique_vs_host = dict(TP=summ["TP"], FP=summ["FP"], FN=summ["FN"], TP_host=summ["TP_host"], FP_host=summ["FP_host"],
                          FN_host=summ["FN_host"], records_where_unique_differs=m.loc[m["tp_fs"] != m["tp_host"], "image_key"].tolist())

    hist_eq = {w: bool(np.array_equal(np.asarray(dsum[f"{w}_hist"]["counts"]), np.asarray(nb[f"{w}_overlap_pairs"]["hist"]["counts"])))
               for w in ("iou", "dice")}
    deg_nb = {"gt": {int(k): v for k, v in nb["gt_one_to_n"].items()}, "pred": {int(k): v for k, v in nb["pred_n_to_one"].items()}}
    deg_fs = {"gt": dsum["overlap_degree"]["gt_one_to_many"]["counts_by_degree"],
              "pred": dsum["overlap_degree"]["pred_many_to_one"]["counts_by_degree"]}

    # per-fold pooled PQ from the notebook's per-record counts (sum of float32 IoUs per record)
    fold_of = meta["fold_of"]
    m["fold"] = m["annotator_image"].map(lambda r: fold_of[r.split("-", 1)[1]])
    df["fold"] = df["image_key"].map(lambda r: fold_of[r.split("-", 1)[1]])
    per_fold = {}
    for f, g in m.groupby("fold"):
        den_nb = g["tp_nb"].sum() + 0.5 * g["fp_nb"].sum() + 0.5 * g["fn_nb"].sum()
        per_fold[str(f)] = dict(notebook_counts_PQ=float(g["sum_iou_f32"].sum() / den_nb),
                                filseg_PQ_pooled_host=aggregate(df[df["fold"] == f])["PQ_pooled_host"],
                                n_records=int(len(g)))

    eval_oof = json.load(open(ROOT / "outputs/exps/b0_r101_s1024/eval_oof_cc_f01234_t0.5_a500.json"))
    out = dict(
        records=len(records), notebook_json=a.nb_json,
        gt_csv_equals_json_records=f"{n_gt_csv_equal}/{len(records)}",
        pq=dict(official_notebook=nb["pq_score"], filseg_PQ_pooled_host=summ["PQ_pooled_host"],
                diff_notebook_minus_filseg=nb["pq_score"] - summ["PQ_pooled_host"],
                filseg_PQ_pooled_standard_unique=summ["PQ_pooled"], diagnostics_notebook_pq=dsum["notebook_pq"]["pq"],
                existing_eval_oof_PQ_pooled_host=eval_oof["all"]["PQ_pooled_host"]),
        counts=dict(official_notebook=dict(TP=nb["tp"], FP=nb["fp"], FN=nb["fn"]),
                    filseg_host=dict(TP=summ["TP_host"], FP=summ["FP_host"], FN=summ["FN_host"]),
                    diagnostics_notebook_pq=dict(TP=dsum["notebook_pq"]["tp"], FP=dsum["notebook_pq"]["fp"], FN=dsum["notebook_pq"]["fn"]),
                    filseg_unique=unique_vs_host),
        per_record_mismatches=mism, per_record_max_abs_sum_iou_diff=float(d_sum.max()),
        per_fold=per_fold,
        histograms=dict(equal_counts=hist_eq,
                        notebook=dict(iou_mean=nb["iou_overlap_pairs"]["mean"], dice_mean=nb["dice_overlap_pairs"]["mean"],
                                      n_pairs=nb["iou_overlap_pairs"]["n"]),
                        diagnostics=dict(iou_mean=dsum["iou_hist"]["mean"], dice_mean=dsum["dice_hist"]["mean"],
                                         n_pairs=dsum["iou_hist"]["n_pairs"])),
        overlap_degrees=dict(notebook=deg_nb, diagnostics=deg_fs,
                             equal=bool(deg_nb["gt"] == deg_fs["gt"] and deg_nb["pred"] == deg_fs["pred"])),
        diagnostics_dice_variants=dsum["dice_variants"],
        filseg_summary={k: summ[k] for k in ("SQ_pooled_host", "RQ_pooled_host", "SQ_pooled", "RQ_pooled", "PQ_img_mean",
                                             "PQ_img_mean_host", "semantic_dice_pooled_px", "semantic_dice_img_mean",
                                             "gt_best_dice_mean", "n_gt", "n_pred")},
    )
    dst = Path(a.out or a.nb_json.replace(".json", "_vs_filseg.json"))
    json.dump(DG.to_jsonable(out), open(dst, "w"), indent=1)
    print(json.dumps(DG.to_jsonable({k: v for k, v in out.items() if k != "diagnostics_dice_variants"}), indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
