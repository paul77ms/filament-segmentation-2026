"""Where the final pipeline misses annotated filaments, and how visible they are. CPU only, analysis only.

For every annotated filament of every annotation set (nested out-of-fold, 707 training images) the outcome under the
final pipeline (filseg.metrics, exclusive cause: merged > split > missed > low IoU; else matched) is combined with the
nested stage-1 map B_ens to name the stage where it is lost:
  0 matched              a prediction matches it at IoU > 0.5
  1a no stage-1 response missed (or low IoU) and the stage-1 probability inside it stays below 0.1
  1b weak stage-1 response  missed, maximum stage-1 probability 0.1-0.5
  2a small pieces removed   missed although stage 1 reaches p > 0.5 there, but only in components < 500 px
  2b lost after the gate    missed although a >= 500 px stage-1 component overlaps it
  3  boundary            overlapped, best IoU <= 0.5
  4  split / 5 merged    matched only in pieces / only together with another filament
Visibility (contrast and size from scripts/error_budget.py, gt_instances.csv): clear = contrast_snr >= 1.61 and
>= 500 px; small but visible = contrast_snr >= 1.61 and < 500 px; faint = contrast_snr < 1.61. Support by the other
annotators: on images with 2-3 annotation sets, whether another set drew the same filament (IoU > 0.5).
  CUDA_VISIBLE_DEVICES="" taskset -c 0-3 .venv/bin/python scripts/miss_decomposition.py run
  ... figure  -> outputs/research/final/{miss_decomposition.csv, miss_summary.json, fig_miss_decomposition.png}
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs" / "research" / "final"
SNR_VISIBLE, GATE = 1.61, 500
STAGES = ["0 matched", "1a no stage-1 response", "1b weak stage-1 response", "2a small pieces removed",
          "2b lost after the gate", "3 boundary", "4 split", "5 merged"]
_Z = {}


def job(args):
    import a_eval as AE
    from filseg.metrics import EvalConfig, evaluate_image
    stem, fold, gts = args
    cfg = EvalConfig(compute_boundary=False, compute_pair_boundary=False, compute_thin=False)
    if fold not in _Z:
        _Z[fold] = np.load(ROOT / "outputs" / "exps" / "b0n_ens" / f"fold{fold}" / "val_probs_last.npz")
    p = _Z[fold][stem].astype(np.float32)
    n, lab, st, _ = cv2.connectedComponentsWithStats((p > 0.5).astype(np.uint8), connectivity=8)
    big = np.zeros(n, bool); big[1:] = st[1:, cv2.CC_STAT_AREA] >= GATE
    A = AE.final_instances(stem, fold)
    rows = []
    for rid, G in gts.items():
        _, pg, _ = evaluate_image(G, A, cfg, image_key=rid, return_instances=True)
        for i, d in enumerate(pg):
            y0, x0, y1, x1 = (int(v) for v in G.boxes[i])
            m = G.crops[i]
            pmax = float(p[y0:y1, x0:x1][m].max())
            comps = np.unique(lab[y0:y1, x0:x1][m & (p[y0:y1, x0:x1] > 0.5)])
            cause = d["fn_cause"] or "tp"
            if cause == "tp":
                s = STAGES[0]
            elif cause in ("missed", "low_iou") and pmax < 0.1:
                s = STAGES[1]
            elif cause == "missed":
                s = STAGES[2] if pmax < 0.5 else (STAGES[4] if big[comps[comps > 0]].any() else STAGES[3])
            else:
                s = {"low_iou": STAGES[5], "split": STAGES[6], "merged": STAGES[7]}[cause]
            rows.append(dict(rid=rid, gt_idx=i, stem=stem, fold=fold, stage=s, pmax_stage1=pmax))
    return rows


def cmd_run(a):
    import multiprocessing as mp

    import a_eval as AE
    from filseg.metrics import single_thread_worker
    single_thread_worker()
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        R = sum(pool.map(job, AE.all_jobs(), chunksize=1), [])
    d = pd.DataFrame(R)
    gt = pd.read_csv(ROOT / "outputs" / "error_budget" / "gt_instances.csv",
                     usecols=["rid", "gt_idx", "area", "contrast_snr", "n_other_sets", "n_other_sets_drawn_iou", "centroid_r_frac"])
    d = d.merge(gt, on=["rid", "gt_idx"], how="left")
    assert d.area.notna().all(), "gt_instances.csv join failed"
    d["visibility"] = np.where(d.contrast_snr >= SNR_VISIBLE, np.where(d.area >= GATE, "clear", "small but visible"), "faint")
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "miss_decomposition.csv", index=False)
    multi = d[d.n_other_sets > 0]
    summ = dict(gt_records=len(d), stage_counts=d.stage.value_counts().reindex(STAGES).fillna(0).astype(int).to_dict(),
                matched_share_by_visibility={v: float((x.stage == STAGES[0]).mean()) for v, x in d.groupby("visibility")},
                small_lt500_matched_share=float((d[d.area < GATE].stage == STAGES[0]).mean()),
                drawn_by_another_annotator={s: float((x.n_other_sets_drawn_iou > 0).mean()) for s, x in multi.groupby("stage")},
                median_stage1_p_no_response=float(d[d.stage == STAGES[1]].pmax_stage1.median()))
    json.dump(summ, open(OUT / "miss_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


def cmd_figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    d = pd.read_csv(OUT / "miss_decomposition.csv")
    col = {STAGES[0]: "#2a78d6", STAGES[1]: "#3d3d3a", STAGES[2]: "#8a8a85", STAGES[3]: "#eda100",
           STAGES[4]: "#f5cf73", STAGES[5]: "#eb6834", STAGES[6]: "#1baf7a", STAGES[7]: "#e87ba4"}
    groups = [("clear", d[d.visibility == "clear"]), ("faint", d[d.visibility == "faint"]), ("small (< 500 px)", d[d.area < GATE]),
              ("all", d)]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for yi, (name, x) in enumerate(groups[::-1]):
        left = 0.0
        sh = x.stage.value_counts(normalize=True).reindex(STAGES).fillna(0)
        for s in STAGES:
            ax[0].barh(yi, sh[s], left=left, color=col[s], edgecolor="white", height=0.62, label=s if yi == 0 else None)
            if sh[s] >= 0.06:
                ax[0].text(left + sh[s] / 2, yi, f"{sh[s]:.0%}", ha="center", va="center", fontsize=8,
                           color="white" if s in (STAGES[0], STAGES[1], STAGES[5]) else "#222")
            left += sh[s]
    ax[0].set_yticks(range(len(groups))); ax[0].set_yticklabels([f"{n} ({len(x):,})" for n, x in groups[::-1]])
    ax[0].set_xlim(0, 1); ax[0].set_xlabel("share of annotated filaments (all annotation sets)")
    ax[0].set_title("What happens to each annotated filament in the final pipeline")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=8)
    multi = d[d.n_other_sets > 0]
    order = [s for s in STAGES if s != STAGES[4]]
    vals = [float((multi[multi.stage == s].n_other_sets_drawn_iou > 0).mean()) for s in order]
    ax[1].barh(range(len(order))[::-1], vals, color=[col[s] for s in order], height=0.62)
    for yi, v in zip(range(len(order))[::-1], vals):
        ax[1].text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8.5)
    ax[1].set_yticks(range(len(order))[::-1]); ax[1].set_yticklabels(order, fontsize=8.5); ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("drawn by another annotator of the same image (IoU > 0.5)")
    ax[1].set_title("Did the other annotators draw it?\n(images with 2-3 annotation sets)")
    fig.tight_layout(); fig.savefig(OUT / "fig_miss_decomposition.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / "fig_miss_decomposition.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "figure"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"run": cmd_run, "figure": cmd_figure}[a.cmd](a)
