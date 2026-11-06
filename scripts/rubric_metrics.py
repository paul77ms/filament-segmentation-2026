"""Quantitative items of the final-evaluation rubric on the nested OOF (707 files, 1,154 annotation-set records), for
the stages of the final pipeline. CPU only, analysis only.

Rubric (overview, "평가"): mean Dice (torchmetrics DiceScore), the leaderboard metric, Dice distribution, IoU
distribution, distribution of one-to-many / many-to-one relations. Definitions follow the official self-evaluation
notebook (src/filseg/scorers.notebook_diag; scripts/official_eval/): per (annotation set, image) record,
  pairs     every (GT, prediction) pair with >= 1 px overlap -> IoU and Dice values (the notebook's histograms)
  1:N       GT filaments touched by >= 2 predictions; N:1 predictions touching >= 2 GT filaments
  sem Dice  foreground Dice of the union masks per record, mean over records (torchmetrics configuration S1 of
            scripts/official_eval/dice_torchmetrics.py; the organisers' exact configuration is not published)
Methods (tune_nested.FileCtx, nested stage 1 B_ens):
  stage1    components of p > 0.5 with >= 500 px (B_ens; the existence gate)
  r2_clean  R2 refinement + assembly + cleanup, no SE
  final_se2 the final submission (+ SE2, threshold m_k x PQ/2, a_eval.SE2_MULT)
  CUDA_VISIBLE_DEVICES="" taskset -c 0 .venv/bin/python scripts/rubric_metrics.py run --workers 1
  ... figure   -> outputs/report/rubric/{summary.json, pairs.npz, fig_rubric.png}
"""
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
from filseg.metrics import single_thread_worker  # noqa: E402

OUT = ROOT / "outputs" / "report" / "rubric"
METHODS = ("stage1", "r2_clean", "final_se2")


def _se2(stem, fold):
    """the final pipeline: base + SE2 at the fold-separated threshold m_k x tau_k (scripts/a_eval.py)."""
    import a_eval as AE
    return AE.final_instances(stem, fold)


def job(args):
    import tune_nested as TN
    from filseg import seg
    from filseg.scorers import iou_matrix, notebook_diag
    stem, fold, gts = args
    ctx = TN.FileCtx(stem, fold)
    P = {"stage1": seg.prob_to_instances(ctx.p, seg.PostCfg(0.5, 500)),
         "r2_clean": ctx.instances(dict(TN.DEFAULT, se_mult=1e9)),
         "final_se2": _se2(stem, fold)}
    out = {}
    for m in METHODS:
        recs, ious, dices = [], [], []
        for rid, G in gts.items():
            d = notebook_diag(G, P[m])
            inter, iou = iou_matrix(G, P[m])
            t = inter > 0
            if t.any():
                ious.append(iou[t])
                dices.append(2 * inter[t] / np.maximum((G.areas[:, None] + P[m].areas[None, :])[t], 1))
            recs.append(dict(rid=rid, **d))
        out[m] = (recs, np.concatenate(ious) if ious else np.zeros(0), np.concatenate(dices) if dices else np.zeros(0))
    return out


def cmd_run(a):
    import multiprocessing as mp
    import a_eval as AE
    single_thread_worker()
    jobs = AE.all_jobs()[:a.limit or None]
    acc = {m: ([], [], []) for m in METHODS}
    t0 = time.time()
    it = map(job, jobs) if a.workers == 1 else \
        mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker).imap_unordered(job, jobs)
    for i, o in enumerate(it):
        for m in METHODS:
            acc[m][0].extend(o[m][0]); acc[m][1].append(o[m][1]); acc[m][2].append(o[m][2])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summ, arrs = {}, {}
    for m in METHODS:
        R = acc[m][0]
        iou = np.concatenate(acc[m][1]); dice = np.concatenate(acc[m][2])
        arrs[f"{m}_iou"], arrs[f"{m}_dice"] = iou.astype(np.float32), dice.astype(np.float32)
        S = lambda k: int(sum(r[k] for r in R))  # noqa: E731
        sem = [2 * r["sem_inter"] / (r["sem_gt"] + r["sem_pred"]) if r["sem_gt"] + r["sem_pred"] else 1.0 for r in R]
        summ[m] = dict(records=len(R), sem_dice_mean=float(np.mean(sem)),
                       sem_dice_pooled=2 * S("sem_inter") / (S("sem_gt") + S("sem_pred")),
                       pairs=int(len(iou)), pair_iou_mean=float(iou.mean()), pair_dice_mean=float(dice.mean()),
                       pair_iou_median=float(np.median(iou)), pair_iou_gt05=float((iou > 0.5).mean()),
                       gt_1to0=S("nb_gt_1to0"), gt_1to1=S("nb_gt_1to1"), gt_1toN=S("nb_gt_1toN"),
                       pred_0to1=S("nb_pred_0to1"), pred_1to1=S("nb_pred_1to1"), pred_Nto1=S("nb_pred_Nto1"),
                       over_frag=S("over_frag"), under_frag=S("under_frag"))
    np.savez_compressed(OUT / "pairs.npz", **arrs)
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), f"{time.time() - t0:.0f}s")


def cmd_figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summ = json.load(open(OUT / "summary.json")); z = np.load(OUT / "pairs.npz")
    lab = {"stage1": "stage 1 only", "r2_clean": "+ R2 + cleanup", "final_se2": "final (+ SE2)"}
    col = {"stage1": "#9aa0a6", "r2_clean": "#5b8def", "final_se2": "#1a3d8f"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.linspace(0, 1, 26)
    for k, (key, name) in enumerate((("iou", "IoU"), ("dice", "Dice"))):
        for m in ("stage1", "final_se2"):
            v = z[f"{m}_{key}"]
            ax[k].hist(v, bins=bins, histtype="step", lw=2, color=col[m], density=True,
                       label=f"{lab[m]} (mean {v.mean():.3f}, n={len(v)})")
        ax[k].set_xlabel(f"{name} of every overlapping (GT, prediction) pair"); ax[k].set_ylabel("density")
        ax[k].set_title(f"{name} distribution (official notebook definition)"); ax[k].legend(fontsize=8, loc="upper left")
    x = np.arange(2); w = 0.27
    for i, m in enumerate(("stage1", "r2_clean", "final_se2")):
        ax[2].bar(x + (i - 1) * w, [summ[m]["gt_1toN"], summ[m]["pred_Nto1"]], w, color=col[m], label=lab[m])
    ax[2].set_xticks(x, ["one-to-many\n(GT split into >= 2 predictions)", "many-to-one\n(prediction joins >= 2 GT)"])
    ax[2].set_ylabel("count (1,154 records)"); ax[2].set_title("one-to-many / many-to-one"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_rubric.png", dpi=110)
    print("wrote", OUT / "fig_rubric.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "figure"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"run": cmd_run, "figure": cmd_figure}[a.cmd](a)
