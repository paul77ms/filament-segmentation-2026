"""Cases: filaments that one annotator did not draw but another did, and that the final pipeline found. CPU only.

Rows (nested out-of-fold, images with 3 annotation sets): every ordered pair of sets (A, B) and every filament of B
that does not overlap anything A drew ("A did not draw it"; 0 shared pixels), with the third set C. For each: found
by the final pipeline (a prediction matches it at IoU > 0.5) and found by C (IoU > 0.5).
Cases: the rows found by the pipeline, the same physical filament seen from several pairs kept once; six are drawn
with numpy rng(2026) (no hand picking). Each row: H-alpha crop | the annotator who did not draw it | the annotator
who drew it (case dashed) | the pipeline's prediction. The bar chart gives the rates over ALL rows, found and not
found, by size, so the cases are not read as typical. (scripts/human_missed.py uses the looser definition "no
filament of A matches it at IoU > 0.5".) This shows agreement with other annotators, not accuracy against the Sun.
  CUDA_VISIBLE_DEVICES="" taskset -c 0 .venv/bin/python scripts/fig_annotator_cases.py
  -> outputs/report/figures_a/fig_annotator_cases.png and fig_annotator_cases.json
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json  # noqa: E402
import sys  # noqa: E402
from itertools import permutations  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs" / "report" / "figures_a"
N_CASES, SEED = 6, 2026
GREEN, ORANGE, BLUE, GREY = "#1baf7a", "#eda100", "#2a78d6", "#8a8a85"


def pool():
    import a_eval as AE
    from filseg.scorers import iou_matrix
    rows, cases = [], []
    for stem, fold, gts in AE.all_jobs():
        if len(gts) != 3:
            continue
        P = AE.final_instances(stem, fold)
        sets = {r: gts[r] for r in sorted(gts)}
        seen = []
        for a, b in permutations(sets, 2):
            c = next(r for r in sets if r not in (a, b))
            A, B, C = sets[a], sets[b], sets[c]
            if not len(B):
                continue
            notA = ~(iou_matrix(B, A)[0] > 0).any(1) if len(A) else np.ones(len(B), bool)
            iouP = iou_matrix(B, P)[1] if len(P) else np.zeros((len(B), 0))
            byP = (iouP > 0.5).any(1) if len(P) else np.zeros(len(B), bool)
            byC = (iou_matrix(B, C)[1] > 0.5).any(1) if len(C) else np.zeros(len(B), bool)
            for i in np.flatnonzero(notA):
                rows.append(dict(area=int(B.areas[i]), pipeline=bool(byP[i]), third=bool(byC[i])))
                if not byP[i]:
                    continue
                m = B.full_mask(int(i))
                if any((m & s_).sum() / (m | s_).sum() > 0.5 for s_ in seen):
                    continue
                seen.append(m)
                cases.append(dict(stem=stem, fold=fold, missed_by=a, drawn_by=b, gt_idx=int(i), area=int(B.areas[i]),
                                  pred_idx=int(iouP[i].argmax()), iou=float(iouP[i].max())))
    return rows, sorted(cases, key=lambda r: (r["stem"], r["drawn_by"], r["gt_idx"]))


def stretch(c):
    lo, hi = np.percentile(c, [1, 99])
    return np.clip((c.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def main():
    import a_eval as AE
    from filseg import data as D
    from filseg import seg
    rows, cand = pool()
    rng = np.random.default_rng(SEED)
    picks = [cand[int(i)] for i in sorted(rng.choice(len(cand), N_CASES, replace=False))]
    area = np.array([r["area"] for r in rows]); fp = np.array([r["pipeline"] for r in rows]); fc = np.array([r["third"] for r in rows])
    groups = [("< 500 px", area < 500), ("500-1,000 px", (area >= 500) & (area < 1000)), ("> 1,000 px", area >= 1000),
              ("all", np.ones(len(rows), bool))]
    rates = {lab: dict(n=int(g.sum()), third=float(fc[g].mean()), pipeline=float(fp[g].mean())) for lab, g in groups}
    fig = plt.figure(figsize=(13, 3.25 * N_CASES + 3.4))
    gs = fig.add_gridspec(N_CASES + 1, 4, height_ratios=[1] * N_CASES + [1.05], hspace=0.28, wspace=0.04, top=0.965, bottom=0.03)
    for r, c in enumerate(picks):
        gts = seg.gt_records([c["stem"] + ".jpeg"])
        A, B = gts[c["missed_by"]], gts[c["drawn_by"]]
        P = AE.final_instances(c["stem"], c["fold"])
        g = B.full_mask(c["gt_idx"]); p = P.full_mask(c["pred_idx"])
        ys, xs = np.nonzero(g | p); cy, cx = int(ys.mean()), int(xs.mean())
        h = int(max(np.ptp(ys), np.ptp(xs)) * 0.75) + 70
        y0, y1, x0, x1 = max(cy - h, 0), min(cy + h, 2048), max(cx - h, 0), min(cx + h, 2048)
        img = stretch(D.read_image(D.train_image_path(c["stem"] + ".jpeg"))[y0:y1, x0:x1])
        ax = [fig.add_subplot(gs[r, k]) for k in range(4)]
        for a in ax:
            a.imshow(img, cmap="gray"); a.set_xticks([]); a.set_yticks([])
        if len(A) and A.union()[y0:y1, x0:x1].any():
            ax[1].contour(A.union()[y0:y1, x0:x1].astype(np.uint8), [0.5], colors=[GREY], linewidths=1.0)
        ax[2].contour(B.union()[y0:y1, x0:x1].astype(np.uint8), [0.5], colors=[ORANGE], linewidths=1.0)
        for a in ax[1:3]:
            a.contour(g[y0:y1, x0:x1].astype(np.uint8), [0.5], colors=[GREEN], linewidths=1.6, linestyles="--")
        ax[3].contour(P.union()[y0:y1, x0:x1].astype(np.uint8), [0.5], colors=[BLUE], linewidths=1.0)
        ax[3].contour(p[y0:y1, x0:x1].astype(np.uint8), [0.5], colors=[BLUE], linewidths=2.0)
        ax[0].set_title(f"{c['stem']} ({c['area']} px)", fontsize=9)
        ax[1].set_title("annotator who did not draw it\n(grey: everything this annotator drew)", fontsize=9)
        ax[2].set_title("annotator who drew it\n(green dashed: the filament)", fontsize=9)
        ax[3].set_title(f"final pipeline (IoU {c['iou']:.2f} with that annotation)", fontsize=9)
    bx = fig.add_subplot(gs[N_CASES, 1:3])
    labels = [f"{lab}\n(n = {v['n']:,})" for lab, v in rates.items()]
    third = [v["third"] for v in rates.values()]; model = [v["pipeline"] for v in rates.values()]
    x = np.arange(len(labels))
    bx.bar(x - 0.18, third, 0.36, color=GREY, label="a third annotator"); bx.bar(x + 0.18, model, 0.36, color=BLUE, label="final pipeline")
    for xi, a_, b_ in zip(x, third, model):
        bx.text(xi - 0.18, a_ + 0.01, f"{a_:.0%}", ha="center", fontsize=8.5); bx.text(xi + 0.18, b_ + 0.01, f"{b_:.0%}", ha="center", fontsize=8.5)
    bx.set_xticks(x); bx.set_xticklabels(labels); bx.set_ylim(0, 0.6); bx.set_ylabel("share found")
    bx.spines[["top", "right"]].set_visible(False); bx.legend(frameon=False, fontsize=9, loc="upper left")
    bx.set_title("All filaments one annotator did not draw at all but another did (3-annotator images): who found them?", fontsize=10)
    fig.suptitle(f"Filaments one annotator did not draw at all, another did, and the pipeline found "
                 f"({N_CASES} drawn at random from {len(cand):,})", fontsize=12, y=0.99)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_annotator_cases.png", dpi=85, bbox_inches="tight"); plt.close(fig)
    json.dump(dict(pool_size=len(cand), rates=rates, cases=picks), open(OUT / "fig_annotator_cases.json", "w"), indent=1)
    print(json.dumps(dict(pool_size=len(cand), rates=rates, cases=[(c["stem"], c["area"], round(c["iou"], 2)) for c in picks])))


if __name__ == "__main__":
    main()
