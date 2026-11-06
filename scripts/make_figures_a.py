"""Report / README figures of the final pipeline (stage 1 -> R2 -> cleanup -> SE2 at m x PQ/2). CPU only.
Every prediction shown is the nested out-of-fold prediction of a validation image (as validated); every number comes
from the records named in the text (scripts/a_eval.py, scripts/eval_dual.py, scripts/island_cleanup.py,
scripts/tune_nested.py, scripts/human_missed.py, scripts/error_budget.py).
  CUDA_VISIBLE_DEVICES="" taskset -c 0 .venv/bin/python scripts/make_figures_a.py   -> outputs/report/figures_a/
  (copied to docs/figures/; `... make_figures_a.py ablation` / `stages` redraws only those figures)
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import se2 as S2  # noqa: E402
import small_exist as SE  # noqa: E402
import tune_nested as TN  # noqa: E402
from a_eval import SE2_MULT  # noqa: E402
from filseg import data as D  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.scorers import iou_matrix  # noqa: E402

OUT = ROOT / "outputs" / "report" / "figures_a"
R = ROOT / "outputs" / "research"
LB_FINAL = 0.37                                   # public leaderboard of the submitted CSV
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a8a85"
SETC = [AQUA, YELLOW, MAGENTA]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11})
_P = {}


def stretch(c):
    lo, hi = np.percentile(c, [0.5, 99.5])
    return np.clip((c.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def win(arr, cy, cx, r):
    return SE.crop_at(arr, int(cy), int(cx), r)


def lab_map(S, idx=None):
    out = np.zeros(SE.HW, np.int32)
    for i in (range(len(S)) if idx is None else idx):
        out[S.full_mask(i)] = i + 1
    return out


def outline(ax, mask, color, lw=1.2, ls="-"):
    if mask.any():
        ax.contour(mask.astype(np.uint8), levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)


def fill(ax, mask, color, alpha=0.45):
    rgba = np.zeros(mask.shape + (4,)); c = matplotlib.colors.to_rgb(color)
    rgba[mask] = (*c, alpha); ax.imshow(rgba)


def pq(df):
    s = df.sum(numeric_only=True)
    return float(s.sum_iou_host / (s.tp_host + .5 * s.fp_host + .5 * s.fn_host))


def v1(I):
    s = I.sum(numeric_only=True)
    return float(s.v1_si / (s.v1_tp + .5 * s.v1_fp + .5 * s.v1_fn))


def se2_scores(stem, n):
    if not _P:
        P = pd.read_csv(R / "se2" / "preds_outer.csv")
        _P.update({(s, int(i)): float(v) for s, i, v in zip(P.stem, P.idx, P.score)})
    return np.array([_P[(stem, q)] for q in range(n)])


def stages(stem, fold):
    """Intermediate results of the final pipeline for one outer-validation file (nested, as validated)."""
    from filseg import refine as RF
    S2._lazy_cand(fold)
    ctx = TN.FileCtx(stem, fold)
    lut = np.zeros(len(ctx.area), bool); lut[1:] = ctx.area[1:] >= 500
    gate = lut[ctx.lab]
    inb = np.bincount(ctx.cmap[gate], minlength=int(ctx.cmap.max()) + 1)
    keep = {j for j in range(1, len(inb)) if inb[j] > 0}
    P0 = RF.assemble({j: v for j, v in ctx.cp.items() if j in keep}, thr=0.5, min_area=300, merge_iou=0.5)
    P1 = TN.clean(P0, 20, 0.05)
    Rf, _ = S2.fragments(ctx.p, ctx.cmap, ctx.cp)
    small = np.zeros(len(ctx.area), bool); small[1:] = (ctx.area[1:] >= 100) & (ctx.area[1:] < 500)
    if Rf is not None:
        sc = se2_scores(stem, len(Rf))
        Q, added = SE.add_only(P1, Rf, np.flatnonzero(sc > SE2_MULT[fold] * ctx.tau))
    else:
        Q, added, sc = P1, [], np.zeros(0)
    return dict(p=ctx.p, gate=gate, small=small[ctx.lab], cand=ctx.cmap > 0, P0=P0, P1=P1, Rf=Rf, sc=sc, Q=Q,
                added=added, tau=ctx.tau)


GREEN, RED = "#1baf7a", "#e34a33"
PRED_FRAME, KEY_FRAME = "#2a78d6", "#6a3d9a"          # (g) prediction, (h) answer key
ANNC = ["#00b4d8", "#f2c200", "#d33fbf"]               # annotation sets 1-3 in the whole-disk panel


def stage_examples():
    """Two validation images picked by a rule, not by eye. Eligible: >= 2 annotation sets, >= 6 annotated filaments
    per set on average, >= 1 SE2 addition (nested OOF records of scripts/a_eval.py). Well-detected = the image whose
    per-image V0 is closest to the 90th percentile of the eligible images, hard = closest to the 10th."""
    r = pd.read_csv(R / "final" / "records.csv")
    g = r.groupby(["stem", "fold"])[["tp_host", "fp_host", "fn_host", "sum_iou_host"]].sum().reset_index()
    g["n_sets"] = r.groupby("stem").size().reindex(g.stem).values
    g["n_add"] = g.stem.map(pd.read_csv(R / "se2_analysis" / "added.csv").groupby("stem").size()).fillna(0)
    g["v0"] = g.sum_iou_host / (g.tp_host + .5 * g.fp_host + .5 * g.fn_host)
    e = g[(g.n_sets >= 2) & ((g.tp_host + g.fn_host) / g.n_sets >= 6) & (g.n_add >= 1)].sort_values("stem")
    pick = lambda q: e.iloc[int((e.v0 - e.v0.quantile(q)).abs().values.argmin())]  # noqa: E731
    return len(e), {"good": pick(0.9), "hard": pick(0.1)}


def outcome(stem, Q):
    """Per annotated filament of every set: caught (a prediction matches it at IoU > 0.5) or missed; and the
    predictions that match no annotation set."""
    sets = [G for _, G in sorted(seg.gt_records([stem + ".jpeg"]).items())]
    caught, missed, hit = np.zeros(SE.HW, bool), np.zeros(SE.HW, bool), np.zeros(len(Q), bool)
    n_c = n_m = 0
    per_set = []
    for G in sets:
        iou = iou_matrix(G, Q)[1] if len(G) and len(Q) else np.zeros((len(G), len(Q)))
        c0 = n_c
        for i in range(len(G)):
            ok = bool(len(Q) and iou[i].max() > 0.5)
            (caught if ok else missed)[G.full_mask(i)] = True
            n_c += ok; n_m += not ok
        per_set.append((len(G), n_c - c0))
        if len(G) and len(Q):
            hit |= (iou > 0.5).any(0)
    nomatch = np.zeros(SE.HW, bool)
    for j in np.flatnonzero(~hit):
        nomatch |= Q.full_mask(int(j))
    return sets, caught, missed, nomatch, n_c, n_m, int((~hit).sum()), per_set


def fig_stages(kind, row, n_elig):
    stem, fold = row.stem, int(row.fold)
    s = stages(stem, fold)
    sets, caught, missed, nomatch, n_c, n_m, n_nm, per_set = outcome(stem, s["Q"])
    n_base = len(s["P1"])
    # zoom: of the 640 px windows centred on an annotated filament (any set) or an SE2 addition, the one with the most
    # annotated pixels; the choice does not look at caught / missed
    ann = caught | missed
    cs = [tuple(int(v) for v in np.argwhere(s["Q"].full_mask(i)).mean(0)) for i in range(n_base, len(s["Q"]))]
    cs += [tuple(int(v) for v in np.argwhere(G.full_mask(i)).mean(0)) for G in sets for i in range(len(G))]
    r = 320
    cy, cx = max(cs, key=lambda c: int(win(ann.astype(np.uint8), c[0], c[1], r).sum()))
    img = D.read_image(D.train_image_path(stem + ".jpeg"))
    fig = plt.figure(figsize=(17, 19.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[2.0, 1, 1])
    top = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4])]
    ax = [fig.add_subplot(gs[1 + k // 4, k % 4]) for k in range(8)]
    # top row: the whole disk, the answer key (every annotation set, its own colour) and the prediction
    from matplotlib.patches import Patch
    disk = stretch(img[120:1928, 120:1928])
    for a_ in top:
        a_.imshow(disk, cmap="gray"); a_.set_xticks([]); a_.set_yticks([])
        a_.add_patch(plt.Rectangle((cx - r - 120, cy - r - 120), 2 * r, 2 * r, fill=False, ec="w", lw=1.5, ls="--"))
    hk = []
    for j, G in enumerate(sets):
        u = G.union()[120:1928, 120:1928] if len(G) else np.zeros((1808, 1808), bool)
        fill(top[0], u, ANNC[j % 3], 0.35); outline(top[0], u, ANNC[j % 3], 1.0)
        hk.append(Patch(fc=ANNC[j % 3], ec=ANNC[j % 3], alpha=0.8,
                        label=f"annotator {j + 1}: {per_set[j][0]} filaments ({per_set[j][1]} caught)"))
    top[0].legend(handles=hk + [Patch(fc="none", ec="w", ls="--", label="zoom window (panels b-h)")], loc="lower left",
                  fontsize=10, framealpha=0.85)
    qb = lab_map(s["Q"], range(n_base))[120:1928, 120:1928] > 0
    qa = lab_map(s["Q"], range(n_base, len(s["Q"])))[120:1928, 120:1928] > 0
    fill(top[1], qb, BLUE, 0.6); outline(top[1], qb, BLUE, 1.0)
    fill(top[1], qa, ORANGE, 0.85); outline(top[1], qa, ORANGE, 1.0)
    top[1].legend(handles=[Patch(fc=BLUE, alpha=0.7, label=f"refined filaments ({n_base})"),
                           Patch(fc=ORANGE, alpha=0.85, label=f"added by SE2 ({len(s['Q']) - n_base})"),
                           Patch(fc="none", ec="w", ls="--", label="zoom window (panels b-h)")], loc="lower left",
                  fontsize=10, framealpha=0.85)
    for a_, t, c_ in ((top[0], "ANSWER KEY: every annotation set, whole disk", KEY_FRAME),
                      (top[1], "PREDICTION: final output, whole disk", PRED_FRAME)):
        a_.set_title(t, fontsize=13, color=c_, fontweight="bold")
        for sp in a_.spines.values():
            sp.set_visible(True); sp.set_color(c_); sp.set_linewidth(5)
    I = stretch(win(img, cy, cx, r))
    titles = ["(a) full disk: caught / missed; box = zoom", "(b) H-alpha input (native resolution)",
              "(c) stage 1: probability (mean of 5 U-Nets)", "(d) existence gate + candidates", "(e) R2 refinement + cleanup",
              "(f) small pieces -> redrawn by R2", "(g) PREDICTION: final output (+ SE2)",
              "(h) ANSWER KEY: annotations, caught / missed"]
    ax[0].imshow(stretch(img[120:1928, 120:1928]), cmap="gray")
    for m, c_, ls in ((caught, GREEN, "-"), (missed, RED, "-"), (nomatch, YELLOW, "--")):
        outline(ax[0], m[120:1928, 120:1928], c_, 0.9, ls)
    ax[0].add_patch(plt.Rectangle((cx - r - 120, cy - r - 120), 2 * r, 2 * r, fill=False, ec="#2a78d6", lw=2))
    for a_ in ax[1:]:
        a_.imshow(I, cmap="gray")
    for a_, t in zip(ax, titles):
        a_.set_title(t, fontsize=11); a_.set_xticks([]); a_.set_yticks([])
    ax[2].imshow(win(s["p"], cy, cx, r), cmap="magma", vmin=0, vmax=1, alpha=0.85)
    fill(ax[3], win(s["gate"].astype(np.uint8), cy, cx, r) > 0, BLUE, 0.5)
    outline(ax[3], win(s["cand"].astype(np.uint8), cy, cx, r) > 0, YELLOW, 0.9)
    L1 = win(lab_map(s["P1"]), cy, cx, r)
    for k in np.unique(L1)[1:]:
        outline(ax[4], L1 == k, BLUE, 1.6)
    fill(ax[5], win(s["small"].astype(np.uint8), cy, cx, r) > 0, YELLOW, 0.7)
    if s["Rf"] is not None:
        LR = win(lab_map(s["Rf"]), cy, cx, r)
        for k in np.unique(LR)[1:]:
            outline(ax[5], LR == k, ORANGE, 1.6)
    pb = win(lab_map(s["Q"], range(n_base)), cy, cx, r) > 0
    pa = win(lab_map(s["Q"], range(n_base, len(s["Q"]))), cy, cx, r) > 0
    fill(ax[6], pb, BLUE, 0.6); outline(ax[6], pb, BLUE, 1.4)
    fill(ax[6], pa, ORANGE, 0.85); outline(ax[6], pa, ORANGE, 1.4)
    wc, wm = win(caught.astype(np.uint8), cy, cx, r) > 0, win(missed.astype(np.uint8), cy, cx, r) > 0
    fill(ax[7], wc, GREEN, 0.45); outline(ax[7], wc, GREEN, 1.4)
    fill(ax[7], wm, RED, 0.45); outline(ax[7], wm, RED, 1.4)
    outline(ax[7], win(nomatch.astype(np.uint8), cy, cx, r) > 0, YELLOW, 1.8, "--")
    for k, c_ in ((6, PRED_FRAME), (7, KEY_FRAME)):           # highlight: what the pipeline outputs vs the answer key
        for sp in ax[k].spines.values():
            sp.set_visible(True); sp.set_color(c_); sp.set_linewidth(5)
        ax[k].title.set_color(c_); ax[k].title.set_fontweight("bold")
    notes = {0: f"whole image, counted per annotator: {n_c} caught (green),\n{n_m} missed (red); per annotator "
                + ", ".join(f"{c}/{n}" for n, c in per_set) + f"\n{n_nm} predictions match no annotation (yellow dashed)",
             2: "bright = filament probability; the faint ends and\nthin parts are weak at 1024 px",
             3: "blue: components >= 500 px confirm a filament\nyellow: candidates p > 0.3 that R2 will redraw",
             4: "each confirmed filament redrawn at native\nresolution; duplicates merged, specks removed",
             5: "yellow: 100-499 px pieces the gate drops\norange: the same pieces redrawn by R2",
             6: f"what the pipeline submits\nblue: refined filaments; orange: added by SE2 ({len(s['added'])} in this image)",
             7: f"{len(sets)} annotation sets. green: caught (IoU > 0.5), red: missed\n(judged per annotator: one filament can be both)\nyellow dashed: a prediction no annotator drew"}
    for k, t in notes.items():                     # below the panel, so that no note covers a filament
        ax[k].set_xlabel(t, fontsize=8.5, color=(PRED_FRAME if k == 6 else KEY_FRAME if k == 7 else "#333"), labelpad=4)
    head = {"good": "A well-detected image", "hard": "A hard image"}[kind]
    q = {"good": "90th", "hard": "10th"}[kind]
    fig.suptitle(f"{head} ({stem}, fold {fold}, never seen by its models; per-image V0 {row.v0:.2f}, the {q} percentile of "
                 f"{n_elig} eligible validation images; zoom: {2 * r} px native window)", fontsize=12, y=1.0)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(OUT / ("fig_pipeline_stages.png" if kind == "good" else "fig_pipeline_stages_hard.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return dict(stem=stem, fold=fold, v0=round(float(row.v0), 4), n_add=len(s["added"]), caught=n_c, missed=n_m, no_match=n_nm)


def fig_stages_both():
    n, ex = stage_examples()
    return {k: fig_stages(k, row, n) for k, row in ex.items()}


def fig_ablation():
    src = [("stage 1 + gate\n(nested B_ens)", R / "eval_dual", "B_ens"), ("+ R2 refinement", R / "eval_dual", "R2nb0"),
           ("+ cleanup", R / "island_cleanup", "R2nb0_isl5"), ("+ SE2\n(final pipeline)", R / "final", "final")]
    rows = []
    for name, d, m in src:
        r = pd.read_csv(d / "records.csv"); i = pd.read_csv(d / "images_v1.csv")
        rows.append(dict(stage=name, V0=pq(r[r.method == m]), V1=v1(i[i.method == m])))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(df))
    for col, c, lab, mk in (("V0", BLUE, "V0: official per-annotation-set PQ", "o"),
                            ("V1", ORANGE, "V1: annotation sets pooled per image", "s")):
        ax.plot(x, df[col], color=c, lw=2, marker=mk, ms=8, label=lab, mec="white", mew=1.5)
        for xi, v in zip(x, df[col]):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color="#333")
    ax.annotate(f"public leaderboard of the\nsubmitted CSV: {LB_FINAL:.2f}", (x[-1], df.V1.iloc[-1]), textcoords="offset points",
                xytext=(-10, -38), ha="center", fontsize=8.5, color="#555")
    ax.set_xticks(x); ax.set_xticklabels(df.stage); ax.set_ylabel("PQ (IoU > 0.5)"); ax.set_ylim(0.33, 0.45)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8); ax.set_axisbelow(True); ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Contribution of each stage (nested 5-fold out-of-fold, 707 training images)")
    fig.savefig(OUT / "fig_ablation.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    df.to_csv(OUT / "ablation.csv", index=False)
    return df.round(4).to_dict("records")


def fig_annotators():
    """One refined fragment of a 3-set image added by SE2: per-set credit and the decision."""
    d = pd.concat([pd.read_csv(R / "se2" / "data" / f"eval_f{k}.csv").assign(outer=k) for k in range(5)])
    p = pd.read_csv(R / "se2" / "preds_outer.csv")
    tau = json.load(open(R / "small_exist" / "train_log.json"))["tau"]["T2"]
    m = d.merge(p[["stem", "idx", "outer", "score"]], on=["stem", "idx", "outer"])
    m = m[(m.n_sets == 3) & (m.score > m.outer.map(lambda k: SE2_MULT[int(k)] * tau[str(k)]))].sort_values("area", ascending=False)
    for _, r in m.iterrows():
        s = stages(r.stem, int(r.fold))
        sets = [g for _, g in sorted(seg.gt_records([r.stem + ".jpeg"]).items())]
        ious = [float(iou_matrix(G, s["Rf"])[1][:, int(r.idx)].max()) if len(G) else 0.0 for G in sets]
        if sum(i > 0.5 for i in ious) == 2:
            break
    q = int(r.idx); mask = s["Rf"].full_mask(q)
    yy, xx = np.nonzero(mask); cy, cx, rr = int(yy.mean()), int(xx.mean()), 80
    img = D.read_image(D.train_image_path(r.stem + ".jpeg"))
    fig = plt.figure(figsize=(10, 4.4))
    ax = fig.add_axes([0.02, 0.06, 0.42, 0.86]); ax.imshow(stretch(win(img, cy, cx, rr)), cmap="gray")
    for j, G in enumerate(sets):
        outline(ax, win(G.union().astype(np.uint8), cy, cx, rr) > 0, SETC[j], 1.6)
    outline(ax, win(s["small"].astype(np.uint8), cy, cx, rr) > 0, "#ffffff", 0.9, ":")
    outline(ax, win(mask.astype(np.uint8), cy, cx, rr) > 0, BLUE, 2.0, "--")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("one redrawn fragment, three annotators")
    tx = fig.add_axes([0.47, 0.06, 0.51, 0.86]); tx.axis("off")
    lines = ["Training target of the SE2 scorer (no fusion, no random choice):", ""]
    gains = []
    for j, (i, nm) in enumerate(zip(ious, ["annotator set 1 (aqua)", "annotator set 2 (yellow)", "annotator set 3 (magenta)"])):
        g = i if i > 0.5 else 0.0; gains.append(g)
        lines.append(f"  {nm}:  IoU {i:.2f}  ->  credit {g:.2f}")
    y = float(np.mean(gains))
    lines += ["", f"  target y = mean credit = {y:.2f}", "",
              "Decision at test time: add the redrawn fragment iff the", "predicted y > 0.6 x PQ/2 = 0.1274 and it overlaps nothing.",
              "", "PQ/2: an added disjoint object changes the pooled per-set", "PQ numerator by n_sets * y and the denominator by",
              "n_sets / 2, so it helps iff y > PQ / 2. The factor 0.6 maximises", "the pooled-set PQ (leaderboard-type) out of fold.",
              "", f"This fragment: predicted y = {r.score:.2f} -> added", "(white dotted: stage-1 pieces; blue dashed: R2 redraw)"]
    tx.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.savefig(OUT / "fig_annotators.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return dict(stem=r.stem, ious=ious, y=y, score=float(r.score))


def fig_human():
    import eval_dual as ED
    ED.OUT = R / "final" / "loao"; ED.LOAO_RATERS = ("final",)
    asp, _ = ED.loao_aspects()
    keys = [("PQ", "PQ (vs the other annotators)"), ("SQ", "matched-pair IoU (SQ)"), ("RQ", "recognition (RQ)"),
            ("small_detected_rate", "small filaments detected (< 500 px)"), ("faint_detected_rate", "faint filaments detected"),
            ("thin_recall", "thin-region recall"), ("thin_precision", "thin-region precision"),
            ("split_gt_rate", "filaments split (lower is better)"), ("fp_spurious_rate", "spurious predictions (lower is better)")]
    h = [float(asp["human"][k]) for k, _ in keys]; m = [float(asp["final"][k]) for k, _ in keys]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    yv = np.arange(len(keys))[::-1]
    ax.barh(yv + 0.19, h, height=0.36, color=GREY, label="a human annotator (leave-one-annotator-out)")
    ax.barh(yv - 0.19, m, height=0.36, color=BLUE, label="final pipeline")
    for yi, a_, b_ in zip(yv, h, m):
        ax.text(a_ + 0.01, yi + 0.19, f"{a_:.2f}", va="center", fontsize=8.5, color="#333")
        ax.text(b_ + 0.01, yi - 0.19, f"{b_:.2f}", va="center", fontsize=8.5, color="#333")
    ax.set_yticks(yv); ax.set_yticklabels([n for _, n in keys]); ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right", frameon=False, fontsize=9); ax.grid(axis="x", color="#e6e6e3", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Scored against the remaining annotators of the same images (296 multi-set images)")
    fig.savefig(OUT / "fig_human_vs_model.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return {k: (round(a_, 4), round(b_, 4)) for (k, _), a_, b_ in zip(keys, h, m)}


def fig_qualitative():
    rc = pd.read_csv(ROOT / "outputs/audit/review_cases.csv")
    cases = [rc[rc.category == c].sort_values("case_id").iloc[0] for c in ["rnd", "thin", "faint", "limb", "pair", "disag"]]
    fig, ax = plt.subplots(3, 4, figsize=(15, 11.6))
    for k, c in enumerate(cases):
        stem, fold = c.file_name[:-5], int(c.fold_v1)
        s = stages(stem, fold)
        x0, y0, w, h = int(c.crop_x), int(c.crop_y), int(c.crop_w), int(c.crop_h)
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        img = stretch(D.read_image(D.train_image_path(stem + ".jpeg"))[sl])
        sets = [g for _, g in sorted(seg.gt_records([stem + ".jpeg"]).items())]
        n_base = len(s["P1"])
        a1, a2 = ax.flat[2 * k], ax.flat[2 * k + 1]
        for a in (a1, a2):
            a.imshow(img, cmap="gray"); a.set_xticks([]); a.set_yticks([])
        for j, G in enumerate(sets):
            outline(a1, G.union()[sl], SETC[j % 3], 1.2)
        fill(a2, lab_map(s["Q"], range(n_base))[sl] > 0, BLUE, 0.5)
        fill(a2, lab_map(s["Q"], range(n_base, len(s["Q"])))[sl] > 0, ORANGE, 0.8)
        a1.set_title(f"{c.case_id}: annotations ({len(sets)} set{'s' if len(sets) > 1 else ''})")
        a2.set_title(f"{c.case_id}: final prediction ({len(s['Q'])} in the whole image)")
    fig.suptitle("Fixed review cases (first of each category, chosen before any model), validation predictions. "
                 "Left: annotator sets (aqua / yellow / magenta). Right: final pipeline (blue; orange = added by SE2).", y=0.995)
    fig.tight_layout(); fig.savefig(OUT / "fig_qualitative.png", dpi=95, bbox_inches="tight"); plt.close(fig)
    return [c.case_id for c in cases]


def fig_thin_recovery():
    """Filaments drawn by >= 2 annotators that stage 1 + gate misses although it sees them (p > 0.5 only in pieces), and
    that SE2 recovers (the matching prediction is an SE2 addition)."""
    g = pd.read_csv(ROOT / "outputs" / "error_budget" / "gt_instances.csv")
    c = g[(g.n_sets >= 2) & (g.n_other_sets_drawn_iou >= 1) & (g.status == "fn_missed") & (g.pmax >= 0.5)]
    picks = []
    for _, r in c.sort_values(["mean_width", "stem"]).iterrows():
        if len(picks) == 3:
            break
        if r.stem in {x[0].stem for x in picks}:
            continue
        s = stages(r.stem, int(r.fold))
        if not s["added"]:
            continue
        G = seg.gt_records([r.stem + ".jpeg"])[r.rid]
        _, iou = iou_matrix(G, s["Q"])
        j = int(iou[int(r.gt_idx)].argmax())
        if j >= len(s["P1"]) and iou[int(r.gt_idx), j] > 0.5:
            picks.append((r, s, G.full_mask(int(r.gt_idx)), s["Q"].full_mask(j), float(iou[int(r.gt_idx), j])))
    fig, ax = plt.subplots(len(picks), 4, figsize=(13, 3.4 * len(picks)), squeeze=False)
    for k, (r, s, gm, qm, b) in enumerate(picks):
        yy, xx = np.nonzero(gm | qm); cy, cx = int(yy.mean()), int(xx.mean())
        rr = max(60, int(max(np.ptp(yy), np.ptp(xx)) * 0.8))
        img = stretch(win(D.read_image(D.train_image_path(r.stem + ".jpeg")), cy, cx, rr))
        for a in ax[k]:
            a.imshow(img, cmap="gray"); a.set_xticks([]); a.set_yticks([])
        sets = [gg for _, gg in sorted(seg.gt_records([r.stem + ".jpeg"]).items())]
        for j, G in enumerate(sets):
            outline(ax[k, 1], win(G.union().astype(np.uint8), cy, cx, rr) > 0, SETC[j % 3], 1.2)
        ax[k, 2].imshow(win(s["p"], cy, cx, rr), cmap="magma", vmin=0, vmax=1, alpha=0.85)
        outline(ax[k, 2], win((s["p"] > 0.5).astype(np.uint8), cy, cx, rr) > 0, AQUA, 0.8)
        fill(ax[k, 3], win(qm.astype(np.uint8), cy, cx, rr) > 0, ORANGE, 0.75)
        ax[k, 0].set_title(f"{r.stem}: width {r.mean_width:.1f} px", fontsize=9)
        ax[k, 1].set_title(f"annotations ({int(r.n_other_sets_drawn_iou) + 1} of {int(r.n_sets)} sets drew it)", fontsize=9)
        ax[k, 2].set_title("stage 1 at 1024: only pieces < 500 px", fontsize=9)
        ax[k, 3].set_title(f"SE2: redrawn and added (IoU {b:.2f})", fontsize=9)
    fig.suptitle("Thin filaments that annotators agree on are lost to the downsampling, not to the model "
                 "(missed by stage 1 + gate; recovered by SE2)", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "fig_thin_recovery.png", dpi=95, bbox_inches="tight"); plt.close(fig)
    return [p[0].stem for p in picks]


def fig_beyond_human():
    g = pd.read_csv(ROOT / "outputs" / "error_budget" / "gt_instances.csv"); m = g[g.n_sets >= 2].copy()
    m["support"] = np.where(m.n_other_sets_drawn_iou == m.n_other_sets, "all", np.where(m.n_other_sets_drawn_cov > 0, "some", "one"))
    bins = ["<250", "250-500", "500-1k", "1k-2k", "2k-5k", ">5k"]
    m["abin"] = pd.cut(m.area, [0, 250, 500, 1000, 2000, 5000, 1e7], labels=bins)
    t = pd.crosstab(m.abin, m.support, normalize="index")[["all", "some", "one"]]
    h = pd.read_csv(R / "se2_analysis" / "human_missed.csv"); h = h[h.third_human.notna()].copy()
    h["sz"] = pd.cut(h.area, [0, 500, 1000, 1e7], labels=["< 500 px", "500-1,000 px", "> 1,000 px"])
    hm = h.groupby("sz", observed=True).agg(third_human=("third_human", lambda v: v.astype(bool).mean()), model=("model_se2", "mean"))
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(bins)); bottom = np.zeros(len(bins))
    for col, c, lab in (("all", BLUE, "drawn by all annotators"), ("some", AQUA, "drawn by some (overlapping)"), ("one", GREY, "drawn by one annotator only")):
        ax[0].bar(x, t[col].values, bottom=bottom, color=c, label=lab, width=0.7, edgecolor="white", lw=1.5)
        for xi, b0, v in zip(x, bottom, t[col].values):
            ax[0].text(xi, b0 + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if col != "some" else "#123")
        bottom += t[col].values
    ax[0].set_xticks(x); ax[0].set_xticklabels([b + " px" for b in bins], fontsize=9); ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("share of annotated filaments"); ax[0].legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.12))
    ax[0].set_title("Annotators agree less on small filaments (multi-set images)", pad=26)
    x2 = np.arange(len(hm))
    ax[1].bar(x2 - 0.18, hm.third_human.values, width=0.36, color=GREY, label="a third annotator", edgecolor="white")
    ax[1].bar(x2 + 0.18, hm.model.values, width=0.36, color=BLUE, label="final pipeline", edgecolor="white")
    for xi, a_, b_ in zip(x2, hm.third_human.values, hm.model.values):
        ax[1].text(xi - 0.18, a_ + 0.01, f"{a_:.0%}", ha="center", fontsize=9); ax[1].text(xi + 0.18, b_ + 0.01, f"{b_:.0%}", ha="center", fontsize=9)
    ax[1].set_xticks(x2); ax[1].set_xticklabels(hm.index.astype(str)); ax[1].set_ylim(0, 0.6)
    ax[1].set_ylabel("share found (IoU > 0.5)"); ax[1].legend(frameon=False, fontsize=9, loc="upper left")
    ax[1].set_title(f"Filaments annotator A missed but annotator B drew ({len(h):,} cases, 3-set images)")
    for a in ax:
        a.grid(axis="y", color="#e6e6e3", lw=0.8); a.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "fig_beyond_human.png", dpi=125, bbox_inches="tight"); plt.close(fig)
    return dict(agreement=t.round(3).to_dict(), human_missed=hm.round(3).to_dict())


def fig_constants():
    """Sensitivity of the six post-processing constants (records of scripts/tune_nested.py). It was measured while the
    small-fragment stage was an earlier scorer of the raw pieces; that stage's constant is not shown."""
    s_ = pd.read_csv(ROOT / "outputs/research/tune_nested/sensitivity.csv").set_index("config")
    names = {"g_min": "existence gate (px)", "a_thr": "R2 mask threshold", "merge": "merge IoU", "a_min": "min filament area (px)",
             "c_part": "cleanup: min part (px)", "i_frac": "cleanup: island fraction"}
    fig, axes = plt.subplots(1, len(names), figsize=(16.5, 3.3), sharey=True)
    for ax, k in zip(axes, names):
        vals = TN.GRID[k]
        cfgs = [TN.cid(dict(TN.DEFAULT, **{k: v})) for v in vals]
        d0 = [float(s_.loc[c, "dV0"]) for c in cfgs]; d1 = [float(s_.loc[c, "dV1"]) for c in cfgs]
        ax.plot(vals, d0, color=BLUE, marker="o", lw=2, ms=6, label="V0", mec="white")
        ax.plot(vals, d1, color=ORANGE, marker="s", lw=2, ms=6, label="V1", mec="white")
        ax.axvline(TN.DEFAULT[k], color=GREY, lw=1, ls="--"); ax.axhline(0, color="#cfcfca", lw=0.8)
        ax.axhspan(-0.002, 0.002, color="#f1f1ee", zorder=0)
        ax.set_title(names[k], fontsize=10); ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    axes[0].set_ylabel("PQ change vs current"); axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Sensitivity of the post-processing constants (nested OOF; dashed = current value; band = +/-0.002)", y=1.04)
    fig.savefig(OUT / "fig_constants.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return list(names)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["ablation"]:
        print(fig_ablation()); sys.exit()
    if sys.argv[1:] == ["stages"]:
        print(fig_stages_both()); sys.exit()
    info = dict(stages=fig_stages_both(), ablation=fig_ablation(), annotators=fig_annotators(), human=fig_human(),
                qualitative=fig_qualitative(), thin=fig_thin_recovery(), beyond=fig_beyond_human(), constants=fig_constants())
    json.dump(info, open(OUT / "figures_info.json", "w"), indent=1, default=str)
    print(json.dumps(info, indent=1, default=str)[:3000])
