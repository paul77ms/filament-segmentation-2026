"""Nested out-of-fold evaluation of the final pipeline (CPU only). Shared helpers of the validation scripts.

  all_jobs()          the 707 training files with their annotation sets, by outer fold (fold k = b0n_ens/fold<k>)
  _eval(...)          V0 records (official per-set matching, scripts/refine_b0exist_checks.REC_KEYS + notebook overlap
                      counts) and V1 counts of one prediction set on one image
  final_instances()   the final pipeline's nested out-of-fold prediction of one image: stage 1 B_ens -> existence gate
                      -> R2n_k refinement -> assembly -> cleanup (tune_nested.FileCtx) -> SE2 with the outer fold's
                      scorer (outputs/research/se2/preds_outer.csv), added iff score > m_k x tau_k and no overlap
                      (m_k: SE2_MULT, chosen without fold k)
  gt_bins, pair_stats, pair_decomp, v1_sq, ratio   per-record pair statistics by annotated width, used by the cleanup
                      reports (scripts/fragment_cleanup.py, scripts/island_cleanup.py)
Commands (outputs/research/final/):
  eval  -> records.csv (one row per annotation-set record), images_v1.csv (one row per image)
  loao  -> loao/ (leave-one-annotator-out aspects of the final pipeline vs the human annotators, scripts/eval_dual.py)
  CUDA_VISIBLE_DEVICES="" taskset -c 0-3 .venv/bin/python scripts/a_eval.py eval --workers 4
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import nested_r2b0_eval as NE  # noqa: E402
import refine_b0exist_checks as RC  # noqa: E402
from filseg import scorers as SC  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.metrics import evaluate_image  # noqa: E402

OUT = ROOT / "outputs" / "research" / "final"
# SE2 threshold of outer fold k = m_k x tau_k; m_k maximises the pooled-set PQ (V1) on the other 4 folds
# (scripts/se2_analysis.py select, records/se2/analysis/threshold_v1.json). Test time: 0.6 x 0.2124.
SE2_MULT = {0: 0.6, 1: 0.8, 2: 0.6, 3: 0.6, 4: 0.6}
METHOD = "final"
WB_EDGES = (0, 6, 9, 13, np.inf)             # annotated width = area / skeleton px
WB_LABELS = ("w<6", "6-9", "9-13", ">=13")


# ============================================================================ jobs and per-image evaluation
def all_jobs():
    jobs = []
    for k in range(5):
        files = (NE.EXPS / "b0n_ens" / f"fold{k}" / "val_files.txt").read_text().split()
        gts = seg.gt_records(files)
        by = {}
        for rid, g in gts.items():
            by.setdefault(rid.split("-", 1)[1], {})[rid] = g
        jobs += [(s, k, by[s]) for s in sorted(by)]
    return jobs


def _eval(name, P, gts, stem, fold, inst=False):
    rec, extra = [], {}
    for rid, G in gts.items():
        r, pg, pp = evaluate_image(G, P, RC.CFG, image_key=rid, return_instances=inst)
        rec.append(dict(method=name, rid=rid, stem=stem, fold=fold, **{k: r.get(k, 0) for k in RC.REC_KEYS},
                        **SC.notebook_diag(G, P)))
        extra[rid] = (pg, pp)
    tp, fp, fn, si = SC.v1_counts([gts[r] for r in sorted(gts)], P)
    img = dict(method=name, stem=stem, fold=fold, n_sets=len(gts), v1_tp=tp, v1_fp=fp, v1_fn=fn, v1_si=si)
    return rec, img, extra


# ============================================================================ the final pipeline, nested OOF
_SCORES = {}


def final_instances(stem, fold):
    import se2 as S2
    import small_exist as SE
    import tune_nested as TN
    if not _SCORES:
        q = pd.read_csv(ROOT / "outputs" / "research" / "se2" / "preds_outer.csv")
        _SCORES.update({(s, int(i)): float(v) for s, i, v in zip(q.stem, q.idx, q.score)})
    S2._lazy_cand(fold)
    ctx = TN.FileCtx(stem, fold)
    base = ctx.instances(dict(TN.DEFAULT, se_mult=1e9))          # stages up to the cleanup, without small fragments
    R, _ = S2.fragments(ctx.p, ctx.cmap, ctx.cp)
    if R is None:
        return base
    sc = np.array([_SCORES[(stem, q)] for q in range(len(R))])
    return SE.add_only(base, R, np.flatnonzero(sc > SE2_MULT[fold] * ctx.tau))[0]


# ============================================================================ per-record pair statistics
def ratio(n, d):
    return float(n / d) if d else float("nan")


def skel_len(m):
    from skimage.morphology import skeletonize
    if not m.any():
        return 0
    return int(skeletonize(m).sum())


def gt_bins(G):
    w = np.array([G.areas[k] / max(skel_len(np.pad(G.crops[k], 1)), 1) for k in range(len(G))], float)
    return w, np.digitize(w, WB_EDGES[1:-1]) if len(w) else np.zeros(0, int)


def pair_stats(G, P, Pb, gb):
    """method P vs its base Pb on one record. Touching pairs: persistent, lost, gained; TP crossings; width bins."""
    def mats(X):
        inter, iou = SC.iou_matrix(G, X)
        dice = 2 * inter / np.maximum(G.areas[:, None] + X.areas[None, :], 1) if inter.size else inter
        return inter, iou, dice
    im, um, dm = mats(P)
    ib, ub, db = mats(Pb)
    tm, tb = im > 0, ib > 0
    pers, lost = tm & tb, tb & ~tm
    r = dict(pers_n=int(pers.sum()), pers_iou_m=float(um[pers].sum()), pers_iou_b=float(ub[pers].sum()),
             pers_dice_m=float(dm[pers].sum()), pers_dice_b=float(db[pers].sum()),
             lost_n=int(lost.sum()), lost_iou_b=float(ub[lost].sum()), lost_dice_b=float(db[lost].sum()),
             gained_n=int((tm & ~tb).sum()),
             cross_up=int(((um > .5) & (ub <= .5)).sum()), cross_down=int(((ub > .5) & (um <= .5)).sum()))
    G2 = np.repeat(gb[:, None], um.shape[1], 1) if um.size else np.zeros(um.shape, int)
    for b in range(len(WB_LABELS)):
        sel = G2 == b
        r[f"wb{b}_ngt"] = int((gb == b).sum())
        tp = sel & (um > .5)
        r[f"wb{b}_tp_n"], r[f"wb{b}_tp_iou"] = int(tp.sum()), float(um[tp].sum())
        pr = sel & tm
        r[f"wb{b}_pair_n"], r[f"wb{b}_pair_iou"], r[f"wb{b}_pair_dice"] = int(pr.sum()), float(um[pr].sum()), float(dm[pr].sum())
        bt = sel & (ub > .5)                      # the base's TP pairs, IoU under base / under this method
        r[f"wb{b}_btp_n"], r[f"wb{b}_btp_iou_b"], r[f"wb{b}_btp_iou_m"] = int(bt.sum()), float(ub[bt].sum()), float(um[bt].sum())
    return r


def pair_decomp(d):
    s = d.sum(numeric_only=True)
    return dict(persistent_n=int(s.pers_n), persistent_iou_base=ratio(s.pers_iou_b, s.pers_n),
                persistent_iou_method=ratio(s.pers_iou_m, s.pers_n), persistent_dice_base=ratio(s.pers_dice_b, s.pers_n),
                persistent_dice_method=ratio(s.pers_dice_m, s.pers_n), lost_n=int(s.lost_n),
                lost_mean_iou_base=ratio(s.lost_iou_b, s.lost_n), lost_mean_dice_base=ratio(s.lost_dice_b, s.lost_n),
                gained_n=int(s.gained_n), tp_cross_up=int(s.cross_up), tp_cross_down=int(s.cross_down))


def v1_sq(I, m):
    x = I[I.method == m]
    return ratio(x.v1_si.sum(), x.v1_tp.sum())


# ============================================================================ commands
def _eval_job(args):
    import small_exist as SE
    stem, fold, gts = args
    Q = final_instances(stem, fold)
    rec, img, _ = _eval(METHOD, Q, gts, stem, fold)
    img["no_set_fp"] = SE.no_set_fp(Q, [gts[r] for r in sorted(gts)]); img["n_pred"] = len(Q)
    return rec, img


def cmd_eval(a):
    import multiprocessing as mp

    from filseg.metrics import single_thread_worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    single_thread_worker()
    R, I = [], []
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for rec, img in pool.imap_unordered(_eval_job, all_jobs()[:a.limit or None], chunksize=1):
            R += rec; I.append(img)
    df = pd.DataFrame(R); sp = RC.split_table().set_index("stem")
    df["group_id"] = df.stem.map(sp.group_id); df["n_sets"] = df.stem.map(sp.n_sets)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "records.csv", index=False); pd.DataFrame(I).to_csv(OUT / "images_v1.csv", index=False)
    s = df.sum(numeric_only=True); v = pd.DataFrame(I).sum(numeric_only=True)
    print(f"{df.stem.nunique()} files, {len(df)} records: V0 {s.sum_iou_host / (s.tp_host + .5 * s.fp_host + .5 * s.fn_host):.4f}, "
          f"V1 {v.v1_si / (v.v1_tp + .5 * v.v1_fp + .5 * v.v1_fn):.4f}")


def _loao_instances(stem, fold):
    return {METHOD: final_instances(stem, fold)}


def _loao_job(args):
    import eval_dual as ED
    ED.instances = _loao_instances
    ED.LOAO_RATERS = (METHOD,)
    return ED.loao_job(args)


def cmd_loao(a):
    import eval_dual as ED
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ED.OUT = OUT / "loao"; ED.LOAO_RATERS = (METHOD,); ED.loao_job = _loao_job
    ED.cmd_loao(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "loao"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"eval": cmd_eval, "loao": cmd_loao}[a.cmd](a)
