"""AT_nested: sensitivity and nested automatic tuning of the post-processing constants (D-074; rule
registered before any configuration was scored). CPU only.

Measured on the nested OOF (707 files) with the earlier small-fragment stage (T2_SE of scripts/small_exist.py): stage-1 B_ens -> existence gate
(components of p > 0.5 with >= g_min px) -> R2n refined candidates that overlap the gate -> refine.assemble(a_thr,
a_min, merge) -> cleanup (parts < c_part px, then < i_frac of the instance; largest part kept) -> SE add-on (score >
se_mult * tau_k). Current values: g_min 500, a_thr 0.5, merge 0.5, a_min 300, c_part 20, i_frac 0.05, se_mult 1.0.
Commands (outputs/research/tune_nested/):
  sweep     every one-at-a-time configuration on all 707 files            -> records.csv, images_v1.csv
  select    nested per-constant choice (training folds) + all-fold choice -> selection.json, sensitivity.csv
  combined  the fold-wise combinations on their held-out fold ("auto") and the all-fold combination ("final")
  loao      LOAO for default vs final;  analyse  the registered rule        -> summary.json
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from filseg.metrics import InstanceSet, single_thread_worker  # noqa: E402

OUT = ROOT / "outputs" / "research" / "tune_nested"
DEFAULT = dict(g_min=500, a_thr=0.5, merge=0.5, a_min=300, c_part=20, i_frac=0.05, se_mult=1.0)
GRID = dict(g_min=[300, 400, 500, 600, 700], a_thr=[0.4, 0.45, 0.5, 0.55, 0.6], merge=[0.3, 0.4, 0.5, 0.6, 0.7],
            a_min=[200, 250, 300, 400, 500], c_part=[0, 10, 20, 50], i_frac=[0, 0.02, 0.05, 0.1],
            se_mult=[0.8, 0.9, 1.0, 1.1, 1.25])


def cid(cfg):
    diff = [f"{k}={cfg[k]}" for k in DEFAULT if cfg[k] != DEFAULT[k]]
    return ";".join(diff) if diff else "default"


def sweep_configs():
    out = [dict(DEFAULT)]
    for k, vals in GRID.items():
        out += [dict(DEFAULT, **{k: v}) for v in vals if v != DEFAULT[k]]
    return out


def _drop_parts(mask, thr_fn):
    n, lab, st = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)[:3]
    if n <= 2:
        return mask
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= thr_fn(areas)
    keep[1 + int(np.argmax(areas))] = True
    return keep[lab]


def clean(P, c_part, i_frac):
    crops = []
    for c in P.crops:
        if c_part:
            c = _drop_parts(c, lambda a: c_part)
        if i_frac:
            c = _drop_parts(c, lambda a: i_frac * a.sum())
        crops.append(c)
    return InstanceSet._build(P.shape, [tuple(b) for b in P.boxes], crops)


class FileCtx:
    """Everything a configuration needs for one outer-validation file (loaded once)."""

    def __init__(self, stem, fold, r2_exp=None):
        import eval_dual as ED
        import island_cleanup as IC
        import nested_r2b0_eval as NE
        import refine_b0exist_checks as RC
        import small_exist as SE
        IC.ensure_stage1(fold)
        ens = NE._load(("ens", fold), lambda: np.load(NE.EXPS / ED.CFG["ens_exp"] / f"fold{fold}" / "val_probs_last.npz"))
        self.p = ens[stem].astype(np.float32)
        n, self.lab, st, _ = cv2.connectedComponentsWithStats((self.p > 0.5).astype(np.uint8), connectivity=8)
        self.area = st[:, cv2.CC_STAT_AREA]
        self.cmap = np.asarray(NE._W[("cand", fold)][NE._W[("fi", fold)][stem + ".jpeg"]])
        z, by = RC._npz(NE.EXPS / (r2_exp or ED.CFG["r2_exp"]) / f"fold{fold}" / "val_cand_probs.npz")
        self.cp = {j: (y0, x0, z[k].astype(np.float32)) for j, y0, x0, k in by.get(stem, [])}
        l2, st2, cand, _ = SE.components(self.p)
        self.Csm = SE.cand_set(l2, st2, cand)
        spr, stau = SE._preds()
        self.ssc = np.array([spr[(stem, i)] for i in range(len(cand))])
        self.tau = stau["T2"][str(fold)]
        self._asm = {}

    def instances(self, cfg):
        import small_exist as SE
        from filseg import refine as RF
        key = (cfg["g_min"], cfg["a_thr"], cfg["merge"], cfg["a_min"])
        if key not in self._asm:
            lut = np.zeros(len(self.area), bool)
            lut[1:] = self.area[1:] >= cfg["g_min"]
            inb = np.bincount(self.cmap[lut[self.lab]], minlength=int(self.cmap.max()) + 1)
            keep = {j for j in range(1, len(inb)) if inb[j] > 0}
            self._asm[key] = RF.assemble({j: v for j, v in self.cp.items() if j in keep}, thr=cfg["a_thr"],
                                         min_area=cfg["a_min"], merge_iou=cfg["merge"])
        P = clean(self._asm[key], cfg["c_part"], cfg["i_frac"])
        Q, _ = SE.add_only(P, self.Csm, np.flatnonzero(self.ssc > cfg["se_mult"] * self.tau))
        return Q


def eval_job(args):
    import a_eval as AE
    import small_exist as SE
    stem, fold, gts, cfgs = args
    ctx = FileCtx(stem, fold)
    sets = [gts[r] for r in sorted(gts)]
    recs, imgs = [], []
    for name, cfg in cfgs:
        Q = ctx.instances(cfg)
        rec, img, _ = AE._eval(name, Q, gts, stem, fold)
        img["no_set_fp"] = SE.no_set_fp(Q, sets)
        recs += rec; imgs.append(img)
    return recs, imgs


def run_jobs(jobs, workers, tag):
    import multiprocessing as mp
    import refine_b0exist_checks as RC
    single_thread_worker()
    t0, R, I = time.time(), [], []
    with mp.get_context("forkserver").Pool(workers, initializer=single_thread_worker) as pool:
        for i, (rec, img) in enumerate(pool.imap_unordered(eval_job, jobs, chunksize=1)):
            R += rec; I += img
            if (i + 1) % 100 == 0:
                print(f"  {tag} {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(R)
    sp = RC.split_table().set_index("stem")
    df["group_id"] = df.stem.map(sp.group_id); df["n_sets"] = df.stem.map(sp.n_sets)
    return df, pd.DataFrame(I)


def cmd_sweep(a):
    import a_eval as AE
    cfgs = [(cid(c), c) for c in sweep_configs()]
    jobs = [(s, f, g, cfgs) for s, f, g in AE.all_jobs()][:a.limit or None]
    df, I = run_jobs(jobs, a.workers, "sweep")
    OUT.mkdir(parents=True, exist_ok=True)
    sfx = "_test" if a.limit else ""
    df.to_csv(OUT / f"records{sfx}.csv", index=False); I.to_csv(OUT / f"images_v1{sfx}.csv", index=False)
    json.dump({n: c for n, c in cfgs}, open(OUT / "configs.json", "w"), indent=1)


def v0(df):
    s = df.sum(numeric_only=True)
    return float(s.sum_iou_host / (s.tp_host + .5 * s.fp_host + .5 * s.fn_host))


def v1(I):
    s = I.sum(numeric_only=True)
    return float(s.v1_si / (s.v1_tp + .5 * s.v1_fp + .5 * s.v1_fn))


def choose(df, folds):
    """per constant: the value with the highest pooled V0 on `folds` (others at current); ties -> current, closer."""
    d = df[df.fold.isin(folds)]
    pick = {}
    for k, vals in GRID.items():
        sc = {}
        for v in vals:
            name = cid(dict(DEFAULT, **{k: v}))
            sc[v] = v0(d[d.method == name])
        best = max(sc.values())
        cands = [v for v in vals if sc[v] >= best - 1e-12]
        pick[k] = DEFAULT[k] if DEFAULT[k] in cands else min(cands, key=lambda v: abs(v - DEFAULT[k]))
    return pick


def cmd_select(a):
    import eval_dual as ED
    df = pd.read_csv(OUT / "records.csv"); I = pd.read_csv(OUT / "images_v1.csv")
    se = pd.read_csv(ROOT / "outputs/research/small_exist/records.csv")
    base = se[se.method == "T2_SE"].sort_values("rid").reset_index(drop=True)
    mine = df[df.method == "default"].sort_values("rid").reset_index(drop=True)
    cols = ["tp_host", "fp_host", "fn_host", "sum_iou_host", "n_pred", "fn_split", "fn_merged"]
    ok = len(base) == len(mine) and (base.rid.values == mine.rid.values).all() and \
        np.allclose(base[cols].values.astype(float), mine[cols].values.astype(float))
    print("base reproduction (default == T2_SE records):", ok)
    assert ok, "STOP: the default configuration does not reproduce the reference records"
    rows = []
    for name in df.method.unique():
        d, i = df[df.method == name], I[I.method == name]
        dg = ED.diag_sums(d)
        rows.append(dict(config=name, V0=v0(d), V1=v1(i), **{f"V0_f{k}": v0(d[d.fold == k]) for k in range(5)},
                         **{k: dg[k] for k in ("fn_missed", "fn_split", "fn_merged", "fp_spurious", "fp_fragment",
                                               "nb_gt_1toN", "nb_pred_Nto1")},
                         pair_iou=dg["pair_iou_mean"], no_set_fp=int(i.no_set_fp.sum())))
    sens = pd.DataFrame(rows)
    d0 = sens[sens.config == "default"].iloc[0]
    sens["dV0"] = sens.V0 - d0.V0; sens["dV1"] = sens.V1 - d0.V1
    sens.to_csv(OUT / "sensitivity.csv", index=False)
    sel = {"per_outer": {k: choose(df, [f for f in range(5) if f != k]) for k in range(5)},
           "final": choose(df, list(range(5)))}
    plateau = {}
    for k, vals in GRID.items():
        sc = {v: float(sens[sens.config == cid(dict(DEFAULT, **{k: v}))].V0.iloc[0]) for v in vals}
        plateau[k] = dict(values=sc, best=max(sc, key=sc.get), current_within_0p002=bool(max(sc.values()) - sc[DEFAULT[k]] <= 0.002))
    sel["plateau"] = plateau
    json.dump(sel, open(OUT / "selection.json", "w"), indent=1)
    pd.set_option("display.width", 220)
    print(sens.sort_values("config")[["config", "V0", "dV0", "V1", "dV1", "fn_missed", "fn_merged", "fn_split", "fp_spurious", "no_set_fp"]].round(4).to_string(index=False))
    print(json.dumps(sel, indent=1))


def cmd_combined(a):
    import a_eval as AE
    sel = json.load(open(OUT / "selection.json"))
    jobs = []
    for s, f, g in AE.all_jobs():
        cfgs = [("auto", dict(DEFAULT, **sel["per_outer"][str(f)])), ("final", dict(DEFAULT, **sel["final"])),
                ("default", dict(DEFAULT))]
        jobs.append((s, f, g, cfgs))
    df, I = run_jobs(jobs, a.workers, "combined")
    df.to_csv(OUT / "records_combined.csv", index=False); I.to_csv(OUT / "images_v1_combined.csv", index=False)


def _loao_instances(stem, fold):
    sel = json.load(open(OUT / "selection.json"))
    ctx = FileCtx(stem, fold)
    return {"default": ctx.instances(dict(DEFAULT)), "final": ctx.instances(dict(DEFAULT, **sel["final"]))}


def loao_wrap(args):
    import eval_dual as ED
    ED.instances = _loao_instances
    ED.LOAO_RATERS = ("default", "final")
    return ED.loao_job(args)


def cmd_loao(a):
    import eval_dual as ED
    ED.OUT = OUT / "loao"; ED.LOAO_RATERS = ("default", "final"); ED.loao_job = loao_wrap
    ED.cmd_loao(a)


def cmd_analyse(a):
    import eval_dual as ED
    import refine_b0exist_checks as RC
    df = pd.read_csv(OUT / "records_combined.csv"); I = pd.read_csv(OUT / "images_v1_combined.csv")
    sel = json.load(open(OUT / "selection.json"))
    sp = RC.split_table(); RC.FOLDS = (0, 1, 2, 3, 4)
    rc = pd.read_csv(ROOT / "outputs/audit/review_cases.csv"); review = set(rc.file_name.str[:-5])
    v0c, _ = RC.compare(df, "default", "auto", sp, review)
    v1c = ED.v1_boot(I, "default", "auto")
    rv = {m: v0(df[(df.method == m) & df.stem.isin(review)]) for m in ("default", "auto")}
    rv1 = {m: v1(I[(I.method == m) & I.stem.isin(review)]) for m in ("default", "auto")}
    newly = None
    if (OUT / "loao" / "loao_records.csv").exists():
        ED.OUT = OUT / "loao"; ED.LOAO_RATERS = ("default", "final")
        asp, BH = ED.loao_aspects()
        f = lambda z: np.nan if z is None else float(z)  # noqa: E731
        newly = [s for s, d in BH.ASPECTS.items() if BH.not_worse(f(asp["default"].get(s)), f(asp["human"].get(s)), d) is True
                 and BH.not_worse(f(asp["final"].get(s)), f(asp["human"].get(s)), d) is False]
    same = all(sel["per_outer"][str(k)] == DEFAULT for k in range(5))
    res = dict(selection=sel, auto_equals_current_everywhere=same,
               auto_vs_current=dict(V0=dict(A=v0c["A"]["PQ"], B=v0c["B"]["PQ"], diff=v0c["file_boot"]["PQ"]["diff"],
                                             ci95=v0c["file_boot"]["PQ"]["ci95"],
                                             per_fold={k: v["diff"] for k, v in v0c["per_fold"].items()}),
                                    V1={k: v1c[k] for k in ("A", "B", "diff", "ci95", "per_fold")},
                                    review=dict(V0=rv, V1=rv1)),
               final_vs_current_all_folds=dict(V0=(v0(df[df.method == "default"]), v0(df[df.method == "final"])),
                                               V1=(v1(I[I.method == "default"]), v1(I[I.method == "final"]))),
               loao_newly_below_human=newly)
    res["rule"] = dict(A1=bool(v0c["file_boot"]["PQ"]["diff"] > 0 and v0c["file_boot"]["PQ"]["ci95"][0] > 0),
                       A2=bool(v1c["diff"] >= 0), A3=None if newly is None else (not newly),
                       A4=bool(rv["auto"] >= rv["default"] and rv1["auto"] >= rv1["default"]))
    res["rule"]["SWITCH"] = all(v is True for v in res["rule"].values())
    json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "select", "combined", "loao", "analyse"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"sweep": cmd_sweep, "select": cmd_select, "combined": cmd_combined, "loao": cmd_loao, "analyse": cmd_analyse}[a.cmd](a)
