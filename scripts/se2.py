"""SE2 (registry row SE2_refined_existence; rule registered before any SE2 scorer was trained): refine the small
fragments that the 500 px gate discards with R2 at native resolution, then judge their existence on the REFINED mask.

Refined fragments of an image (stage-1 map p, candidate map cmap, refined candidate probabilities cp):
  gate      8-connected components of p > 0.5 with >= 500 px; a candidate touching one is already used by R2b0
  small     components of p > 0.5 with 100-499 px (what SE scores)
  fragments candidates (p > 0.3, >= 20 px) that overlap a small component and do not touch the gate, refined by R2
            and assembled exactly as R2b0 (refine.assemble: p > 0.5, merge IoU 0.5) with >= 100 px
Scorer input: 96x96 crop at the refined instance's box centre, channels (H-alpha, rint(p*255), refined mask, gate
mask); SE architecture and recipe (small_exist.fit_predict). Target (deterministic, all annotation sets): y = mean over
the image's sets of (IoU of the refined mask with the set's best instance if > 0.5 else 0). Decision as SE: add iff
score > tau_k (= PQ_V0 of the base on the outer-train folds / 2) and zero overlap with the base instances.
Sources (strict nesting; nothing used for outer fold k ever saw fold k):
  eval_f<k>      fold k: B_ens stage 1 + R2n_context/fold<k> refinements (tune_nested.FileCtx)
  inner_k<k>_f<j> fold j != k: stage 1 of b0n hold{k,j} + R2nn_context/fold<k>_in<j> refinements
Commands (outputs/research/se2/):  data --source eval|inner ; train ; eval ; loao ; analyse
"""
import os

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

OUT = ROOT / "outputs" / "research" / "se2"
EXPS = ROOT / "outputs" / "exps"
GATE, A_LO, A_HI, MIN_FRAG = 500, 100, 500, 100
METHODS = ("current", "se2")


def cand_map(p):
    """Candidate label map exactly as refine._build_one / island_cleanup._LazyCand (u8 = rint(p*255))."""
    from filseg import refine as RF
    u8 = np.clip(np.rint(p.astype(np.float32) * 255), 0, 255).astype(np.uint8)
    q = u8.astype(np.float32) / 255.0
    n, lab, stats = cv2.connectedComponentsWithStats((q > RF.CAND_THR).astype(np.uint8), connectivity=8)[:3]
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= RF.CAND_MIN]
    lut = np.zeros(n, np.uint16)
    for j, k in enumerate(keep, 1):
        lut[k] = j
    return lut[lab]


def fragments(p, cmap, cp):
    """-> (refined fragment InstanceSet or None, gate mask)."""
    from filseg import refine as RF
    n, lab, st, _ = cv2.connectedComponentsWithStats((p > 0.5).astype(np.uint8), connectivity=8)
    area = st[:, cv2.CC_STAT_AREA]
    gate_lut = np.zeros(n, bool); gate_lut[1:] = area[1:] >= GATE
    small_lut = np.zeros(n, bool); small_lut[1:] = (area[1:] >= A_LO) & (area[1:] < A_HI)
    gate = gate_lut[lab]
    touch_gate = set(np.unique(cmap[gate]).tolist()) - {0}
    js = set(np.unique(cmap[small_lut[lab]]).tolist()) - {0} - touch_gate
    js = {j for j in js if j in cp}
    if not js:
        return None, gate
    R = RF.assemble({j: cp[j] for j in sorted(js)}, thr=0.5, min_area=MIN_FRAG, merge_iou=0.5)
    return (R if len(R) else None), gate


def crops(R, img, p, gate):
    import small_exist as SE
    pu8 = np.clip(np.rint(p * 255), 0, 255).astype(np.uint8)
    g8 = gate.astype(np.uint8) * 255
    X = []
    for q in range(len(R)):
        y0, x0, y1, x1 = (int(v) for v in R.boxes[q])
        m = np.zeros(p.shape, np.uint8); m[y0:y1, x0:x1] = R.crops[q].astype(np.uint8) * 255
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        X.append(np.stack([SE.crop_at(img, cy, cx), SE.crop_at(pu8, cy, cx), SE.crop_at(m, cy, cx), SE.crop_at(g8, cy, cx)]))
    return np.stack(X) if X else np.zeros((0, 4, SE.CROP, SE.CROP), np.uint8)


def _rows(R, sets, stem, fold, source):
    import small_exist as SE
    y, tp_any, best = SE.targets(R, sets)
    return [dict(source=source, stem=stem, fold=fold, idx=q, area=int(R.areas[q]), y=float(y[q]), tp_any=bool(tp_any[q]),
                 best_iou=float(best[q]), n_sets=len(sets)) for q in range(len(R))]


def _lazy_cand(fold):
    """island_cleanup.ensure_stage1 skips installing its in-memory candidate map while a rebuilt nested cache exists
    on disk (run_r2nn.sh rebuilds them one at a time); install the identical in-memory map unconditionally."""
    import eval_dual as ED
    import island_cleanup as IC
    import nested_r2b0_eval as NE
    if ("cand", fold) in NE._W:
        return
    files = (NE.EXPS / ED.CFG["ens_exp"] / f"fold{fold}" / "val_files.txt").read_text().split()
    NE._W[("idx", fold)] = dict(files=files, meta=IC._NoMeta())
    NE._W[("fi", fold)] = {f: i for i, f in enumerate(files)}
    NE._W[("cand", fold)] = IC._LazyCand(fold, files)


def eval_job(args):
    import tune_nested as TN
    from filseg import data as D
    stem, fold, gts = args
    _lazy_cand(fold)
    ctx = TN.FileCtx(stem, fold)
    R, gate = fragments(ctx.p, ctx.cmap, ctx.cp)
    if R is None:
        return f"eval_f{fold}", np.zeros((0, 4, 96, 96), np.uint8), []
    img = D.read_image(D.train_image_path(stem + ".jpeg"))
    return f"eval_f{fold}", crops(R, img, ctx.p, gate), _rows(R, [gts[r] for r in sorted(gts)], stem, fold, f"eval_f{fold}")


_Z = {}


def inner_job(args):
    import refine_b0exist_checks as RC
    from filseg import data as D
    k, j, stem, gts = args
    a, b = sorted((k, j))
    key = f"hold{a}{b}"
    if key not in _Z:
        _Z[key] = np.load(EXPS / "b0n_r101_s1024" / key / "val_probs_last.npz")
    p = _Z[key][stem].astype(np.float32)
    z, by = RC._npz(EXPS / "R2nn_context" / f"fold{k}_in{j}" / "val_cand_probs.npz")
    cp = {jj: (y0, x0, z[kk].astype(np.float32)) for jj, y0, x0, kk in by.get(stem, [])}
    R, gate = fragments(p, cand_map(p), cp)
    src = f"inner_k{k}_f{j}"
    if R is None:
        return src, np.zeros((0, 4, 96, 96), np.uint8), []
    img = D.read_image(D.train_image_path(stem + ".jpeg"))
    return src, crops(R, img, p, gate), _rows(R, [gts[r] for r in sorted(gts)], stem, j, src)


def _save(acc, d):
    d.mkdir(parents=True, exist_ok=True)
    for name, (c, r) in acc.items():
        np.savez(d / f"{name}.npz", X=np.concatenate(c) if c else np.zeros((0, 4, 96, 96), np.uint8))
        pd.DataFrame(r).to_csv(d / f"{name}.csv", index=False)


def cmd_data(a):
    import multiprocessing as mp
    import a_eval as AE
    from filseg import seg
    from filseg.metrics import single_thread_worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    single_thread_worker()
    t0, acc = time.time(), {}
    if a.source == "eval":
        jobs, fn = AE.all_jobs()[:a.limit or None], eval_job
    else:
        sp = pd.read_csv(ROOT / "outputs" / "splits" / "folds_v1.csv")
        gts_all = seg.gt_records()
        by = {}
        for rid, g in gts_all.items():
            by.setdefault(rid.split("-", 1)[1], {})[rid] = g
        jobs = []
        for k in range(5):
            for j in range(5):
                if j == k or (a.pairs and f"{k}{j}" not in a.pairs.split(",")):
                    continue
                assert (EXPS / "R2nn_context" / f"fold{k}_in{j}" / "val_cand_probs.npz").exists(), (k, j)
                files = sorted(sp[sp.fold == j].file_name)[:a.limit or None]
                jobs += [(k, j, f[:-5], by[f[:-5]]) for f in files]
        fn = inner_job
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for i, (name, X, rows) in enumerate(pool.imap(fn, jobs, chunksize=4)):
            c, r = acc.setdefault(name, ([], []))
            c.append(X); r += rows
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    _save(acc, OUT / ("data_test" if a.limit else "data"))
    print({k: len(v[1]) for k, v in acc.items()}, f"{time.time() - t0:.0f}s")


def load(name):
    d = OUT / "data"
    return np.load(d / f"{name}.npz")["X"], pd.read_csv(d / f"{name}.csv")


def cmd_train(a):
    import torch
    import small_exist as SE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    rows, log = [], {}
    for k in range(5):
        parts = [load(f"inner_k{k}_f{j}") for j in range(5) if j != k]
        Xtr = np.concatenate([p[0] for p in parts]); ytr = np.concatenate([p[1].y.values for p in parts])
        Xev, mev = load(f"eval_f{k}")
        assert not set(pd.concat([p[1] for p in parts]).stem) & set(mev.stem), "outer fold leaked"
        t0 = time.time()
        _, pr = SE.fit_predict(Xtr, ytr, Xev, device)
        rows.append(mev[["stem", "idx"]].assign(outer=k, score=pr))
        log[f"outer{k}"] = dict(n_train=int(len(Xtr)), pos_train=int((ytr > 0).sum()), n_eval=int(len(Xev)), sec=round(time.time() - t0, 1))
        print(f"outer {k}: train {len(Xtr)} (pos {(ytr > 0).sum()}), eval {len(Xev)}, {time.time() - t0:.0f}s", flush=True)
    pd.concat(rows).to_csv(OUT / "preds_outer.csv", index=False)
    # final (test) scorer: every fold's eval source (stage 1 = mean of 4 inner models, closer to the test-time mean of 5
    # than a single fold model; refinement by R2n_k, which never saw fold k)
    parts = [load(f"eval_f{k}") for k in range(5)]
    Xall = np.concatenate([p[0] for p in parts]); yall = np.concatenate([p[1].y.values for p in parts])
    t0 = time.time()
    net, _ = SE.fit_predict(Xall, yall, Xall[:1], device)
    torch.save(net.state_dict(), OUT / "scorer_final.pt")
    log["final"] = dict(n_train=int(len(Xall)), pos_train=int((yall > 0).sum()), sec=round(time.time() - t0, 1))
    json.dump(dict(device=device, logs=log), open(OUT / "train_log.json", "w"), indent=1)


_P = {}


def _instances(stem, fold, methods=METHODS):
    import small_exist as SE
    import tune_nested as TN
    if not _P:
        P = pd.read_csv(OUT / "preds_outer.csv")
        _P.update({(s, int(i)): float(v) for s, i, v in zip(P.stem, P.idx, P.score)})
    _lazy_cand(fold)
    ctx = TN.FileCtx(stem, fold)
    out = {"current": ctx.instances(dict(TN.DEFAULT))}
    if "se2" in methods:
        base = ctx.instances(dict(TN.DEFAULT, se_mult=1e9))
        R, _ = fragments(ctx.p, ctx.cmap, ctx.cp)
        if R is None:
            out["se2"] = base
        else:
            sc = np.array([_P[(stem, q)] for q in range(len(R))])
            out["se2"] = SE.add_only(base, R, np.flatnonzero(sc > ctx.tau))[0]
    return {m: out[m] for m in methods}


def eval_rec_job(args):
    import a_eval as AE
    import small_exist as SE
    stem, fold, gts = args
    sets = [gts[r] for r in sorted(gts)]
    recs, imgs = [], []
    for m, Q in _instances(stem, fold).items():
        rec, img, _ = AE._eval(m, Q, gts, stem, fold)
        img["no_set_fp"] = SE.no_set_fp(Q, sets)
        recs += rec; imgs.append(img)
    return recs, imgs


def cmd_eval(a):
    import multiprocessing as mp
    import a_eval as AE
    import refine_b0exist_checks as RC
    from filseg.metrics import single_thread_worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    single_thread_worker()
    jobs = AE.all_jobs()[:a.limit or None]
    R, I, t0 = [], [], time.time()
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for rec, img in pool.imap_unordered(eval_rec_job, jobs, chunksize=1):
            R += rec; I += img
    df = pd.DataFrame(R)
    sp = RC.split_table().set_index("stem")
    df["group_id"] = df.stem.map(sp.group_id); df["n_sets"] = df.stem.map(sp.n_sets)
    df.to_csv(OUT / "records.csv", index=False); pd.DataFrame(I).to_csv(OUT / "images_v1.csv", index=False)
    print(f"eval done {time.time() - t0:.0f}s")


def _loao_instances(stem, fold):
    return _instances(stem, fold, ("current", "se2"))


def loao_wrap(args):
    import eval_dual as ED
    ED.instances = _loao_instances
    ED.LOAO_RATERS = ("current", "se2")
    return ED.loao_job(args)


def cmd_loao(a):
    import eval_dual as ED
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ED.OUT = OUT / "loao"; ED.LOAO_RATERS = ("current", "se2"); ED.loao_job = loao_wrap
    ED.cmd_loao(a)


def cmd_analyse(a):
    import eval_dual as ED
    import refine_b0exist_checks as RC
    import tune_nested as TN
    df = pd.read_csv(OUT / "records.csv"); I = pd.read_csv(OUT / "images_v1.csv")
    sp = RC.split_table(); RC.FOLDS = (0, 1, 2, 3, 4)
    rc = pd.read_csv(ROOT / "outputs/audit/review_cases.csv"); review = set(rc.file_name.str[:-5])
    allm = {m: dict(V0=TN.v0(df[df.method == m]), V1=TN.v1(I[I.method == m]), no_set_fp=int(I[I.method == m].no_set_fp.sum()),
                    fn_missed=int(df[df.method == m].fn_missed.sum()), n_to_1=int(df[df.method == m].nb_pred_Nto1.sum()),
                    one_to_n=int(df[df.method == m].nb_gt_1toN.sum())) for m in METHODS}
    v0c, _ = RC.compare(df, "current", "se2", sp, review)
    v1c = ED.v1_boot(I, "current", "se2")
    rv = {m: TN.v0(df[(df.method == m) & df.stem.isin(review)]) for m in METHODS}
    newly, small = None, None
    if (OUT / "loao" / "loao_records.csv").exists():
        ED.OUT = OUT / "loao"; ED.LOAO_RATERS = ("current", "se2")
        asp, BH = ED.loao_aspects()
        f = lambda z: np.nan if z is None else float(z)  # noqa: E731
        newly = [s for s, d in BH.ASPECTS.items() if BH.not_worse(f(asp["current"].get(s)), f(asp["human"].get(s)), d) is True
                 and BH.not_worse(f(asp["se2"].get(s)), f(asp["human"].get(s)), d) is False]
        small = {r: asp[r].get("small_detected_rate") for r in ("human", "current", "se2")}
    d0 = v0c["file_boot"]["PQ"]; pf = {k: v["diff"] for k, v in v0c["per_fold"].items()}
    c, s2 = allm["current"], allm["se2"]
    res = dict(methods=allm, V0=dict(diff=d0["diff"], ci95=d0["ci95"], per_fold=pf), V1={k: v1c[k] for k in ("A", "B", "diff", "ci95", "per_fold")},
               review_V0=rv, loao_newly_below_human=newly, loao_small_detection=small)
    res["rule"] = dict(E1=bool(d0["ci95"][0] > -0.002 and sum(v > 0 for v in pf.values()) >= 3),
                       E2=bool(v1c["diff"] >= 0),
                       E3=bool((s2["no_set_fp"] - c["no_set_fp"]) <= 0.10 * c["no_set_fp"]),
                       E4=None if newly is None else bool(not newly and small["se2"] > small["current"]),
                       E5=bool(rv["se2"] >= rv["current"]),
                       E6=bool((s2["n_to_1"] - c["n_to_1"]) <= 0.10 * c["n_to_1"]))
    res["rule"]["ADOPT"] = all(v is True for v in res["rule"].values())
    json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "train", "eval", "loao", "analyse"])
    ap.add_argument("--source", choices=["eval", "inner"], default="eval")
    ap.add_argument("--pairs", default="", help="inner: only these k j pairs, e.g. '01,02'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"data": cmd_data, "train": cmd_train, "eval": cmd_eval, "loao": cmd_loao, "analyse": cmd_analyse}[a.cmd](a)
