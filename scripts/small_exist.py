"""Experiment SE: existence scoring of small stage-1 candidates, add-only (GPT idea 4; docs/small_exist.md;
registry SE_nested; rule registered before any result).

Candidates (fixed): 8-connected components of a stage-1 probability map p > 0.5 with 100 <= area < 500 px (native
2048 grid). These are exactly the components the frozen min_area 500 filter removes (below 100 px no component was
TP-worthy in docs/minarea_analysis.md). Components >= 500 px form the "base mask" (context channel).
Input of the scorer: a 96 x 96 native crop centred on the candidate's box centre, 4 channels (uint8): H-alpha image,
stage-1 probability (rint(p * 255)), candidate mask, base mask. Target y = mean over the image's annotation sets of
(IoU with that set's best GT instance if > 0.5 else 0): the exact per-component V0 gain term, because an added disjoint
component changes the pooled V0 numerator by n_sets * y and the denominator by n_sets / 2 (docs/E3_component_rescore.md
§1). So adding it raises V0 iff y > PQ / 2. A candidate drawn by any one annotator gets credit; nothing that only some
sets drew is treated as certain background.
Scorer: ResNet-101 stem + layer1 + layer2 (ImageNet weights, local file), 4-channel stem (new channels initialised with
the mean RGB filter), global average pool, linear -> logit; BCE with the soft target y. AdamW 1e-4, wd 1e-4, cosine,
25 epochs, bs 64, flips + rot90; seed 2026; no early stopping, no model selection.
Decision (fixed, not tuned): add a candidate iff sigmoid(logit) > tau_k = PQ_V0(base on the outer-train folds) / 2 and
it has zero pixel overlap with the base pipeline's instances. Base instances are never changed.
Nested protocol (no fold sees its own labels through stage 1):
  outer fold k evaluation candidates  = from b0n_ens/fold<k> (mean of the 4 inner models that never saw fold k);
  outer fold k training candidates    = folds j != k, each from inner model hold{k,j} (never saw k or j);
  final (test) scorer                 = trained on the b0 5-fold OOF candidates of all folds.
Bases: T2 = the final pipeline before its small-fragment stage (R2nb0_isl5, island_cleanup) - primary; T1 = B_ens
t0.5/a500 - secondary reference. SE2 (scripts/se2.py) reuses the per-fold break-even tau_k of T2 and this scorer recipe.
Commands (outputs/research/small_exist/):
  cands [--limit N]   candidate crops + targets for every source  -> cands/<source>.npz, cands/<source>.csv
  train               5 nested scorers + the final scorer (GPU)    -> preds_outer.csv, scorer_final.pt, train_log.json
  cands/train --final_only   only the b0 5-fold OOF candidates and the final scorer (no nested inner models needed)
  table               add-only evaluation on the 707 outer files   -> records.csv, images_v1.csv, added.csv
  loao                eval_dual's LOAO path for the 4 methods       -> loao/*.csv
  analyse             comparisons + the registered rule             -> summary.json
  CUDA_VISIBLE_DEVICES="" TMPDIR=$PWD/tmp taskset -c 0-3 .venv/bin/python scripts/small_exist.py cands
"""
from __future__ import annotations

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
from filseg import data as D  # noqa: E402
from filseg import scorers as SC  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.metrics import InstanceSet, single_thread_worker  # noqa: E402

OUT = ROOT / "outputs" / "research" / "small_exist"
EXPS = ROOT / "outputs" / "exps"
THR, A_LO, A_HI, CROP = 0.5, 100, 500, 96
HW = (2048, 2048)
SEED = 2026
PAIRS = [(a, b) for a in range(5) for b in range(a + 1, 5)]


def sources(final_only=False):
    """(name, exp, run, fold of the files used) for every candidate source.

    final_only: only the b0 5-fold OOF sources b0_f<k>, the training data of the final (test) scorer; they do not need
    the nested inner models. Each source's file is written separately, so b0_f<k> is the same either way."""
    out = []
    if not final_only:
        for a, b in PAIRS:
            for f in (a, b):
                out.append((f"inner_h{a}{b}_f{f}", "b0n_r101_s1024", f"hold{a}{b}", f))
    for k in range(5):
        if not final_only:
            out.append((f"bens_f{k}", "b0n_ens", f"fold{k}", k))
        out.append((f"b0_f{k}", "b0_r101_s1024", f"fold{k}", k))
    return out


def fold_files(fold):
    sp = pd.read_csv(ROOT / "outputs" / "splits" / "folds_v1.csv")
    return sorted(sp[sp.fold == fold].file_name)


# ============================================================================ candidates
def components(p):
    """label map, stats of p > THR (8-conn), candidate labels (A_LO <= area < A_HI), base mask (area >= A_HI)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((p > THR).astype(np.uint8), connectivity=8)
    area = st[:, cv2.CC_STAT_AREA]
    cand = [k for k in range(1, n) if A_LO <= area[k] < A_HI]
    big = np.zeros(n, bool); big[1:] = area[1:] >= A_HI
    return lab, st, cand, big[lab]


def cand_set(lab, st, cand):
    """InstanceSet of the candidate components (bbox crops), in candidate order."""
    boxes, crops = [], []
    for k in cand:
        x, y, w, h = st[k, :4]
        boxes.append((y, x, y + h, x + w)); crops.append(lab[y:y + h, x:x + w] == k)
    return InstanceSet._build(HW, boxes, crops)


def crop_at(arr, cy, cx, r=CROP // 2):
    out = np.zeros((2 * r, 2 * r), arr.dtype)
    y0, x0 = cy - r, cx - r
    sy0, sx0, sy1, sx1 = max(y0, 0), max(x0, 0), min(y0 + 2 * r, HW[0]), min(x0 + 2 * r, HW[1])
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def targets(C, sets):
    """per candidate: y (mean over sets of best IoU if > 0.5 else 0), tp_any, best IoU over all sets, n_sets."""
    best = np.zeros((len(sets), len(C)))
    for i, G in enumerate(sets):
        if len(G) and len(C):
            best[i] = SC.iou_matrix(G, C)[1].max(0)
    gain = np.where(best > 0.5, best, 0.0)
    return gain.mean(0), (best > 0.5).any(0), best.max(0) if len(sets) else np.zeros(len(C))


_W = {}


def _npz(exp, run):
    key = (exp, run)
    if key not in _W:
        _W.clear()
        _W[key] = np.load(EXPS / exp / run / "val_probs_last.npz")
    return _W[key]


def extract(p, img, sets):
    """-> crops (N, 4, CROP, CROP) uint8, rows (list of dicts), candidate InstanceSet."""
    lab, st, cand, base = components(p)
    C = cand_set(lab, st, cand)
    y, tp_any, best = targets(C, sets) if sets is not None else (np.full(len(cand), np.nan),) * 3
    pu8 = np.clip(np.rint(p * 255), 0, 255).astype(np.uint8)
    crops, rows = [], []
    for i, k in enumerate(cand):
        x, yy, w, h = st[k, :4]
        cy, cx = int(yy + h // 2), int(x + w // 2)
        ch = [crop_at(img, cy, cx), crop_at(pu8, cy, cx), crop_at((lab == k).astype(np.uint8) * 255, cy, cx),
              crop_at(base.astype(np.uint8) * 255, cy, cx)]
        crops.append(np.stack(ch))
        rows.append(dict(cand=i, label=int(k), area=int(st[k, cv2.CC_STAT_AREA]), cy=cy, cx=cx,
                         y=float(y[i]), tp_any=bool(tp_any[i]), best_iou=float(best[i]),
                         r_frac=float(np.hypot(cy - 1023.5, cx - 1023.5) / 904.0)))
    return (np.stack(crops) if crops else np.zeros((0, 4, CROP, CROP), np.uint8)), rows, C


def cands_job(args):
    name, exp, run, fold, fn, sets = args
    stem = fn[:-5]
    p = _npz(exp, run)[stem].astype(np.float32)
    img = D.read_image(D.train_image_path(fn))
    crops, rows, _ = extract(p, img, sets)
    for r in rows:
        r.update(source=name, stem=stem, fold=fold, n_sets=len(sets))
    return name, crops, rows


def cmd_cands(a):
    import multiprocessing as mp
    single_thread_worker()
    gts = seg.gt_records()
    by = {}
    for rid, g in gts.items():
        by.setdefault(rid.split("-", 1)[1], []).append((rid, g))
    jobs = []
    for name, exp, run, fold in sources(a.final_only):
        files = fold_files(fold)
        have = set((EXPS / exp / run / "val_files.txt").read_text().split())
        assert set(files) <= have, (name, "stage-1 run does not cover the fold")
        for fn in files[:a.limit or None]:
            jobs.append((name, exp, run, fold, fn, [g for _, g in sorted(by[fn[:-5]])]))
    t0, acc = time.time(), {}
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for i, (name, crops, rows) in enumerate(pool.imap(cands_job, jobs, chunksize=4)):   # ordered: per-source npz
            c, r = acc.setdefault(name, ([], []))
            c.append(crops); r += rows
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    d = OUT / ("cands_test" if a.limit else "cands"); d.mkdir(parents=True, exist_ok=True)
    summ = {}
    for name, (c, r) in acc.items():
        X = np.concatenate(c) if c else np.zeros((0, 4, CROP, CROP), np.uint8)
        np.savez(d / f"{name}.npz", X=X)
        df = pd.DataFrame(r); df.to_csv(d / f"{name}.csv", index=False)
        assert len(df) == len(X)
        summ[name] = dict(n=int(len(df)), pos=int(df.tp_any.sum()) if len(df) else 0)
    json.dump(dict(sec=round(time.time() - t0), sources=summ), open(d / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in list(summ.items())[:6]}), f"... {time.time() - t0:.0f}s")


# ============================================================================ scorer
def build_scorer():
    import torch
    import torchvision
    m = torchvision.models.resnet101(weights=None)
    sd = seg.load_legacy_state_dict(Path.home() / ".cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth")
    m.load_state_dict(sd)
    w = m.conv1.weight.data                                   # (64, 3, 7, 7)
    conv = torch.nn.Conv2d(4, 64, 7, 2, 3, bias=False)
    conv.weight.data = w.mean(1, keepdim=True).repeat(1, 4, 1, 1)
    net = torch.nn.Sequential(conv, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2,
                              torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(512, 1))
    return net


def to_input(X):
    import torch
    x = torch.from_numpy(X.astype(np.float32) / 255.0)
    x[:, 0] = (x[:, 0] - 0.30) / 0.22
    return x


def fit_predict(Xtr, ytr, Xev, device, seed=SEED, epochs=25, bs=64, log=None):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    net = build_scorer().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(Xtr) / bs))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    xt, yt = to_input(Xtr), torch.from_numpy(ytr.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), generator=g)
        tl = 0.0
        for i in range(0, len(xt), bs):
            idx = perm[i:i + bs]
            xb = xt[idx]
            if torch.rand(1, generator=g) < 0.5:
                xb = xb.flip(-1)
            if torch.rand(1, generator=g) < 0.5:
                xb = xb.flip(-2)
            xb = torch.rot90(xb, int(torch.randint(0, 4, (1,), generator=g)), (-2, -1))
            out = net(xb.to(device)).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yt[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tl += float(loss) * len(idx)
        if log is not None:
            log.append(dict(epoch=ep + 1, loss=tl / len(xt)))
    net.eval()
    preds = []
    with torch.no_grad():
        xe = to_input(Xev)
        for i in range(0, len(xe), 256):
            preds.append(torch.sigmoid(net(xe[i:i + 256].to(device)).squeeze(1)).cpu().numpy())
    return net, (np.concatenate(preds) if preds else np.zeros(0))


def load_src(name, d=None):
    d = d or OUT / "cands"
    return np.load(d / f"{name}.npz")["X"], pd.read_csv(d / f"{name}.csv")


def outer_train_sources(k):
    return [f"inner_h{min(k, j)}{max(k, j)}_f{j}" for j in range(5) if j != k]


def tau_table():
    """tau_k = pooled V0 PQ of each base on the outer-train folds / 2 (registered numbers, from existing records)."""
    import refine_b0exist_checks as RC
    t2 = pd.read_csv(ROOT / "outputs" / "research" / "island_cleanup" / "records.csv")
    t2 = t2[t2.method == "R2nb0_isl5"]
    t1 = pd.read_csv(ROOT / "outputs" / "research" / "eval_dual" / "records.csv")
    t1 = t1[t1.method == "B_ens"]
    out = {}
    for name, df in (("T2", t2), ("T1", t1)):
        out[name] = {k: float(RC.summary_counts(df[df.fold != k])["PQ"]) / 2 for k in range(5)}
    return out


def cmd_train(a):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logs, rows = {}, []
    for k in ([] if a.final_only else range(5)):
        parts = [load_src(s) for s in outer_train_sources(k)]
        Xtr = np.concatenate([p[0] for p in parts]); ytr = np.concatenate([p[1].y.values for p in parts])
        Xev, mev = load_src(f"bens_f{k}")
        assert not set(pd.concat([p[1] for p in parts]).stem) & set(mev.stem), "outer fold leaked into training"
        log = []
        t0 = time.time()
        _, pr = fit_predict(Xtr, ytr, Xev, device, log=log)
        logs[f"outer{k}"] = dict(n_train=int(len(Xtr)), pos_train=int((ytr > 0).sum()), n_eval=int(len(Xev)),
                                 sec=round(time.time() - t0), loss=log)
        rows.append(mev.assign(score=pr, outer=k))
        print(f"outer {k}: train {len(Xtr)} (pos {(ytr > 0).sum()}), eval {len(Xev)}, {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.concat(rows).to_csv(OUT / "preds_outer.csv", index=False)
    parts = [load_src(f"b0_f{j}") for j in range(5)]
    Xtr = np.concatenate([p[0] for p in parts]); ytr = np.concatenate([p[1].y.values for p in parts])
    log = []
    net, _ = fit_predict(Xtr, ytr, Xtr[:1], device, log=log)
    torch.save(net.state_dict(), OUT / "scorer_final.pt")
    logs["final"] = dict(n_train=int(len(Xtr)), pos_train=int((ytr > 0).sum()), loss=log)
    if a.final_only:   # the nested records behind tau_table() may not exist; the test tau is fixed (D-066)
        json.dump(dict(device=device, logs=logs), open(OUT / "train_log_final_only.json", "w"), indent=1)
    else:
        json.dump(dict(device=device, tau=tau_table(), logs=logs), open(OUT / "train_log.json", "w"), indent=1)


# ============================================================================ add-only evaluation
def add_only(P: InstanceSet, C: InstanceSet, keep_idx):
    """P plus the kept candidates that have zero pixel overlap with P (and with each other: components are disjoint)."""
    occ = P.union()
    boxes, crops, added = [tuple(b) for b in P.boxes], list(P.crops), []
    for i in keep_idx:
        y0, x0, y1, x1 = C.boxes[i]
        if np.logical_and(occ[y0:y1, x0:x1], C.crops[i]).any():
            continue
        boxes.append((y0, x0, y1, x1)); crops.append(C.crops[i]); added.append(i)
    return InstanceSet._build(P.shape, boxes, crops), added


def no_set_fp(P, sets):
    """predictions that match NO annotation set of the image at IoU > 0.5."""
    if not len(P):
        return 0
    hit = np.zeros(len(P), bool)
    for G in sets:
        if len(G):
            hit |= (SC.iou_matrix(G, P)[1] > 0.5).any(0)
    return int((~hit).sum())


_PRED = {}


def _preds():
    if "p" not in _PRED:
        df = pd.read_csv(OUT / "preds_outer.csv")
        _PRED["p"] = {(s, int(c)): float(v) for s, c, v in zip(df.stem, df.cand, df.score)}
        _PRED["tau"] = json.load(open(OUT / "train_log.json"))["tau"]
    return _PRED["p"], _PRED["tau"]


def methods_for(stem, fold):
    import island_cleanup as IC
    M = IC._loao_instances(stem, fold)
    ens = np.load(EXPS / "b0n_ens" / f"fold{fold}" / "val_probs_last.npz")[stem].astype(np.float32)
    lab, st, cand, _ = components(ens)
    C = cand_set(lab, st, cand)
    pr, tau = _preds()
    sc = np.array([pr[(stem, i)] for i in range(len(cand))])
    out, info = {}, []
    for name, base in (("T2", M[IC.MAIN]), ("T1", M["B_ens"])):
        out[name] = base
        keep = np.flatnonzero(sc > tau[name][str(fold)])
        out[f"{name}_SE"], added = add_only(base, C, keep)
        info += [dict(stem=stem, fold=fold, base=name, cand=int(i), score=float(sc[i]), area=int(C.areas[i]))
                 for i in added]
    return out, info


def table_job(args):
    import a_eval as AE
    stem, fold, gts = args
    M, info = methods_for(stem, fold)
    sets = [gts[r] for r in sorted(gts)]
    recs, imgs = [], []
    for name, P in M.items():
        rec, img, _ = AE._eval(name, P, gts, stem, fold)
        img["no_set_fp"] = no_set_fp(P, sets)
        recs += rec; imgs.append(img)
    return recs, imgs, info


def cmd_table(a):
    import multiprocessing as mp
    import a_eval as AE
    import refine_b0exist_checks as RC
    single_thread_worker()
    jobs = AE.all_jobs()
    if a.limit:
        jobs = jobs[:a.limit]
    t0, R, I, A = time.time(), [], [], []
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        for i, (rec, img, info) in enumerate(pool.imap_unordered(table_job, jobs, chunksize=2)):
            R += rec; I += img; A += info
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(R)
    sp = RC.split_table().set_index("stem")
    df["group_id"] = df.stem.map(sp.group_id); df["n_sets"] = df.stem.map(sp.n_sets)
    sfx = "_test" if a.limit else ""
    df.to_csv(OUT / f"records{sfx}.csv", index=False)
    pd.DataFrame(I).to_csv(OUT / f"images_v1{sfx}.csv", index=False)
    pd.DataFrame(A).to_csv(OUT / f"added{sfx}.csv", index=False)
    print(f"table: {len(df)} rows, {df.stem.nunique()} files, {time.time() - t0:.0f}s")


def _loao_instances(stem, fold):
    M, _ = methods_for(stem, fold)
    return M


def loao_wrap(args):
    import eval_dual as ED
    ED.instances = _loao_instances
    ED.LOAO_RATERS = ("T2", "T2_SE", "T1", "T1_SE")
    return ED.loao_job(args)


def cmd_loao(a):
    import eval_dual as ED
    ED.OUT = OUT / "loao"
    ED.LOAO_RATERS = ("T2", "T2_SE", "T1", "T1_SE")
    ED.loao_job = loao_wrap
    ED.cmd_loao(a)


def cmd_analyse(a):
    import eval_dual as ED
    import refine_b0exist_checks as RC
    df = pd.read_csv(OUT / "records.csv"); I = pd.read_csv(OUT / "images_v1.csv")
    add = pd.read_csv(OUT / "added.csv")
    sp = RC.split_table()
    RC.FOLDS = (0, 1, 2, 3, 4)
    rc = pd.read_csv(ROOT / "outputs/audit/review_cases.csv")
    review = set(rc.file_name.str[:-5])
    ED.OUT = OUT / "loao"
    ED.LOAO_RATERS = ("T2", "T2_SE", "T1", "T1_SE")
    asp, BH = ED.loao_aspects() if (OUT / "loao" / "loao_records.csv").exists() else (None, None)
    res = {}
    for base in ("T2", "T1"):
        main = f"{base}_SE"
        v0, _ = RC.compare(df, base, main, sp, review)
        v1 = ED.v1_boot(I, base, main)
        rv = I[I.stem.isin(review)]
        rv1 = ED.v1_boot(rv, base, main)
        rv0 = {m: RC.summary_counts(df[(df.method == m) & df.stem.isin(review)])["PQ"] for m in (base, main)}
        fp = {m: int(I[I.method == m].no_set_fp.sum()) for m in (base, main)}
        r = dict(V0=dict(A=v0["A"]["PQ"], B=v0["B"]["PQ"], diff=v0["file_boot"]["PQ"]["diff"],
                         ci95=v0["file_boot"]["PQ"]["ci95"], per_fold={k: v["diff"] for k, v in v0["per_fold"].items()}),
                 V1=v1, review=dict(V0=rv0, V1=dict(A=rv1["A"], B=rv1["B"])), no_set_fp=fp,
                 n_added=int((add.base == base).sum()), diagnostics={m: ED.diag_sums(df[df.method == m]) for m in (base, main)})
        folds_pos = sum(v > 0 for v in r["V0"]["per_fold"].values())
        newly, small = None, None
        if asp is not None:
            newly = []
            for s, d in BH.ASPECTS.items():
                vb, vm, vh = (asp[x].get(s) for x in (base, main, "human"))
                f = lambda z: np.nan if z is None else float(z)  # noqa: E731
                if BH.not_worse(f(vb), f(vh), d) is True and BH.not_worse(f(vm), f(vh), d) is False:
                    newly.append(s)
            small = dict(base=asp[base].get("small_detected_rate"), main=asp[main].get("small_detected_rate"),
                         human=asp["human"].get("small_detected_rate"))
            r["loao_small_detected"] = small
            r["loao_newly_below_human"] = newly
        r["rule"] = dict(
            E1_v0_gain=bool(r["V0"]["diff"] > 0 and r["V0"]["ci95"][0] > -0.002 and folds_pos >= 3),
            E2_v1_not_lower=bool(v1["diff"] >= 0),
            E3_noset_fp_le_10pct=bool(fp[main] <= 1.10 * fp[base]),
            E4_loao=None if asp is None else bool(small["main"] > small["base"] and not newly),
            E5_review_not_lower=bool(rv0[main] >= rv0[base] and rv1["B"] >= rv1["A"]))
        r["rule"]["PASS"] = all(v is True for v in r["rule"].values())
        res[base] = r
    json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({b: dict(V0=res[b]["V0"], V1={k: res[b]["V1"][k] for k in ("A", "B", "diff", "ci95")},
                              rule=res[b]["rule"], n_added=res[b]["n_added"]) for b in res}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cands", "train", "table", "loao", "analyse"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--final_only", action="store_true",
                    help="cands/train: only the b0 5-fold OOF candidates and the final scorer (no nested inner models)")
    a = ap.parse_args()
    if a.cmd != "train":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    {"cands": cmd_cands, "train": cmd_train, "table": cmd_table, "loao": cmd_loao, "analyse": cmd_analyse}[a.cmd](a)
