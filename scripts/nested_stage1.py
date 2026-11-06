"""Nested-CV stage 1 for the refinement experiment R2b0 (decision D-035).

Problem fixed: in refine_v1 the stage-1 probability of a TRAINING image in fold j came from b0 fold-j, which was
trained on folds != j, i.e. INCLUDING the validation fold k. So the validation fold's labels shaped the refinement
model's training inputs. Training R2 only on more folds does not remove this.

Design. Outer fold k = validation; outer-train = folds != k. Inner model b0n{k,j} (j != k) is the unchanged b0 recipe
trained on the three folds outside {k, j} (scripts/train_seg.py --exp b0n_r101_s1024 --holdout k,j) and validated on
folds k AND j. The holdout pair is symmetric, so one model is the inner model of (outer k, inner fold j) and of
(outer j, inner fold k): 10 models serve the 5 outer folds.

Stage 1 of outer fold k (cache outputs/cache/refine_nested_o<k>):
  file in fold j != k (outer-train, R2n_k training input) : b0n{k,j} probability            (never saw k or j)
  file in fold k      (outer-validation)                  : mean of b0n{k,j} over the 4 j != k (none saw k)
This mirrors the final submission, where R2_final is trained on all 707 files with the 5-fold b0 OOF (refine_v1:
fold-j file <- b0 fold j) and applied to test images with the MEAN of the 5 b0 fold models. The matched baseline of
outer fold k is that same 4-model mean thresholded like the submission (t0.5, 8-conn, min_area 500).

Subcommands (CPU only):
  plan                 inner runs (train/val folds, outer folds served, status) and the queue order
  ens    --outer k     outputs/exps/b0n_ens/fold<k>/{val_probs_last.npz, val_files.txt, config.json}
  cache  --outer k     outputs/cache/refine_nested_o<k> (prob_u8 + candidates; images.npy/gt.npy symlinked to refine_v1)
  verify --outer k     leakage + consistency checks -> outputs/nested/verify_o<k>.json (non-zero exit on failure)
Paths can be redirected (--inner_exp/--ens_exp/--cache_prefix/--r2_exp/--files_per_fold) for wiring tests.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from filseg import refine as RF  # noqa: E402

EXPS = Path(os.environ.get("FILSEG_EXPS", ROOT / "outputs" / "exps"))   # redirected only by wiring tests
FOLDS = ROOT / "outputs" / "splits" / "folds_v1.csv"
K = 5
# queue order: outer folds 2-4 first (not seen when the R2b0 rule was conceived on folds 0/1), then 0/1
PAIR_ORDER = [(2, 3), (2, 4), (0, 2), (1, 2), (3, 4), (0, 3), (1, 3), (0, 4), (1, 4), (0, 1)]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["plan", "ens", "cache", "verify"])
    p.add_argument("--outer", type=int)
    p.add_argument("--inner_exp", default="b0n_r101_s1024")
    p.add_argument("--ens_exp", default="b0n_ens")
    p.add_argument("--r2_exp", default="R2n_context")
    p.add_argument("--cache_prefix", default=str(ROOT / "outputs" / "cache" / "refine_nested_o"))
    p.add_argument("--static_from", default=str(RF.CACHE), help="cache whose images/gt are reused ('' = write)")
    p.add_argument("--files_per_fold", type=int, default=0, help="wiring tests only: first N files of each fold")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=str(ROOT / "outputs" / "nested"))
    return p.parse_args()


def split(files_per_fold=0):
    sp = pd.read_csv(FOLDS)
    if files_per_fold:
        sp = sp.sort_values("file_name").groupby("fold", group_keys=False).head(files_per_fold)
    return sp


def run_name(k, j):
    a, b = sorted((k, j))
    return f"hold{a}{b}"


def inner_runs(k):
    """[(inner fold j, run name)] for outer fold k."""
    return [(j, run_name(k, j)) for j in range(K) if j != k]


def done(exp, run):
    return (EXPS / exp / run / "summary.json").exists()


class NpzCache:
    def __init__(self):
        self.z = {}

    def get(self, path, stem):
        if path not in self.z:
            self.z[path] = np.load(path)
        return self.z[path][stem].astype(np.float32)


def ens_mean(a, k, files, nz):
    """4-model mean on the outer-validation fold k (float32 mean of the stored float16 probabilities)."""
    out = {}
    for f in files:
        st = f[:-5]
        out[st] = np.mean([nz.get(EXPS / a.inner_exp / r / "val_probs_last.npz", st) for _, r in inner_runs(k)],
                          axis=0)
    return out


def cmd_plan(a):
    rows = []
    for (k, j) in PAIR_ORDER:
        r = run_name(k, j)
        rows.append(dict(run=r, train_folds=[f for f in range(K) if f not in (k, j)], val_folds=sorted((k, j)),
                         serves=f"outer {k} (inner OOF of fold {j}) + outer {j} (inner OOF of fold {k})",
                         done=done(a.inner_exp, r)))
    print(pd.DataFrame(rows).to_string(index=False))
    for k in range(K):
        need = [r for _, r in inner_runs(k)]
        print(f"outer {k}: inner runs {need} -> ready={all(done(a.inner_exp, r) for r in need)}; "
              f"ens={(EXPS / a.ens_exp / f'fold{k}' / 'val_probs_last.npz').exists()}; "
              f"cache={(Path(a.cache_prefix + str(k)) / 'index.json').exists()}; R2n={done(a.r2_exp, f'fold{k}')}")


def cmd_ens(a):
    k = a.outer
    for _, r in inner_runs(k):
        assert done(a.inner_exp, r), f"inner run {r} not finished"
    sp = split(a.files_per_fold)
    files = sorted(sp.file_name[sp.fold == k])
    t0 = time.time()
    probs = ens_mean(a, k, files, NpzCache())
    out = EXPS / a.ens_exp / f"fold{k}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "val_probs_last.npz", **{st: p.astype(np.float16) for st, p in probs.items()})
    (out / "val_files.txt").write_text("\n".join(files))
    json.dump(dict(outer=k, members=[f"{a.inner_exp}/{r}" for _, r in inner_runs(k)], combine="mean of probabilities",
                   n_files=len(files), note="matched stage 1 of outer fold k (D-035); baseline = t0.5/8-conn/a500"),
              open(out / "config.json", "w"), indent=1)
    print(f"ens outer {k}: {len(files)} files, {time.time() - t0:.0f}s -> {out}")


def stage1_sources(a, k, sp):
    """file -> list of 'exp/run/npz' sources whose (mean) probability is the stage 1 of that file for outer fold k."""
    fold_of = dict(zip(sp.file_name, sp.fold))
    src = {}
    for f in sorted(sp.file_name):
        j = int(fold_of[f])
        if j == k:
            src[f] = [f"{a.ens_exp}/fold{k}/val_probs_last.npz"]
        else:
            src[f] = [f"{a.inner_exp}/{run_name(k, j)}/val_probs_last.npz"]
    return src


def cmd_cache(a):
    k = a.outer
    sp = split(a.files_per_fold)
    assert (EXPS / a.ens_exp / f"fold{k}" / "val_probs_last.npz").exists(), "run 'ens' first"
    src = stage1_sources(a, k, sp)
    nz = NpzCache()

    def prob(fname):
        (s,) = src[fname]
        return nz.get(EXPS / s, fname[:-5])

    t0 = time.time()
    d = RF.build_cache(stage1_exp=f"nested_outer{k}", n_workers=a.workers, cache=a.cache_prefix + str(k),
                       stage1_probs=prob, files=sorted(sp.file_name), static_from=a.static_from or None,
                       extra=dict(outer=k, exist_exp=a.ens_exp, stage1_sources=src))
    idx = json.load(open(Path(d) / "index.json"))
    nw = sum(len(m["windows"]) for f, m in idx["meta"].items() if idx["fold"][f] != k)
    print(f"cache outer {k}: {d}, {time.time() - t0:.0f}s, train windows {nw}")


def cmd_verify(a):
    """Checks (all must hold):
    A inner runs: trained exactly on the folds outside {k, j}; no fold-k and no fold-j file among train_files;
      validated on folds {k, j}.
    B stage-1 source of every file is the expected inner model / outer ensemble, and no source model saw fold k.
    C cached prob_u8 == recomputed from the sources (every file), candidates of the outer-validation files ==
      recomputed from prob_u8 alone (p > CAND_THR, >= CAND_MIN px).
    D the outer ensemble == mean of the 4 inner models (recomputed).
    E (if present) R2n_k: trained only on files of folds != k, from this cache, existence source = this ensemble."""
    k = a.outer
    sp = split(a.files_per_fold)
    fold_files = {f: set(sp.file_name[sp.fold == f]) for f in range(K)}
    res, ok = dict(outer=k), True

    def check(name, cond, detail=None):
        nonlocal ok
        res[name] = dict(ok=bool(cond), **({"detail": detail} if detail is not None else {}))
        ok &= bool(cond)

    for j, r in inner_runs(k):
        d = EXPS / a.inner_exp / r
        cfg = json.load(open(d / "config.json"))
        tr = set((d / "train_files.txt").read_text().split())
        va = set((d / "val_files.txt").read_text().split())
        exp_train = set().union(*(fold_files[f] for f in range(K) if f not in (k, j)))
        if a.files_per_fold:          # wiring tests use a file subset; the run lists may be full
            tr, va = tr & set(sp.file_name), va & set(sp.file_name)
        check(f"A_{r}_train_folds", sorted(cfg["train_folds"]) == [f for f in range(K) if f not in (k, j)],
              cfg["train_folds"])
        check(f"A_{r}_no_outer_val_file_in_train", not (tr & fold_files[k]), len(tr & fold_files[k]))
        check(f"A_{r}_no_inner_fold_file_in_train", not (tr & fold_files[j]), len(tr & fold_files[j]))
        check(f"A_{r}_train_files_exact", tr == exp_train, len(tr ^ exp_train))
        check(f"A_{r}_val_files", va == fold_files[k] | fold_files[j], len(va ^ (fold_files[k] | fold_files[j])))
    cache = Path(a.cache_prefix + str(k))
    idx = json.load(open(cache / "index.json"))
    src = idx["stage1_sources"]
    check("B_outer", idx["outer"] == k and idx["exist_exp"] == a.ens_exp)
    ens_cfg = json.load(open(EXPS / a.ens_exp / f"fold{k}" / "config.json"))
    check("B_ensemble_members", sorted(ens_cfg["members"]) == sorted(f"{a.inner_exp}/{r}" for _, r in inner_runs(k)),
          ens_cfg["members"])
    bad = []
    for f, s in src.items():
        (s,) = s
        j = idx["fold"][f]
        want = f"{a.ens_exp}/fold{k}/val_probs_last.npz" if j == k else f"{a.inner_exp}/{run_name(k, j)}/val_probs_last.npz"
        if s != want:
            bad.append(f)
        if j != k:   # the source model never saw fold k nor the file's own fold
            cfg = json.load(open(EXPS / a.inner_exp / run_name(k, j) / "config.json"))
            if k in cfg["train_folds"] or j in cfg["train_folds"]:
                bad.append(f)
    check("B_sources", not bad and set(src) == set(idx["files"]), bad[:5])
    nz = NpzCache()
    P = np.load(cache / "prob_u8.npy", mmap_mode="r")
    C = np.load(cache / "cand.npy", mmap_mode="r")
    mism, cand_mism = [], []
    for i, f in enumerate(idx["files"]):
        (s,) = src[f]
        q = np.clip(np.rint(nz.get(EXPS / s, f[:-5]) * 255), 0, 255).astype(np.uint8)
        if not np.array_equal(q, P[i]):
            mism.append(f)
        if idx["fold"][f] == k:
            p = P[i].astype(np.float32) / 255.0
            n, lab, st = cv2.connectedComponentsWithStats((p > RF.CAND_THR).astype(np.uint8), connectivity=8)[:3]
            keep = [c for c in range(1, n) if st[c, cv2.CC_STAT_AREA] >= RF.CAND_MIN]
            lut = np.zeros(n, np.uint16)
            lut[keep] = np.arange(1, len(keep) + 1, dtype=np.uint16)
            if not np.array_equal(lut[lab], C[i]):
                cand_mism.append(f)
    check("C_prob_u8_equals_sources", not mism, mism[:5])
    check("C_val_candidates_from_prob_only", not cand_mism, cand_mism[:5])
    vf = sorted(fold_files[k])
    ens = np.load(EXPS / a.ens_exp / f"fold{k}" / "val_probs_last.npz")
    re = ens_mean(a, k, vf, nz)
    dmax = max(float(np.abs(ens[f[:-5]].astype(np.float32) - re[f[:-5]].astype(np.float16).astype(np.float32)).max())
               for f in vf)
    check("D_ensemble_is_mean_of_4", dmax == 0.0, dmax)
    r2 = EXPS / a.r2_exp / f"fold{k}"
    if (r2 / "config.json").exists():
        cfg = json.load(open(r2 / "config.json"))
        tr = set((r2 / "train_files.txt").read_text().split())
        check("E_r2_train_has_no_outer_val_file", not (tr & fold_files[k]), len(tr & fold_files[k]))
        check("E_r2_train_folds", sorted(cfg["train_folds"]) == [f for f in range(K) if f != k], cfg["train_folds"])
        check("E_r2_cache", Path(cfg["cache_dir"]).resolve() == cache.resolve(), cfg["cache_dir"])
        check("E_r2_exist_exp", cfg["exist_exp"] == a.ens_exp, cfg["exist_exp"])
    else:
        res["E_r2"] = "not trained yet"
    res["all_ok"] = bool(ok)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"verify_o{k}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


def main():
    a = parse()
    if a.cmd != "plan":
        assert a.outer is not None and 0 <= a.outer < K
    {"plan": cmd_plan, "ens": cmd_ens, "cache": cmd_cache, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
