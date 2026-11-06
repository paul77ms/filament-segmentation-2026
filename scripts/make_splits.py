#!/usr/bin/env python
"""S1: observation grouping, temporal-leakage analysis and leakage-free CV split.

Outputs
  outputs/splits/folds_v1.csv                     one row per TRAIN file (the CV split)
  outputs/splits/analysis/*.csv                   supporting tables (gaps, similarity, balance)
  docs/figures/split_*.png                        supporting figures

Usage
  PYTHONPATH=src .venv/bin/python scripts/make_splits.py            # full run (CPU only, <=16 workers)
  ... --skip-image-analysis                                         # time-only (fast) run
  ... --orientation-check                                           # geometry check (after a full run)

Rules respected
  * CPU only. Images opened read-only. No test labels exist / none used.
  * 'spine' and 'category_id' are NOT used. Train polygons are used only for the
    leakage *analysis* (re-observation of the same filament), never as a training target.

Geometry notes (verified in this script, see docs/split_design.md)
  * Images are 2048x2048, disk centre ~(1021.5,1022.6) px, R ~901 px (limb ring fit).
  * Orientation: solar north up, west right (rotation direction checked empirically).
  * Derotation: Snodgrass & Ulrich (1990) sidereal rate minus 0.9856 deg/day, B0 from
    Meeus (Astronomical Algorithms, ch. 29); P angle assumed already removed.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root (path-independent)

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

PROJ = _REPO
DATA = f"{PROJ}/MAGFiLO_1.0_Kaggle_2026"
TRAIN_JSON = f"{DATA}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_DIR = f"{DATA}/train/train_images"
TEST_DIR = f"{DATA}/test/test_images"
OUT_DIR = f"{PROJ}/outputs/splits"
ANA_DIR = f"{OUT_DIR}/analysis"
FIG_DIR = f"{PROJ}/docs/figures"

FNAME_RE = re.compile(r"^(\d{14})([A-Za-z]{2})\.jpe?g$")
SITE_NAMES = {"Bh": "Big Bear", "Mh": "Mauna Loa", "Lh": "Learmonth", "Uh": "Udaipur",
              "Th": "El Teide", "Ch": "Cerro Tololo"}
GAP_THRESH_D = [1 / 24, 6 / 24, 1, 3, 7, 14, 27]
GAP_THRESH_LBL = ["<1h", "<6h", "<1d", "<3d", "<7d", "<14d", "<27d"]
DT_BINS = [0, 1 / 24, 6 / 24, 1, 2, 3, 5, 7, 10, 14, 20, 25, 30, 40]
SYNODIC_CARRINGTON_D = 27.2753
MIN_OVERLAP = 0.05      # min co-visible fraction (mu>0.26) of A's disk for a derotated correlation

G_FEAT = None      # (n,256,256) float32 feature maps, set in parent before fork
G_GEO = None       # (n,3) cx,cy,R on the 256 grid
G_B0 = None        # (n,) radians
G_T = None         # (n,) days
G_MASK = None      # list of (512,512) uint8 union masks (train only; None for test)
G_INST = None      # list of (512,512) uint8 instance label maps of the first annotation set
G_GEO512 = None    # (n,3) cx,cy,R on the 512 grid
G_MASK_DIL = None  # list of dilated union masks (512 grid)


# --------------------------------------------------------------------------- metadata
def parse_name(fn: str):
    m = FNAME_RE.match(fn)
    if m is None:
        raise ValueError(f"unexpected file name {fn}")
    return pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S"), m.group(2)


def to_days(ts) -> np.ndarray:
    return pd.to_datetime(pd.Series(ts)).values.astype("datetime64[s]").astype(np.int64) / 86400.0


def load_metadata():
    coco = json.load(open(TRAIN_JSON))
    tr_files = sorted(os.listdir(TRAIN_DIR))
    te_files = sorted(os.listdir(TEST_DIR))
    ids_by_file = defaultdict(list)
    for im in coco["images"]:
        ids_by_file[im["file_name"]].append(im["id"])
        # date_captured must match the file name timestamp
        assert pd.to_datetime(im["date_captured"]) == parse_name(im["file_name"])[0], im
    n_ann = defaultdict(int)
    for a in coco["annotations"]:
        n_ann[a["image_id"]] += 1
    assert set(ids_by_file) == set(tr_files), "JSON file_names != train_images directory"
    assert not (set(tr_files) & set(te_files)), "file name shared by train and test"
    rows = []
    for f in tr_files:
        dt, site = parse_name(f)
        ids = sorted(ids_by_file[f])
        assert len(set(i.split("-", 1)[1] for i in ids)) == 1
        rows.append(dict(file_name=f, image_ids=";".join(ids), n_sets=len(ids),
                         n_instances_total=int(sum(n_ann[i] for i in ids)),
                         n_instances_per_set=";".join(str(n_ann[i]) for i in ids),
                         datetime=dt, site=site, year=dt.year,
                         batch_groups=";".join(sorted(set(i[:4] for i in ids))), split="train"))
    for f in te_files:
        dt, site = parse_name(f)
        rows.append(dict(file_name=f, image_ids="", n_sets=0, n_instances_total=0,
                         n_instances_per_set="", datetime=dt, site=site, year=dt.year,
                         batch_groups="", split="test"))
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    df["t_days"] = to_days(df.datetime)
    return coco, df


# --------------------------------------------------------------------------- time gaps
def nn_gap(src_t: np.ndarray, ref_t: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """|time| to the nearest element of ref_t for every element of src_t (days)."""
    ref = np.sort(ref_t)
    idx = np.searchsorted(ref, src_t)
    out = np.full(len(src_t), np.inf)
    for off in (-2, -1, 0, 1):
        j = np.clip(idx + off, 0, len(ref) - 1)
        d = np.abs(src_t - ref[j])
        if exclude_self:
            d = np.where(d == 0, np.inf, d)  # exact duplicate timestamps do not occur (asserted)
        out = np.minimum(out, d)
    return out


def gap_table(gaps: dict) -> pd.DataFrame:
    rows = []
    for name, g in gaps.items():
        g = np.asarray(g)
        r = dict(set=name, n=len(g), min_h=g.min() * 24, p10_d=np.quantile(g, .1), median_d=np.median(g),
                 p90_d=np.quantile(g, .9), max_d=g.max())
        for th, lb in zip(GAP_THRESH_D, GAP_THRESH_LBL):
            r[lb] = int((g < th).sum())
            r[lb + "_pct"] = round(100 * (g < th).mean(), 1)
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- solar geometry
def sun_B0(t_days_unix: np.ndarray) -> np.ndarray:
    """Heliographic latitude of disk centre (B0, radians). Meeus ch.29 (low precision, ~0.01 deg)."""
    jd = np.asarray(t_days_unix) + 2440587.5
    T = (jd - 2451545.0) / 36525.0
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T ** 2
    M = np.deg2rad(357.52911 + 35999.05029 * T - 0.0001537 * T ** 2)
    C = ((1.914602 - 0.004817 * T - 0.000014 * T ** 2) * np.sin(M)
         + (0.019993 - 0.000101 * T) * np.sin(2 * M) + 0.000289 * np.sin(3 * M))
    om = np.deg2rad(125.04 - 1934.136 * T)
    lam = np.deg2rad(L0 + C - 0.00569 - 0.00478 * np.sin(om))
    K = np.deg2rad(73.6667 + 1.3958333 * (jd - 2396758.0) / 36525.0)
    return np.arcsin(np.sin(lam - K) * np.sin(np.deg2rad(7.25)))


def omega_syn(lat):
    """Synodic differential rotation rate, rad/day (Snodgrass & Ulrich 1990 minus Earth orbit)."""
    s2 = np.sin(lat) ** 2
    return np.deg2rad(14.713 - 2.396 * s2 - 1.787 * s2 ** 2 - 0.9856)


def derot_map(geo_a, geo_b, b0a, b0b, dt, n, mu_min=0.26, rmax=0.97):
    """For every pixel of image A (n x n grid) return where the same solar surface point is in B.
    dt = t_B - t_A (days). valid: on disk in both and mu=cos(angle from disk centre) > mu_min."""
    cxa, cya, ra = geo_a
    cxb, cyb, rb = geo_b
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    X = (xx - cxa) / ra
    Y = -(yy - cya) / ra
    rho2 = X * X + Y * Y
    on = rho2 < rmax ** 2
    Z = np.sqrt(np.clip(1 - rho2, 0, 1))
    sphi = np.clip(Y * np.cos(b0a) + Z * np.sin(b0a), -1, 1)
    phi = np.arcsin(sphi)
    L = np.arctan2(X, Z * np.cos(b0a) - Y * np.sin(b0a))
    Lb = L + omega_syn(phi) * dt
    cphi = np.cos(phi)
    Xb = cphi * np.sin(Lb)
    Yb = sphi * np.cos(b0b) - cphi * np.cos(Lb) * np.sin(b0b)
    Zb = sphi * np.sin(b0b) + cphi * np.cos(Lb) * np.cos(b0b)
    valid = on & (Z > mu_min) & (Zb > mu_min) & (Xb * Xb + Yb * Yb < rmax ** 2)
    return (cxb + rb * Xb).astype(np.float32), (cyb - rb * Yb).astype(np.float32), valid, (on & (Z > mu_min))


def heliographic_lat(x_full, y_full, geo_full, b0):
    cx, cy, R = geo_full
    X = (x_full - cx) / R
    Y = -(y_full - cy) / R
    Z = np.sqrt(np.clip(1 - X * X - Y * Y, 0, 1))
    return np.arcsin(np.clip(Y * np.cos(b0) + Z * np.sin(b0), -1, 1))


# --------------------------------------------------------------------------- image features
def _fit_circle(p):
    A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
    s = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    return s[0], s[1], np.sqrt(s[2] + s[0] ** 2 + s[1] ** 2)


def limb_fit(s4: np.ndarray):
    """Fit the bright limb ring on a 512x512 (1/4 scale) image. Returns full-res cx,cy,R,rms,inlier_frac.
    (A threshold mask is NOT used for geometry: the processed off-limb halo has variable size.)"""
    m = s4 > 0.35 * np.percentile(s4, 99)
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    th = s4 - cv2.blur(s4, (9, 9))
    ang = np.deg2rad(np.arange(0, 360, 1.0))
    rs = np.arange(170, 265, 0.5)
    for _ in range(2):
        X = cx + np.cos(ang)[:, None] * rs[None]
        Y = cy + np.sin(ang)[:, None] * rs[None]
        v = cv2.remap(th, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR, borderValue=-1e3)
        r = rs[v.argmax(1)]
        ok = v.max(1) > 5
        p = np.c_[cx + np.cos(ang) * r, cy + np.sin(ang) * r][ok]
        keep = np.ones(len(p), bool)
        for _ in range(5):
            c = _fit_circle(p[keep])
            d = np.abs(np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) - c[2])
            keep = d < max(2.0, 3 * np.median(d[keep]))
        cx, cy, R = c
    rms = float(np.sqrt(np.mean(d[keep] ** 2)))
    # 1/4-scale pixel centre -> full-res pixel centre
    return (cx + 0.5) * 4 - 0.5, (cy + 0.5) * 4 - 0.5, R * 4, rms * 4, keep.sum() / 360.0


def image_stats(path: str):
    raw = open(path, "rb").read()
    md5 = hashlib.md5(raw).hexdigest()
    s4 = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_4).astype(np.float32)
    return md5, limb_fit(s4)


def feature_map(path: str, geo_full, sigma=1.5, ks=31, n=256):
    """High-pass, robust-normalised on-disk map on an n x n grid (default 1/8 scale)."""
    s = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_8).astype(np.float32)
    assert s.shape == (n, n)
    sc = n / 2048.0
    cx, cy, R = (geo_full[0] + 0.5) * sc - 0.5, (geo_full[1] + 0.5) * sc - 0.5, geo_full[2] * sc
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - cx, yy - cy) / R
    s2 = s.copy()
    s2[rr > 0.97] = np.median(s[(rr > 0.9) & (rr < 0.97)])     # keep the halo/limb ring out of the background
    bg = cv2.medianBlur(np.clip(s2, 0, 255).astype(np.uint8), ks).astype(np.float32)
    res = cv2.GaussianBlur(s2 - bg, (0, 0), sigma)
    inn = rr < 0.9
    mu = np.median(res[inn])
    sd = 1.4826 * np.median(np.abs(res[inn] - mu)) + 1e-6
    res = np.clip((res - mu) / sd, -4, 4)
    res[rr >= 0.97] = 0
    return res.astype(np.float32), (cx, cy, R)


def _feat_worker(args):
    return feature_map(*args)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def pair_similarity(pair):
    """Derotated and non-derotated correlation of feature maps (256 grid) for pair (i, j); dt = t_j - t_i."""
    i, j = pair
    dt = G_T[j] - G_T[i]
    mx, my, valid, base = derot_map(G_GEO[i], G_GEO[j], G_B0[i], G_B0[j], dt, 256)
    out = dict(i=i, j=j, dt=dt, overlap=valid.sum() / max(base.sum(), 1))
    # a correlation over a thin limb-adjacent sliver is dominated by limb residuals (observed: strongly
    # negative values when overlap < ~1%), so require >= MIN_OVERLAP of A's usable disk to be co-visible
    if out["overlap"] >= MIN_OVERLAP and valid.sum() >= 500:
        bs = cv2.remap(G_FEAT[j], mx, my, cv2.INTER_LINEAR, borderValue=0)
        out["corr_derot"] = _corr(G_FEAT[i][valid], bs[valid])
    else:
        out["corr_derot"] = np.nan
    # no derotation: same pixel position (images share geometry to <~2 px at full res)
    mx0, my0, valid0, _ = derot_map(G_GEO[i], G_GEO[j], G_B0[i], G_B0[j], 0.0, 256)
    b0 = cv2.remap(G_FEAT[j], mx0, my0, cv2.INTER_LINEAR, borderValue=0)
    out["corr_raw"] = _corr(G_FEAT[i][valid0], b0[valid0])
    # label-based re-observation (train-train only)
    if G_INST is not None and G_INST[i] is not None and G_MASK[j] is not None:
        mx5, my5, valid5, _ = derot_map(G_GEO512[i], G_GEO512[j], G_B0[i], G_B0[j], dt, 512)
        lab = G_INST[i]
        ids, cnt_all = np.unique(lab[lab > 0], return_counts=True)
        tgt = cv2.remap(G_MASK_DIL[j], mx5, my5, cv2.INTER_NEAREST, borderValue=0)
        elig = reobs = 0
        a_pix = hit_pix = 0
        for k, c in zip(ids, cnt_all):
            sel = lab == k
            vis = sel & valid5
            if vis.sum() < 0.5 * c:          # less than half of this filament co-visible
                continue
            elig += 1
            h = (tgt[vis] > 0).sum()
            a_pix += vis.sum()
            hit_pix += h
            if h >= 0.5 * vis.sum():
                reobs += 1
        out.update(inst_elig=elig, inst_reobs=reobs, area_elig=a_pix, area_hit=hit_pix)
    return out


# --------------------------------------------------------------------------- grouping & folds
def time_components(t_days: np.ndarray, T: float) -> np.ndarray:
    """Connected components of the graph linking observations with |dt| < T (1-D => cut at gaps >= T)."""
    order = np.argsort(t_days)
    ts = t_days[order]
    comp_sorted = np.r_[0, np.cumsum(np.diff(ts) >= T)]
    comp = np.empty(len(t_days), int)
    comp[order] = comp_sorted
    return comp


def group_features(df_tr: pd.DataFrame, group_col: str):
    sites = sorted(df_tr.site.unique())
    years = sorted(df_tr.year.unique())
    bgs = sorted(set(";".join(df_tr.batch_groups).split(";")))
    gids = np.sort(df_tr[group_col].unique())
    blocks, names, w = [], [], []
    g = df_tr.groupby(group_col)
    blocks.append(np.c_[g.size().reindex(gids).values, g.n_sets.sum().reindex(gids).values,
                        g.n_instances_total.sum().reindex(gids).values]); names += ["files", "sets", "instances"]; w += [6, 6, 6]
    blocks.append(pd.crosstab(df_tr[group_col], df_tr.site).reindex(index=gids, columns=sites, fill_value=0).values)
    names += [f"site_{s}" for s in sites]; w += [1.0] * len(sites)
    blocks.append(pd.crosstab(df_tr[group_col], df_tr.year).reindex(index=gids, columns=years, fill_value=0).values)
    names += [f"year_{y}" for y in years]; w += [1.0] * len(years)
    blocks.append(pd.crosstab(df_tr[group_col], df_tr.n_sets).reindex(index=gids, columns=[1, 2, 3], fill_value=0).values)
    names += ["nsets_1", "nsets_2", "nsets_3"]; w += [1.0] * 3
    bg_mat = np.zeros((len(gids), len(bgs)))
    gpos = {gg: k for k, gg in enumerate(gids)}
    bpos = {b: k for k, b in enumerate(bgs)}
    for gg, s in zip(df_tr[group_col], df_tr.batch_groups):
        for b in s.split(";"):
            bg_mat[gpos[gg], bpos[b]] += 1
    blocks.append(bg_mat); names += [f"batch_{b}" for b in bgs]; w += [0.3] * len(bgs)
    return gids, np.hstack(blocks).astype(float), names, np.array(w)


def assign_folds(F: np.ndarray, w: np.ndarray, k: int, seed: int, restarts: int = 300, swaps: bool = True):
    """Deterministic grouped assignment minimising a weighted chi-square imbalance:
    sum_f sum_c w_c (S_fc - E_c)^2 / max(E_c, 1).  Greedy (largest first, random tie-break) + move/swap search."""
    rng = np.random.default_rng(seed)
    E = F.sum(0) / k
    den = np.maximum(E, 1.0)
    n = len(F)

    def cost(S):
        return float((w * (S - E) ** 2 / den).sum())

    best, best_cost = None, np.inf
    for _ in range(restarts):
        order = np.lexsort((rng.random(n), -F[:, 0] * rng.uniform(0.8, 1.2, n)))
        a = np.full(n, -1)
        S = np.zeros((k, F.shape[1]))
        for gi in order:
            c = []
            for f in range(k):
                S[f] += F[gi]
                c.append(cost(S))
                S[f] -= F[gi]
            f = int(np.argmin(c))
            a[gi] = f
            S[f] += F[gi]
        cur = cost(S)
        improved = True
        while improved:
            improved = False
            for gi in rng.permutation(n):
                fi = a[gi]
                for f in range(k):          # move
                    if f == fi:
                        continue
                    S[fi] -= F[gi]; S[f] += F[gi]
                    c = cost(S)
                    if c < cur - 1e-9:
                        a[gi] = f; cur = c; improved = True; fi = f
                    else:
                        S[f] -= F[gi]; S[fi] += F[gi]
                for gj in (range(n) if swaps else ()):        # swap
                    fj = a[gj]
                    if fj == fi:
                        continue
                    S[fi] += F[gj] - F[gi]; S[fj] += F[gi] - F[gj]
                    c = cost(S)
                    if c < cur - 1e-9:
                        a[gi], a[gj] = fj, fi; cur = c; improved = True; fi = fj
                    else:
                        S[fi] -= F[gj] - F[gi]; S[fj] -= F[gi] - F[gj]
        if cur < best_cost - 1e-9:
            best, best_cost = a.copy(), cur
    return best, best_cost



def orientation_check(n_pairs=80, seed=0, workers=16):
    """Checks the geometry assumptions used for derotation (north up, west right, B0 on, no residual P):
    on train pairs 0.5-3 d apart, compare the derotated correlation of the model against alternatives.
    Needs outputs/splits/analysis/files_all.csv from a full run."""
    a = pd.read_csv(f"{ANA_DIR}/files_all.csv")
    tr = a[a.split == "train"].reset_index(drop=True)
    med = np.median(tr[["limb_cx", "limb_cy", "limb_R"]].values, 0)
    geo = np.where(tr.limb_fallback.values[:, None], med[None], tr[["limb_cx", "limb_cy", "limb_R"]].values)
    t = tr.t_days.values
    b0 = sun_B0(t)
    rng = np.random.default_rng(seed)
    cand = [(i, j) for i in range(len(tr)) for j in range(i + 1, min(len(tr), i + 6)) if 0.5 < t[j] - t[i] < 3]
    pairs = [cand[k] for k in rng.choice(len(cand), n_pairs, replace=False)]
    idx = sorted(set(p for pr in pairs for p in pr))
    with Pool(workers) as p:
        fm = dict(zip(idx, p.starmap(feature_map, [(f"{TRAIN_DIR}/{tr.file_name[i]}", geo[i]) for i in idx])))

    def corr(i, j, dt, alpha=0.0, b0on=True):
        A, ga = fm[i]
        B, gb = fm[j]
        if alpha:
            B = cv2.warpAffine(B, cv2.getRotationMatrix2D((gb[0], gb[1]), alpha, 1.0), (256, 256))
        mx, my, v, _ = derot_map(ga, gb, b0[i] if b0on else 0.0, b0[j] if b0on else 0.0, dt, 256)
        return _corr(A[v], cv2.remap(B, mx, my, cv2.INTER_LINEAR)[v])

    variants = [("model: +omega, B0, no extra rotation", dict()), ("no derotation (dt=0)", dict(dt=0.0)),
                ("reversed rotation (-omega)", dict(sign=-1)), ("B0 ignored", dict(b0on=False))]
    variants += [(f"image B rotated {a_:+d} deg", dict(alpha=a_)) for a_ in (-6, -3, -1, 1, 3, 6)]
    res = {}
    for name, kw in variants:
        vals = []
        for i, j in pairs:
            dt = t[j] - t[i]
            if "dt" in kw:
                dt = kw["dt"]
            if kw.get("sign") == -1:
                dt = -dt
            vals.append(corr(i, j, dt, alpha=kw.get("alpha", 0.0), b0on=kw.get("b0on", True)))
        res[name] = np.array(vals)
    base = res[variants[0][0]]
    tbl = pd.DataFrame([dict(variant=k, median_corr=round(float(np.median(v)), 3),
                             model_better_pct=round(100 * float(np.mean(base > v)), 1), n_pairs=len(v))
                        for k, v in res.items()])
    tbl.to_csv(f"{ANA_DIR}/orientation_check.csv", index=False)
    print(tbl.to_string())

# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=14.0, help="grouping threshold in days")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--restarts", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-dt-pairs", type=float, default=40.0, help="pairs analysed for similarity: |dt| < this")
    ap.add_argument("--n-null", type=int, default=1500)
    ap.add_argument("--skip-image-analysis", action="store_true")
    ap.add_argument("--orientation-check", action="store_true", help="only run the geometry check (after a full run)")
    args = ap.parse_args()
    assert args.workers <= 16
    if args.orientation_check:
        return orientation_check(workers=args.workers)
    os.makedirs(ANA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    t_start = time.time()
    global G_FEAT, G_GEO, G_B0, G_T, G_MASK, G_INST, G_GEO512, G_MASK_DIL

    # ---------------- 1. metadata
    coco, df = load_metadata()
    assert df.datetime.duplicated().sum() == 0, "identical timestamps"
    is_tr = (df.split == "train").values
    is_te = ~is_tr
    print(f"[meta] train files {is_tr.sum()}  test files {is_te.sum()}  image entries {len(coco['images'])} "
          f"annotations {len(coco['annotations'])}")
    yr = pd.crosstab(df.year, df.split)
    st = pd.crosstab(df.site, df.split)
    yr.to_csv(f"{ANA_DIR}/counts_by_year.csv")
    st.to_csv(f"{ANA_DIR}/counts_by_site.csv")
    pd.crosstab(df.year, [df.split, df.site]).to_csv(f"{ANA_DIR}/counts_by_year_site.csv")
    print(yr.T.to_string()); print(st.T.to_string())
    rng_tbl = df.groupby("split").datetime.agg(["min", "max", "count"])
    rng_tbl.to_csv(f"{ANA_DIR}/date_range.csv")
    print(rng_tbl)

    # ---------------- 2. time gaps
    t = df.t_days.values
    gap_tr_tr = nn_gap(t[is_tr], t[is_tr], exclude_self=True)
    gap_te_tr = nn_gap(t[is_te], t[is_tr])
    gap_te_te = nn_gap(t[is_te], t[is_te], exclude_self=True)
    gap_all = nn_gap(t, t, exclude_self=True)
    df["nn_gap_any_d"] = gap_all
    df.loc[is_tr, "nn_gap_train_d"] = gap_tr_tr
    df.loc[is_te, "nn_gap_train_d"] = gap_te_tr
    gt = gap_table({"train->nearest other train": gap_tr_tr, "test->nearest train": gap_te_tr,
                    "test->nearest other test": gap_te_te})
    # near-simultaneous pairs (<1 h) across the whole set
    srt = df.sort_values("t_days")
    close = []
    for a, b in zip(srt.index[:-1], srt.index[1:]):
        if df.t_days[b] - df.t_days[a] < 1 / 24:
            close.append(dict(file_a=df.file_name[a], split_a=df.split[a], file_b=df.file_name[b],
                              split_b=df.split[b], gap_s=round((df.t_days[b] - df.t_days[a]) * 86400)))
    close = pd.DataFrame(close)

    # ---------------- 3. image analysis (exact duplicates, geometry, similarity vs dt)
    paths = [f"{TRAIN_DIR if s == 'train' else TEST_DIR}/{f}" for f, s in zip(df.file_name, df.split)]
    sim = None
    if not args.skip_image_analysis:
        t0 = time.time()
        with Pool(args.workers) as p:
            st_out = p.map(image_stats, paths, chunksize=8)
        df["md5"] = [o[0] for o in st_out]
        geo = np.array([o[1] for o in st_out])
        df[["limb_cx", "limb_cy", "limb_R", "limb_rms", "limb_inlier"]] = geo
        med = np.median(geo[:, :3], 0)
        bad = (np.abs(geo[:, 2] - med[2]) > 5) | (geo[:, 3] > 5) | (np.hypot(geo[:, 0] - med[0], geo[:, 1] - med[1]) > 5)
        df["limb_fallback"] = bad
        geo_use = np.where(bad[:, None], med[None], geo[:, :3])
        print(f"[geom] limb fit median cx,cy,R = {med.round(2)}; R p1/p99 = {np.percentile(geo[~bad, 2], [1, 99]).round(2)}; "
              f"fallback to median geometry for {bad.sum()} file(s): {list(df.file_name[bad])}  ({time.time()-t0:.1f}s)")
        dup = df[df.md5.duplicated(keep=False)]
        print(f"[dup] exact byte-duplicates: {len(dup)} files")
        # (all-pairs near-duplicate check is done below on the background-removed feature maps;
        #  raw thumbnails are dominated by limb-darkening / instrument fixed patterns)
        # feature maps
        t0 = time.time()
        with Pool(args.workers) as p:
            fo = p.map(_feat_worker, [(pth, g) for pth, g in zip(paths, geo_use)], chunksize=8)
        G_FEAT = np.stack([o[0] for o in fo])
        G_GEO = np.array([o[1] for o in fo])
        G_GEO512 = np.array([((g[0] + .5) / 4 - .5, (g[1] + .5) / 4 - .5, g[2] / 4) for g in geo_use])
        G_B0 = sun_B0(t)
        G_T = t
        df["B0_deg"] = np.rad2deg(G_B0)
        print(f"[feat] feature maps {G_FEAT.shape} ({time.time()-t0:.1f}s)")
        # all-pairs, no derotation, feature maps pooled to 128x128 inside 0.9 R: catches duplicates under any name
        fp = G_FEAT.reshape(len(df), 128, 2, 128, 2).mean((2, 4))
        yy, xx = np.mgrid[:128, :128]
        c128 = ((med[0] + .5) / 16 - .5, (med[1] + .5) / 16 - .5, med[2] / 16)
        tmask = np.hypot(xx - c128[0], yy - c128[1]) < 0.9 * c128[2]
        hp = fp[:, tmask]
        hp = (hp - hp.mean(1, keepdims=True)) / (hp.std(1, keepdims=True) + 1e-6)
        C = hp @ hp.T / hp.shape[1]
        iu = np.triu_indices(len(df), 1)
        cv = C[iu]
        gaph = np.abs(t[iu[0]] - t[iu[1]]) * 24
        order = np.argsort(-cv)[:60]
        top = pd.DataFrame(dict(file_a=df.file_name.values[iu[0][order]], split_a=df.split.values[iu[0][order]],
                                file_b=df.file_name.values[iu[1][order]], split_b=df.split.values[iu[1][order]],
                                feat_corr=cv[order].round(4), gap_h=gaph[order].round(3)))
        top.to_csv(f"{ANA_DIR}/top_allpairs_feature_correlations.csv", index=False)
        far = gaph > 24 * 30
        print("[dup] all-pairs feature-map correlation (no derotation): top-12\n" + top.head(12).to_string())
        print(f"[dup] pairs >30 d apart: max {cv[far].max():.3f}, p99.9 {np.quantile(cv[far], .999):.3f}; "
              f"pairs <1 h apart: min {cv[gaph < 1].min():.3f}, median {np.median(cv[gaph < 1]):.3f}")
        dup_summary = dict(max_far=float(cv[far].max()), p999_far=float(np.quantile(cv[far], .999)),
                           min_near=float(cv[gaph < 1].min()), median_near=float(np.median(cv[gaph < 1])))
        pd.Series(dup_summary).to_csv(f"{ANA_DIR}/duplicate_check_summary.csv")
        del C, hp, fp

        # train masks at 1/4 scale (union of all sets) + instance map of the first set; latitude of filaments
        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        G_MASK, G_INST, G_MASK_DIL = [None] * len(df), [None] * len(df), [None] * len(df)
        kdil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))   # +-4 px at 1/4 scale = +-16 px full res
        lat_rows = []
        for idx in np.where(is_tr)[0]:
            ids = df.image_ids[idx].split(";")
            un = np.zeros((512, 512), np.uint8)
            inst = np.zeros((512, 512), np.uint8)
            for si, iid in enumerate(ids):
                for k, a in enumerate(anns_by_img[iid]):
                    assert len(a["segmentation"]) == 1
                    pts = np.asarray(a["segmentation"][0], float).reshape(-1, 2)
                    q = np.round(((pts + 0.5) / 4 - 0.5) * 16).astype(np.int32)
                    cv2.fillPoly(un, [q], 1, lineType=cv2.LINE_8, shift=4)
                    if si == 0:
                        assert k + 1 < 255
                        cv2.fillPoly(inst, [q], k + 1, lineType=cv2.LINE_8, shift=4)
                        cxy = pts.mean(0)
                        lat_rows.append(dict(file_name=df.file_name[idx], image_id=iid, ann=a["id"],
                                             lat_deg=np.rad2deg(heliographic_lat(cxy[0], cxy[1], geo_use[idx], G_B0[idx])),
                                             r_over_R=np.hypot(cxy[0] - geo_use[idx][0], cxy[1] - geo_use[idx][1]) / geo_use[idx][2]))
            G_MASK[idx] = un
            G_INST[idx] = inst
            G_MASK_DIL[idx] = cv2.dilate(un, kdil)
        lat = pd.DataFrame(lat_rows)
        lat.to_csv(f"{ANA_DIR}/filament_latitudes_first_set.csv", index=False)
        alat = lat.lat_deg.abs()
        print(f"[lat] |latitude| of filament centroids (first set, n={len(lat)}): "
              f"p50 {alat.median():.1f}  p90 {alat.quantile(.9):.1f}  p95 {alat.quantile(.95):.1f}  p99 {alat.quantile(.99):.1f} deg")

        # pair list: all pairs with |dt| < max_dt + random null pairs (|dt| > 90 d, train-train)
        n = len(df)
        order_t = np.argsort(t)
        split_arr = df.split.values
        pairs, kinds = [], []
        for a in range(n):
            ia = order_t[a]
            for b in range(a + 1, n):
                ib = order_t[b]
                if t[ib] - t[ia] >= args.max_dt_pairs:
                    break
                ka, kb = split_arr[ia], split_arr[ib]
                kind = "train-train" if ka == kb == "train" else ("test-test" if ka == kb else "test-train")
                if kind == "test-test":
                    continue
                if kind == "test-train" and ka == "train":   # put the test file first so i=test
                    ia_, ib_ = ib, ia
                else:
                    ia_, ib_ = ia, ib
                pairs.append((ia_, ib_)); kinds.append(kind)
        rng = np.random.default_rng(args.seed)
        tr_idx = np.where(is_tr)[0]
        nnull = 0
        while nnull < args.n_null:
            i, j = rng.choice(tr_idx, 2, replace=False)
            if abs(t[j] - t[i]) > 90:
                pairs.append((i, j)); kinds.append("null>90d"); nnull += 1
        t0 = time.time()
        with Pool(args.workers) as p:
            res = p.map(pair_similarity, pairs, chunksize=32)
        sim = pd.DataFrame(res)
        sim["kind"] = kinds
        sim["abs_dt"] = sim.dt.abs()
        sim["file_i"] = df.file_name.values[sim.i]
        sim["file_j"] = df.file_name.values[sim.j]
        sim.drop(columns=["i", "j"]).to_csv(f"{ANA_DIR}/pair_similarity.csv", index=False)
        print(f"[sim] {len(sim)} pairs ({time.time()-t0:.1f}s)")
        sim["i"] = [p_[0] for p_ in pairs]
        sim["j"] = [p_[1] for p_ in pairs]

        # decay table
        rows = []
        null = sim[sim.kind == "null>90d"]
        for lo, hi in zip(DT_BINS[:-1], DT_BINS[1:]):
            for kind in ["train-train", "test-train"]:
                s = sim[(sim.kind == kind) & (sim.abs_dt >= lo) & (sim.abs_dt < hi)]
                if len(s) == 0:
                    continue
                r = dict(kind=kind, dt_lo_d=round(lo, 3), dt_hi_d=hi, n_pairs=len(s), n_pairs_corr_valid=int(s.corr_derot.notna().sum()),
                         corr_derot_median=s.corr_derot.median(), corr_derot_p90=s.corr_derot.quantile(.9),
                         corr_raw_median=s.corr_raw.median(), overlap_median=s.overlap.median())
                if kind == "train-train":
                    r["inst_reobs_rate"] = s.inst_reobs.sum() / max(s.inst_elig.sum(), 1)
                    r["inst_eligible"] = int(s.inst_elig.sum())
                    r["area_reobs_frac"] = s.area_hit.sum() / max(s.area_elig.sum(), 1)
                rows.append(r)
        rows.append(dict(kind="null>90d", dt_lo_d=90, dt_hi_d=np.inf, n_pairs=len(null), n_pairs_corr_valid=int(null.corr_derot.notna().sum()),
                         corr_derot_median=null.corr_derot.median(), corr_derot_p90=null.corr_derot.quantile(.9),
                         corr_raw_median=null.corr_raw.median(), overlap_median=null.overlap.median(),
                         inst_reobs_rate=null.inst_reobs.sum() / max(null.inst_elig.sum(), 1),
                         inst_eligible=int(null.inst_elig.sum()),
                         area_reobs_frac=null.area_hit.sum() / max(null.area_elig.sum(), 1)))
        decay = pd.DataFrame(rows)
        decay.to_csv(f"{ANA_DIR}/similarity_vs_dt.csv", index=False)
        print(decay.round(3).to_string())
        null_p95 = null.corr_derot.quantile(.95)
        null_p99 = null.corr_derot.quantile(.99)
        print(f"[sim] null derotated corr p95 {null_p95:.3f} p99 {null_p99:.3f}")

    # ---------------- 4. grouping
    tr = df[is_tr].copy().reset_index(drop=True)
    comp_rows = []
    for T in [0.25, 1, 2, 3, 5, 7, 10, 12, 14, 15, 16, 17, 18, 20, 27]:
        c = time_components(tr.t_days.values, T)
        sz = np.bincount(c)
        comp_rows.append(dict(T_days=T, n_groups=len(sz), max_group=int(sz.max()),
                              max_group_frac=round(sz.max() / len(tr), 3),
                              p90_group=float(np.quantile(sz, .9)), singletons=int((sz == 1).sum())))
    comp_tbl = pd.DataFrame(comp_rows)
    comp_tbl.to_csv(f"{ANA_DIR}/components_vs_T.csv", index=False)
    print(comp_tbl.to_string())
    tr["group_id"] = time_components(tr.t_days.values, args.T)
    gsz = tr.groupby("group_id").size()
    gspan = tr.groupby("group_id").t_days.agg(lambda x: x.max() - x.min())
    print(f"[group] T={args.T} d: {len(gsz)} groups; size p50 {gsz.median()} p90 {gsz.quantile(.9)} max {gsz.max()}; "
          f"span max {gspan.max():.1f} d")
    max_frac = gsz.max() / len(tr)
    assert max_frac <= 1.0 / args.n_folds, f"largest group {max_frac:.2f} exceeds one fold"

    # ---------------- 5. folds (leakage-free, grouped) + diagnostic file-level folds
    gids, F, names, w = group_features(tr, "group_id")
    a, cst = assign_folds(F, w, args.n_folds, args.seed, args.restarts)
    # relabel folds by earliest member time for a stable, readable ordering
    g2f = dict(zip(gids, a))
    tr["fold"] = tr.group_id.map(g2f)
    first = tr.groupby("fold").t_days.min().sort_values()
    remap = {old: new for new, old in enumerate(first.index)}
    tr["fold"] = tr.fold.map(remap)
    print(f"[fold] grouped assignment cost {cst:.2f}")

    tr["file_group"] = np.arange(len(tr))
    gids2, F2, _, _ = group_features(tr, "file_group")
    a2, cst2 = assign_folds(F2, w, args.n_folds, args.seed + 1, restarts=5, swaps=False)
    tr["diag_fold_filelevel"] = a2
    print(f"[fold] diagnostic file-level assignment cost {cst2:.2f}")

    # ---------------- 6. assertions (leakage-free property)
    assert tr.file_name.is_unique
    all_ids = [i for s in tr.image_ids for i in s.split(";")]
    assert len(all_ids) == len(set(all_ids)) == len(coco["images"]), "image id coverage"
    assert set(all_ids) == set(im["id"] for im in coco["images"])
    assert (tr.groupby("group_id").fold.nunique() == 1).all(), "a group spans >1 fold"
    assert (tr.groupby("file_name").fold.nunique() == 1).all(), "a file spans >1 fold"
    assert tr.fold.between(0, args.n_folds - 1).all() and tr.fold.nunique() == args.n_folds
    val_gap = np.empty(len(tr))
    val_gap_diag = np.empty(len(tr))
    for f in range(args.n_folds):
        m = (tr.fold == f).values
        val_gap[m] = nn_gap(tr.t_days.values[m], tr.t_days.values[~m])
        m2 = (tr.diag_fold_filelevel == f).values
        val_gap_diag[m2] = nn_gap(tr.t_days.values[m2], tr.t_days.values[~m2])
    assert val_gap.min() >= args.T, f"val file within T of a training file: {val_gap.min()}"
    tr["val_gap_to_train_d"] = val_gap
    tr["diag_val_gap_to_train_d"] = val_gap_diag
    print(f"[assert] OK: no file/group spans 2 folds; min val->train gap {val_gap.min():.2f} d >= T={args.T}")

    # ---------------- 7. balance + gap comparison tables
    bal = tr.groupby("fold").agg(files=("file_name", "size"), groups=("group_id", "nunique"),
                                 sets=("n_sets", "sum"), instances=("n_instances_total", "sum"),
                                 first=("datetime", "min"), last=("datetime", "max"),
                                 val_gap_min_d=("val_gap_to_train_d", "min"),
                                 val_gap_median_d=("val_gap_to_train_d", "median"))
    bal["inst_per_set"] = (bal.instances / bal.sets).round(2)
    bal = bal.join(pd.crosstab(tr.fold, tr.site)).join(pd.crosstab(tr.fold, tr.n_sets).add_prefix("nsets_"))
    bal.to_csv(f"{ANA_DIR}/fold_balance.csv")
    pd.crosstab(tr.year, tr.fold).to_csv(f"{ANA_DIR}/fold_by_year.csv")
    print(bal.to_string()); print(pd.crosstab(tr.year, tr.fold).T.to_string())
    gt2 = gap_table({"test->nearest train (official)": gap_te_tr,
                     f"CV v1 (T={args.T:g}d) val->nearest train-fold": val_gap,
                     "diag file-level CV val->nearest train-fold": val_gap_diag,
                     "train->nearest other train": gap_tr_tr,
                     "test->nearest other test": gap_te_te})
    gt2.to_csv(f"{ANA_DIR}/gap_comparison.csv", index=False)
    print(gt2[["set", "n", "min_h", "median_d", "p90_d"] + GAP_THRESH_LBL].to_string())
    gsizes = pd.DataFrame(dict(group_id=gsz.index, n_files=gsz.values, span_d=gspan.values,
                               fold=tr.groupby("group_id").fold.first().values,
                               first=tr.groupby("group_id").datetime.min().values,
                               last=tr.groupby("group_id").datetime.max().values))
    gsizes.to_csv(f"{ANA_DIR}/groups_T{args.T:g}.csv", index=False)
    if len(close):
        close.to_csv(f"{ANA_DIR}/near_simultaneous_pairs.csv", index=False)

    # content similarity of each held-out file to its training data: test vs CV
    if sim is not None:
        f_of = dict(zip(tr.file_name, tr.fold))
        fd_of = dict(zip(tr.file_name, tr.diag_fold_filelevel))
        rows = []
        tt = sim[sim.kind.isin(["train-train", "test-train"])]
        # symmetrise train-train pairs
        both = pd.concat([tt[["file_i", "file_j", "corr_derot", "abs_dt", "kind"]],
                          tt[tt.kind == "train-train"].rename(columns={"file_i": "file_j", "file_j": "file_i"})
                          [["file_i", "file_j", "corr_derot", "abs_dt", "kind"]]])
        grp = dict(tuple(both.groupby("file_i")))
        empty = both.iloc[:0]
        for fi in df.file_name:
            g = grp.get(fi, empty)
            if fi in f_of:
                gv = g[g.kind == "train-train"]
                m_cv = gv[gv.file_j.map(f_of) != f_of[fi]]
                m_dg = gv[gv.file_j.map(fd_of) != fd_of[fi]]
                rows.append(dict(file=fi, set="CV v1 val", max_corr=m_cv.corr_derot.max()))
                rows.append(dict(file=fi, set="diag file-level CV val", max_corr=m_dg.corr_derot.max()))
            else:
                rows.append(dict(file=fi, set="test (official)", max_corr=g.corr_derot.max()))
        mc = pd.DataFrame(rows)
        mc["no_train_within_max_dt"] = mc.max_corr.isna()
        mc["max_corr"] = mc.max_corr.fillna(0.0)   # no training file within max_dt (or no co-visible area) -> 0
        mc.to_csv(f"{ANA_DIR}/max_similarity_to_training.csv", index=False)
        null = sim[sim.kind == "null>90d"]
        thr = null.corr_derot.quantile(.99)
        summ = mc.groupby("set").max_corr.describe(percentiles=[.25, .5, .75, .9]).round(3)
        summ[f"frac_above_null_p99({thr:.3f})"] = mc.groupby("set").max_corr.apply(lambda x: round((x > thr).mean(), 3))
        summ["frac_above_0.3"] = mc.groupby("set").max_corr.apply(lambda x: round((x > 0.3).mean(), 3))
        summ.to_csv(f"{ANA_DIR}/max_similarity_summary.csv")
        print(summ.to_string())

    # ---------------- 8. write split
    out = tr[["file_name", "image_ids", "n_sets", "n_instances_total", "datetime", "site", "group_id", "fold",
              "year", "n_instances_per_set", "batch_groups", "val_gap_to_train_d", "diag_fold_filelevel"]].copy()
    out["datetime"] = out.datetime.dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out.sort_values("datetime").reset_index(drop=True)
    out.to_csv(f"{OUT_DIR}/folds_v1.csv", index=False)
    # re-read and re-check the written file
    chk = pd.read_csv(f"{OUT_DIR}/folds_v1.csv")
    assert (chk.groupby("group_id").fold.nunique() == 1).all() and chk.file_name.is_unique
    assert len(chk) == len(tr) and chk.n_sets.sum() == len(coco["images"])
    if not args.skip_image_analysis:      # time-only runs must not clobber the geometry columns
        df.to_csv(f"{ANA_DIR}/files_all.csv", index=False)
    gt.to_csv(f"{ANA_DIR}/gap_nearest.csv", index=False)
    print(f"[done] wrote {OUT_DIR}/folds_v1.csv ({len(chk)} files)  total {time.time()-t_start:.0f}s")

    # ---------------- 9. figures
    try:
        make_figures(df, tr, gap_te_tr, val_gap, val_gap_diag, sim, args)
    except Exception as e:  # figures are auxiliary
        print("[fig] failed:", repr(e))


def make_figures(df, tr, gap_te_tr, val_gap, val_gap_diag, sim, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    # (a) timeline: folds as rows, test as its own row
    fig, ax = plt.subplots(figsize=(13, 3.6))
    for f in range(args.n_folds):
        s = tr[tr.fold == f]
        ax.scatter(s.datetime, np.full(len(s), f), s=9, color=c1, linewidths=0)
    te = df[df.split == "test"]
    ax.scatter(te.datetime, np.full(len(te), args.n_folds + 0.8), s=9, color=c2, linewidths=0)
    ax.set_yticks(list(range(args.n_folds)) + [args.n_folds + 0.8])
    ax.set_yticklabels([f"fold {f} (val)" for f in range(args.n_folds)] + ["test"])
    ax.grid(axis="x", color=grid, lw=0.8)
    ax.set_title(f"Train files by CV fold (time groups, T={args.T:g} d) and test files", loc="left", color=ink)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/split_timeline.png", dpi=130); plt.close(fig)

    # (b) gap CDF: test->train vs CV val->train-fold
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for g, lab, col in [(gap_te_tr, "test → nearest train (official split)", c2),
                        (val_gap, f"CV v1 val → nearest train-fold (T={args.T:g} d)", c1),
                        (val_gap_diag, "diagnostic file-level CV val → train-fold", c3)]:
        x = np.sort(np.maximum(g, 1e-4))
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col, lw=2, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("time gap to nearest training observation (days, log)")
    ax.set_ylabel("cumulative fraction of held-out files")
    for v, lb in [(1 / 24, "1 h"), (1, "1 d"), (args.T, f"T={args.T:g} d")]:
        ax.axvline(v, color=grid, lw=1, zorder=0); ax.text(v, 1.02, lb, color=ink2, ha="center", fontsize=8)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(axis="y", color=grid, lw=0.8)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/split_gap_cdf.png", dpi=130); plt.close(fig)

    if sim is None:
        return
    # (c) similarity decay vs dt (two panels, one measure each)
    null = sim[sim.kind == "null>90d"]
    edges = np.array(DT_BINS)
    mids = np.sqrt(np.maximum(edges[:-1], 1 / 96) * edges[1:])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axs[0]
    ax.axhspan(null.corr_derot.quantile(.05), null.corr_derot.quantile(.95), color=grid, zorder=0)
    ax.text(mids[0], null.corr_derot.quantile(.95) + 0.015, "null: pairs >90 d apart, 5–95%", color=ink2, fontsize=8)
    for kind, col, lab in [("train-train", c1, "train–train pairs"), ("test-train", c2, "test–train pairs")]:
        tt = sim[sim.kind == kind]
        med, q1, q3 = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = tt[(tt.abs_dt >= lo) & (tt.abs_dt < hi)]
            ok = len(s) > 0 and s.corr_derot.notna().mean() >= 0.5   # bins dominated by no-overlap pairs are masked
            med.append(s.corr_derot.median() if ok else np.nan)
            q1.append(s.corr_derot.quantile(.25) if ok else np.nan)
            q3.append(s.corr_derot.quantile(.75) if ok else np.nan)
        if kind == "train-train":
            ax.fill_between(mids, q1, q3, color=col, alpha=0.15, lw=0)
        ax.plot(mids, med, color=col, lw=2, marker="o", ms=4, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("|Δt| between observations (days, log)")
    ax.set_ylabel("correlation of background-removed maps")
    ax.set_title("Image similarity after differential-rotation compensation\n(median; band = train–train IQR; "
                 "10–14 d bin masked: <50% of pairs co-visible)", loc="left", fontsize=10)
    ax.axvline(args.T, color=ink2, lw=1, ls="--"); ax.text(args.T * 1.05, 0.85, f"T={args.T:g} d", color=ink2, fontsize=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.92)); ax.grid(axis="y", color=grid, lw=0.8)
    ax = axs[1]
    tt = sim[sim.kind == "train-train"]
    rate = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = tt[(tt.abs_dt >= lo) & (tt.abs_dt < hi)]
        rate.append(s.inst_reobs.sum() / s.inst_elig.sum() if len(s) and s.inst_elig.sum() >= 50 else np.nan)
    nr = null.inst_reobs.sum() / max(null.inst_elig.sum(), 1)
    ax.plot(mids, rate, color=c1, lw=2, marker="o", ms=4)
    ax.axhline(nr, color=ink2, lw=1, ls=":"); ax.text(mids[0], nr + 0.015, f"chance level (pairs >90 d apart) {nr:.3f}", color=ink2, fontsize=8)
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("|Δt| between train observations (days, log)")
    ax.set_ylabel("re-found fraction of co-visible filaments")
    ax.set_title("Annotated filaments (first set) re-found at the derotated position\nin the other file's annotation "
                 "(±16 px; bins with <50 eligible masked)", loc="left", fontsize=10)
    ax.axvline(args.T, color=ink2, lw=1, ls="--"); ax.grid(axis="y", color=grid, lw=0.8)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/split_similarity_vs_dt.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
