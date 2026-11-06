"""Dense reference for boundary-F and thin-region diagnostics (scipy EDT, full-image morphology)."""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repository root (path-independent)
import sys, math, numpy as np, cv2
from scipy import ndimage
sys.path.insert(0, _REPO + "/src"); sys.path.insert(0, ".")
cv2.setNumThreads(1)
from filseg.metrics import evaluate_image, EvalConfig
import synth_compare as sc
CROSS = np.array([[0,1,0],[1,1,1],[0,1,0]], bool)
def inner_b(m):
    p = np.pad(m, 1); er = ndimage.binary_erosion(p, CROSS, border_value=0)
    return (p & ~er)[1:-1, 1:-1]
def thin(m, r):
    pad = r + 1; p = np.pad(m.astype(np.uint8), pad)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1)).astype(bool)
    op = ndimage.binary_opening(p.astype(bool), el, border_value=0)
    return (p.astype(bool) & ~op)[pad:-pad, pad:-pad]
def within(t, d):
    if not t.any(): return np.zeros_like(t)
    return ndimage.distance_transform_edt(~t) <= d
rng = np.random.default_rng(5); cases = sc.make_cases(600, seed=99)
cfg = EvalConfig(); bad = []; n = 0
for key, shape, gts, preds in cases:
    if min(shape) < 3: continue
    row, pg, pp = evaluate_image(gts, preds, cfg, shape=shape); n += 1
    bg = np.zeros(shape, bool); bp = np.zeros(shape, bool)
    for m in gts: bg |= inner_b(m)
    for m in preds: bp |= inner_b(m)
    ng_, np_ = int(bg.sum()), int(bp.sum())
    if ng_ or np_:
        P = (bp & within(bg, 2)).sum() / np_ if (np_ and ng_) else 0.0
        Rr = (bg & within(bp, 2)).sum() / ng_ if (np_ and ng_) else 0.0
        F = 2*P*Rr/(P+Rr) if P + Rr > 0 else 0.0
        if not abs(F - row["boundary_f"]) <= 1e-12: bad.append((key, "bf", F, row["boundary_f"]))
    tg = np.zeros(shape, bool); tp = np.zeros(shape, bool)
    ug = np.zeros(shape, bool); up = np.zeros(shape, bool)
    for m in gts: tg |= thin(m, 3); ug |= m
    for m in preds: tp |= thin(m, 3); up |= m
    rec = (tg & up).sum() / tg.sum() if tg.sum() else float("nan")
    prec = (tp & ug).sum() / tp.sum() if tp.sum() else float("nan")
    for nm, a, b in (("thin_rec", rec, row["thin_region_recall"]), ("thin_prec", prec, row["thin_region_precision"])):
        if not ((math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12): bad.append((key, nm, a, b))
    a_g, a_p, a_i = int(ug.sum()), int(up.sum()), int((ug & up).sum())
    sd = 2*a_i/(a_g+a_p) if a_g + a_p else float("nan")
    if not ((math.isnan(sd) and math.isnan(row["semantic_dice"])) or abs(sd - row["semantic_dice"]) <= 1e-12): bad.append((key, "sdice", sd, row["semantic_dice"]))
print("cases", n, "mismatches", len(bad), bad[:5])
