"""Experiment SE (small-candidate existence scoring) - small input checks before any run. CPU only.
  CUDA_VISIBLE_DEVICES="" TMPDIR=$PWD/tmp taskset -c 0-3 .venv/bin/python tests/test_small_exist.py
(a) candidates are exactly the 100-499 px components of p > 0.5; the base mask is the >= 500 px components;
(b) the target is the mean over sets of (best IoU if > 0.5 else 0) - a candidate drawn by 1 of 2 sets gets 0.5 x IoU;
(c) the decision rule: adding a disjoint component raises pooled V0 iff y > PQ / 2 (checked with evaluate_image);
(d) add-only never changes base instances and never creates overlap;
(e) nested sources: outer fold k is never in the training files of the stage-1 runs used for its training candidates,
    and its training candidates come only from folds j != k;
(f) real-data smoke: one image of bens_f0 gives crops (N, 4, 96, 96) whose candidate channel is non-empty;
(g) the scorer runs forward on CPU.
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import small_exist as SE  # noqa: E402
from filseg.metrics import InstanceSet  # noqa: E402

H = SE.HW


def blob(y, x, h, w):
    m = np.zeros(H, bool); m[y:y + h, x:x + w] = True
    return m


def test_a_candidates():
    p = np.zeros(H, np.float32)
    p[blob(100, 100, 5, 10)] = 0.9          # 50 px  -> ignored
    p[blob(300, 300, 10, 15)] = 0.9         # 150 px -> candidate
    p[blob(600, 600, 20, 30)] = 0.9         # 600 px -> base
    p[blob(900, 900, 10, 15)] = 0.4         # below the threshold
    lab, st, cand, base = SE.components(p)
    assert [int(st[k, 4]) for k in cand] == [150]
    assert base.sum() == 600 and not base[300:310, 300:315].any()


def test_b_target():
    c = blob(300, 300, 10, 15)
    C = InstanceSet.from_masks([c])
    s1 = InstanceSet.from_masks([c, blob(50, 50, 10, 10)])     # draws the candidate exactly
    s2 = InstanceSet.from_masks([blob(800, 800, 10, 10)])       # does not
    y, tp_any, best = SE.targets(C, [s1, s2])
    assert np.isclose(y[0], 0.5) and tp_any[0] and np.isclose(best[0], 1.0)
    s3 = InstanceSet.from_masks([blob(300, 300, 10, 40)])       # IoU 150/400 < 0.5 -> no credit
    y, tp_any, _ = SE.targets(C, [s3])
    assert y[0] == 0 and not tp_any[0]


def test_c_decision_rule():
    from filseg.metrics import evaluate_image
    import refine_b0exist_checks as RC
    rng = np.random.default_rng(0)
    for trial in range(3):
        G = InstanceSet.from_masks([blob(100 * i + 50, 200, 20, 60) for i in range(8)])
        G2 = InstanceSet.from_masks([blob(100 * i + 50, 200, 20, 60) for i in range(5)] + [blob(1500, 1500, 12, 12)])
        P = InstanceSet.from_masks([blob(100 * i + 52, 205, 18, 50) for i in range(6)] + [blob(1800, 100, 30, 30)])

        def pq(P_):
            s = [evaluate_image(G_, P_, RC.CFG, image_key="x")[0] for G_ in (G, G2)]
            n = sum(r["sum_iou_host"] for r in s); d = sum(r["tp_host"] + .5 * r["fp_host"] + .5 * r["fn_host"] for r in s)
            return n / d
        base = pq(P)
        cand = blob(1500, 1500, 12, 12 + trial)            # matches G2's extra instance only
        C = InstanceSet.from_masks([cand])
        y = SE.targets(C, [G, G2])[0][0]
        P2, added = SE.add_only(P, C, [0])
        assert added == [0]
        assert (pq(P2) > base) == (y > base / 2), (trial, y, base, pq(P2))


def test_d_add_only():
    P = InstanceSet.from_masks([blob(100, 100, 30, 30)])
    C = InstanceSet.from_masks([blob(120, 120, 20, 20), blob(500, 500, 12, 12)])   # first overlaps the base
    Q, added = SE.add_only(P, C, [0, 1])
    assert added == [1] and len(Q) == 2
    assert np.array_equal(Q.full_mask(0), P.full_mask(0))
    assert sum(Q.full_mask(i).astype(int) for i in range(len(Q))).max() == 1


def test_e_nested_sources():
    sp = pd.read_csv(ROOT / "outputs" / "splits" / "folds_v1.csv")
    for k in range(5):
        kfiles = set(sp[sp.fold == k].file_name)
        for name in SE.outer_train_sources(k):
            a, b, j = int(name[7]), int(name[8]), int(name[-1])
            assert j != k and k in (a, b) and j in (a, b)
            tr = set((ROOT / "outputs" / "exps" / "b0n_r101_s1024" / f"hold{a}{b}" / "train_files.txt").read_text().split())
            assert not tr & kfiles, (k, name)
        ens_cfg = (ROOT / "outputs" / "exps" / "b0n_ens" / f"fold{k}" / "val_files.txt").read_text().split()
        assert set(ens_cfg) == kfiles


def test_f_real_smoke():
    from filseg import data as D
    from filseg import seg
    fn = sorted(SE.fold_files(0))[3]
    p = np.load(ROOT / "outputs" / "exps" / "b0n_ens" / "fold0" / "val_probs_last.npz")[fn[:-5]].astype(np.float32)
    img = D.read_image(D.train_image_path(fn))
    sets = [g for _, g in sorted((r, g) for r, g in seg.gt_records([fn]).items())]
    X, rows, C = SE.extract(p, img, sets)
    assert X.shape[1:] == (4, 96, 96) and len(rows) == len(X) == len(C)
    for i in range(len(X)):
        assert X[i, 2].max() == 255
        assert 100 <= rows[i]["area"] < 500


def test_g_scorer_forward():
    import torch
    net = SE.build_scorer().eval()
    with torch.no_grad():
        out = net(SE.to_input(np.zeros((2, 4, 96, 96), np.uint8)))
    assert out.shape == (2, 1)
    n = sum(p.numel() for p in net.parameters())
    assert n < 2_000_000, n


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
