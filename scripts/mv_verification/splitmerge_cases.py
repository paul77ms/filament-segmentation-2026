import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repository root (path-independent)
import sys, numpy as np
sys.path.insert(0, _REPO + "/src")
from filseg.metrics import evaluate_image, EvalConfig
S = (256, 256)
def R(y0, x0, y1, x1):
    m = np.zeros(S, bool); m[y0:y1, x0:x1] = True; return m
def show(name, g, p, cfg=EvalConfig()):
    r, pg, pp = evaluate_image(g, p, cfg, shape=S)
    keys = ["tp","fp","fn","n_split_gt","n_split_fragments","n_merge_pred","n_merged_gt","n_missed_gt","n_spurious_pred"]
    fn = {k[3:]: r[k] for k in r if k.startswith("fn_") and r[k]}; fp = {k[3:]: r[k] for k in r if k.startswith("fp_") and r[k]}
    print(f"{name:58s}", {k: r[k] for k in keys}, "FN:", fn, "FP:", fp, "best_iou_gt:", [round(x["best_iou"], 3) for x in pg])
    return r
g = R(100, 20, 103, 220)                       # thin GT 3 x 200 = 600 px
show("A thin GT split in 2 by exact-width preds", [g], [R(100, 20, 103, 120), R(100, 120, 103, 220)])
show("B thin GT split in 2 by preds 3x wider (fat pred)", [g], [R(97, 20, 106, 120), R(97, 120, 106, 220)])
show("C thin GT split in 2 by preds 2x wider", [g], [R(98, 20, 104, 120), R(98, 120, 104, 220)])
show("D thin GT split in 3 (60/70/70 px long)", [g], [R(100, 20, 103, 80), R(100, 80, 103, 150), R(100, 150, 103, 220)])
show("E thin GT shattered into 40 pieces of 3x5=15 px", [g], [R(100, 20 + 5 * k, 103, 25 + 5 * k) for k in range(40)])
show("F thin GT shattered into 67 pieces of 3x3=9 px (<min_px)", [g], [R(100, 20 + 3 * k, 103, 23 + 3 * k) for k in range(66)])
g1, g2 = R(100, 20, 103, 120), R(100, 124, 103, 220)
show("G two thin GTs (4 px gap) merged by exact pred", [g1, g2], [R(100, 20, 103, 220)])
show("H two thin GTs merged by 3x-wide pred", [g1, g2], [R(97, 20, 106, 220)])
show("I big GT + 9-px GT merged (small GT < min_px)", [R(50, 50, 80, 150), R(82, 60, 85, 63)], [R(50, 50, 85, 150)])
show("J pred covers GT fully + 49% of 2nd GT", [g1, R(100, 124, 103, 224)], [R(100, 20, 103, 173)])
show("K over-dilated single pred (3x area)", [g], [R(99, 20, 105, 220) | R(98, 20, 99, 220) | R(105, 20, 106, 220)])
show("L GT1 split; one fragment also swallows GT2", [R(100, 20, 103, 220), R(110, 120, 113, 160)], [R(100, 20, 103, 120), R(100, 120, 113, 220)])
show("M crossing: GT1,GT2 each split, middle halves merged", [R(100, 20, 103, 120), R(100, 130, 103, 230)], [R(100, 20, 103, 70), R(100, 70, 103, 180), R(100, 180, 103, 230)])
show("N duplicate preds of the same GT", [g], [g.copy(), g.copy()])
show("O pred exactly half of GT (IoU 0.5) + rest missing", [g], [R(100, 20, 103, 120)])
show("P empty pred mask", [g], [np.zeros(S, bool)])
