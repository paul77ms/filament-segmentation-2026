"""Scorer variants and organiser diagnostics (docs/organiser_perspective.md, docs/improvement_plan.md §1).

V0 = the official self-evaluation notebook: one record per (annotation set, image), every IoU > 0.5 pair is a TP,
     PQ pooled over records (identical to filseg.metrics host counts; use evaluate_image for V0).
V1 = scorer hypothesis that reproduces both public LB scores (D-044): one record per IMAGE whose GT is the
     concatenation of all annotation sets (overlapping copies kept), predictions matched ONE-TO-ONE (maximum total
     IoU among pairs with IoU > 0.5), PQ pooled over images.
Notebook diagnostics (cell 5 of the official notebook): overlap = intersection > 0 (1 px is enough); per record
     the number of GT touched by >= 2 predictions (1:n), predictions touching >= 2 GT (n:1), and the organisers'
     over-/under-fragmentation sums (Bitbucket mleco code): sum over GT of (N-1) for N >= 2 touching predictions,
     sum over predictions of (N-1) for N >= 2 touching GT; pairwise IoU/Dice of all touching pairs.
"""
from __future__ import annotations

import numpy as np

from .metrics import InstanceSet, intersection_matrix


def iou_matrix(G: InstanceSet, P: InstanceSet):
    """(intersection, IoU) matrices, shape (len(G), len(P))."""
    if not len(G) or not len(P):
        z = np.zeros((len(G), len(P)))
        return z, z
    inter = intersection_matrix(G, P).astype(np.float64)
    return inter, inter / np.maximum(G.areas[:, None] + P.areas[None, :] - inter, 1)


def v1_counts(sets: list, P: InstanceSet):
    """V1 counts for one image: sets = the image's annotation sets (InstanceSet each). -> (tp, fp, fn, sum_iou)."""
    from scipy.optimize import linear_sum_assignment
    mats = [iou_matrix(G, P)[1] for G in sets]
    M = np.concatenate(mats, 0) if mats else np.zeros((0, len(P)))
    n_gt, n_p = M.shape
    if not n_gt or not n_p:
        return 0, n_p, n_gt, 0.0
    W = np.where(M > 0.5, M, 0.0)
    r, c = linear_sum_assignment(-W)
    ok = W[r, c] > 0.5
    tp = int(ok.sum())
    return tp, n_p - tp, n_gt - tp, float(W[r, c][ok].sum())


def notebook_diag(G: InstanceSet, P: InstanceSet):
    """Overlap diagnostics of one record (official notebook cell 5 + organisers' fragmentation counts)."""
    inter, iou = iou_matrix(G, P)
    touch = inter > 0
    ng = touch.sum(1) if len(P) else np.zeros(len(G), int)
    npr = touch.sum(0) if len(G) else np.zeros(len(P), int)
    pair_iou = iou[touch]
    pair_dice = 2 * inter[touch] / np.maximum((G.areas[:, None] + P.areas[None, :])[touch], 1) if touch.any() else np.zeros(0)
    gu = G.union() if len(G) else None
    pu = P.union() if len(P) else None
    if gu is not None and pu is not None:
        si, sg, sp = int((gu & pu).sum()), int(gu.sum()), int(pu.sum())
    else:
        si, sg, sp = 0, int(gu.sum()) if gu is not None else 0, int(pu.sum()) if pu is not None else 0
    return dict(nb_gt_1to0=int((ng == 0).sum()), nb_gt_1to1=int((ng == 1).sum()), nb_gt_1toN=int((ng >= 2).sum()),
                nb_pred_0to1=int((npr == 0).sum()), nb_pred_1to1=int((npr == 1).sum()), nb_pred_Nto1=int((npr >= 2).sum()),
                over_frag=int(np.maximum(ng - 1, 0).sum()), under_frag=int(np.maximum(npr - 1, 0).sum()),
                pair_n=int(touch.sum()), pair_iou_sum=float(pair_iou.sum()), pair_dice_sum=float(pair_dice.sum()),
                sem_inter=si, sem_gt=sg, sem_pred=sp)
