"""Deliberately simple brute-force PQ reference, written independently of filseg.metrics.

- instances are python frozensets of flat pixel indices (no bboxes, no crops, no numpy matmul)
- IoU = len(g & p) / (len(g) + len(p) - len(g & p)); 0 if union == 0
- candidate iff IoU > 0.5 (strict)
- 'unique' (standard PQ): one-to-one matching restricted to candidate edges that maximises the
  total IoU, found by exhaustive enumeration inside each connected component of the candidate
  graph (exponential, fine for small components)
- 'host': every candidate pair is a TP; FP = preds with no candidate; FN = gts with no candidate
"""
from __future__ import annotations

import math

import numpy as np


def pixset(m) -> frozenset:
    return frozenset(np.flatnonzero(np.asarray(m).ravel() != 0).tolist())


def iou_table(G, P):
    T = []
    for g in G:
        row = []
        for p in P:
            inter = len(g & p)
            union = len(g) + len(p) - inter
            row.append(inter / union if union > 0 else 0.0)
        T.append(row)
    return T


def _components(edges, ng):
    """edges: list of (i, j). Returns list of edge lists, one per connected component."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for i, j in edges:
        a, b = find(("g", i)), find(("p", j))
        if a != b:
            parent[a] = b
    comp = {}
    for i, j in edges:
        comp.setdefault(find(("g", i)), []).append((i, j))
    return list(comp.values())


def _enumerate_best(comp_edges, T):
    """Exhaustive search over all matchings (sets of vertex-disjoint edges) of one component.
    Returns (best_sum, best_matching, set of cardinalities attaining best_sum)."""
    best = [-1.0, [], set()]
    n = len(comp_edges)
    leaves = [0]

    def rec(k, used_g, used_p, cur, s):
        if k == n:
            leaves[0] += 1
            if leaves[0] > 5_000_000:
                raise RuntimeError("too many matchings for brute force")
            if s > best[0] + 1e-12:
                best[0], best[1], best[2] = s, list(cur), {len(cur)}
            elif abs(s - best[0]) <= 1e-12:
                best[2].add(len(cur))
            return
        i, j = comp_edges[k]
        rec(k + 1, used_g, used_p, cur, s)                      # skip edge k
        if i not in used_g and j not in used_p:                 # take edge k
            used_g.add(i); used_p.add(j); cur.append((i, j))
            rec(k + 1, used_g, used_p, cur, s + T[i][j])
            cur.pop(); used_g.discard(i); used_p.discard(j)

    rec(0, set(), set(), [], 0.0)
    return best[0], best[1], best[2]


def bf_record(G, P, thr=0.5):
    """G, P: lists of pixel sets. Returns dict of per-record terms for both conventions."""
    ng, npr = len(G), len(P)
    T = iou_table(G, P)
    edges = [(i, j) for i in range(ng) for j in range(npr) if T[i][j] > thr]
    # ---- unique
    pairs, cards_ambiguous = [], False
    for comp in _components(edges, ng):
        s, m, cards = _enumerate_best(comp, T)
        pairs += m
        cards_ambiguous |= len(cards) > 1
    tp = len(pairs)
    s_iou = 0.0
    for i, j in sorted(pairs):
        s_iou += T[i][j]
    fp, fn = npr - tp, ng - tp
    den = tp + 0.5 * fp + 0.5 * fn
    # ---- host
    tp_h = len(edges)
    s_h = 0.0
    for i, j in edges:
        s_h += T[i][j]
    fp_h = sum(1 for j in range(npr) if not any(T[i][j] > thr for i in range(ng)))
    fn_h = sum(1 for i in range(ng) if not any(T[i][j] > thr for j in range(npr)))
    den_h = tp_h + 0.5 * fp_h + 0.5 * fn_h
    nan = float("nan")
    n_half = sum(1 for i in range(ng) for j in range(npr) if T[i][j] == 0.5)
    return dict(n_gt=ng, n_pred=npr, tp=tp, fp=fp, fn=fn, sum_iou=s_iou,
                pq=(s_iou / den if den > 0 else nan), pq_empty1=(s_iou / den if den > 0 else 1.0),
                sq=(s_iou / tp if tp else nan), rq=(tp / den if den > 0 else nan),
                tp_host=tp_h, fp_host=fp_h, fn_host=fn_h, sum_iou_host=s_h,
                pq_host=(s_h / den_h if den_h > 0 else nan),
                card_tie=cards_ambiguous, n_iou_exact_half=n_half, n_cand=len(edges),
                iou=T)


def bf_aggregate(recs):
    S = lambda k: sum(r[k] for r in recs)
    den = S("tp") + 0.5 * S("fp") + 0.5 * S("fn")
    den_h = S("tp_host") + 0.5 * S("fp_host") + 0.5 * S("fn_host")
    nan = float("nan")
    pqs = [r["pq"] for r in recs if not math.isnan(r["pq"])]
    pqh = [r["pq_host"] for r in recs if not math.isnan(r["pq_host"])]
    return dict(PQ_pooled=(S("sum_iou") / den if den else nan),
                SQ_pooled=(S("sum_iou") / S("tp") if S("tp") else nan),
                RQ_pooled=(S("tp") / den if den else nan),
                PQ_pooled_host=(S("sum_iou_host") / den_h if den_h else nan),
                SQ_pooled_host=(S("sum_iou_host") / S("tp_host") if S("tp_host") else nan),
                RQ_pooled_host=(S("tp_host") / den_h if den_h else nan),
                PQ_img_mean=(sum(pqs) / len(pqs) if pqs else nan),
                PQ_img_mean_empty1=(sum(r["pq_empty1"] for r in recs) / len(recs) if recs else nan),
                PQ_img_mean_host=(sum(pqh) / len(pqh) if pqh else nan),
                TP=S("tp"), FP=S("fp"), FN=S("fn"), TP_host=S("tp_host"), FP_host=S("fp_host"),
                FN_host=S("fn_host"), n_records_both_empty=sum(1 for r in recs if r["n_gt"] == 0 and r["n_pred"] == 0))


def bf_matched_pairs(G, P, T, thr=0.5):
    edges = [(i, j) for i in range(len(G)) for j in range(len(P)) if T[i][j] > thr]
    pairs = []
    for comp in _components(edges, len(G)):
        pairs += _enumerate_best(comp, T)[1]
    return pairs


def bf_diag(G, P, tau=0.5, min_px=10, miss=0.1, thr=0.5):
    """Split/merge/missed/spurious and exclusive FN/FP causes exactly as written in docs/metrics.md s.5."""
    ng, npr = len(G), len(P)
    T = iou_table(G, P)
    I = [[len(g & p) for p in P] for g in G]
    frag = [[I[i][j] >= min_px and I[i][j] >= tau * len(P[j]) for j in range(npr)] for i in range(ng)]
    cont = [[I[i][j] >= min_px and I[i][j] >= tau * len(G[i]) for j in range(npr)] for i in range(ng)]
    n_frag = [sum(frag[i]) for i in range(ng)]
    split_gt = [n >= 2 for n in n_frag]
    n_cont = [sum(cont[i][j] for i in range(ng)) for j in range(npr)]
    merge_pred = [n >= 2 for n in n_cont]
    merged_gt = [any(cont[i][j] and merge_pred[j] for j in range(npr)) for i in range(ng)]
    fragment_pred = [any(frag[i][j] and split_gt[i] for i in range(ng)) for j in range(npr)]
    gbest = [max(T[i]) if npr else 0.0 for i in range(ng)]
    pbest = [max(T[i][j] for i in range(ng)) if ng else 0.0 for j in range(npr)]
    pairs = bf_matched_pairs(G, P, T, thr)
    gm = {i for i, _ in pairs}
    pm = {j for _, j in pairs}
    fn_c, fp_c = {}, {}
    for i in range(ng):
        if i in gm:
            continue
        c = "merged" if merged_gt[i] else "split" if split_gt[i] else "missed" if gbest[i] < miss else "low_iou"
        fn_c[c] = fn_c.get(c, 0) + 1
    for j in range(npr):
        if j in pm:
            continue
        c = ("duplicate" if pbest[j] > thr else "merge" if merge_pred[j] else "fragment" if fragment_pred[j]
             else "spurious" if pbest[j] < miss else "low_iou")
        fp_c[c] = fp_c.get(c, 0) + 1
    out = dict(n_split_gt=sum(split_gt), n_split_fragments=sum(n for n, s in zip(n_frag, split_gt) if s),
               n_merge_pred=sum(merge_pred), n_merged_gt=sum(merged_gt),
               n_missed_gt=sum(b < miss for b in gbest), n_spurious_pred=sum(b < miss for b in pbest))
    for c in ("missed", "merged", "split", "low_iou"):
        out[f"fn_{c}"] = fn_c.get(c, 0)
    for c in ("spurious", "duplicate", "merge", "fragment", "low_iou"):
        out[f"fp_{c}"] = fp_c.get(c, 0)
    return out
