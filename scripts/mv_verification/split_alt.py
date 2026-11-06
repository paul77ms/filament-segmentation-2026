import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repository root (path-independent)
import sys, json, numpy as np
sys.path.insert(0, _REPO + "/src")
import cv2; cv2.setNumThreads(1)
from filseg.metrics import InstanceSet, intersection_matrix, split_merge_diagnostics
ANN = _REPO + "/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
d = json.load(open(ANN)); by = {}
for a in d["annotations"]: by.setdefault(a["image_id"], []).append(a)
st = {}
for rid in by: st.setdefault(rid.split("-", 1)[1], []).append(rid)
tot = dict(records=0, split_containment=0, split_coverage=0, merge_containment=0, merge_coverage=0,
           assigned_pairs=0, assigned_pairs_fail_containment=0, split_cov_not_cont=0)
for stem, ids in st.items():
    if len(ids) < 2: continue
    S = {i: InstanceSet.from_coco_anns(by[i]) for i in ids}
    for a in ids:
        for b in ids:
            if a == b: continue
            G, P = S[a], S[b]
            I = intersection_matrix(G, P).astype(float)
            sm = split_merge_diagnostics(I, G.areas, P.areas, 0.5, 10)
            # coverage-based alternative: p assigned to the gt it overlaps most; p covers >= 10% of g (and >= 10 px)
            assign_p = I.argmax(0) if len(G) else np.zeros(len(P), int)
            cov = (I >= 10) & (I >= 0.1 * G.areas[:, None]) & (np.arange(len(G))[:, None] == assign_p[None, :])
            split_cov = cov.sum(1) >= 2
            assign_g = I.argmax(1) if len(P) else np.zeros(len(G), int)
            covm = (I >= 10) & (I >= 0.1 * P.areas[None, :]) & (np.arange(len(P))[None, :] == assign_g[:, None])
            merge_cov = covm.sum(0) >= 2
            tot["records"] += 1
            tot["split_containment"] += int(sm["split_gt"].sum()); tot["split_coverage"] += int(split_cov.sum())
            tot["merge_containment"] += int(sm["merge_pred"].sum()); tot["merge_coverage"] += int(merge_cov.sum())
            big = (I >= 10) & (I >= 0.2 * G.areas[:, None]) & (np.arange(len(G))[:, None] == assign_p[None, :])
            tot["assigned_pairs"] += int(big.sum())
            tot["assigned_pairs_fail_containment"] += int((big & (I < 0.5 * P.areas[None, :])).sum())
            tot["split_cov_not_cont"] += int((split_cov & ~sm["split_gt"]).sum())
print(json.dumps(tot, indent=1))
