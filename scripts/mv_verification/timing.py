import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repository root (path-independent)
import os, sys, time, json, statistics
os.environ["MPLBACKEND"] = "Agg"
import numpy as np, cv2, torch, pandas as pd
cv2.setNumThreads(1); torch.set_num_threads(1)
sys.path.insert(0, _REPO + "/src")
from pycocotools import mask as mask_util
from filseg.metrics import InstanceSet, EvalConfig, evaluate_image, match_instances, intersection_matrix
from filseg import rle
ANN = _REPO + "/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
d = json.load(open(ANN)); by = {}
for a in d["annotations"]: by.setdefault(a["image_id"], []).append(a)
rid = sorted([k for k, v in by.items() if len(v) == 10])[0]
others = [a for k, v in by.items() if k.split("-")[1] != rid.split("-")[1] for a in v][:200:20]
dec = lambda a: mask_util.decode(mask_util.merge(mask_util.frPyObjects(a["segmentation"], 2048, 2048))).astype(bool)
gts = [dec(a) for a in by[rid]]
k3 = np.ones((3, 3), np.uint8)
preds = [cv2.dilate(g.astype(np.uint8), k3).astype(bool) for g in gts] + [np.roll(g, 2, 1) for g in gts] + [dec(a) for a in others[:10]]
print("record", rid, "n_gt", len(gts), "n_pred", len(preds), "gt areas", [int(g.sum()) for g in gts])
gt_rles = [rle.encode(m) for m in gts]; pr_rles = [rle.encode(m) for m in preds]
def tm(f, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return round(statistics.median(ts) * 1000, 1)
G = InstanceSet.from_rles(gt_rles); P = InstanceSet.from_rles(pr_rles)
res = dict(
  build_from_full_masks_ms=tm(lambda: (InstanceSet.from_masks(gts), InstanceSet.from_masks(preds))),
  build_from_rles_ms=tm(lambda: (InstanceSet.from_rles(gt_rles), InstanceSet.from_rles(pr_rles))),
  intersection_matrix_ms=tm(lambda: intersection_matrix(G, P)),
  match_unique_ms=tm(lambda: match_instances(G, P)),
  evaluate_image_pq_only_ms=tm(lambda: evaluate_image(G, P, EvalConfig(compute_thin=False, compute_boundary=False), return_instances=False)),
  evaluate_image_full_ms=tm(lambda: evaluate_image(G, P, EvalConfig())),
  encode_30_masks_ms=tm(lambda: [rle.encode(m) for m in preds]),
)
ns = {}; exec(open(_REPO + "/docs/thirdparty_metric/ooLemon0-0_self-evaluation-notebook_cells.py").read(), ns)
gdf = pd.DataFrame({"filament_id": [f"{rid}_{k}" for k in range(len(gt_rles))], "segmentation_rle": gt_rles})
pdf = pd.DataFrame({"filament_id": [f"{rid.split('-')[1]}_{k}" for k in range(len(pr_rles))], "segmentation_rle": pr_rles})
res["host_notebook_get_overlap_df_ms"] = tm(lambda: ns["get_overlap_df"](gdf, pdf), n=3)
r, _, _ = evaluate_image(G, P, EvalConfig())
od = ns["get_overlap_df"](gdf, pdf)
res["pq_m1_unique"], res["pq_m1_host"], res["pq_notebook"] = r["pq"], r["pq_host"], ns["get_pq_score"](od)
import resource; res["peak_rss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
res["loadavg"] = os.getloadavg()
print(json.dumps(res, indent=1))
