"""Mask-level comparison of two submission CSVs + integrity checks + overlay pages for visual review. CPU only.

  unchanged  a new mask bit-identical to a reference mask of the same image
  changed    a new mask that overlaps a reference mask but is not identical (each such new mask counted once)
  added      a new mask with no pixel in common with any reference mask
  removed    a reference mask with no pixel in common with any new mask
Checks on the new CSV: unique filament_id, no two masks of an image overlap, no two identical masks.
Overlay pages: for every added (and every removed) mask a 2-panel tile (original crop | crop with the mask outline in
cyan and the other predictions of the image in white), 4 tiles per row.
  .venv/bin/python scripts/compare_submission_masks.py --new NEW.csv --ref REF.csv --out DIR
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as M

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filseg import data as D  # noqa: E402

H = W = 2048


def read(path):
    out = {}
    with open(path) as f:
        r = csv.reader(f); next(r)
        for fid, rle in r:
            out.setdefault(fid.split("_")[0], []).append((fid, rle))
    return out


def dec(rle):
    return M.decode({"size": [H, W], "counts": rle.encode()}).astype(bool)


def tiles(items, title, path, per_page=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for p0 in range(0, len(items), per_page):
        chunk = items[p0:p0 + per_page]
        rows = (len(chunk) + 3) // 4
        fig, ax = plt.subplots(rows, 8, figsize=(16, 2.2 * rows), squeeze=False)
        for a in ax.ravel():
            a.axis("off")
        for t, (stem, fid, m, others) in enumerate(chunk):
            img = cv2.imread(str(D.TEST_DIR / f"{stem}.jpeg") if hasattr(D, "TEST_DIR") else
                             str(ROOT / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images" / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
            ys, xs = np.nonzero(m)
            cy, cx = int(ys.mean()), int(xs.mean())
            h = max(48, int(max(np.ptp(ys), np.ptp(xs)) * 0.9))
            Y0, Y1, X0, X1 = max(cy - h, 0), min(cy + h, H), max(cx - h, 0), min(cx + h, W)
            crop = img[Y0:Y1, X0:X1]
            ov = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            c, _ = cv2.findContours(others[Y0:Y1, X0:X1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, c, -1, (255, 255, 255), 1)
            c, _ = cv2.findContours(m[Y0:Y1, X0:X1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, c, -1, (0, 230, 255), 1)
            r, k = divmod(t, 4)
            ax[r, 2 * k].imshow(crop, cmap="gray"); ax[r, 2 * k].set_title(f"{fid} ({int(m.sum())} px)", fontsize=7)
            ax[r, 2 * k + 1].imshow(ov); ax[r, 2 * k + 1].set_title("cyan: this mask, white: others", fontsize=7)
        fig.suptitle(f"{title} ({p0 + 1}-{p0 + len(chunk)} of {len(items)})", fontsize=10)
        fig.tight_layout()
        out = path.with_name(f"{path.stem}_{p0 // per_page + 1:02d}.png")
        fig.savefig(out, dpi=80); plt.close(fig); paths.append(str(out))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    new, ref = read(a.new), read(a.ref)
    ids = [fid for rows in new.values() for fid, _ in rows]
    res = dict(rows_new=len(ids), rows_ref=sum(len(v) for v in ref.values()), duplicate_ids=len(ids) - len(set(ids)),
               overlapping_pairs=0, identical_pairs=0, unchanged=0, changed=0, added=0, removed=0,
               images_with_change=0)
    added, removed = [], []
    for stem in sorted(set(new) | set(ref)):
        nm = [(fid, dec(r)) for fid, r in new.get(stem, [])]
        rm = [(fid, dec(r)) for fid, r in ref.get(stem, [])]
        for i in range(len(nm)):            # integrity of the new CSV
            for j in range(i + 1, len(nm)):
                if (nm[i][1] & nm[j][1]).any():
                    res["overlapping_pairs"] += 1
                if np.array_equal(nm[i][1], nm[j][1]):
                    res["identical_pairs"] += 1
        runion = np.zeros((H, W), bool)
        for _, m in rm:
            runion |= m
        nunion = np.zeros((H, W), bool)
        for _, m in nm:
            nunion |= m
        changed_here = False
        for fid, m in nm:
            if any(np.array_equal(m, r) for _, r in rm):
                res["unchanged"] += 1
            elif (m & runion).any():
                res["changed"] += 1; changed_here = True
            else:
                res["added"] += 1; changed_here = True
                added.append((stem, fid, m, nunion & ~m))
        for fid, m in rm:
            if not (m & nunion).any():
                res["removed"] += 1; changed_here = True
                removed.append((stem, fid, m, runion & ~m))
        res["images_with_change"] += int(changed_here)
    res["added_pages"] = tiles(added, "ADDED masks (new CSV only)", out / "added.png")
    res["removed_pages"] = tiles(removed, "REMOVED masks (reference CSV only)", out / "removed.png")
    res["added_ids"] = [x[1] for x in added]; res["removed_ids"] = [x[1] for x in removed]
    json.dump(res, open(out / "comparison.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_ids")}, indent=1))


if __name__ == "__main__":
    main()
