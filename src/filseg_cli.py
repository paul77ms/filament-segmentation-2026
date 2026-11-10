"""Command-line entry point: predict a folder of GONG H-alpha images with the weights stored in this repository.

  filseg-predict --data MAGFiLO_1.0_Kaggle_2026 --out outputs/submission_r2b0_se2.csv
  python src/filseg_cli.py ...                                                      # the same, without installing
  python src/filseg_cli.py --data /path/to/new_images --out predictions.csv --masks out/masks --overlays out/overlays
                           # any folder of 2048x2048 GONG H-alpha JPEGs; masks/overlays are optional

It does exactly what notebooks/reproduce_submission.ipynb does in section 4: rebuild the checkpoints from
weights/*.partNN (SHA-256 checked), run the final pipeline, write `filament_id,segmentation_rle`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="filseg-predict", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="any folder of 2048x2048 GONG H-alpha JPEGs, or the competition data "
                    "(test_images folder, the MAGFiLO_1.0_Kaggle_2026 folder, or its parent)")
    ap.add_argument("--masks", default="", help="optional folder: one 16-bit PNG label map per image "
                    "(0 = background, k = filament <stem>_<k-1>)")
    ap.add_argument("--overlays", default="", help="optional folder: one JPEG per image with the filament outlines")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--weights", default="weights", help="folder with *.partNN, manifests and SHA256SUMS (default: weights)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = torch default)")
    a = ap.parse_args(argv)
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import filseg_infer_r2b0 as R2B0
    import reproduce_utils as U
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    w = Path(a.weights)
    paths = U.prepare_weights(w)
    b0 = [p for p in paths if p.name.startswith("b0_fold")]
    test_dir = U.find_test_dir(Path(a.data))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s = R2B0.run(test_dir, b0, w / "r2_final.pt", w / "se2_scorer.pt", a.out, limit=a.limit)
    print(f"wrote {a.out} in {time.time() - t0:.0f} s: {s.get('rows', '?')} filaments")
    if a.masks or a.overlays:
        n = U.export_masks(a.out, test_dir, a.masks or None, a.overlays or None, limit=a.limit)
        print(f"wrote masks / overlays for {n} images:", ", ".join(x for x in (a.masks, a.overlays) if x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
