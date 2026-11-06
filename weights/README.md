# Model weights

There are seven fp32 `state_dict` checkpoints, about 1.2 GB in total. GitHub rejects files over 100 MiB, so each
checkpoint is stored in plain Git as byte-range parts under 50 MB (`<name>.partNN`), with a manifest
(`<name>.manifest.json`: the size and SHA-256 of every part and of the whole file). `SHA256SUMS` lists the same
hashes.

`src/reproduce_utils.prepare_weights("weights")` does four things, and the notebook calls it first:
1. checks every part;
2. concatenates the parts;
3. checks the whole file;
4. writes `weights/<name>` (git-ignored).

The manual equivalent is `python src/weight_parts.py join weights/<name>.manifest.json`.

| File | Bytes | Parts | Model | Written by (`entry_points.md`) | SHA-256 |
|---|---|---|---|---|---|
| `b0_fold0.pt` | 206,654,336 | 5 | stage 1, U-Net ResNet-101, 1 channel, fold 0 | 3.1 `outputs/exps/b0_r101_s1024/fold0/last.pt` | `d7ac64c6…41129dd` |
| `b0_fold1.pt` | 206,654,336 | 5 | stage 1, fold 1 | 3.1 | `5bd418d8…643fe035` |
| `b0_fold2.pt` | 206,654,336 | 5 | stage 1, fold 2 | 3.1 | `3c17ca1c…e8aff` |
| `b0_fold3.pt` | 206,654,336 | 5 | stage 1, fold 3 | 3.1 | `c26cfa50…832254a` |
| `b0_fold4.pt` | 206,654,336 | 5 | stage 1, fold 4 | 3.1 | `4bdc5e35…83497` |
| `r2_final.pt` | 206,767,232 | 5 | R2 refinement, U-Net ResNet-101, 10 channels, all 707 images | 3.3 `outputs/exps/R2_final_context/all/last.pt` | `922a848c…44c6f` |
| `se2_scorer.pt` | 5,882,364 | 1 | SE2 scorer (final submission), same architecture, input = R2-refined fragments | 3.7 `outputs/research/se2/scorer_final.pt` (`scripts/se2.py train`) | `bff904de…deb9c` |

The full hashes are in `SHA256SUMS`. Each file is byte-identical to the file the training run wrote.

**Pretrained weights.** For training only, every encoder was initialised from torchvision's original ImageNet
ResNet-101 checkpoint `resnet101-5d3b4d8f.pth`, read from `~/.cache/torch/hub/checkpoints/`
(https://download.pytorch.org/models/resnet101-5d3b4d8f.pth). Inference needs no download.
