# Rahul Joseph Bejoy (CH24BTECH11031): Model ablation

Trained U-Net variants under one identical recipe on Prabhav's 5-fold tile package, then scored every checkpoint on 8 hand-labelled tiles of the held-out video 3 (102 granules).

Two rounds of work:

1. **Original labels** (granule area 4 to 400 px, about 22,283 granules): 3 models x 5 folds x 3 seeds = 45 runs.
2. **Stricter labels** (granule area 60 to 1500 px, about 8,000 granules): 5 models x 5 folds, seed 0 = 25 runs. A 30 to 1500 px version is being trained (results to be added).

## Models

The ablation chain adds one component at a time, all with standard 3x3 convolutions. The depthwise models are a side study on size.

| Model | What it is |
| --- | --- |
| `NoSkip` | Encoder-decoder with no skip connections (FCN-like) |
| `UNet` | NoSkip + skip connections |
| `AttUNet` | UNet + attention gates on the skip connections |
| `DWUNet` | UNet with depthwise-separable convolutions (depthwise 3x3 + pointwise 1x1) |
| `DWAttUNet` | DWUNet + attention gates |

## Contents

| Path | What it is |
| --- | --- |
| `code/rahul_5fold_ablation.py` | Model definitions, training, metrics and the full resumable experiment. `--models` chooses from all five |
| `code/score_video3.py` | Scores every saved checkpoint, the pseudo-labeller and any label-free method on the hand-labelled video-3 tiles |
| `code/score_old_models_on_new_labels.py` | Scores the 45 first-round checkpoints against the new, stricter label package |
| `code/requirements.txt` | Python packages |
| `code/early_exploration/` | First version (single split, Colab). Kept for history; final numbers do not come from it |
| `results/runs_extra.csv` | First round (original 4-400 px labels): one row per run with all metrics, size and speed |
| `results/60_1500px/` | Second round: `runs.csv`, `runs_extra.csv`, per-run training curves in `history/`, and the hand-label scores (`v3_tiles.csv`, `v3_tiles_extra.csv`, `pseudo_vs_hand.csv`, `extra_methods_vs_hand.csv`) |
| `results/old_models_on_60_1500_labels/` | The first-round models scored on the new labels (`scores.csv`, `summary.csv`) |
| `figures/01_violins_5fold.png`, `figures/02_video3_tile_*.png` | First-round violin plot and one video-3 tile with each method's prediction |
| `figures/overlays_60_1500px/` | Second-round hand labels vs predictions per tile (green = correct, red = extra, yellow = missed) |
| `presentation/Group7_Final_Presentation_Phase1.pdf` | Final group presentation (ablation: slides 23-32) |

Statistics (paired tests, violins) and the DoG baseline are in Vedant's folder. Data preparation, pseudo-labels, folds and hand labels are in Prabhav's folder. Data and checkpoints are not in this repository because of their size.

## Running it

Needs the `sg_dataset_5fold*/` package from Prabhav. Run from the folder that contains it.

```
pip install -r code/requirements.txt

# quick check (a few minutes)
python rahul_5fold_ablation.py --root sg_dataset_5fold_v2 --out test_run --folds 1 --seeds 0 --max-epochs 1 --models UNet NoSkip AttUNet

# full experiment (resumable; about 12 min per run on a MacBook Air M3)
python rahul_5fold_ablation.py --root sg_dataset_5fold_v2 --out rahul_5fold_outputs_v2 --seeds 0 --models UNet NoSkip AttUNet DWUNet DWAttUNet

# score the checkpoints on the hand-labelled video-3 tiles
python score_video3.py --coco roboflow_v3/coco/unzipped/test --images roboflow_v3/images \
    --pseudo roboflow_v3/pseudo_masks --ckpt rahul_5fold_outputs_v2 --root sg_dataset_5fold_v2 \
    --out rahul_v3_outputs_v2 --extra DoG=masks_methodA
```

## Training recipe (identical for all models)

Base width 16, loss = Dice + 0.5 x BCE, Adam at 1e-3, batch 16, up to 30 epochs with early stopping on validation Dice (patience 6), flips, 90-degree rotations and brightness/contrast jitter, 1x resolution, ReLU + BatchNorm in every block, sigmoid output thresholded at 0.5. Run on a MacBook Air M3 (MPS). Folds are grouped by tile position so near-identical neighbouring frames never sit in train and test together.

## Key numbers

**Round 1: original labels, 4-400 px** (45 runs, test tiles of each fold)

| | UNet | DWUNet | DWAttUNet |
| --- | --- | --- | --- |
| Dice | 0.778 | 0.767 | 0.769 |
| F1 at IoU 0.5 | 0.715 | 0.688 | 0.690 |
| Parameters | 1.94 M | 0.39 M | 0.41 M |

On 8 hand-labelled tiles these models reach Dice about 0.52 and F1 about 0.32 (recall about 73%, precision about 21%, about 250 to 280 false positives). The pseudo-labeller itself scores F1 0.381 there. So the labels over-detected, and the models copied it.

**Scoring the round-1 models on the stricter 60-1500 labels** (test tiles): UNet F1 0.405 (precision 0.28, recall 0.74), DWUNet 0.395, DWAttUNet 0.397. They find most granules but mark about 3 times too many blobs.

**Round 2: 60-1500 labels** (25 runs, seed 0, mean over 5 folds, test tiles)

| | Dice | Matched IoU | F1 | AJI | AP50 | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| NoSkip | 0.652 | 0.719 | 0.554 | 0.430 | 0.400 | 1.75 M |
| UNet | 0.701 | 0.808 | 0.625 | 0.513 | 0.521 | 1.94 M |
| AttUNet | 0.712 | 0.815 | 0.651 | 0.519 | 0.540 | 1.97 M |
| DWUNet | 0.701 | 0.817 | 0.610 | 0.509 | 0.512 | 0.39 M |
| DWAttUNet | 0.691 | 0.810 | 0.594 | 0.494 | 0.489 | 0.41 M |

Adding skip connections gives the largest gain; attention gives a small extra gain that has not been tested for significance (one seed, 5 folds).

**Round 2 on the 8 hand-labelled tiles** (mean over the 5 folds' checkpoints; 102 true granules)

| | Dice | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| NoSkip | 0.534 | 0.361 | 32.6 | 36.2 | 69.4 |
| UNet | 0.515 | 0.368 | 38.4 | 40.8 | 63.6 |
| AttUNet | 0.523 | 0.353 | 39.6 | 54.0 | 62.4 |
| DWUNet | 0.524 | 0.349 | 38.0 | 47.2 | 64.0 |
| DWAttUNet | 0.529 | 0.360 | 37.8 | 45.4 | 64.2 |

False positives fell from about 250 to about 36 to 54 per model and precision roughly doubled (about 0.21 to 0.42 to 0.49), but recall fell from about 0.73 to about 0.32 to 0.39. On human labels the five models are within noise of each other (F1 0.349 to 0.368). The 30-1500 version is being trained to see whether keeping smaller granules recovers recall.

## Label versions compared

Prabhav generated three label versions by granule area. The scores on each version's own test tiles are not comparable with each other (different ground truth); the hand-label columns are the common yardstick. UNet shown.

| Label version (area in px) | Granules, whole package | Test F1 on its own labels | Hand-label F1 | Hand-label precision | Hand-label recall |
| --- | --- | --- | --- | --- | --- |
| 4-400 (original) | about 22,283 | 0.715 | 0.335 | 0.23 | 0.73 |
| 60-1500 | about 8,000 | 0.625 | 0.368 | 0.49 | 0.38 |
| 30-1500 | training in progress | - | - | - | - |

## Limits

- The ground truth for the 5-fold runs is an automatic labeller, not a human.
- The hand-label test has 8 tiles, 102 granules and one annotator.
- Round 2 uses one seed (5 folds), so differences between models are indicative until paired tests are run.
- Touching granules merge, because instances come from connected components.
