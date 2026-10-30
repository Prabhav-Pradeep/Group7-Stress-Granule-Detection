# Rahul Joseph Bejoy (CH24BTECH11031): Model ablation

Trained three U-Net variants under one identical recipe on Prabhav's 5-fold tile package (3 models x 5 folds x 3 seeds = 45 runs), then scored all 45 checkpoints on 8 hand-labelled tiles of the held-out video 3.

| Model | Change from the previous one |
| --- | --- |
| `UNet` | Standard 3x3 convolutions (baseline) |
| `DWUNet` | Convolutions replaced by depthwise-separable ones (depthwise 3x3 + pointwise 1x1) |
| `DWAttUNet` | DWUNet plus attention gates on the skip connections |

## Contents

| Path | What it is |
| --- | --- |
| `code/rahul_5fold_ablation.py` | Model definitions, training, metrics and the full 45-run experiment (resumable) |
| `code/score_video3.py` | Scores every saved checkpoint, the pseudo-labeller and any label-free method on the hand-labelled video-3 tiles |
| `code/requirements.txt` | Python packages |
| `code/early_exploration/` | First version (single train/val/test split, Colab). Kept for history; the final numbers do NOT come from it |
| `results/runs_extra.csv` | One row per run: dice, iou, f1, aji, ap50, tp, fp, fn, precision, recall, localisation error, epochs, train minutes, parameters, speed |
| `figures/01_violins_5fold.png` | Distribution of every metric over the 15 runs per model |
| `figures/02_video3_tile_*.png` | One video-3 tile: hand labels, pseudo-labels, each model and DoG (green = correct, red = extra, yellow = missed) |
| `presentation/Group7_Final_Presentation_Phase1.pdf` | Final group presentation (ablation: slides 23-32) |

The statistics and the DoG baseline are in Vedant's folder. The data preparation, pseudo-labels, folds and hand labels are in Prabhav's folder. Data and checkpoints are not in this repository because of their size.

## Running it

Needs the `sg_dataset_5fold/` package from Prabhav's notebook. Run from the folder that contains it.

```
pip install -r code/requirements.txt

# quick check (a few minutes)
python rahul_5fold_ablation.py --folds 1 --seeds 0 --max-epochs 1 --out test_run

# the full experiment (several hours on a laptop; resumable)
python rahul_5fold_ablation.py

# score the checkpoints on the hand-labelled video-3 tiles
python score_video3.py --coco roboflow_v3/coco/unzipped/test --images roboflow_v3/images \
    --pseudo roboflow_v3/pseudo_masks --ckpt rahul_5fold_outputs --root sg_dataset_5fold
```

Outputs go to `rahul_5fold_outputs/` (runs.csv, runs_extra.csv, history, checkpoints, predictions) and `rahul_v3_outputs/` (v3_tiles.csv, overlays).

## Training recipe (identical for all three models)

Base width 16, loss = Dice + 0.5 x BCE, Adam at 1e-3, batch 16, up to 30 epochs with early stopping on validation Dice (patience 6), flips, 90-degree rotations and brightness/contrast jitter, 1x resolution, ReLU + BatchNorm in every block, sigmoid output thresholded at 0.5. Run on a MacBook Air M3 (MPS).

## Key numbers

| | UNet | DWUNet | DWAttUNet |
| --- | --- | --- | --- |
| Dice (5-fold, pseudo-label test sets) | 0.778 | 0.767 | 0.769 |
| F1 at IoU 0.5 (5-fold) | 0.715 | 0.688 | 0.690 |
| Parameters | 1.94 M | 0.39 M | 0.41 M |
| Mean training time per run (min) | 12.6 | 13.3 | 15.3 |

- No pairwise difference is significant after Holm correction (closest: F1, p = 0.071); see Vedant's folder.
- On 8 human-labelled video-3 tiles all three models drop to Dice about 0.52 and F1 about 0.32 (recall about 73%, precision about 21%).
- The ground truth for the 5-fold runs is the classical pseudo-labeller, not human annotation. The pseudo-labeller itself scores F1 0.381 against the hand labels.
