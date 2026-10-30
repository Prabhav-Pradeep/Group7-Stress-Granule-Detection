# Vedant P K (CH24BTECH11042): Statistics and label-free methods

Statistical tests and plots on Rahul's 45 training runs, and a label-free baseline (DoG) scored on the same 8 hand-labelled tiles of the held-out video 3.

## Contents

| Path | What it is |
| --- | --- |
| `code/vedant_stats.py` | Paired t-tests (Holm-corrected), Friedman test, Cohen's dz and violin plots for one results CSV |
| `code/vedant_all.py` | One command that runs `vedant_stats.py` on the 5-fold and video-3 results and writes `summary.md` and the tables |
| `code/dog_label_free_method.py` | DoG baseline: difference of Gaussians, robust threshold, watershed. No training, no labels |
| `code/Statistics_Run1_and_DoG_Method.ipynb` | Colab notebook: first statistics run, and the cell that generates the DoG masks |
| `code/Statistics_Run2_All_Methods.ipynb` | Colab notebook: the statistics re-run that includes the DoG scores (the all-methods table) |
| `results/summary.md` | Auto-written summary: tables, significance sentences, caveats |
| `results/all_methods_video3.csv` | Every approach scored against the hand labels (networks, pseudo-labeller, DoG) |
| `results/stats_folds/`, `results/stats_video3/` | `summary.csv`, `pairwise_tests.csv`, `friedman.csv` for the 5-fold and the video-3 comparison |
| `results/masks_methodA/` | The DoG masks for the 8 video-3 tiles (PNG) |
| `figures/01_violins_5fold.png`, `figures/02_violins_video3.png` | Violin plots of all five metrics per model |
| `figures/overlays/` | Hand labels vs predictions per tile (green = correct, red = extra, yellow = missed) |

The training code and the model results are in Rahul's folder. The data preparation, pseudo-labels, folds and hand labels are in Prabhav's folder. The two Colab notebooks differ only in that the second one was re-run after the DoG scores existed. The input `for_vedant.zip` is not in this repository.

## How the tests work

- **5-fold results:** the 3 seeds in a fold are averaged first, so each model has 5 paired values (one per fold). Paired t-tests on every pair of models, Holm-corrected within each metric. Violins show all 15 runs per model.
- **Video-3 results:** the 15 checkpoints of a model are averaged for each tile, so each model has 8 paired values (one per tile).
- A Wilcoxon test is not used: with 5 folds its smallest possible p-value is 0.0625.

## Running it

```
python vedant_all.py
```

Run it in the folder that holds `vedant_stats.py`, `rahul_5fold_outputs/runs.csv`, `rahul_5fold_outputs/runs_extra.csv` and `rahul_v3_outputs/` (from Rahul's `score_video3.py`). It writes `slides_assets/` with the tables and plots.

## Key results

- **5-fold (pseudo-label ground truth):** no pairwise difference is significant after Holm correction (closest: UNet vs DWUNet on F1, p = 0.071). The Friedman test on F1 is significant (p = 0.015), which says the three models differ somewhere, but no single pair reaches significance with only 5 folds.
- **Video 3 (hand labels, 8 tiles):** the only significant pair is UNet over DWUNet on F1 (Holm p = 0.0203). With 8 tiles and one annotator this is indicative only.
- **All approaches against the hand labels:**

| Method | Dice | F1 | Precision | Recall |
| --- | --- | --- | --- | --- |
| UNet | 0.526 | 0.335 | 0.229 | 0.731 |
| DWUNet | 0.515 | 0.302 | 0.207 | 0.715 |
| DWAttUNet | 0.529 | 0.319 | 0.212 | 0.729 |
| Pseudo-labeller | 0.471 | 0.381 | 0.333 | 0.637 |
| DoG (label-free) | 0.502 | 0.224 | 0.102 | 0.794 |

DoG finds the most granules (highest recall) but its noise-relative threshold has no floor on dark tiles, so it produces about 714 false detections. Numbers for the networks are means over their 15 checkpoints; counts for all methods are summed over the 8 tiles.
