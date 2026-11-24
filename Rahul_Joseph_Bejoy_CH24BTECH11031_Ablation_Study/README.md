# Rahul Joseph Bejoy (CH24BTECH11031): Architecture ablation and blob filter (final)

Question: what does each part of a U-Net contribute to stress-granule segmentation? Five networks, one identical training recipe, grouped 5-fold cross-validation on videos 1+2 (ground truth = the automatic V4 labels), and a second test on 8 hand-outlined tiles of the held-out video 3 (102 granules, never used in training).

| Model | What it tests | Parameters |
| --- | --- | --- |
| `NoSkip` | encoder-decoder without skip connections (FCN-style baseline) | 1,747,921 |
| `UNet` | + skip connections | 1,943,761 |
| `AttUNet` | + attention gates on the skips | 1,965,885 |
| `DWUNet` | depthwise-separable convolutions (depthwise 3x3 + pointwise 1x1) | 389,435 |
| `DWAttUNet` | depthwise-separable + attention | 411,559 |

5 models x 5 folds x 1 seed (seed 0) = 25 checkpoints. This replaces the earlier 3-model, 3-seed version in this folder (`results/runs_extra.csv`, trained on older labels); numbers from that version are superseded.

## Contents

| Path | What it is |
| --- | --- |
| `code/rahul_5fold_ablation.py` | Models, loss, training, metrics and the full 5-fold experiment (resumable) |
| `code/score_video3.py` | Scores every checkpoint, the automatic labeller and label-free methods (DoG, Cellpose) on the hand tiles |
| `code/score_old_models_on_new_labels.py` | Re-scores checkpoints trained on older label versions on the same hand tiles |
| `code/plot_loss_curves.py` | Training loss and validation Dice curves (mean over folds) from the history CSVs |
| `results/final/` | 5-fold and hand-tile results for the final (V4) labels: per-run metrics, per-tile scores, summary tables |
| `results/earlier_label_versions/` | The same networks trained on label versions V1-V3 (how the labels evolved) |
| `figures/loss_curves.png` | Training loss and validation Dice, mean over 5 folds |
| `figures/01_violins_5fold.png`, `02_video3_tile_*.png` | Earlier figures (older labels) |
| `slides/Group7_Ablation_Statistics.pdf` | Earlier ablation/statistics slides (older labels) |

Statistics and the DoG baseline are in Vedant's folder. Data preparation, labels, folds and hand labels are in Prabhav's folder. Data and checkpoints are not in this repository because of their size.

## Recipe (identical for all five models)

Base width 16 (channels 16/32/64/128/256, 256 px to 16 px), conv-BatchNorm-ReLU blocks, 1x1 convolution + sigmoid output thresholded at 0.5. Loss = (1 - mean per-image soft Dice, smoothing 1) + 0.5 x BCE. Adam, learning rate 1e-3, batch 16, at most 30 epochs, early stopping with patience 6 on validation Dice (best epoch restored). Augmentation: flips, 90-degree rotations, brightness/contrast jitter. Validation = last 15% of frames of each video. MacBook Air M3 (MPS).

Metrics: granules are connected components, matched one-to-one (Hungarian) at IoU >= 0.5; precision, recall, F1, pixel Dice, AJI, AP50. `iou` is the mean IoU of the matched granules (not pixel IoU).

Blob filter: after thresholding, blobs smaller than N px are deleted, with no retraining. N = 20 was chosen on the validation tiles only.

## Key results (mean +/- std over 5 folds; hand tiles: mean over tiles)

| Model | 5-fold F1 (N=0) | 5-fold F1 (N=20) | matched IoU (N=20) | hand F1 (N=0 to 20) |
| --- | --- | --- | --- | --- |
| NoSkip | 0.559 +/- 0.044 | 0.587 +/- 0.043 | 0.716 | 0.478 to 0.507 |
| UNet | 0.662 +/- 0.045 | 0.699 +/- 0.039 | 0.824 | 0.522 to 0.567 |
| AttUNet | 0.671 +/- 0.050 | 0.703 +/- 0.047 | 0.827 | 0.541 to 0.564 |
| DWUNet | 0.645 +/- 0.049 | 0.686 +/- 0.045 | 0.833 | 0.509 to 0.549 |
| DWAttUNet | 0.646 +/- 0.044 | 0.686 +/- 0.048 | 0.820 | 0.510 to 0.550 |

- Skip connections are the only significant factor (NoSkip vs each other model: Holm p < 0.01 on F1; see Vedant's folder). UNet, AttUNet, DWUNet and DWAttUNet are not separable.
- DWUNet has about 5x fewer parameters. On this hardware it trained 43% longer per fold, so the gain is memory, not speed.
- The 20 px filter raises every network's F1; for UNet, hand-tile false positives fall from 75.4 to 55.8 at unchanged recall.
- One seed and 8 hand tiles: treat small differences as indicative only.

## Reproduce

```
pip install -r code/requirements.txt
cd code
python rahul_5fold_ablation.py                       # 5 models x 5 folds (several hours on a laptop; resumable)
python plot_loss_curves.py --hist rahul_5fold_outputs_final/history --out loss_curves.png --wide
python score_video3.py --coco <roboflow coco test folder> --images <tiles folder> \
    --pseudo <labeller masks folder> --ckpt rahul_5fold_outputs_final --root sg_dataset_5fold
```

Needs the `sg_dataset_5fold/` package and the video-3 tiles from Prabhav's folder.

Blob-filter code: Prabhav_Pradeep_CH24BTECH11029_Data_Labelling_Final_Submission/code/analysis/blob_filter_eval.py
