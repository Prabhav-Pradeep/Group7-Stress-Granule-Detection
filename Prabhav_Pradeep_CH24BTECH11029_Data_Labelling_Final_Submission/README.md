# Prabhav Pradeep (CH24BTECH11029) — Data labelling, final submission

Final version of the granule labels used by the whole group: the automatic labeller (V4), the
hand labels of the held-out video 3, the code that packages the labelled data, the final presentation and the Data
Labelling section of the IEEE report. The earlier data-preparation work (Phase 1) is in
[`../Prabhav_Pradeep_CH24BTECH11029_Data_Preparation/`](../Prabhav_Pradeep_CH24BTECH11029_Data_Preparation/).

## Contents

| Path | What it is |
|---|---|
| `code/DataLabelling_V4_Final_5fold_Dataset_and_Roboflow_Tiles.ipynb` | **Final labelling code** (Google Colab, CPU). Raw videos → cell masks → granule labels (V4) → 256 px tiles → grouped 5-fold split → video-3 tiles for Roboflow. Saved with outputs. Settings in the `GRANULE_KW` line of the CONFIG cell |
| `code/build_labelled_data_package.py` | Builds the labelled-data zip (`Group7_Labelled_Data_V4_auto_and_hand.zip`, on Drive) from the notebook output, the Roboflow tiles and the hand-label export; decodes and checks the hand labels |
| `code/analysis/blob_filter_eval.py` | Post-processing test: delete predicted blobs under N px, N chosen on validation tiles only (N = 20) |
| `code/analysis/size_binned_detection.py` | Scores by granule size and perimeter, perimeter statistics, size over time |
| `code/analysis/granule_perimeter_heatmap.py` | Granule perimeters and heatmaps from any model's tile predictions |
| `code/earlier_versions/` | Labeller V1 (4–400 px) and V3 (30–1500 px) notebooks, kept to show how the labels evolved |
| `results/Label_Versions_V1-V4_All_Results.xlsx` | Every number behind the slides: each label version, every network, 5-fold, blob filter, size, perimeter, time |
| `results/Label_Versions_Settings_and_Scores.xlsx` | Labeller settings of each version and what each change did |
| `results/Blob_Filter_Minimum_Size_Sweep_Results.zip` | Output of `blob_filter_eval.py` |
| `results/Granule_Size_and_Perimeter_Binned_Results.zip` | Output of `size_binned_detection.py` |
| `presentation/Group7_Final_Presentation_StressGranules.pdf` | Final group presentation (35 slides; backup tables from slide 21) |
| `slides/Group7_Final_Presentation_LaTeX_source/` | Its source: compile `main.tex` (Beamer; `backup.tex` holds the backup tables) |
| `docs/IEEE_Report_DataLabelling_Section/` | Data Labelling section of the IEEE report: `.tex`, plain-text copy, PDF preview, figures |
| `figures/TopHat_Filter_Interactive_Demo.html` | Interactive demo of the top-hat filter (open in a browser) |

## Data (Google Drive)

The data is too large for this repository and is shared through the Google Drive link:

* **Input**: the three raw microscope videos (`.avi`, about 800 MB).
* **Output**: `Group7_Labelled_Data_V4_auto_and_hand.zip` (223 MB), built by
  `code/build_labelled_data_package.py`. Its contents:

```
automatic_labels/   sg_dataset_5fold.zip   videos 1+2: 1,487 tiles (256x256), 13,507 granules, 5 folds
                                           per tile: images, masks, instances, cells, YOLO boxes
                    v1_pl.npz, v2_pl.npz, v3_pl.npz   full-frame labels per video
video3_test_tiles/  images/                the 20 tiles uploaded to Roboflow (never used for training)
                    auto_labels_V4/        V4 automatic masks for those 20 tiles
                    hand_labels/coco/      manual Roboflow export: 8 tiles, 102 granules
                    hand_labels/masks/     the same hand labels as 0/255 masks
                    tiles.csv              frame, position, hand-labelled?, granules (hand / auto)
```

## The labeller (V4)

1. **Cell mask**: 3-component Gaussian mixture on (√I, log(1 + optical-flow magnitude)), with an
   intensity gate so motion alone cannot make a pixel a cell.
2. **Top-hat contrast**: S = (I − b) / (b + 0.05), with b = smoothed grey-scale opening (disk radius r).
3. **Hessian blobs**: B = max over σ of σ²·max(0, −λ₁), seeds above median + k·1.4826·MAD.
4. **Growth**: watershed to half the peak height, radius cap, size and shape filters.

| Version | Area (px) | k | Min contrast | Top-hat r | Blob σ up to | Granules | U-Net F1 (hand tiles) |
|---|---|---|---|---|---|---|---|
| V1 | 4–400 | 4 | 0.5 | 8 | 3.5 | 22,283 | 0.34 |
| V2 | 60–1500 | 4 | 0.5 | 8 | 3.5 | 7,998 | 0.37 |
| V3 | 30–1500 | 5 | 0.7 | 8 | 3.5 | 13,039 | 0.46 |
| **V4** | 30–1500 | 5 | 0.7 | **20** | **7** | 13,507 | **0.52** (0.57 with the 20 px blob filter) |

## How to reproduce

1. Open `code/DataLabelling_V4_Final_5fold_Dataset_and_Roboflow_Tiles.ipynb` in Colab, put the three
   `.avi` videos in `MyDrive/SGTracking/content`, run all cells.
2. Download the output folder (`SGTracking_5FOLD_v3_30-1500px`) as a zip, then:

```
python code/build_labelled_data_package.py \
    --drive  SGTracking_5FOLD_v3_30-1500px.zip \
    --tiles  images.zip \
    --coco   <Roboflow COCO-segmentation export>.zip \
    --out    Group7_Labelled_Data_V4_auto_and_hand.zip
```

The analysis scripts in `code/analysis/` (except `granule_perimeter_heatmap.py`) import Rahul's
`rahul_5fold_ablation.py`; run them from his code folder, as described at the top of each script.
