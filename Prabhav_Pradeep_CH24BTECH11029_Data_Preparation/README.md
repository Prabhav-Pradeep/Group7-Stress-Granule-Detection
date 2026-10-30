# Prabhav Pradeep (CH24BTECH11029) — Data preparation

Turned three raw microscope videos into a ready-to-train, leak-free dataset for the group:
cleaning and alignment, automatic labels (pseudo-labels) for cells and granules, 256 px tiles,
a 5-fold split grouped by tile position, a checksummed team package, and 20 tiles of the
held-out video 3 for hand labelling in Roboflow.

## Contents

| Path | What it is |
|---|---|
| `presentation/Group7_Final_Presentation_Phase1.pdf` | Final group presentation. Data preparation = slides 1–22 |
| `code/SG_Data_Preparation_5fold_and_Roboflow_Tiles.ipynb` | **Main notebook** (Google Colab, CPU, about 10 min), saved with outputs |
| `code/sgtrack/preprocess.py` | Load video, photobleach correction, denoise, registration (phase correlation + ECC), valid region, optical flow |
| `code/sgtrack/pseudolabel.py` | Cell masks (3-component Gaussian mixture on brightness × movement), cell instances, granule pseudo-labels |
| `code/sgtrack/pipeline.py` | Runs each stage per video; every cached `.npz` is re-read after saving |
| `code/sgtrack/share.py` | Tiling, grouped 5-fold split (heaviest substack → lightest fold), video-3 Roboflow tiles, SHA-256 team package |
| `code/requirements.txt` | Python packages |
| `figures/01_pipeline_one_frame.png` | Aligned frame, optical flow, cells and granules (video 1, frame 10) |
| `figures/02_training_targets_per_tile.png` | One tile as image, mask, instances and YOLO boxes |
| `figures/03_tiling_and_registration.png` | The 4 × 4 tile grid on the registered frame |
| `figures/04_tile_position_over_time.png` | One grid position across frames (why a random split leaks) |
| `figures/05_yolo_training_samples.png` | YOLO training samples (from the earlier 2-fold package) |
| `figures/06_counts_over_time.png` | Granules per frame, videos 1–3 |
| `figures/07_video3_roboflow_tiles.png` | The 20 video-3 tiles with their auto-labels |

The `.py` files are the `%%writefile` cells of the notebook, extracted unchanged.
`pipeline.track_video` refers to a `tracking` module that is not part of this notebook; it is not used here.

## Running the notebook

1. Open it in Google Colab (a CPU runtime is enough).
2. Put the three `.avi` videos in `MyDrive/SGTracking/content` and check the names in the CONFIG cell.
3. Run all cells. Everything is written to the new folder `MyDrive/SGTracking_5FOLD`:
   * `DATA_FOR_TEAM_5FOLD/`: `sg_dataset_5fold.zip`, `v1/v2_pre.npz`, `v1/v2_pl.npz`,
     `MANIFEST.sha256`, `verify_download.py` (about 614 MB)
   * `ROBOFLOW_V3/`: `images.zip` (the 20 tiles), `pseudo_masks/`, `tiles.csv`

The raw videos and the generated datasets are not in this repository because of their size.

## Key numbers

| | |
|---|---|
| Videos | v1: 21 frames, v2: 86 frames (training pool, frame 0 skipped); v3: 148 frames (final test only) |
| Granules per frame (mean) | v1: 287, v2: 203, v3: 229 |
| Tiles | 1,516 tiles of 256 × 256 px, 22,283 granules, 25 grid positions (substacks) |
| Folds | 5 folds, 5 test positions each, 4,388–4,524 test granules per fold |
| Leakage check | passed: in every fold no test position is in train or val |
| Video 3 | 20 tiles (5 frames × 4), 7 grid positions, for hand labels in Roboflow |
