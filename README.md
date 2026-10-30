# Group 7 — Stress-granule detection

Data preparation, model ablation and statistics for detecting stress granules in
fluorescence microscopy videos.

| Member | Roll no. | Part | Folder |
|---|---|---|---|
| Prabhav Pradeep | CH24BTECH11029 | Data preparation: cleaning, pseudo-labels, tiling, 5-fold split, video-3 tiles for Roboflow | [`Prabhav_Pradeep_CH24BTECH11029_Data_Preparation/`](Prabhav_Pradeep_CH24BTECH11029_Data_Preparation/) |
| Rahul Joseph Bejoy | CH24BTECH11031 | Ablation study: U-Net vs depthwise U-Net vs depthwise attention U-Net | [`Rahul_Joseph_Bejoy_CH24BTECH11031_Ablation_Study/`](Rahul_Joseph_Bejoy_CH24BTECH11031_Ablation_Study/) |
| Vedant P K | CH24BTECH11042 | Statistics and label-free methods: violin plots, hypothesis tests, DoG / pseudo-labeller baselines | [`Vedant_PK_CH24BTECH11042_Statistics_and_Label_Free_Methods/`](Vedant_PK_CH24BTECH11042_Statistics_and_Label_Free_Methods/) |

The final group presentation is in every member's folder under `presentation/`
(`Group7_Final_Presentation_Phase1.pdf`, identical copies).

## How this repository is organised

* One folder per member, holding only that member's work.
* Every commit is labelled with the member's name and part, as described in
  [`SUBMISSION_PROTOCOL.md`](SUBMISSION_PROTOCOL.md). `git log` shows who did what.
* Large files (`.zip`, `.npz`, `.pdf`, `.pptx`, model weights) are stored with
  [Git LFS](https://git-lfs.com). Run `git lfs install` once before cloning to get the real files.

## Pipeline in one line

Videos 1 + 2 → clean and align → auto-label cells and granules → 256 px tiles (1,516) →
5 folds grouped by tile position → U-Net ablation (45 runs) → statistics; video 3 → 20 tiles
labelled by hand in Roboflow → final test.
