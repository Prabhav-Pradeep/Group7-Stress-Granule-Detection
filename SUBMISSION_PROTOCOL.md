# Submission protocol

Everyone follows the same rules so the teachers can see who did what.

## 1. Only touch your own folder

| Member | Folder |
|---|---|
| Prabhav Pradeep | `Prabhav_Pradeep_CH24BTECH11029_Data_Preparation/` |
| Rahul Joseph Bejoy | `Rahul_Joseph_Bejoy_CH24BTECH11031_Ablation_Study/` |
| Vedant P K | `Vedant_PK_CH24BTECH11042_Statistics_and_Label_Free_Methods/` |

The final presentation is already in each folder under `presentation/`. Do not change it.

## 2. Folder layout inside your folder

```
<your folder>/
  README.md        what you did, what each file is, how to run your code
  presentation/    Group7_Final_Presentation_Phase1.pdf (already there)
  code/            notebooks and .py files (highest priority)
  slides/          your own slide sources or drafts, if any
  figures/         plots and example images
  results/         CSVs, metric tables (e.g. runs.csv, v3_tiles.csv)
  data/            zips or other data files (large files go through Git LFS automatically)
  docs/            any other write-ups
```

Leave out folders you don't need. Name files so a stranger understands them:
`UNet_Ablation_5fold_Training.ipynb`, not `Untitled(3).ipynb`.

## 3. Commit messages

```
[<Full Name> | <Part>] <Add|Update|Fix>: <what, in plain words>

Member: <Full Name> (<Roll no.>)
Part: <your part of the project>
Files: <the main files in this commit>
<one or two lines on what they are, if useful>
```

Example:

```
[Rahul Joseph Bejoy | Ablation Study] Add: U-Net, DWUNet and DWAttUNet training notebook

Member: Rahul Joseph Bejoy (CH24BTECH11031)
Part: Ablation study
Files: code/UNet_Ablation_5fold_Training.ipynb
Trains the three U-Net variants on 5 folds x 3 seeds and writes runs.csv.
```

Make one commit per kind of thing (code, results, figures, data, README), not one huge commit.

## 4. Before every commit: no sensitive content

Do not commit:

* swearing, rants, or comments about other people (check notebook comments, markdown
  cells, printed outputs, slide speaker notes and file names);
* passwords, API keys or tokens (Roboflow, Weights & Biases, Hugging Face, Google);
* personal data such as phone numbers or home addresses;
* chat exports or screenshots of private messages.

## 5. Set your name once

```
git config user.name  "<Full Name>"
git config user.email "<the email on your GitHub account>"
```

The email must match your GitHub account, or GitHub won't show the commits as yours.
