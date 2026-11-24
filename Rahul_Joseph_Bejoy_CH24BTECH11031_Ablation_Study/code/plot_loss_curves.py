"""
Make loss_curves.png (training loss and validation Dice per epoch, mean over the 5 folds, one line per network)
from the training histories of the final runs.

Run from ~/granule_project with the venv active:
    python plot_loss_curves.py --hist rahul_5fold_outputs_final/history --out loss_curves.png
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"NoSkip": "#2a78d6", "UNet": "#eb6834", "AttUNet": "#1baf7a", "DWUNet": "#eda100", "DWAttUNet": "#e87ba4"}
ORDER = ["NoSkip", "UNet", "AttUNet", "DWUNet", "DWAttUNet"]

ap = argparse.ArgumentParser()
ap.add_argument("--hist", default="rahul_5fold_outputs_final/history")
ap.add_argument("--out", default="loss_curves.png")
ap.add_argument("--wide", action="store_true", help="two panels side by side (for a wide slide figure)")
a = ap.parse_args()

runs = {}
for f in sorted(glob.glob(os.path.join(a.hist, "*.csv"))):
    m = re.match(r"^(\w+?)_fold(\d+)_seed(\d+)\.csv$", os.path.basename(f))
    if m:
        runs.setdefault(m.group(1), []).append(pd.read_csv(f))
if not runs:
    raise SystemExit(f"no history CSVs found in {a.hist}")

def mean_curve(dfs, col):
    n = max(len(d) for d in dfs)
    arr = np.full((len(dfs), n), np.nan)
    for i, d in enumerate(dfs):
        arr[i, :len(d)] = d[col].values
    return np.nanmean(arr, axis=0)

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#bdbdbd", "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a"})
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2)) if a.wide else plt.subplots(2, 1, figsize=(3.9, 4.1), sharex=True)
axs = list(axs)
for ax, col, title in zip(axs, ["train_loss", "val_dice"], ["Training loss (lower is better)", "Validation Dice (higher is better)"]):
    for m in ORDER:
        if m in runs:
            y = mean_curve(runs[m], col)
            ax.plot(range(1, len(y) + 1), y, color=COL[m], linewidth=2, label=m)
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(color="#e3e3e3", linewidth=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
for ax in (axs if a.wide else axs[1:]):
    ax.set_xlabel("epoch")
axs[-1 if a.wide else 0].legend(fontsize=9 if a.wide else 7.5, frameon=False, ncol=1 if a.wide else 2, loc="lower right" if a.wide else "upper right")
fig.text(0.01, 0.005, "mean over 5 folds", fontsize=7.5, color="#5a5a5a")
fig.tight_layout()
fig.savefig(a.out, dpi=200, facecolor="white")
print("saved", a.out, "| networks:", ", ".join(m for m in ORDER if m in runs), "| runs per network:", {m: len(v) for m, v in runs.items()})
