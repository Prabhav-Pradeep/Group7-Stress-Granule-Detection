"""
Statistics + violin plots for the 5-fold ablation (UNet vs DWUNet vs DWAttUNet).

Two modes
---------
1) 5-fold results (Rahul's runs.csv: model,fold,seed,dice,iou,f1,aji,ap50,tp,fp,fn)
       python vedant_stats.py --file runs.csv --mode folds --out stats_folds
   Pairing unit = FOLD. The 3 seeds inside a fold are averaged first, so each
   model has 5 paired values (one per fold). Violins show all 15 runs.

2) Video-3 hand-labelled tiles (v3_tiles.csv: model,fold,seed,tile,dice,iou,f1,aji,ap50,tp,fp,fn)
       python vedant_stats.py --file v3_tiles.csv --mode tiles --out stats_v3
   Pairing unit = TILE. For each tile the scores of the 15 checkpoints
   (5 folds x 3 seeds) of a model are averaged, so each model has one value per tile.

Outputs (in --out): summary.csv, pairwise_tests.csv, friedman.csv, violins.png
Needs: pandas numpy scipy matplotlib
"""
import argparse, itertools, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["UNet", "DWUNet", "DWAttUNet"]
METRICS = ["dice", "iou", "f1", "aji", "ap50"]


def holm(p):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--mode", choices=["folds", "tiles"], default="folds")
    ap.add_argument("--out", default="stats_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.file)
    models = [m for m in MODELS if m in set(df.model)]
    metrics = [m for m in METRICS if m in df.columns]
    unit = "fold" if a.mode == "folds" else "tile"
    if unit not in df.columns:
        raise SystemExit(f"column '{unit}' not found in {a.file}")

    # ---- 1. summary over all individual runs (what the violins show) ----
    rows = []
    for m in models:
        d = df[df.model == m]
        for k in metrics:
            rows.append(dict(model=m, metric=k, n_rows=len(d),
                             mean=d[k].mean(), std=d[k].std(ddof=1),
                             median=d[k].median(), min=d[k].min(), max=d[k].max()))
    pd.DataFrame(rows).round(4).to_csv(f"{a.out}/summary.csv", index=False)

    # ---- 2. one paired value per model per unit (fold or tile) ----
    paired = df.groupby(["model", unit])[metrics].mean().reset_index()
    n_units = paired.groupby("model")[unit].nunique().min()

    pw, fr = [], []
    for k in metrics:
        wide = paired.pivot(index=unit, columns="model", values=k)[models].dropna()
        # Friedman: do the 3 models differ at all? (repeated measures, non-parametric)
        if len(wide) >= 3 and len(models) >= 3:
            chi, p = stats.friedmanchisquare(*[wide[m] for m in models])
            fr.append(dict(metric=k, n_units=len(wide), chi2=chi, p=p))
        # paired t-tests on every pair, Holm-corrected within this metric
        rec = []
        for m1, m2 in itertools.combinations(models, 2):
            diff = wide[m1] - wide[m2]
            t, p = stats.ttest_rel(wide[m1], wide[m2])
            dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
            try:
                pw_w = stats.wilcoxon(wide[m1], wide[m2]).pvalue
            except ValueError:
                pw_w = np.nan
            rec.append(dict(metric=k, model_a=m1, model_b=m2, n_units=len(wide),
                            mean_a=wide[m1].mean(), mean_b=wide[m2].mean(),
                            mean_diff_a_minus_b=diff.mean(), t=t, p_raw=p,
                            cohens_dz=dz, wilcoxon_p=pw_w))
        adj = holm([r["p_raw"] for r in rec])
        for r, pa in zip(rec, adj):
            r["p_holm"] = pa
            r["significant_0.05"] = bool(pa < 0.05)
        pw.extend(rec)
    pd.DataFrame(pw).round(4).to_csv(f"{a.out}/pairwise_tests.csv", index=False)
    pd.DataFrame(fr).round(4).to_csv(f"{a.out}/friedman.csv", index=False)

    # ---- 3. violin plots: every individual run/row as a dot ----
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4.2))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    rng = np.random.default_rng(0)
    for ax, k in zip(np.atleast_1d(axes), metrics):
        data = [df.loc[df.model == m, k].values for m in models]
        parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.8)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c); body.set_alpha(0.45)
        for i, (v, c) in enumerate(zip(data, colors), 1):
            ax.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, s=9, color=c,
                       edgecolor="k", linewidth=0.3, zorder=3)
        ax.set_xticks(range(1, len(models) + 1)); ax.set_xticklabels(models, rotation=20)
        ax.set_title(k); ax.grid(axis="y", alpha=0.3)
    unit_txt = "15 runs per model (5 folds x 3 seeds)" if a.mode == "folds" else "all checkpoints x tiles"
    fig.suptitle(f"{os.path.basename(a.file)}  -  {unit_txt}", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{a.out}/violins.png", dpi=200, bbox_inches="tight")

    # ---- console summary ----
    print(f"paired units per model ({unit}s): {n_units}")
    print("\nMEAN +- STD over all rows")
    s = pd.DataFrame(rows)
    print(s.pivot(index="model", columns="metric", values="mean").loc[models, metrics].round(3))
    print("\nFriedman (any difference among the 3 models?)")
    print(pd.DataFrame(fr).round(4).to_string(index=False))
    print("\nPairwise paired t-tests (Holm-corrected p)")
    print(pd.DataFrame(pw)[["metric", "model_a", "model_b", "mean_diff_a_minus_b",
                            "p_raw", "p_holm", "cohens_dz"]].round(4).to_string(index=False))
    print(f"\nSaved to {a.out}/: summary.csv, pairwise_tests.csv, friedman.csv, violins.png")
    if n_units < 8:
        print("\nNOTE: only %d paired units. With so few, even a real difference may not reach "
              "significance; the smallest possible Wilcoxon p is 2/2^n." % n_units)


if __name__ == "__main__":
    main()
