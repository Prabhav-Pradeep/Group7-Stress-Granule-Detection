"""
Post-processing test: drop predicted blobs smaller than N px, then score again. No retraining.

Why: the training labels never contain a granule smaller than 30 px (the labeller's size filter), yet
about 44 % of the networks' false positives on the hand-labelled tiles are blobs under 30 px.

What it does, for every saved checkpoint <Model>_fold<j>_seed<s>.pt:
  1. VALIDATION (chooses N): predicts fold j's val tiles (training videos only, never test) and scores
     them with every candidate N. The N with the best mean validation F1 over all checkpoints is chosen.
  2. 5-FOLD TEST: the saved test predictions (preds/<tag>.npz; predicted again if missing), scored with every N.
  3. HAND LABELS (video 3): predicts the hand-labelled tiles, scored per tile with every N.
The chosen N is fixed by step 1 before steps 2 and 3 are looked at. N = 0 reproduces runs.csv and v3_tiles.csv.

Put this file next to rahul_5fold_ablation.py and score_video3.py and run it from that folder:

  python blob_filter_eval.py --ckpt rahul_5fold_outputs_final --root sg_dataset_5fold_final \
      --coco roboflow_v3/coco/unzipped/test --images roboflow_v3/images --out blob_filter_results

Writes to --out:
  sweep.csv                 every checkpoint x split x N: dice, iou, f1, aji, ap50, tp, fp, fn, precision, recall
  summary.csv               mean over checkpoints per model, split and N
  v3_tiles_filtered.csv     same columns as v3_tiles.csv, at the chosen N  (for Vedant's statistics)
  runs_filtered.csv         same columns as runs.csv, at the chosen N      (for Vedant's statistics)
  sweep.png                 validation / test / hand-label F1 against N
"""
import argparse, glob, json, os, re, sys

import numpy as np
import pandas as pd
import torch
from skimage.measure import label as cc_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import rahul_5fold_ablation as R

COLS = ["dice", "iou", "f1", "aji", "ap50", "tp", "fp", "fn", "precision", "recall"]


def drop_small(preds, min_size):
    """Remove connected blobs smaller than min_size px from every tile (same 8-connectivity as the scoring)."""
    if min_size <= 0:
        return preds
    out = np.empty(preds.shape, bool)
    for k, p in enumerate(preds):
        lab = cc_label(p > 0, connectivity=2)
        keep = np.bincount(lab.ravel()) >= min_size
        keep[0] = False
        out[k] = keep[lab]
    return out


def score(preds, insts, sizes):
    rows = []
    for n in sizes:
        m = R.evaluate_tiles(drop_small(preds, n), insts)
        rows.append(dict(min_size=n, **{k: m[k] for k in COLS}))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", required=True, help="5-fold output folder (ckpt/, preds/, config.json)")
    ap.add_argument("--root", required=True, help="the 5-fold dataset the models were trained on")
    ap.add_argument("--coco", help="folder with _annotations.coco.json (hand labels); leave out to skip video 3")
    ap.add_argument("--images", help="roboflow_v3/images")
    ap.add_argument("--sizes", type=int, nargs="+", default=[0, 10, 20, 30, 40, 50, 60, 80])
    ap.add_argument("--base", type=int, default=None)
    ap.add_argument("--out", default="blob_filter_results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sizes = sorted(set([0] + a.sizes))

    cfg = os.path.join(a.ckpt, "config.json")
    base = a.base or (json.load(open(cfg))["base"] if os.path.exists(cfg) else 16)
    pat = re.compile(r"^(\w+)_fold(\d+)_seed(\d+)\.pt$")
    cks = [p for p in sorted(glob.glob(os.path.join(a.ckpt, "ckpt", "*.pt"))) if pat.match(os.path.basename(p))]
    if not cks:
        raise SystemExit(f"no <Model>_fold<j>_seed<s>.pt in {a.ckpt}/ckpt")

    hand = None
    if a.coco:
        import score_video3 as S
        hand = S.load_hand_labels(a.coco, a.images)
        hand_imgs = np.stack([t["img"] for t in hand])

    rows, v3_rows, cache = [], [], {}
    for n_ck, path in enumerate(cks, 1):
        mt = pat.match(os.path.basename(path))
        name, fold, seed = mt.group(1), int(mt.group(2)), int(mt.group(3))
        meta = dict(model=name, fold=fold, seed=seed)
        model = R.UNet(base=base, **R.MODELS[name])
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.to(R.DEVICE)
        if fold not in cache:                                   # one fold's tiles in memory at a time
            va = R.load_fold(a.root, fold, "val", with_instances=True)
            te = R.load_fold(a.root, fold, "test", with_instances=True)
            cache = {fold: dict(va_x=va[0], va_inst=va[2], te_x=te[0], te_inst=te[2],
                                te_names=[os.path.splitext(s)[0] for s in te[3]])}
        d = cache[fold]
        # 1) validation
        rows += [dict(**meta, split="val", **r) for r in score(R.predict_tiles(model, d["va_x"]), d["va_inst"], sizes)]
        # 2) 5-fold test: reuse the saved predictions when they match the test tiles
        npz = os.path.join(a.ckpt, "preds", os.path.basename(path)[:-3] + ".npz")
        te_pred = None
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=True)
            if [os.path.splitext(str(s))[0] for s in z["names"]] == d["te_names"]:
                te_pred = z["pred"] > 0
        if te_pred is None:
            te_pred = R.predict_tiles(model, d["te_x"])
        rows += [dict(**meta, split="test", **r) for r in score(te_pred, d["te_inst"], sizes)]
        # 3) hand labels, per tile (the convention of v3_tiles.csv)
        if hand is not None:
            hp = R.predict_tiles(model, hand_imgs)
            for t, p in zip(hand, hp):
                for r in score(p[None], t["inst"][None], sizes):
                    v3_rows.append(dict(**meta, tile=t["name"], **r))
        print(f"[{n_ck}/{len(cks)}] {os.path.basename(path)}", flush=True)

    sw = pd.DataFrame(rows)
    if v3_rows:
        v3 = pd.DataFrame(v3_rows)
        # one row per checkpoint and N: scores = mean over tiles, counts = sum over tiles (as in score_video3.py)
        agg = v3.groupby(["model", "fold", "seed", "min_size"]).agg(
            **{k: (k, "mean") for k in ["dice", "iou", "f1", "aji", "ap50"]}, **{k: (k, "sum") for k in ["tp", "fp", "fn"]}).reset_index()
        agg["precision"] = agg.tp / (agg.tp + agg.fp).clip(lower=1)
        agg["recall"] = agg.tp / (agg.tp + agg.fn).clip(lower=1)
        sw = pd.concat([sw, agg.assign(split="hand")], ignore_index=True)
    sw.to_csv(os.path.join(a.out, "sweep.csv"), index=False)

    summ = sw.groupby(["split", "model", "min_size"])[COLS].mean().round(4).reset_index()
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    val = sw[sw.split == "val"].groupby("min_size").f1.mean()
    best = int(val.idxmax())
    print("\nmean validation F1 over all checkpoints, per minimum blob size:")
    print(val.round(4).to_string())
    print(f"\nCHOSEN on validation: drop blobs smaller than {best} px")

    pd.set_option("display.width", 200)
    for split in ("test", "hand"):
        s = summ[(summ.split == split) & summ.min_size.isin(sorted({0, best, 30}))]
        if len(s):
            print(f"\n=== {split}: N = 0 (as now) vs chosen N = {best} (and 30) - mean over checkpoints ===")
            print(s[["model", "min_size", "precision", "recall", "f1", "dice", "tp", "fp", "fn"]].to_string(index=False))

    # files in the original formats at the chosen N, so the statistics can be re-run unchanged
    t = sw[(sw.split == "test") & (sw.min_size == best)]
    t[["model", "fold", "seed"] + COLS[:8]].to_csv(os.path.join(a.out, "runs_filtered.csv"), index=False)
    if v3_rows:
        v = v3[v3.min_size == best]
        v[["model", "fold", "seed", "tile"] + COLS[:8]].to_csv(os.path.join(a.out, "v3_tiles_filtered.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    splits = [s for s in ("val", "test", "hand") if s in set(sw.split)]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.6 * len(splits), 3.8), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        for m, g in summ[summ.split == sp].groupby("model"):
            ax.plot(g.min_size, g.f1, marker="o", label=m)
        ax.axvline(best, color="k", ls="--", lw=1)
        ax.set_title({"val": "validation (chooses N)", "test": "5-fold test", "hand": "hand labels (video 3)"}[sp])
        ax.set_xlabel("drop blobs smaller than N px"); ax.set_ylabel("F1"); ax.grid(alpha=0.3)
    axes[0, -1].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "sweep.png"), dpi=150); plt.close(fig)
    print(f"\nSaved to {a.out}/")


if __name__ == "__main__":
    main()
