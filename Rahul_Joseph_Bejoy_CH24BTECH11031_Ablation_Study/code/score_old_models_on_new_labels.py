"""
Score the 45 EXISTING checkpoints (trained on the old pseudo-labels) against the NEW label package.

Why: Prabhav regenerated the labels with a stricter threshold. Before retraining, this shows how far the
old models are from the new labels. It is NOT a measure of how good a model is: the old models were taught
the old (over-detecting) labels, so a low score here mostly means "the labels changed a lot".

Run from the folder that contains rahul_5fold_ablation.py (same folder as before):

    python score_old_models_on_new_labels.py --root sg_dataset_5fold_v2 --ckpt rahul_5fold_outputs

For every checkpoint (model, fold, seed) it scores the tiles of THAT fold's roles on the new labels:
    test  = the tiles this fold's model never saw (the fair number)
    val, train = the other two roles (the model saw the OLD labels of these tiles during training)
    (test + val + train together cover the whole package, which is what Prabhav asked for)

Writes into --out (default rahul_oldmodels_newlabels):
    scores.csv        model,fold,seed,role,n_tiles,dice,iou,f1,aji,ap50,tp,fp,fn,precision,recall
    summary.csv       mean over checkpoints per model and role
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import torch

import rahul_5fold_ablation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="NEW package folder that contains fold_1 ... fold_5")
    ap.add_argument("--ckpt", default="rahul_5fold_outputs", help="folder with ckpt/<model>_fold<j>_seed<s>.pt")
    ap.add_argument("--out", default="rahul_oldmodels_newlabels")
    ap.add_argument("--base", type=int, default=None, help="default: read from <ckpt>/config.json, else 16")
    ap.add_argument("--roles", nargs="+", default=["test", "val", "train"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.base is None:
        cfg = os.path.join(a.ckpt, "config.json")
        a.base = json.load(open(cfg)).get("base", 16) if os.path.exists(cfg) else 16
    print(f"base width {a.base}", flush=True)

    cache, rows = {}, []
    files = sorted(glob.glob(os.path.join(a.ckpt, "ckpt", "*.pt")))
    if not files:
        raise SystemExit(f"no checkpoints in {a.ckpt}/ckpt")
    print(f"Device: {R.DEVICE} | checkpoints: {len(files)}", flush=True)
    for f in files:
        tag = os.path.basename(f)[:-3]                      # UNet_fold3_seed1
        name, fold, seed = tag.split("_")
        j, s = int(fold[4:]), int(seed[4:])
        model = R.UNet(base=a.base, **R.MODELS[name]).to(R.DEVICE)
        model.load_state_dict(torch.load(f, map_location=R.DEVICE))
        for role in a.roles:
            key = (j, role)
            if key not in cache:
                x, _, inst, _ = R.load_fold(a.root, j, role, with_instances=True)
                cache[key] = (x, inst)
            x, inst = cache[key]
            m = R.evaluate_tiles(R.predict_tiles(model, x), inst)
            rows.append(dict(model=name, fold=j, seed=s, role=role, n_tiles=len(x), **m))
        last = [r for r in rows if r["model"] == name and r["fold"] == j and r["seed"] == s and r["role"] == "test"]
        if last:
            r = last[0]
            print(f"{tag}: TEST on new labels  dice {r['dice']:.3f}  f1 {r['f1']:.3f}  "
                  f"precision {r['precision']:.3f}  recall {r['recall']:.3f}  tp/fp/fn {r['tp']}/{r['fp']}/{r['fn']}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "scores.csv"), index=False)

    df = pd.DataFrame(rows)
    keep = ["dice", "iou", "f1", "aji", "ap50", "precision", "recall", "tp", "fp", "fn"]
    summ = df.groupby(["model", "role"])[keep].mean().round(3).reset_index()
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print("\nMean over checkpoints:")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
