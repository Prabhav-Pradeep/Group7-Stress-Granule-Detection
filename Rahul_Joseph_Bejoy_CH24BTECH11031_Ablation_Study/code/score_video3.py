"""
Score every saved 5-fold checkpoint on the hand-labelled video-3 tiles (Roboflow COCO export).

Run from ~/granule_project (the folder that holds rahul_5fold_ablation.py):

  python score_video3.py \
      --coco   roboflow_v3/coco/unzipped/test \
      --images roboflow_v3/images \
      --pseudo roboflow_v3/pseudo_masks \
      --ckpt   rahul_5fold_outputs \
      --root   sg_dataset_5fold

Writes into --out (default rahul_v3_outputs):
  v3_tiles.csv        model,fold,seed,tile,dice,iou,f1,aji,ap50,tp,fp,fn   <- send this to Vedant
  v3_tiles_extra.csv  same + precision, recall, loc_err, n_gt, n_pred
  pseudo_vs_hand.csv  the classical pseudo-labels scored against the hand labels (how good are the pseudo-labels?)
  extra_methods_vs_hand.csv  label-free methods given with --extra Name=folder (same metrics)
  overlays/<tile>.png hand labels + each model; TP green, FP red, FN yellow  (error-cases slide)

The ground truth here is HAND-DRAWN (Roboflow), unlike the 5-fold test sets (pseudo-labels).
"""
import argparse, glob, json, os, re, sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from skimage.io import imread, imsave
from skimage.measure import label as cc_label
from skimage.draw import polygon as draw_polygon
from skimage.segmentation import find_boundaries

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import rahul_5fold_ablation as R          # same model code + same metric function as the 5-fold runs


# ----------------------------------------------------------------------------- COCO label decoding
def rle_string_to_counts(s):
    """Decode COCO compressed-RLE string into a list of run lengths (same algorithm as pycocotools)."""
    if isinstance(s, bytes):
        s = s.decode()
    counts, p, m = [], 0, 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= (-1) << (5 * k)
        if m > 2:
            x += counts[m - 2]
        counts.append(int(x))
        m += 1
    return counts


def decode_rle(rle, h, w):
    counts = rle["counts"]
    if isinstance(counts, (str, bytes)):
        counts = rle_string_to_counts(counts)
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in counts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape((w, h)).T            # COCO RLE is column-major


def decode_segmentation(seg, h, w):
    if isinstance(seg, dict):
        hh, ww = seg.get("size", [h, w])
        return decode_rle(seg, hh, ww)
    m = np.zeros((h, w), np.uint8)
    for poly in seg:                          # list of polygons [x1,y1,x2,y2,...]
        xy = np.asarray(poly, float).reshape(-1, 2)
        rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape=(h, w))
        m[rr, cc] = 1
    return m


def tile_name_from_roboflow(fn):
    """'v3_t110_y0256_x0512_png.rf.2b14ea...png' -> 'v3_t110_y0256_x0512'"""
    base = os.path.basename(fn)
    if "_png.rf." in base:
        return base.split("_png.rf.")[0]
    return os.path.splitext(base)[0]


def load_hand_labels(coco_dir, images_dir):
    d = json.load(open(os.path.join(coco_dir, "_annotations.coco.json")))
    by_img = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    cats = {}
    for a in d["annotations"]:
        cats[a["category_id"]] = cats.get(a["category_id"], 0) + 1
    print(f"COCO: {len(d['images'])} images, {len(d['annotations'])} annotations, categories used: {cats}")
    if len(cats) > 1:
        print("  WARNING: more than one category is used - all of them are treated as granules.")
    tiles = []
    for im in d["images"]:
        name = tile_name_from_roboflow(im["file_name"])
        h, w = im["height"], im["width"]
        inst = np.zeros((h, w), np.uint16)
        for k, a in enumerate(by_img.get(im["id"], []), 1):
            inst[decode_segmentation(a["segmentation"], h, w) > 0] = k
        path = os.path.join(images_dir, name + ".png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - the tile '{name}' from the COCO file is not in --images")
        img = R.read_gray(path)
        if img.dtype != np.uint8:
            print(f"  WARNING: {name}.png is {img.dtype}, training tiles were uint8 - check the scaling!")
        img = img.astype(np.uint8)
        if img.shape != inst.shape:
            raise ValueError(f"{name}: image {img.shape} vs labels {inst.shape} - was the export resized?")
        # sanity: Roboflow's own copy of the image should equal the original
        exp = glob.glob(os.path.join(coco_dir, name + "_png.rf.*"))
        if exp:
            e = R.read_gray(exp[0]).astype(np.float32)
            if e.shape == img.shape:
                diff = float(np.abs(e - img).mean())
                if diff > 2:
                    print(f"  WARNING: {name}: Roboflow's copy differs from images/ (mean abs diff {diff:.1f})")
        tiles.append(dict(name=name, img=img, inst=inst, n_gt=int(len(np.unique(inst)) - (0 in inst))))
    tiles.sort(key=lambda t: t["name"])
    return tiles


# ----------------------------------------------------------------------------- error classification for overlays
def classify(pred_bin, gt_inst):
    p_lbl, n_p = cc_label(pred_bin, return_num=True)
    p_lbl = p_lbl.astype(np.int64)
    g_lbl, n_g = R.compact_labels(gt_inst)
    tp_p, tp_g = set(), set()
    if n_p and n_g:
        cont = np.bincount((p_lbl * (n_g + 1) + g_lbl).ravel(),
                           minlength=(n_p + 1) * (n_g + 1)).reshape(n_p + 1, n_g + 1)
        area_p, area_g, inter = cont.sum(1)[1:], cont.sum(0)[1:], cont[1:, 1:]
        iou = inter / np.maximum(area_p[:, None] + area_g[None, :] - inter, 1)
        r, c = linear_sum_assignment(1 - iou)
        for i, j in zip(r, c):
            if iou[i, j] >= 0.5:
                tp_p.add(i + 1); tp_g.add(j + 1)
    fp_p = set(range(1, n_p + 1)) - tp_p
    fn_g = set(range(1, n_g + 1)) - tp_g
    return p_lbl, g_lbl, tp_p, fp_p, fn_g


def paint(img, layers, scale=3):
    g = img.astype(np.float32)
    lo, hi = np.percentile(g, 1), np.percentile(g, 99.5)
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.repeat(np.repeat(np.stack([g] * 3, -1), scale, 0), scale, 1)
    for lbl, ids, colour in layers:
        if not ids:
            continue
        up = np.repeat(np.repeat(lbl, scale, 0), scale, 1)
        edge = find_boundaries(np.isin(up, list(ids)) * up, mode="inner") & np.isin(up, list(ids))
        rgb[edge] = colour
    return (rgb * 255).astype(np.uint8)


GREEN, RED, YELLOW, CYAN = (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 0.9, 1)


def make_overlay(tile, preds_by_model, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g_lbl, _ = R.compact_labels(tile["inst"])
    panels = [("hand labels (cyan)", paint(tile["img"], [(g_lbl, set(range(1, g_lbl.max() + 1)), CYAN)]))]
    for name, pred in preds_by_model.items():
        p_lbl, g, tp_p, fp_p, fn_g = classify(pred, tile["inst"])
        panels.append((f"{name}\nTP {len(tp_p)}  FP {len(fp_p)}  FN {len(fn_g)}",
                       paint(tile["img"], [(p_lbl, tp_p, GREEN), (p_lbl, fp_p, RED), (g, fn_g, YELLOW)])))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    for ax, (title, im) in zip(np.atleast_1d(axes), panels):
        ax.imshow(im); ax.set_title(title, fontsize=10); ax.axis("off")
    fig.suptitle(f"{tile['name']}   (green = TP, red = FP, yellow = FN)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.savefig(out_path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="folder with _annotations.coco.json (the unzipped 'test' folder)")
    ap.add_argument("--images", required=True, help="roboflow_v3/images (original 256x256 tiles)")
    ap.add_argument("--pseudo", default=None, help="roboflow_v3/pseudo_masks (optional, for pseudo-vs-hand)")
    ap.add_argument("--ckpt", default="rahul_5fold_outputs", help="the output folder of the 5-fold run (has ckpt/ and config.json)")
    ap.add_argument("--root", default=None, help="sg_dataset_5fold (optional, only for the intensity sanity check)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="label-free methods to score too, each as Name=folder_with_one_mask_png_per_tile, e.g. DoG=masks_methodA")
    ap.add_argument("--out", default="rahul_v3_outputs")
    ap.add_argument("--base", type=int, default=None, help="override base width (default: read from config.json)")
    ap.add_argument("--overlay-fold", type=int, default=1)
    ap.add_argument("--overlay-seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "overlays"), exist_ok=True)

    tiles = load_hand_labels(a.coco, a.images)
    imgs = np.stack([t["img"] for t in tiles])
    gts = [t["inst"] for t in tiles]
    print(f"{len(tiles)} hand-labelled tiles:", ", ".join(f"{t['name']} ({t['n_gt']})" for t in tiles))

    # intensity sanity check: video-3 tiles vs training tiles
    print(f"video-3 tiles: mean {imgs.mean():.1f}  p99 {np.percentile(imgs, 99):.0f}")
    if a.root:
        ref = sorted(glob.glob(os.path.join(a.root, "fold_1", "train", "images", "*.png")))[:60]
        if ref:
            r = np.stack([R.read_gray(f) for f in ref]).astype(np.float32)
            print(f"training tiles: mean {r.mean():.1f}  p99 {np.percentile(r, 99):.0f}")
            if not 0.5 < imgs.mean() / max(r.mean(), 1e-6) < 2:
                print("  WARNING: brightness differs a lot between video-3 and training tiles - ask Prabhav about the preprocessing.")

    base = a.base
    cfg = os.path.join(a.ckpt, "config.json")
    if base is None:
        base = json.load(open(cfg))["base"] if os.path.exists(cfg) else 16
    print(f"device {R.DEVICE} | base width {base}")

    # ---- score all checkpoints
    ck = sorted(glob.glob(os.path.join(a.ckpt, "ckpt", "*.pt")))
    pat = re.compile(r"^(UNet|DWUNet|DWAttUNet)_fold(\d+)_seed(\d+)\.pt$")
    rows, overlay_preds = [], {}
    for n, path in enumerate(ck, 1):
        mt = pat.match(os.path.basename(path))
        if not mt:
            continue
        name, fold, seed = mt.group(1), int(mt.group(2)), int(mt.group(3))
        model = R.UNet(base=base, **R.MODELS[name])
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.to(R.DEVICE)
        preds = R.predict_tiles(model, imgs)
        for t, p, g in zip(tiles, preds, gts):
            m = R.evaluate_tiles(p[None], g[None])
            rows.append(dict(model=name, fold=fold, seed=seed, tile=t["name"], **m,
                             n_gt=t["n_gt"], n_pred=int(cc_label(p).max())))
        if fold == a.overlay_fold and seed == a.overlay_seed:
            overlay_preds[name] = preds
        print(f"[{n}/{len(ck)}] {name} fold {fold} seed {seed}", flush=True)
    if not rows:
        raise SystemExit(f"no checkpoints matched in {a.ckpt}/ckpt")
    df = pd.DataFrame(rows)
    cols = ["model", "fold", "seed", "tile", "dice", "iou", "f1", "aji", "ap50", "tp", "fp", "fn"]
    df[cols].to_csv(os.path.join(a.out, "v3_tiles.csv"), index=False)
    df.to_csv(os.path.join(a.out, "v3_tiles_extra.csv"), index=False)

    # ---- masks from other methods (pseudo-labeller, your label-free methods) scored against the hand labels
    prow, extra_rows, extra_preds = [], [], {}

    def score_mask_folder(name, folder, strict):
        arr, rows = [], []
        for t, g in zip(tiles, gts):
            f = os.path.join(folder, t["name"] + ".png")
            if not os.path.exists(f):
                if strict:
                    raise SystemExit(f"{f} not found: method '{name}' needs one mask PNG per hand-labelled tile, named like the tile")
                return None, None
            pm = R.read_gray(f) > 0
            if pm.shape != g.shape:
                raise SystemExit(f"{f}: mask is {pm.shape}, tile is {g.shape}")
            arr.append(pm)
            rows.append(dict(model=name, tile=t["name"], **R.evaluate_tiles(pm[None], g[None])))
        return np.stack(arr), rows

    if a.pseudo:
        arr, rows = score_mask_folder("Pseudo", a.pseudo, strict=False)
        if rows:
            prow = rows
            extra_preds["Pseudo"] = arr
            pd.DataFrame(prow).to_csv(os.path.join(a.out, "pseudo_vs_hand.csv"), index=False)
    for spec in a.extra:
        name, folder = spec.split("=", 1)
        arr, rows = score_mask_folder(name, folder, strict=True)
        extra_rows += rows
        extra_preds[name] = arr
    if extra_rows:
        pd.DataFrame(extra_rows).to_csv(os.path.join(a.out, "extra_methods_vs_hand.csv"), index=False)

    # ---- overlays for the error-cases slide (networks of one fold/seed + pseudo + extra methods)
    if overlay_preds:
        order = [m for m in R.MODELS if m in overlay_preds]
        for i, t in enumerate(tiles):
            panels = {m: overlay_preds[m][i] for m in order}
            panels.update({n: p[i] for n, p in extra_preds.items()})
            make_overlay(t, panels, os.path.join(a.out, "overlays", t["name"] + ".png"))
        print(f"overlays saved for fold {a.overlay_fold} seed {a.overlay_seed}")
    else:
        print("no checkpoint for the overlay fold/seed - use --overlay-fold / --overlay-seed")

    # ---- console summary
    show = ["dice", "iou", "f1", "aji", "ap50", "tp", "fp", "fn"]
    print("\n=== video-3 (hand labels): mean over tiles, then over 15 checkpoints per model ===")
    per_model = df.groupby(["model", "fold", "seed"])[show].agg(
        {**{k: "mean" for k in ["dice", "iou", "f1", "aji", "ap50"]}, **{k: "sum" for k in ["tp", "fp", "fn"]}})
    print(per_model.groupby("model").mean().round(3).to_string())
    others = pd.DataFrame(prow + extra_rows)
    if len(others):
        print("\nPseudo-labeller / label-free methods vs hand labels (mean over tiles; counts summed over tiles):")
        agg = {**{k: "mean" for k in ["dice", "iou", "f1", "aji", "ap50"]}, **{k: "sum" for k in ["tp", "fp", "fn"]}}
        print(others.groupby("model")[show].agg(agg).round(3).to_string())
    print(f"\nSaved to {a.out}/. Send v3_tiles.csv to Vedant.")


if __name__ == "__main__":
    main()
