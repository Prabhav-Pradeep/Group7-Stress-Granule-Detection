"""
Scores split by granule SIZE (area and perimeter) + perimeter statistics + size over time + red heatmaps.

Put this file next to rahul_5fold_ablation.py and score_video3.py and run it from that folder.
It uses the same models, the same COCO loader and the same matching rule as those scripts
(connected components, one-to-one Hungarian matching, IoU >= 0.5); it only measures every granule,
so the "all" rows equal the tp / fp / fn in v3_tiles.csv and runs.csv.

  # 1) final test: hand-labelled video-3 tiles (runs every checkpoint, plus the labeller and DoG masks)
  python size_binned_detection.py v3 \
      --coco   roboflow_v3/coco/unzipped/test \
      --images roboflow_v3/images \
      --ckpt   rahul_5fold_outputs_final \
      --pseudo roboflow_v3/pseudo_masks --extra DoG=masks_methodA \
      --out    size_bins_v3

  # 2) 5-fold test sets (pseudo-label ground truth); uses the predictions saved in <ckpt>/preds/, no GPU
  python size_binned_detection.py folds --ckpt rahul_5fold_outputs_final --root sg_dataset_5fold_final --out size_bins_folds

Size bins (change with --area-bins / --perim-bins; an open last bin is always added):
  area      0-30, 30-60, 60-150, 150-400, 400-1500, >=1500 px
  perimeter 0-20, 20-30, 30-45, 45-70, 70-140, >=140 px   (a disc of 30 / 60 / 150 / 400 / 1500 px has
                                                         a perimeter of about 19 / 27 / 43 / 71 / 137 px)
Metrics per bin:
  n_gt, n_pred  ground-truth granules / predicted blobs in the bin
  tp, fn        ground-truth granules of that size that were matched / missed
  fp            predicted blobs of that size that match nothing
  recall        tp / n_gt                          (ground truth binned by ITS size)
  precision     matched blobs / n_pred             (predictions binned by THEIR size)
  f1            2 * precision * recall / (precision + recall)
  capture       >= 50 % of the granule's pixels predicted as granule (counts granules found but drawn
                too big or merged with a neighbour, which recall does not)
  merged        captured, but the predicted blob also covers another ground-truth granule
  perim_err     median |P_pred - P_gt| / P_gt of matched pairs (%); perim_bias: median signed (%)
Perimeter = skimage.measure.perimeter (4-neighbourhood) on each granule's own mask. Granules cut by the
tile edge are scored normally but left out of the perimeter statistics and the time trend.

Time: frame -> minutes from VIDEO_MINUTES below (taken from the video file names: v1 "0 to 1 hrs",
v2 "1 to 2 hrs", v3 "1 to 1h42m"); edit it if the real frame interval is known.

Writes to --out:
  bins_area_summary.csv, bins_perimeter_summary.csv   mean +- std over checkpoints per model and bin  <- slide tables
  bins_area_per_ckpt.csv, bins_perimeter_per_ckpt.csv per checkpoint (= per fold) and bin
  perimeter_distribution.csv   % of granules per perimeter range + mean / median perimeter and area
  size_over_time.csv           per video and frame: granules per tile, mean / median perimeter and area
  size_trend.csv               slope (px per hour), Pearson r, p and Spearman rho of mean perimeter vs time
  granules_gt.csv, granules_pred.csv   one row per granule per checkpoint (everything above comes from these)
  scores_by_size.png, perimeter_distribution.png, size_over_time.png
  heatmap_red_tiles.png        every tested tile: image + granules in red, for ground truth and each method
  heatmap_red_occupancy.png    (folds) full frame: how often each pixel is granule, red scale
  heatmap_red_frame_<video>.png (folds) one whole frame stitched from its tiles, granules in red
"""
import argparse, glob, json, os, re, sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as cc_label, perimeter, regionprops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import rahul_5fold_ablation as R               # models, predict_tiles, read_gray, compact_labels

AREA_BINS = [0, 30, 60, 150, 400, 1500]
PERIM_BINS = [0, 20, 30, 45, 70, 140]
VIDEO_MINUTES = {"v1": (0, 60), "v2": (60, 120), "v3": (60, 102)}     # (start, end) of each video in minutes
NAME_RE = re.compile(r"^(v\d+)_t(\d+)_y(\d+)_x(\d+)")
GT = "ground truth"


def bin_labels(edges):
    return [f">={lo:g}" if np.isinf(hi) else f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]


def parse_name(name):
    m = NAME_RE.match(name)
    return (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))) if m else (None, None, 0, 0)


# ----------------------------------------------------------------------------- per-granule matching
def _shape_rows(lbl, n):
    H, W = lbl.shape
    rows = [None] * n
    for r in regionprops(lbl):
        y0, x0, y1, x1 = r.bbox
        rows[r.label - 1] = dict(area=int(r.area), perimeter=float(perimeter(np.pad(r.image, 1), 4)),
                                 eq_diam=float(r.equivalent_diameter_area),
                                 edge=bool(y0 == 0 or x0 == 0 or y1 == H or x1 == W))
    return rows


def match_tile(pred_bin, gt_inst, iou_thresh=0.5):
    """Same matching as R.evaluate_tiles; returns one record per GT granule and per predicted blob."""
    p_lbl, n_p = cc_label(pred_bin, return_num=True)
    p_lbl = p_lbl.astype(np.int64)
    g_lbl, n_g = R.compact_labels(gt_inst)
    gt_rows, pr_rows = _shape_rows(g_lbl, n_g), _shape_rows(p_lbl, n_p)
    for r in gt_rows:
        r.update(matched=False, iou=0.0, captured=False, merged=False, pred_perimeter=np.nan)
    for r in pr_rows:
        r.update(matched=False, iou=0.0)
    if n_p and n_g:
        cont = np.bincount((p_lbl * (n_g + 1) + g_lbl).ravel(),
                           minlength=(n_p + 1) * (n_g + 1)).reshape(n_p + 1, n_g + 1)
        area_p, area_g, inter = cont.sum(1)[1:], cont.sum(0)[1:], cont[1:, 1:]
        iou = inter / np.maximum(area_p[:, None] + area_g[None, :] - inter, 1)
        r, c = linear_sum_assignment(1 - iou)
        for i, j in zip(r, c):
            if iou[i, j] >= iou_thresh:
                gt_rows[j].update(matched=True, iou=float(iou[i, j]), pred_perimeter=pr_rows[i]["perimeter"])
                pr_rows[i].update(matched=True, iou=float(iou[i, j]))
        covered, gts_per_blob = inter.sum(0), (inter > 0).sum(1)
        for j in range(n_g):
            if covered[j] >= 0.5 * area_g[j]:
                gt_rows[j]["captured"] = True
                gt_rows[j]["merged"] = bool(gts_per_blob[int(np.argmax(inter[:, j]))] > 1)
            if not gt_rows[j]["matched"]:
                gt_rows[j]["iou"] = float(iou[:, j].max())
    return gt_rows, pr_rows


# ----------------------------------------------------------------------------- collecting results
class Collector:
    """Per-granule rows + red-heatmap accumulators, filled tile by tile."""

    def __init__(self, keep_frames=None):
        self.gt_rows, self.pr_rows = [], []
        self.occ, self.cover = {}, {}              # (source, video) -> full-frame arrays
        self.keep_frames = keep_frames or {}       # video -> frame whose tiles are kept for the stitched figure
        self.frame_tiles = {}                      # (source, video) -> {(y, x): mask}
        self.gt_seen = set()

    def _accumulate(self, source, name, mask):
        video, frame, y, x = parse_name(name)
        if video is None:
            return
        key = (source, video)
        if key not in self.occ:
            self.occ[key] = np.zeros((1024, 1024), np.float32); self.cover[key] = np.zeros((1024, 1024), np.float32)
        h, w = mask.shape
        if y + h > self.occ[key].shape[0] or x + w > self.occ[key].shape[1]:
            pad = ((0, max(0, y + h - self.occ[key].shape[0])), (0, max(0, x + w - self.occ[key].shape[1])))
            self.occ[key] = np.pad(self.occ[key], pad); self.cover[key] = np.pad(self.cover[key], pad)
        self.occ[key][y:y + h, x:x + w] += mask > 0
        self.cover[key][y:y + h, x:x + w] += 1
        if self.keep_frames.get(video) == frame:
            self.frame_tiles.setdefault(key, {})[(y, x)] = mask > 0

    def add(self, model, ckpt, names, preds, gts, fold=None):
        for name, p, g in zip(names, preds, gts):
            gr, prr = match_tile(p, g)
            video, frame, y, x = parse_name(name)
            meta = dict(model=model, ckpt=ckpt, fold=fold, tile=name, video=video, frame=frame)
            self.gt_rows += [{**meta, **r} for r in gr]
            self.pr_rows += [{**meta, **r} for r in prr]
            self._accumulate(model + "|" + ckpt, name, p)
            if name not in self.gt_seen:            # ground truth once per tile
                self.gt_seen.add(name)
                self._accumulate(GT, name, g > 0)


# ----------------------------------------------------------------------------- tables
def bin_table(gt, pr, key, edges):
    labels = bin_labels(edges)
    gb = pd.cut(gt[key], edges, right=False, labels=labels)
    pb = pd.cut(pr[key], edges, right=False, labels=labels)
    rows = []
    for b in labels + ["all"]:
        g = gt if b == "all" else gt[gb == b]
        p = pr if b == "all" else pr[pb == b]
        tp, n_gt, n_pred = int(g.matched.sum()), len(g), len(p)
        rec = tp / n_gt if n_gt else np.nan
        prec = p.matched.sum() / n_pred if n_pred else np.nan
        m = g[g.matched]
        rel = (m.pred_perimeter - m.perimeter) / m.perimeter
        rows.append(dict(bin=b, n_gt=n_gt, n_pred=n_pred, tp=tp, fn=n_gt - tp, fp=int((~p.matched).sum()),
                         recall=rec, precision=prec,
                         f1=2 * prec * rec / (prec + rec) if (prec + rec) > 0 else (0.0 if n_gt and n_pred else np.nan),
                         capture=g.captured.mean() if n_gt else np.nan, merged=g.merged.mean() if n_gt else np.nan,
                         perim_err=100 * rel.abs().median() if len(m) else np.nan,
                         perim_bias=100 * rel.median() if len(m) else np.nan))
    return pd.DataFrame(rows)


def model_order(models):
    known = [m for m in list(R.MODELS) + ["Pseudo", "DoG"] if m in models]
    return known + sorted(set(models) - set(known))


def binned(gt, pr, key, edges):
    per = []
    for (model, ck), g in gt.groupby(["model", "ckpt"], sort=False):
        p = pr[(pr.model == model) & (pr.ckpt == ck)]
        per.append(bin_table(g, p, key, edges).assign(model=model, ckpt=ck, fold=g.fold.iloc[0]))
    per = pd.concat(per, ignore_index=True)
    labels = bin_labels(edges) + ["all"]
    rows = []
    for (model, b), g in per.groupby(["model", "bin"], sort=False):
        row = dict(model=model, bin=b, ckpts=len(g))
        for k in ("n_gt", "n_pred", "tp", "fp", "fn"):
            row[k] = round(g[k].mean(), 1)
        for k in ("recall", "precision", "f1", "capture", "merged", "perim_err", "perim_bias"):
            v = g[k].dropna()
            row[k] = round(v.mean(), 3) if len(v) else np.nan
            row[k + "_std"] = round(v.std(), 3) if len(v) > 1 else np.nan
        rows.append(row)
    s = pd.DataFrame(rows)
    s["model"] = pd.Categorical(s.model, model_order(set(s.model)), ordered=True)
    s["bin"] = pd.Categorical(s.bin, labels, ordered=True)
    per["bin"] = pd.Categorical(per.bin, labels, ordered=True)
    return per.sort_values(["model", "ckpt", "bin"]), s.sort_values(["model", "bin"]).reset_index(drop=True)


def granule_sources(gt, pr):
    """{source: DataFrame of granules}: the ground truth once per tile, each model's predictions (all ckpts)."""
    first = gt.drop_duplicates("tile")[["tile", "model", "ckpt"]]        # the GT rows of one run per tile
    out = {GT: gt.merge(first, on=["tile", "model", "ckpt"])}
    for m in model_order(set(pr.model)):
        out[m] = pr[pr.model == m]
    return {k: v[~v.edge] for k, v in out.items()}                     # whole granules only


def perimeter_distribution(src, edges):
    labels = bin_labels(edges)
    rows = []
    for name, d in src.items():
        share = pd.cut(d.perimeter, edges, right=False, labels=labels).value_counts(normalize=True).reindex(labels).fillna(0)
        rows.append(dict(source=name, granules=len(d), mean_perimeter=d.perimeter.mean(), median_perimeter=d.perimeter.median(),
                         p10=d.perimeter.quantile(0.1), p90=d.perimeter.quantile(0.9), mean_area=d.area.mean(),
                         median_area=d.area.median(), **{f"% P {l}": 100 * share[l] for l in labels}))
    return pd.DataFrame(rows).round(2)


def to_minutes(video, frame, n_frames):
    lo, hi = VIDEO_MINUTES.get(video, (0, n_frames - 1))
    return lo + (hi - lo) * frame / max(n_frames - 1, 1)


def size_over_time(src, n_frames):
    rows = []
    for name, d in src.items():
        d = d.dropna(subset=["video"])
        tiles = d.groupby(["video", "frame"]).tile.nunique()
        for (v, f), g in d.groupby(["video", "frame"]):
            rows.append(dict(source=name, video=v, frame=int(f), minutes=to_minutes(v, f, n_frames[v]),
                             granules=len(g), granules_per_tile=len(g) / tiles[(v, f)],
                             mean_perimeter=g.perimeter.mean(), median_perimeter=g.perimeter.median(),
                             mean_area=g.area.mean(), median_area=g.area.median()))
    return pd.DataFrame(rows)


def size_trend(t):
    rows = []
    groups = [(s, v, g) for (s, v), g in t.groupby(["source", "video"])]
    groups += [(s, "all videos", g) for s, g in t.groupby("source") if g.video.nunique() > 1]
    for s, v, g in groups:
        if len(g) < 3:
            rows.append(dict(source=s, video=v, frames=len(g), note="fewer than 3 frames - no trend")); continue
        for k in ("mean_perimeter", "mean_area", "granules_per_tile"):
            lr = stats.linregress(g.minutes, g[k])
            rho, p_s = stats.spearmanr(g.minutes, g[k])
            rows.append(dict(source=s, video=v, frames=len(g), quantity=k, slope_per_hour=60 * lr.slope,
                             start=lr.intercept + lr.slope * g.minutes.min(), end=lr.intercept + lr.slope * g.minutes.max(),
                             pearson_r=lr.rvalue, p=lr.pvalue, spearman_rho=rho, spearman_p=p_s,
                             trend=("increases" if lr.slope > 0 else "decreases") if lr.pvalue < 0.05 else "no significant change"))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_scores(summaries, path, title):
    plt = _plt()
    keys = ("recall", "precision", "f1", "capture")
    fig, axes = plt.subplots(len(summaries), len(keys), figsize=(4.2 * len(keys), 3.6 * len(summaries)), squeeze=False)
    for row, (by, s) in zip(axes, summaries.items()):
        s = s[s.bin != "all"]
        labels = list(s.bin.cat.categories[:-1])
        n = s.groupby("bin", observed=True).n_gt.max().reindex(labels)
        x = np.arange(len(labels))
        for ax, k in zip(row, keys):
            for model, g in s.groupby("model", observed=True, sort=False):
                g = g.set_index("bin").reindex(labels)
                ax.errorbar(x, g[k], yerr=g[k + "_std"], marker="o", ms=4, capsize=2, label=str(model))
            ax.set_xticks(x)
            ax.set_xticklabels([f"{l}\nn={0 if pd.isna(c) else int(c)}" for l, c in zip(labels, n)], fontsize=7)
            ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.set_title(f"{k} by {by}", fontsize=10)
            ax.set_xlabel(f"ground-truth {by} (px)" if k != "precision" else f"predicted {by} (px)", fontsize=8)
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_distribution(src, edges, path, title):
    plt = _plt()
    labels = bin_labels(edges)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    w = 0.8 / len(src); x = np.arange(len(labels))
    for k, (name, d) in enumerate(src.items()):
        share = 100 * pd.cut(d.perimeter, edges, right=False, labels=labels).value_counts(normalize=True).reindex(labels).fillna(0)
        axes[0].bar(x + (k - (len(src) - 1) / 2) * w, share, w, label=name,
                    color="#c62828" if name == GT else None)
        axes[1].hist(d.perimeter, bins=np.linspace(0, 200, 51), histtype="step", lw=2 if name == GT else 1.2,
                     density=True, label=f"{name} (median {d.perimeter.median():.0f})", color="#c62828" if name == GT else None)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels); axes[0].set_xlabel("perimeter (px)")
    axes[0].set_ylabel("% of granules"); axes[0].legend(fontsize=7, frameon=False); axes[0].set_title("granules per perimeter range")
    axes[1].set_xlabel("perimeter (px)"); axes[1].set_ylabel("density"); axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_title("perimeter distribution (granules not cut by the tile edge)")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_time(t, path, title):
    plt = _plt()
    qs = [("mean_perimeter", "mean perimeter (px)"), ("mean_area", "mean area (px)"), ("granules_per_tile", "granules per tile")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (k, lab) in zip(axes, qs):
        for s, g in t.groupby("source", sort=False):
            for v, gv in g.groupby("video"):
                gv = gv.sort_values("minutes")
                ax.plot(gv.minutes, gv[k], marker="o" if len(gv) < 10 else None, ms=3,
                        color="#c62828" if s == GT else None, lw=2 if s == GT else 1,
                        label=s if v == g.video.min() else None)
        ax.set_xlabel("time (min, from the video names)"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def red_overlay(img, mask, alpha=0.6):
    g = img.astype(np.float32)
    lo, hi = np.percentile(g, 1), np.percentile(g, 99.5)
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([g] * 3, -1)
    m = mask > 0
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array([1.0, 0.0, 0.0])
    return rgb


def plot_red_tiles(tiles, panels, path, title, max_tiles=12):
    """tiles: [(name, img, gt_inst)]; panels: {method: [mask per tile]}"""
    plt = _plt()
    tiles = tiles[:max_tiles]
    cols = ["image", GT] + list(panels)
    fig, axes = plt.subplots(len(tiles), len(cols), figsize=(2.3 * len(cols), 2.4 * len(tiles)), squeeze=False)
    for i, (name, img, g) in enumerate(tiles):
        ims = [red_overlay(img, np.zeros_like(g)), red_overlay(img, g)] + [red_overlay(img, panels[m][i]) for m in panels]
        for j, (ax, im) in enumerate(zip(axes[i], ims)):
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(name, fontsize=7)
    fig.suptitle(title + "   (red = granule)"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_red_occupancy(col, sources, path, title):
    plt = _plt()
    videos = sorted({v for (_, v) in col.occ})
    fig, axes = plt.subplots(len(videos), len(sources), figsize=(3.3 * len(sources), 3.3 * len(videos)), squeeze=False)
    for i, v in enumerate(videos):
        for j, s in enumerate(sources):
            ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
            if (s, v) not in col.occ:
                continue
            occ = np.where(col.cover[(s, v)] > 0, col.occ[(s, v)] / np.maximum(col.cover[(s, v)], 1), np.nan)
            im = ax.imshow(np.ma.masked_invalid(occ), cmap="Reds", vmin=0, vmax=np.nanpercentile(occ, 99.5) or 1)
            if i == 0:
                ax.set_title(s.split("|")[0], fontsize=9)
            if j == 0:
                ax.set_ylabel(v)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="fraction of frames the pixel is granule")
    fig.suptitle(title); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_red_frame(col, video, frame, images, sources, path):
    plt = _plt()
    canvas = np.zeros((1024, 1024), np.float32); seen = np.zeros((1024, 1024), bool)
    for (y, x), im in images.items():
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im; seen[y:y + im.shape[0], x:x + im.shape[1]] = True
    fig, axes = plt.subplots(1, len(sources), figsize=(4.6 * len(sources), 4.8), squeeze=False)
    for ax, s in zip(axes[0], sources):
        m = np.zeros_like(seen)
        for (y, x), t in col.frame_tiles.get((s, video), {}).items():
            m[y:y + t.shape[0], x:x + t.shape[1]] = t
        rgb = red_overlay(canvas, m); rgb[~seen] = 0.15
        ax.imshow(rgb); ax.set_title(s.split("|")[0], fontsize=10); ax.axis("off")
    fig.suptitle(f"{video} frame {frame}: whole frame stitched from its test tiles (red = granule, grey = tile not in any test set)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------- the two modes
def run_v3(a, col):
    import torch
    import score_video3 as S
    tiles = S.load_hand_labels(a.coco, a.images)
    imgs = np.stack([t["img"] for t in tiles]); gts = [t["inst"] for t in tiles]; names = [t["name"] for t in tiles]
    cfg = os.path.join(a.ckpt, "config.json")
    base = a.base or (json.load(open(cfg))["base"] if os.path.exists(cfg) else 16)
    pat = re.compile(r"^(\w+)_fold(\d+)_seed(\d+)\.pt$")
    ck = [p for p in sorted(glob.glob(os.path.join(a.ckpt, "ckpt", "*.pt"))) if pat.match(os.path.basename(p))]
    if not ck:
        raise SystemExit(f"no <Model>_fold<j>_seed<s>.pt files in {a.ckpt}/ckpt")
    panels = {}
    for n, path in enumerate(ck, 1):
        mt = pat.match(os.path.basename(path))
        model = R.UNet(base=base, **R.MODELS[mt.group(1)])
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.to(R.DEVICE)
        preds = R.predict_tiles(model, imgs)
        col.add(mt.group(1), f"fold{mt.group(2)}_seed{mt.group(3)}", names, preds, gts, fold=int(mt.group(2)))
        if int(mt.group(2)) == a.overlay_fold and int(mt.group(3)) == a.overlay_seed:
            panels[mt.group(1)] = list(preds)
        print(f"[{n}/{len(ck)}] {os.path.basename(path)}", flush=True)
    for spec in ([f"Pseudo={a.pseudo}"] if a.pseudo else []) + list(a.extra):
        name, folder = spec.split("=", 1)
        keep = [k for k, t in enumerate(names) if os.path.exists(os.path.join(folder, t + ".png"))]
        if len(keep) < len(names):
            print(f"WARNING: {name} has masks for {len(keep)} of {len(names)} tiles; scored on those only")
        masks = {k: R.read_gray(os.path.join(folder, names[k] + ".png")) > 0 for k in keep}
        col.add(name, "single", [names[k] for k in keep], [masks[k] for k in keep], [gts[k] for k in keep])
        panels[name] = [masks.get(k, np.zeros(gts[k].shape, bool)) for k in range(len(names))]
    order = model_order(set(panels))
    plot_red_tiles([(t["name"], t["img"], t["inst"]) for t in tiles], {m: panels[m] for m in order},
                   os.path.join(a.out, "heatmap_red_tiles.png"),
                   f"video-3 hand-labelled tiles: hand labels vs fold {a.overlay_fold} seed {a.overlay_seed} models")
    return "video-3 hand labels", None


def run_folds(a, col):
    pat = re.compile(r"^(\w+)_fold(\d+)_seed(\d+)\.npz$")
    files = [p for p in sorted(glob.glob(os.path.join(a.ckpt, "preds", "*.npz"))) if pat.match(os.path.basename(p))]
    if not files:
        raise SystemExit(f"no <Model>_fold<j>_seed<s>.npz files in {a.ckpt}/preds")
    # one example frame per video (the middle test frame) for the stitched red figure
    all_names = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(a.root, "fold_*", "test", "images", "*.png"))]
    frames = {}
    for n in all_names:
        v, f, _, _ = parse_name(n)
        if v:
            frames.setdefault(v, set()).add(f)
    col.keep_frames = {v: sorted(fs)[len(fs) // 2] for v, fs in frames.items()}
    present = {pat.match(os.path.basename(p)).group(1) for p in files}
    ov_model = a.overlay_model if a.overlay_model in present else model_order(present)[0]
    sample, gt_cache = [], {}
    for n, path in enumerate(files, 1):
        mt = pat.match(os.path.basename(path))
        model, fold, seed = mt.group(1), int(mt.group(2)), int(mt.group(3))
        z = np.load(path, allow_pickle=True)
        names = [str(s) for s in z["names"]]
        if fold not in gt_cache:                                           # one fold in memory at a time
            d = os.path.join(a.root, f"fold_{fold}", "test", "instances")
            gt_cache = {fold: {s: R.read_gray(os.path.join(d, s + ".png")) for s in names}}
        preds = z["pred"] > 0
        col.add(model, f"fold{fold}_seed{seed}", names, preds, [gt_cache[fold][s] for s in names], fold=fold)
        if seed == a.overlay_seed and model == ov_model:
            idx = np.random.default_rng(fold).choice(len(names), size=min(2, len(names)), replace=False)
            img_dir = os.path.join(a.root, f"fold_{fold}", "test", "images")
            sample += [(names[k], R.read_gray(os.path.join(img_dir, names[k] + ".png")), gt_cache[fold][names[k]], preds[k]) for k in idx]
        print(f"[{n}/{len(files)}] {os.path.basename(path)}", flush=True)
    if sample:
        plot_red_tiles([(s[0], s[1], s[2]) for s in sample], {ov_model: [s[3] for s in sample]},
                       os.path.join(a.out, "heatmap_red_tiles.png"), f"5-fold test tiles (2 per fold): pseudo-labels vs {ov_model}")
    n_frames = {v: max(fs) + 1 for v, fs in frames.items()}
    return "5-fold test sets (pseudo-label ground truth)", n_frames


def main():
    ap = argparse.ArgumentParser(description="Scores split by granule size + perimeter statistics + red heatmaps")
    ap.add_argument("mode", choices=["v3", "folds"])
    ap.add_argument("--ckpt", required=True, help="5-fold output folder (has ckpt/ and preds/)")
    ap.add_argument("--coco", help="v3: folder with _annotations.coco.json")
    ap.add_argument("--images", help="v3: roboflow_v3/images")
    ap.add_argument("--pseudo", help="v3: roboflow_v3/pseudo_masks (optional)")
    ap.add_argument("--extra", nargs="*", default=[], help="v3: Name=mask_folder, e.g. DoG=masks_methodA")
    ap.add_argument("--root", help="folds: the 5-fold dataset folder (fold_<j>/test/images and instances)")
    ap.add_argument("--base", type=int, default=None)
    ap.add_argument("--area-bins", type=float, nargs="+", default=AREA_BINS)
    ap.add_argument("--perim-bins", type=float, nargs="+", default=PERIM_BINS)
    ap.add_argument("--overlay-fold", type=int, default=1, help="v3: which fold's models to draw in the red figure")
    ap.add_argument("--overlay-seed", type=int, default=0)
    ap.add_argument("--overlay-model", default="UNet", help="folds: model drawn in the red tile / frame figures")
    ap.add_argument("--out", default="size_bins")
    a = ap.parse_args()
    a_edges = sorted(a.area_bins) + [np.inf]
    p_edges = sorted(a.perim_bins) + [np.inf]
    os.makedirs(a.out, exist_ok=True)

    col = Collector()
    if a.mode == "v3":
        if not (a.coco and a.images):
            raise SystemExit("v3 mode needs --coco and --images")
        title, n_frames = run_v3(a, col)
    else:
        if not a.root:
            raise SystemExit("folds mode needs --root")
        title, n_frames = run_folds(a, col)

    gt, pr = pd.DataFrame(col.gt_rows), pd.DataFrame(col.pr_rows)
    gt.to_csv(os.path.join(a.out, "granules_gt.csv"), index=False)
    pr.to_csv(os.path.join(a.out, "granules_pred.csv"), index=False)

    # 1) scores per size bin
    summaries = {}
    for by, key, edges in (("area", "area", a_edges), ("perimeter", "perimeter", p_edges)):
        per, summ = binned(gt, pr, key, edges)
        per.to_csv(os.path.join(a.out, f"bins_{by}_per_ckpt.csv"), index=False)
        summ.to_csv(os.path.join(a.out, f"bins_{by}_summary.csv"), index=False)
        summaries[by] = summ
    plot_scores(summaries, os.path.join(a.out, "scores_by_size.png"), f"Scores by granule size - {title}")

    # 2) perimeter statistics
    src = granule_sources(gt, pr)
    dist = perimeter_distribution(src, p_edges)
    dist.to_csv(os.path.join(a.out, "perimeter_distribution.csv"), index=False)
    plot_distribution(src, p_edges, os.path.join(a.out, "perimeter_distribution.png"), f"Granule perimeters - {title}")

    # 3) size over time
    if n_frames is None:
        n_frames = {v: int(g.frame.max()) + 1 for v, g in gt.dropna(subset=["video"]).groupby("video")}
        n_frames = {v: max(n, 148 if v == "v3" else n) for v, n in n_frames.items()}   # video 3 has 148 frames
    t = size_over_time(src, n_frames)
    trend = size_trend(t) if len(t) else pd.DataFrame()
    t.to_csv(os.path.join(a.out, "size_over_time.csv"), index=False)
    trend.to_csv(os.path.join(a.out, "size_trend.csv"), index=False)
    if len(t):
        plot_time(t, os.path.join(a.out, "size_over_time.png"), f"Granule size over time - {title}")

    # 4) red heatmaps over the whole frame (folds: every tile of v1/v2 is in exactly one test set)
    if a.mode == "folds":
        # every v1/v2 tile is in exactly one fold's test set: pool each model's folds into one full-frame map
        pooled = {}
        for (s, v) in [k for k in col.occ if k[0] != GT]:
            m = s.split("|")[0]
            o, c = pooled.get((m, v), (0, 0))
            pooled[(m, v)] = (o + col.occ[(s, v)], c + col.cover[(s, v)])
        for (m, v), (o, c) in pooled.items():
            col.occ[(m, v)], col.cover[(m, v)] = o, c
        models = model_order({m for m, _ in pooled})
        plot_red_occupancy(col, [GT] + models, os.path.join(a.out, "heatmap_red_occupancy.png"),
                           "Where granules are: fraction of frames each pixel is granule (test folds pooled)")
        ov = a.overlay_model if a.overlay_model in models else models[0]
        for v, f in col.keep_frames.items():
            imgs = {}
            for p in glob.glob(os.path.join(a.root, "fold_*", "test", "images", f"{v}_t{f:03d}_*.png")):
                _, _, y, x = parse_name(os.path.splitext(os.path.basename(p))[0])
                imgs[(y, x)] = R.read_gray(p)
            merged = {}                     # the frame's tiles were predicted by different folds' models
            for (s, vv), d in list(col.frame_tiles.items()):
                if vv == v and s.startswith(ov + "|") and s.endswith(f"seed{a.overlay_seed}"):
                    merged.update(d)
            col.frame_tiles[(ov, v)] = merged
            plot_red_frame(col, v, f, imgs, [GT, ov], os.path.join(a.out, f"heatmap_red_frame_{v}.png"))

    # console summary
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    show = ["model", "bin", "n_gt", "n_pred", "tp", "fp", "fn", "recall", "precision", "f1", "capture", "merged", "perim_err"]
    for by, s in summaries.items():
        print(f"\n=== {title}: scores by {by} (mean over checkpoints) ===")
        print(s[show].to_string(index=False))
    print("\n=== perimeter statistics (granules not cut by the tile edge) ===")
    print(dist.to_string(index=False))
    if len(trend) and "quantity" in trend:
        print("\n=== size over time (linear fit of the per-frame mean vs minutes) ===")
        print(trend[trend.quantity.isin(["mean_perimeter", "mean_area"])][
            ["source", "video", "frames", "quantity", "start", "end", "slope_per_hour", "pearson_r", "p", "trend"]].round(4).to_string(index=False))
    print(f"\nSaved to {a.out}/")


if __name__ == "__main__":
    main()
