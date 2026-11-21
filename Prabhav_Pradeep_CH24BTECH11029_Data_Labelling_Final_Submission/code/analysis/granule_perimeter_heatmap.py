"""Granule perimeters + heatmaps from ANY segmentation model's output on 256 x 256 tiles.

Standalone: needs only numpy, scipy, scikit-image, opencv-python, pandas, matplotlib.
It does not care which model made the prediction. Whatever the model returns per tile, give it to
`to_instances()`; it accepts

    * a probability map      float H x W in [0, 1]       (U-Net sigmoid output)   -> thresholded
    * a binary mask          bool / uint8 0-1 / 0-255    (saved mask PNG)
    * an instance image      int H x W, 0 = background, 1..N = granules (Mask R-CNN, StarDist, ...)
    * a list of polygons     [ (K x 2) array of (x, y) points, ... ]   (YOLO-seg, Roboflow, SAM)

The boundary of every granule is then measured on its own (touching granules in a plain mask are
split first with a distance-map watershed).

-------------------------------------------------------------------------------------------------
Plug in your model (Python):

    import granule_perimeter_heatmap as GP

    def predict(img_uint8):                 # H x W uint8 tile -> any of the formats above
        ...                                 # e.g. torch.sigmoid(model(x))[0, 0].cpu().numpy()
        return prob

    tables, inst = GP.measure_folder("fold_1/test/images", predict=predict, out_dir="results_fold1",
                                     pixel_um=None)     # set pixel_um (micrometre per pixel) if known

Or from masks the model already saved (one PNG / .npy per tile, same file name as the image):

    python granule_perimeter_heatmap.py --images fold_1/test/images --pred my_pred_masks --out results_fold1

Sanity check without a model (the pseudo-label instances act as the "prediction"):

    python granule_perimeter_heatmap.py --images fold_1/test/images --pred fold_1/test/instances --out check

Outputs in --out:
    granules.csv          one row per granule: tile, id, area, perimeter, circularity, centroid, ...
    tiles.csv             one row per tile: granule count, median / mean perimeter
    summary.txt           overall numbers (granules touching the tile edge are left out, their
                          outline is cut by the tile border)
    overlays/<tile>.png   each granule filled with a colour = its perimeter (the "per-tile heatmap")
    heatmap_occupancy.png fraction of tiles in which each pixel of the full frame is granule
    heatmap_density.png   granule centres per tile, Gaussian-smoothed, on the full frame
    heatmap_perimeter.png mean perimeter of the granules found around each place in the frame
    perimeter_hist.png    distribution of perimeters
-------------------------------------------------------------------------------------------------
Perimeter: skimage.measure.perimeter (4-neighbourhood) on the pixel mask, which is within ~5 % of
the true length for digital discs of radius 3-22 px (30-1500 px area). For polygon input the
polygon's own length is reported as well (`perimeter_polygon_px`).
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, measure, segmentation

TILE_RE = re.compile(r"_y(\d+)_x(\d+)")        # <video>_t<frame>_y<row>_x<col> -> tile position


# ============================================================================ model output -> instances
def to_instances(pred, threshold=0.5, min_area=30, max_area=1500, split=True, min_distance=4,
                 shape=None):
    """Any model output (see module doc) -> (instance image int32, {id: polygon length} or {})."""
    poly_len = {}
    if isinstance(pred, (list, tuple)):                       # polygons
        if shape is None:
            raise ValueError("polygons need shape=(H, W)")
        inst = np.zeros(shape, np.int32)
        for i, p in enumerate(pred, 1):
            p = np.asarray(p, np.float64).reshape(-1, 2)
            cv2.fillPoly(inst, [np.round(p).astype(np.int32)], i)
            poly_len[i] = float(np.sum(np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)))
        return _filter(inst, min_area, max_area, poly_len)
    pred = np.asarray(pred)
    if pred.ndim == 3:                                        # (1, H, W) / (H, W, 1) / RGB mask
        pred = pred.squeeze() if 1 in pred.shape else pred[..., 0]
    if np.issubdtype(pred.dtype, np.floating):
        mask = pred >= threshold
    elif pred.dtype == bool:
        mask = pred
    elif pred.max() <= 1 or (pred.dtype == np.uint8 and set(np.unique(pred)) <= {0, 255}):
        mask = pred > 0
    else:                                                     # already instance labels
        return _filter(pred.astype(np.int32), min_area, max_area, poly_len)
    return _filter(split_touching(mask, min_distance) if split else measure.label(mask, connectivity=2),
                   min_area, max_area, poly_len)


def split_touching(mask, min_distance=4):
    """Binary mask -> instances; touching granules split by watershed on the distance map."""
    mask = ndi.binary_fill_holes(mask)
    lab = measure.label(mask, connectivity=2)
    dist = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(dist, min_distance=min_distance, labels=lab, exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    out = segmentation.watershed(-dist, markers, mask=mask)
    missed = (lab > 0) & (out == 0)                           # components with no peak: keep whole
    if missed.any():
        out[missed] = measure.label(missed)[missed] + out.max()
    return out


def _filter(inst, min_area, max_area, poly_len):
    """Keep granules with min_area <= area <= max_area (same rule as the pseudo-labels), renumber."""
    out = np.zeros_like(inst, np.int32)
    lens, n = {}, 0
    for r in measure.regionprops(inst):
        if min_area <= r.area <= max_area:
            n += 1
            out[tuple(r.coords.T)] = n
            if r.label in poly_len:
                lens[n] = poly_len[r.label]
    return out, lens


# ============================================================================ measurements
def granule_table(inst, tile_name="", pixel_um=None, poly_len=None):
    """One row per granule: area, perimeter (two estimators), circularity, size, position."""
    H, W = inst.shape
    rows = []
    for r in measure.regionprops(inst):
        m = np.pad(r.image, 1)                                # pad so the bbox edge is not "border"
        per = measure.perimeter(m, neighborhood=4)
        y0, x0, y1, x1 = r.bbox
        row = dict(tile=tile_name, granule=r.label, area_px=int(r.area),
                   perimeter_px=per, perimeter_crofton_px=measure.perimeter_crofton(m, directions=4),
                   equiv_diameter_px=r.equivalent_diameter_area,
                   circularity=min(1.0, 4 * np.pi * r.area / per ** 2) if per > 0 else np.nan,
                   centroid_y=r.centroid[0], centroid_x=r.centroid[1],
                   touches_border=bool(y0 == 0 or x0 == 0 or y1 == H or x1 == W))
        if poly_len and r.label in poly_len:
            row["perimeter_polygon_px"] = poly_len[r.label]
        if pixel_um:
            row["area_um2"] = row["area_px"] * pixel_um ** 2
            row["perimeter_um"] = per * pixel_um
            row["equiv_diameter_um"] = row["equiv_diameter_px"] * pixel_um
        rows.append(row)
    return pd.DataFrame(rows)


def tile_offset(name):
    """(y, x) of the tile in the full frame, read from '..._y0512_x0256'; (0, 0) if absent."""
    m = TILE_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


# ============================================================================ heatmaps
def perimeter_overlay(img, inst, table, vmax=None, ax=None, title=""):
    """Per-tile heatmap: each granule filled with a colour = its perimeter, outline drawn."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    ax.imshow(img, cmap="gray")
    per = np.full(int(inst.max()) + 1, np.nan)
    if len(table):
        per[table.granule.values] = table.perimeter_px.values
    val = np.where(inst > 0, per[inst], np.nan)
    vmax = vmax or (np.nanmax(per) if np.isfinite(per).any() else 1)
    im = ax.imshow(np.ma.masked_invalid(val), cmap="inferno", vmin=0, vmax=vmax, alpha=0.75)
    ax.contour(inst > 0, levels=[0.5], colors="cyan", linewidths=0.6)
    ax.set_title(title, fontsize=8); ax.axis("off")
    return im


def frame_heatmaps(tables, masks, frame_shape=None, tile=256, sigma=8.0):
    """Heatmaps over the full frame from many tiles (tile positions read from the names).

    occupancy : fraction of the tiles covering a pixel in which that pixel is granule
    density   : granule centres per 100 x 100 px per tile, Gaussian-smoothed (sigma px)
    perimeter : mean perimeter of granules whose centre lies near each pixel (NaN = none)
    """
    offs = {n: tile_offset(n) for n in masks}
    if frame_shape is None:
        frame_shape = (max(y for y, _ in offs.values()) + tile, max(x for _, x in offs.values()) + tile)
    occ = np.zeros(frame_shape); cover = np.zeros(frame_shape)
    cnt = np.zeros(frame_shape); psum = np.zeros(frame_shape)
    for n, inst in masks.items():
        y, x = offs[n]; h, w = inst.shape
        occ[y:y + h, x:x + w] += inst > 0
        cover[y:y + h, x:x + w] += 1
    df = pd.concat([t for t in tables.values() if len(t)], ignore_index=True) if tables else pd.DataFrame()
    for r in df.itertuples():
        y, x = offs[r.tile]
        cy = min(frame_shape[0] - 1, int(round(y + r.centroid_y)))
        cx = min(frame_shape[1] - 1, int(round(x + r.centroid_x)))
        cnt[cy, cx] += 1; psum[cy, cx] += r.perimeter_px
    with np.errstate(invalid="ignore", divide="ignore"):
        occupancy = np.where(cover > 0, occ / cover, np.nan)
        c_s, p_s = ndi.gaussian_filter(cnt, sigma), ndi.gaussian_filter(psum, sigma)
        cov_s = ndi.gaussian_filter(cover, sigma)
        density = np.where(cover > 0, 1e4 * c_s / np.maximum(cov_s, 1e-9), np.nan)   # per 100 x 100 px per tile
        perimeter = np.where((cover > 0) & (c_s > 1e-3 * max(c_s.max(), 1e-12)), p_s / np.maximum(c_s, 1e-12), np.nan)
    return dict(occupancy=occupancy, density=density, perimeter=perimeter, coverage=cover)


def save_heatmap(arr, path, title, label, cmap="magma", tile=256):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap)
    for k in range(tile, max(arr.shape), tile):               # tile grid
        ax.axhline(k - 0.5, color="w", lw=0.4, alpha=0.4); ax.axvline(k - 0.5, color="w", lw=0.4, alpha=0.4)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_title(title); ax.axis("off"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ============================================================================ whole folder
def _read_pred(path):
    if path.endswith(".npy"):
        return np.load(path)
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def measure_folder(images_dir, predict=None, pred_dir=None, out_dir="perimeter_results",
                   threshold=0.5, min_area=30, max_area=1500, split=True, min_distance=4,
                   pixel_um=None, n_overlays=24, sigma=8.0):
    """Run on every tile in `images_dir`, using `predict(img)` or saved predictions in `pred_dir`.
    Returns (granules DataFrame, {tile: instance image})."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert (predict is None) != (pred_dir is None), "give exactly one of predict / pred_dir"
    os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    assert paths, "no .png tiles in " + images_dir
    tables, masks, imgs = {}, {}, {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if predict is not None:
            pred = predict(img)
        else:
            hits = [os.path.join(pred_dir, name + e) for e in (".png", ".npy", ".tif")]
            hits = [h for h in hits if os.path.exists(h)]
            if not hits:
                print("no prediction for", name, "- skipped"); continue
            pred = _read_pred(hits[0])
        inst, plen = to_instances(pred, threshold, min_area, max_area, split, min_distance, img.shape)
        tables[name] = granule_table(inst, name, pixel_um, plen)
        masks[name], imgs[name] = inst, img

    df = pd.concat([t for t in tables.values() if len(t)], ignore_index=True) if any(len(t) for t in tables.values()) \
        else pd.DataFrame(columns=["tile", "granule", "area_px", "perimeter_px", "touches_border"])
    df.to_csv(os.path.join(out_dir, "granules.csv"), index=False)
    inner = df[~df.touches_border.astype(bool)] if len(df) else df
    per_tile = pd.DataFrame([dict(tile=n, granules=len(t),
                                  granules_inner=int((~t.touches_border).sum()) if len(t) else 0,
                                  median_perimeter_px=t.perimeter_px[~t.touches_border].median() if len(t) else np.nan,
                                  mean_perimeter_px=t.perimeter_px[~t.touches_border].mean() if len(t) else np.nan)
                             for n, t in tables.items()])
    per_tile.to_csv(os.path.join(out_dir, "tiles.csv"), index=False)

    unit = " (%.4g um/px)" % pixel_um if pixel_um else ""
    lines = ["tiles: %d" % len(tables), "granules: %d (%d not touching the tile edge)" % (len(df), len(inner))]
    if len(inner):
        q = inner.perimeter_px.quantile([0.1, 0.5, 0.9])
        lines += ["perimeter px  : median %.1f, mean %.1f, 10-90%% %.1f-%.1f" % (q[0.5], inner.perimeter_px.mean(), q[0.1], q[0.9]),
                  "area px       : median %.0f" % inner.area_px.median(),
                  "circularity   : median %.2f (1 = perfect circle)" % inner.circularity.median()]
        if pixel_um:
            lines.append("perimeter um%s: median %.2f, mean %.2f" % (unit, inner.perimeter_um.median(), inner.perimeter_um.mean()))
    open(os.path.join(out_dir, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # per-tile perimeter overlays, one shared colour scale
    vmax = float(inner.perimeter_px.quantile(0.98)) if len(inner) else None
    for n in list(tables)[:n_overlays]:
        fig, ax = plt.subplots(figsize=(4.2, 4))
        im = perimeter_overlay(imgs[n], masks[n], tables[n], vmax, ax, "%s  (%d granules)" % (n, len(tables[n])))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="perimeter (px)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "overlays", n + ".png"), dpi=130); plt.close(fig)

    # full-frame heatmaps (only meaningful when names carry _y####_x####)
    hm = frame_heatmaps({n: t[~t.touches_border] if len(t) else t for n, t in tables.items()}, masks, sigma=sigma)
    save_heatmap(hm["occupancy"], os.path.join(out_dir, "heatmap_occupancy.png"),
                 "granule occupancy (%d tiles)" % len(masks), "fraction of tiles that are granule here", "viridis")
    save_heatmap(hm["density"], os.path.join(out_dir, "heatmap_density.png"),
                 "granule density (centres, sigma %g px)" % sigma, "granules per 100 x 100 px per tile", "magma")
    save_heatmap(hm["perimeter"], os.path.join(out_dir, "heatmap_perimeter.png"),
                 "mean granule perimeter", "perimeter (px)", "inferno")
    if len(inner):
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.hist(inner.perimeter_px, bins=40, color="#4c72b0")
        ax.axvline(inner.perimeter_px.median(), color="k", ls="--", lw=1, label="median %.1f px" % inner.perimeter_px.median())
        ax.set_xlabel("perimeter (px)"); ax.set_ylabel("granules"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "perimeter_hist.png"), dpi=150); plt.close(fig)
    print("written to", os.path.abspath(out_dir))
    return df, masks


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", required=True, help="folder of tile PNGs (e.g. fold_1/test/images)")
    ap.add_argument("--pred", required=True, help="folder of predictions, same file names (.png/.npy/.tif)")
    ap.add_argument("--out", default="perimeter_results")
    ap.add_argument("--threshold", type=float, default=0.5, help="for probability maps")
    ap.add_argument("--min-area", type=int, default=30)
    ap.add_argument("--max-area", type=int, default=1500)
    ap.add_argument("--no-split", action="store_true", help="do not split touching granules")
    ap.add_argument("--pixel-um", type=float, default=None, help="micrometres per pixel, if known")
    ap.add_argument("--sigma", type=float, default=8.0, help="smoothing of the density heatmap (px)")
    a = ap.parse_args()
    measure_folder(a.images, pred_dir=a.pred, out_dir=a.out, threshold=a.threshold, min_area=a.min_area,
                   max_area=a.max_area, split=not a.no_split, pixel_um=a.pixel_um, sigma=a.sigma)
