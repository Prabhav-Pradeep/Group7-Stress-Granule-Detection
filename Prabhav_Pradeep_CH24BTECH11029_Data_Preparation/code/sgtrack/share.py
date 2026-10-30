"""Leave-one-video-out (LOVO) labelled dataset + checksummed team package.

Dataset layout written by `export_lovo_dataset` (all PNG, 256x256 tiles):

    sg_dataset/
      fold_test_v1/                 <- one folder per video held out
        train/  val/  test/
          images/       <name>.png  uint8 grayscale registered frame tile
          masks/        <name>.png  uint8 binary granule mask (0 / 255)
          instances/    <name>.png  uint16 granule instance labels (0 = background)
          cells/        <name>.png  uint16 cell instance labels
          labels_yolo/  <name>.txt  YOLO boxes (class 0 = granule)
      fold_test_v2/ ...
      to_label/        <name>.png   tiles for manual annotation (Roboflow)
      to_label_pseudo/ <name>.png   their pseudo-masks, for comparison
      tiles.csv  README.md
  <name> = <video>_t<frame>_y<row>_x<col>

Fold "test_vK": test = every tile of video K. Train / val come only from the OTHER videos;
val = the last `val_frac` of each training video's frames (a time block, used only for early
stopping / model selection). No frame of the test video is ever in train or val.
"""
import os
import json
import hashlib
import shutil
import zipfile
import numpy as np
import pandas as pd
import cv2
from skimage import measure, segmentation


SUBDIRS = ("images", "masks", "instances", "cells", "labels_yolo")


def _yolo_lines(inst, tile):
    out = []
    for r in measure.regionprops(inst):
        y0, x0, y1, x1 = r.bbox
        out.append("0 %.6f %.6f %.6f %.6f" % ((x0 + x1) / 2 / tile, (y0 + y1) / 2 / tile,
                                              (x1 - x0) / tile, (y1 - y0) / tile))
    return out


def lovo_roles(frames_by_video, val_frac=0.15):
    """{fold: {video: {frame: 'train'|'val'|'test'}}} for leave-one-video-out."""
    roles = {}
    for test_v in frames_by_video:
        fold = {}
        for v, frames in frames_by_video.items():
            frames = list(frames)
            if v == test_v:
                fold[v] = {f: "test" for f in frames}
            else:
                k = max(1, int(round(len(frames) * val_frac)))
                fold[v] = {f: ("val" if i >= len(frames) - k else "train") for i, f in enumerate(frames)}
        roles["test_" + test_v] = fold
    return roles


def export_lovo_dataset(pre, plb, frames_by_video, out_dir, tile=256, min_valid=0.9,
                        keep_empty=0.2, val_frac=0.15, n_to_label=8, seed=0):
    """Cut every video into tiles once, then lay them out in one folder per LOVO fold."""
    rng = np.random.default_rng(seed)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    stash = os.path.join(out_dir, "_tiles")
    rows = []
    for v, frames in frames_by_video.items():
        reg, G, C, valid = pre[v]["reg"], plb[v]["granules"], plb[v]["cells"], pre[v]["valid"]
        H, W = reg.shape[1:]
        for sub in SUBDIRS:
            os.makedirs(os.path.join(stash, v, sub), exist_ok=True)
        for f in frames:
            for y in range(0, H - tile + 1, tile):
                for x in range(0, W - tile + 1, tile):
                    if valid[y:y + tile, x:x + tile].mean() < min_valid:
                        continue
                    inst, _, _ = segmentation.relabel_sequential(G[f, y:y + tile, x:x + tile].astype(np.int32))
                    n = int(inst.max())
                    if n == 0 and rng.random() > keep_empty:
                        continue
                    name = "%s_t%03d_y%04d_x%04d" % (v, f, y, x)
                    img = (np.clip(reg[f, y:y + tile, x:x + tile].astype(np.float32), 0, 1) * 255).round().astype(np.uint8)
                    c, _, _ = segmentation.relabel_sequential(C[f, y:y + tile, x:x + tile].astype(np.int32))
                    d = os.path.join(stash, v)
                    cv2.imwrite(os.path.join(d, "images", name + ".png"), img)
                    cv2.imwrite(os.path.join(d, "masks", name + ".png"), ((inst > 0) * 255).astype(np.uint8))
                    cv2.imwrite(os.path.join(d, "instances", name + ".png"), inst.astype(np.uint16))
                    cv2.imwrite(os.path.join(d, "cells", name + ".png"), c.astype(np.uint16))
                    with open(os.path.join(d, "labels_yolo", name + ".txt"), "w") as fh:
                        fh.write("\n".join(_yolo_lines(inst, tile)))
                    rows.append(dict(name=name, video=v, frame=f, y=y, x=x, n_granules=n,
                                     granule_px=int((inst > 0).sum()), n_cells=int(c.max())))
    df = pd.DataFrame(rows)

    roles = lovo_roles(frames_by_video, val_frac)
    for fold, fr in roles.items():
        col = "role_" + fold
        df[col] = [fr[v][f] for v, f in zip(df.video, df.frame)]
        for role in ("train", "val", "test"):
            for sub in SUBDIRS:
                os.makedirs(os.path.join(out_dir, "fold_" + fold, role, sub), exist_ok=True)
        for name, v, role in zip(df.name, df.video, df[col]):
            for sub in SUBDIRS:
                ext = ".txt" if sub == "labels_yolo" else ".png"
                shutil.copyfile(os.path.join(stash, v, sub, name + ext),
                                os.path.join(out_dir, "fold_" + fold, role, sub, name + ext))

    # tiles for manual labelling: spread evenly over the videos, only tiles with granules
    lab, ps = os.path.join(out_dir, "to_label"), os.path.join(out_dir, "to_label_pseudo")
    os.makedirs(lab); os.makedirs(ps)
    vids = list(frames_by_video)
    per = {v: n_to_label // len(vids) + (1 if i < n_to_label % len(vids) else 0) for i, v in enumerate(vids)}
    picked = []
    for v in vids:
        cand = df[(df.video == v) & (df.n_granules > 0)]
        if len(cand):
            picked += list(cand.name.iloc[rng.choice(len(cand), min(per[v], len(cand)), replace=False)])
    for name in picked:
        v = name.split("_")[0]
        shutil.copyfile(os.path.join(stash, v, "images", name + ".png"), os.path.join(lab, name + ".png"))
        shutil.copyfile(os.path.join(stash, v, "masks", name + ".png"), os.path.join(ps, name + ".png"))
    df["to_label"] = df.name.isin(set(picked))
    shutil.rmtree(stash)

    df.to_csv(os.path.join(out_dir, "tiles.csv"), index=False)
    _write_readme(df, roles, out_dir, tile, val_frac)
    return df


def fold_summary(df):
    """Tiles / granules per fold and role."""
    out = []
    for col in [c for c in df.columns if c.startswith("role_")]:
        g = df.groupby(col).agg(tiles=("name", "size"), granules=("n_granules", "sum"))
        for role, r in g.iterrows():
            vids = sorted(df[df[col] == role].video.unique())
            out.append(dict(fold=col[5:], role=role, videos="+".join(vids), tiles=int(r.tiles), granules=int(r.granules)))
    return pd.DataFrame(out)


def _write_readme(df, roles, out_dir, tile, val_frac):
    txt = ["# Stress-granule labelled dataset (Group 7) - leave-one-video-out", "",
           "Tiles of %d x %d px from registered, denoised confocal frames. Labels are pseudo-labels" % (tile, tile),
           "(bivariate-histogram cell masks + Hessian-blob granule masks). Manual ground truth for",
           "`to_label/` is added separately (Roboflow).", "",
           "## Folds",
           "One folder per video held out (`fold_test_<video>`): test = all tiles of that video;",
           "train/val = the other videos, val = last %d%% of each training video's frames (time block," % round(val_frac * 100),
           "for early stopping only). The test video never appears in train or val.", "",
           fold_summary(df).to_string(index=False), "",
           "## Files per tile",
           "images/ (uint8), masks/ (0/255 binary granules), instances/ (uint16 granule ids),",
           "cells/ (uint16 cell ids), labels_yolo/ (class 0 boxes, normalised xywh).",
           "tiles.csv: one row per tile with its role in every fold (columns role_test_<video>).", ""]
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write("\n".join(txt))


# ============================================================================ grouped k-fold
def cut_tiles(pre, plb, frames_by_video, stash, tile=256, min_valid=0.9, keep_empty=0.2, seed=0):
    """Cut every frame of every video into tiles once; write them under stash/<video>/<sub>/."""
    rng = np.random.default_rng(seed)
    rows = []
    for v, frames in frames_by_video.items():
        reg, G, C, valid = pre[v]["reg"], plb[v]["granules"], plb[v]["cells"], pre[v]["valid"]
        H, W = reg.shape[1:]
        for sub in SUBDIRS:
            os.makedirs(os.path.join(stash, v, sub), exist_ok=True)
        for f in frames:
            for y in range(0, H - tile + 1, tile):
                for x in range(0, W - tile + 1, tile):
                    if valid[y:y + tile, x:x + tile].mean() < min_valid:
                        continue
                    inst, _, _ = segmentation.relabel_sequential(G[f, y:y + tile, x:x + tile].astype(np.int32))
                    n = int(inst.max())
                    if n == 0 and rng.random() > keep_empty:
                        continue
                    name = "%s_t%03d_y%04d_x%04d" % (v, f, y, x)
                    img = (np.clip(reg[f, y:y + tile, x:x + tile].astype(np.float32), 0, 1) * 255).round().astype(np.uint8)
                    c, _, _ = segmentation.relabel_sequential(C[f, y:y + tile, x:x + tile].astype(np.int32))
                    d = os.path.join(stash, v)
                    cv2.imwrite(os.path.join(d, "images", name + ".png"), img)
                    cv2.imwrite(os.path.join(d, "masks", name + ".png"), ((inst > 0) * 255).astype(np.uint8))
                    cv2.imwrite(os.path.join(d, "instances", name + ".png"), inst.astype(np.uint16))
                    cv2.imwrite(os.path.join(d, "cells", name + ".png"), c.astype(np.uint16))
                    with open(os.path.join(d, "labels_yolo", name + ".txt"), "w") as fh:
                        fh.write("\n".join(_yolo_lines(inst, tile)))
                    rows.append(dict(name=name, video=v, frame=f, y=y, x=x, n_granules=n,
                                     granule_px=int((inst > 0).sum()), n_cells=int(c.max())))
    return pd.DataFrame(rows)


def assign_position_folds(df, k=5, seed=0):
    """Deal the (video, y, x) tile positions into k folds with about equal granule counts.
    A position shows the same cells in every frame, so all its frames stay in one fold."""
    rng = np.random.default_rng(seed)
    g = df.groupby(["video", "y", "x"]).n_granules.sum().reset_index(name="n")
    g = g.iloc[rng.permutation(len(g))].sort_values("n", ascending=False, kind="stable")
    load = np.zeros(k)
    fold = {}
    for _, r in g.iterrows():                      # biggest group -> lightest fold
        j = int(np.argmin(load))
        fold[(r.video, r.y, r.x)] = j + 1           # folds are numbered 1..k
        load[j] += r.n
    return np.array([fold[(v, y, x)] for v, y, x in zip(df.video, df.y, df.x)])


def export_grouped_kfold(pre, plb, frames_by_video, out_dir, k=5, val_frac=0.15, tile=256,
                         min_valid=0.9, keep_empty=0.2, seed=0):
    """k-fold dataset grouped by tile position, one folder per fold:
        fold_<j>/test  = all tiles of the positions in fold j
        fold_<j>/train, val = the other positions; val = last `val_frac` of each video's frames
                              (a time block, for early stopping only)
    No position (= no set of cells) is ever in both train/val and test of the same fold."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    stash = os.path.join(out_dir, "_tiles")
    df = cut_tiles(pre, plb, frames_by_video, stash, tile, min_valid, keep_empty, seed)
    df["fold"] = assign_position_folds(df, k, seed)
    last = {}
    for v, frames in frames_by_video.items():
        frames = list(frames)
        n_val = max(1, int(round(len(frames) * val_frac)))
        last[v] = set(frames[-n_val:])
    for j in range(1, k + 1):
        col = "role_fold_%d" % j
        df[col] = ["test" if fo == j else ("val" if f in last[v] else "train")
                   for fo, v, f in zip(df.fold, df.video, df.frame)]
        for role in ("train", "val", "test"):
            for sub in SUBDIRS:
                os.makedirs(os.path.join(out_dir, "fold_%d" % j, role, sub), exist_ok=True)
        for name, v, role in zip(df.name, df.video, df[col]):
            for sub in SUBDIRS:
                ext = ".txt" if sub == "labels_yolo" else ".png"
                shutil.copyfile(os.path.join(stash, v, sub, name + ext),
                                os.path.join(out_dir, "fold_%d" % j, role, sub, name + ext))
    shutil.rmtree(stash)
    df.to_csv(os.path.join(out_dir, "tiles.csv"), index=False)
    summ = kfold_summary(df, k)
    txt = ["# Stress-granule labelled dataset (Group 7) - %d-fold, grouped by tile position" % k, "",
           "Tiles of %d x %d px from registered, denoised frames of %s. Labels are pseudo-labels" % (tile, tile, " and ".join(frames_by_video)),
           "(bivariate-histogram cell masks + Hessian-blob granule masks).", "",
           "Each frame is a 4 x 4 grid of tiles; a grid position shows the same cells in every frame, so all",
           "frames of a position are in the same fold. fold_<j>/test = the positions of fold j; train/val = all",
           "other positions, val = last %d%% of each video's frames (early stopping only)." % round(val_frac * 100),
           "Final test: human-labelled video-3 tiles, shared separately.", "",
           summ.to_string(index=False), "",
           "Files per tile: images/ (uint8), masks/ (0/255 granules), instances/ (uint16 granule ids),",
           "cells/ (uint16 cell ids), labels_yolo/ (class 0 boxes). tiles.csv: fold and role_fold_<j> per tile.", ""]
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write("\n".join(txt))
    return df


def kfold_summary(df, k):
    out = []
    for j in range(1, k + 1):
        col = "role_fold_%d" % j
        for role in ("train", "val", "test"):
            d = df[df[col] == role]
            out.append(dict(fold=j, role=role, tiles=len(d), granules=int(d.n_granules.sum()),
                            positions=len(d.groupby(["video", "y", "x"]))))
    return pd.DataFrame(out)


# ============================================================================ Roboflow tiles (held-out video)
def export_label_tiles(pre, plb, video, frames, out_dir, n_frames=5, per_frame=4, tile=256,
                       min_valid=0.9, seed=0):
    """Tiles of a held-out video for manual labelling: `n_frames` frames spread evenly over the
    video, `per_frame` tiles each. Positions are spread as evenly as possible (the least-used grid
    position is always picked next), so tiles show different cells wherever the video has enough
    positions; a position is reused only in a different, distant frame. Only tiles with at least
    one pseudo-labelled granule are picked.
        images/          PNGs to upload to Roboflow (names carry video, frame and position)
        pseudo_masks/    the classical labeller's masks for the same tiles (do NOT upload)
        images.zip, tiles.csv, README.md"""
    rng = np.random.default_rng(seed)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "images")); os.makedirs(os.path.join(out_dir, "pseudo_masks"))
    reg, G, valid = pre["reg"], plb["granules"], pre["valid"]
    H, W = reg.shape[1:]
    frames = [f for f in frames if G[f].max() > 0]          # frames that actually have labels
    pick_f = sorted(set(frames[int(round(i))] for i in np.linspace(0, len(frames) - 1, n_frames)))
    positions = [(y, x) for y in range(0, H - tile + 1, tile) for x in range(0, W - tile + 1, tile)
                 if valid[y:y + tile, x:x + tile].mean() >= min_valid]
    uses = {p: 0 for p in positions}
    taken = set()                                 # (frame, position) pairs already exported
    rows = []
    for _ in range(per_frame):                    # round-robin over the chosen frames
        for f in pick_f:
            cand = [p for p in positions if (f, p) not in taken
                    and G[f, p[0]:p[0] + tile, p[1]:p[1] + tile].max() > 0]
            if not cand:
                continue
            low = min(uses[p] for p in cand)      # least-used positions first -> different cells
            cand = [p for p in cand if uses[p] == low]
            y, x = cand[int(rng.integers(len(cand)))]
            uses[(y, x)] += 1; taken.add((f, (y, x)))
            name = "%s_t%03d_y%04d_x%04d" % (video, f, y, x)
            img = (np.clip(reg[f, y:y + tile, x:x + tile].astype(np.float32), 0, 1) * 255).round().astype(np.uint8)
            inst, _, _ = segmentation.relabel_sequential(G[f, y:y + tile, x:x + tile].astype(np.int32))
            cv2.imwrite(os.path.join(out_dir, "images", name + ".png"), img)
            cv2.imwrite(os.path.join(out_dir, "pseudo_masks", name + ".png"), ((inst > 0) * 255).astype(np.uint8))
            rows.append(dict(name=name, video=video, frame=f, y=y, x=x, pseudo_granules=int(inst.max())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "tiles.csv"), index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "images.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for n in df.name:
            z.write(os.path.join(out_dir, "images", n + ".png"), n + ".png")
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write("\n".join([
            "# Video-3 tiles for manual labelling (Roboflow)", "",
            "%d tiles of %d x %d px from frames %s, spread over %d different grid positions." %
            (len(df), tile, tile, ", ".join(map(str, pick_f)), len(set(zip(df.y, df.x)))),
            "Upload images/ (or images.zip) to Roboflow. Keep the file names: they encode video, frame and position.",
            "Draw a polygon around every granule; export as COCO Segmentation with NO resize/augmentation.",
            "pseudo_masks/ = the classical labeller's masks for comparison - do not upload them.", ""]))
    return df


# ============================================================================ integrity
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def verify_npz(path):
    """True if every array in the .npz reads back (zip CRC checked)."""
    try:
        z = np.load(path)
        for k in z.files:
            z[k]
        return True
    except Exception:
        return False


VERIFY_PY = '''"""Check a downloaded team package:  python verify_download.py  (run inside the folder)."""
import hashlib, os, sys
ok = True
for line in open("MANIFEST.sha256"):
    h, name = line.split(None, 1); name = name.strip()
    if not os.path.exists(name):
        print("MISSING ", name); ok = False; continue
    d = hashlib.sha256()
    with open(name, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): d.update(b)
    good = d.hexdigest() == h
    ok &= good
    print("OK      " if good else "CORRUPT ", name, "(%.1f MB)" % (os.path.getsize(name) / 1e6))
print("ALL FILES COMPLETE" if ok else "SOME FILES ARE BAD - download them again"); sys.exit(0 if ok else 1)
'''


def build_team_package(cache_dir, dataset_dir, out_dir, videos, zip_name="sg_dataset.zip"):
    """Local package: v*_pre.npz, v*_pl.npz, sg_dataset.zip, MANIFEST.sha256, verify_download.py.
    Refuses to include an .npz that does not read back completely."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    files = []
    for v in videos:
        for kind in ("pre", "pl"):
            src = os.path.join(cache_dir, "%s_%s.npz" % (v, kind))
            if not verify_npz(src):
                raise IOError("%s is damaged - delete it and re-run sections 1 and 3" % src)
            dst = os.path.join(out_dir, os.path.basename(src))
            shutil.copyfile(src, dst); files.append(dst)
    zpath = os.path.join(out_dir, zip_name)
    base = os.path.dirname(os.path.abspath(dataset_dir))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(dataset_dir):
            for f in sorted(fs):
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, base))
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad:
            raise IOError("zip self-test failed at " + bad)
    files.append(zpath)
    with open(os.path.join(out_dir, "verify_download.py"), "w") as fh:
        fh.write(VERIFY_PY)
    with open(os.path.join(out_dir, "MANIFEST.sha256"), "w") as fh:
        for p in files:
            fh.write("%s  %s\n" % (sha256(p), os.path.basename(p)))
    return read_manifest(out_dir)


def read_manifest(folder):
    out = {}
    for line in open(os.path.join(folder, "MANIFEST.sha256")):
        h, name = line.split(None, 1)
        out[name.strip()] = h
    return out


def verify_folder(folder, manifest=None):
    """Re-hash every file of the manifest inside `folder`. Returns {name: 'OK'|'CORRUPT'|'MISSING'}."""
    manifest = manifest or read_manifest(folder)
    res = {}
    for name, h in manifest.items():
        p = os.path.join(folder, name)
        res[name] = "MISSING" if not os.path.exists(p) else ("OK" if sha256(p) == h else "CORRUPT")
    return res
