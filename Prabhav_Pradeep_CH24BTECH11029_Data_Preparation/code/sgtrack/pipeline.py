"""Glue code: run a whole stage over a video and cache the results to .npz files."""
import os
import numpy as np
from . import preprocess as P
from . import pseudolabel as PL


# ----------------------------------------------------------------------------- safe cache I/O
def load_npz_checked(path, keys=None):
    """Load an .npz completely (every array is read, so the zip CRC of each one is checked).
    Returns a dict, or None if the file is missing, truncated/corrupt, or lacks `keys`.
    A bad file is renamed to *.corrupt so it is rebuilt instead of crashing the run."""
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        out = {k: z[k] for k in z.files}
        if keys is not None and set(keys) - set(out):
            raise KeyError("missing arrays: %s" % sorted(set(keys) - set(out)))
        return out
    except Exception as e:
        bad = path + ".corrupt"
        print("[cache] %s is unreadable (%s: %s) -> moved to %s, rebuilding"
              % (os.path.basename(path), type(e).__name__, str(e)[:80], os.path.basename(bad)))
        os.replace(path, bad)
        return None


def save_npz_checked(path, **arrays):
    """Write to local disk first, copy to the destination (e.g. Google Drive), then re-read the
    destination completely. Writing a large .npz straight onto Drive can leave a truncated file
    if the sync is interrupted - this is what damaged v2_pre.npz."""
    import shutil
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        shutil.copyfile(tmp, path + ".partial")
        os.replace(path + ".partial", path)
    finally:
        os.remove(tmp)
    chk = load_npz_checked(path, keys=arrays.keys())
    if chk is None:
        raise IOError("could not verify %s after writing" % path)
    print("[cache] saved and verified %s (%.1f MB)" % (os.path.basename(path), os.path.getsize(path) / 1e6))


PRE_KEYS = ("reg", "flow", "T", "valid", "gains", "levels")
PL_KEYS = ("cells", "granules")


def preprocess_video(path, cache, register_mode="euclidean", sigma=1.0):
    """load -> photobleach correction -> denoise -> register -> optical flow. Cached."""
    cached = load_npz_checked(cache, PRE_KEYS)
    if cached is not None:
        return cached
    raw = P.load_video(path)
    corr, gains, levels = P.correct_photobleaching(raw)
    den = P.denoise(corr, sigma)
    reg, T = P.register_stack(den, mode=register_mode)
    valid = P.valid_region(T, reg.shape[1:])
    flow = P.optical_flow_magnitude(reg)
    out = dict(reg=reg.astype(np.float16), flow=flow.astype(np.float16), T=T, valid=valid,
               gains=gains, levels=levels)
    save_npz_checked(cache, **out)
    return out


def pseudolabel_video(pre, cache, frames=None, clf=None, granule_kw=None, cell_kw=None):
    """Bivariate-histogram cell masks + granule pseudo-masks for every frame. Cached.

    Returns dict with cells (T,H,W uint16 cell instances), granules (T,H,W uint16 granule
    instances) and the fitted BivariateCellClassifier (not cached; refit if needed)."""
    reg = pre["reg"].astype(np.float32)
    flow = pre["flow"].astype(np.float32)
    frames = list(range(len(reg))) if frames is None else list(frames)
    if clf is None:
        clf = PL.BivariateCellClassifier().fit(reg[frames], flow[frames])
    cached = load_npz_checked(cache, PL_KEYS)
    if cached is not None:
        return dict(cells=cached["cells"], granules=cached["granules"], clf=clf)
    granule_kw, cell_kw = granule_kw or {}, cell_kw or {}
    T, H, W = reg.shape
    cells = np.zeros((T, H, W), np.uint16)
    gran = np.zeros((T, H, W), np.uint16)
    prev, next_id = None, None
    for t in frames:
        cm = PL.cell_mask(clf.prob_cell(reg[t], flow[t])) & pre["valid"]
        if prev is None:   # first frame: independent segmentation; later: propagate IDs
            cells[t] = PL.cell_instances(reg[t], cm, **cell_kw)
            next_id = int(cells[t].max()) + 1
        else:
            cells[t], next_id = PL.propagate_cells(reg[t], cm, prev, next_id=next_id)
        prev = cells[t].astype(np.int32)
        gran[t] = PL.granule_instances_blob(reg[t], cm, **granule_kw)
    save_npz_checked(cache, cells=cells, granules=gran)
    return dict(cells=cells, granules=gran, clf=clf)


def track_video(labels, reg, cells=None, frames=None, frame_interval=1.0, max_dist=12,
                max_gap=3, event_kw=None, tracker_kw=None):
    """Granule tracking + events + cell assignment for one video.

    labels : (T,H,W) granule instance labels (model predictions or pseudo-labels)
    cells  : (T,H,W) cell instance labels (optional) -> granules get a persistent cell id
    Returns dict(tracks=DataFrame, events=DataFrame (filtered), raw_events=DataFrame,
                 cells=list of cell-tracked label images, per_cell=DataFrame counts).
    """
    import pandas as pd
    from . import tracking as TR
    frames = list(range(len(labels))) if frames is None else list(frames)
    cell_tr = None
    if cells is not None:   # IDs already consistent (propagate_cells); IoU matching as a safety net
        cell_tr = TR.track_cells([cells[f].astype(np.int32) for f in frames])
    dets = []
    for k, f in enumerate(frames):
        dets.append(TR.detections_from_labels(labels[f].astype(np.int32), f,
                                              reg[f].astype(np.float32),
                                              None if cell_tr is None else cell_tr[k]))
    trk = TR.GranuleTracker(max_dist=max_dist, max_gap=max_gap, **(tracker_kw or {}))
    df, raw = trk.run(dets, frames)
    ev = TR.filter_events(df, raw, **(event_kw or {}))
    cent = None
    if cell_tr is not None:
        from skimage import measure
        cent = {(f, r.label): r.centroid for k, f in enumerate(frames)
                for r in measure.regionprops(cell_tr[k])}
    df = TR.add_kinematics(df, frame_interval, cent)
    per_cell = None
    if cell_tr is not None:
        per_cell = (df[df.cell > 0].groupby(["frame", "cell"]).size()
                    .rename("n_granules").reset_index())
    return dict(tracks=df, events=ev, raw_events=raw, cells=cell_tr, per_cell=per_cell)
