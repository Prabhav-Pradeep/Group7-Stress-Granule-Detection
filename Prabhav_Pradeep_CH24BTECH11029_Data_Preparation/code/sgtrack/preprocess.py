"""Stage 1 - loading, photobleach correction, stabilisation (registration) and optical flow.

Every function works on a stack shaped (T, H, W) of float32 images in [0, 1].
"""
import numpy as np
import cv2
from skimage.registration import phase_cross_correlation


# ----------------------------------------------------------------------------- loading
def load_video(path, channel=None, min_frames=3):
    """Read an AVI/MP4 into a (T, H, W) float32 stack in [0, 1] (ffmpeg backend).

    The microscopy videos are pseudo-coloured red, so by default we take the per-pixel max
    over RGB. Raises a clear error if the file decodes to fewer than `min_frames` frames -
    the usual cause is an incomplete upload / Google Drive sync (a truncated raw AVI simply
    decodes to fewer frames without any error).
    """
    import os
    import imageio.v2 as iio2
    frames = np.asarray(iio2.mimread(path, format="FFMPEG", memtest=False))
    if frames.ndim == 4:
        frames = frames[..., channel] if channel is not None else frames.max(-1)
    if len(frames) < min_frames:
        raise ValueError(
            "%s decoded to only %d frame(s) (file size %.1f MB). The file is probably "
            "incomplete - re-upload it to Drive, wait until the upload finishes, then delete "
            "the cache/*_pre.npz files and re-run." % (os.path.basename(path), len(frames),
                                                         os.path.getsize(path) / 1e6))
    return frames.astype(np.float32) / 255.0


# ----------------------------------------------------------------------------- photobleaching
def correct_photobleaching(stack, fg_percentile=75):
    """Exponential-fit photobleaching correction.

    Photobleaching makes the signal decay roughly as exp(-k t); a fixed threshold would then
    lose granules in late frames. We measure each frame's foreground level (median of the
    non-saturated pixels above the `fg_percentile`-th percentile), fit log(level) = a - k t,
    and multiply frame t by exp(k t).

    If the fit shows no decay (k <= 0, i.e. the signal *rises*, e.g. because granules are
    forming) we do not touch the data - dimming a real biological increase would be wrong.
    Returns (corrected_stack, gains, levels).
    """
    levels = []
    for f in stack:
        thr = np.percentile(f, fg_percentile)
        sel = f[(f > thr) & (f < 0.999)]
        levels.append(np.median(sel) if sel.size else thr)
    levels = np.array(levels) + 1e-6
    t = np.arange(len(stack))
    ok = np.isfinite(levels) & (levels > 1e-5)
    if ok.sum() >= 3:
        slope, _ = np.polyfit(t[ok], np.log(levels[ok]), 1)
        k = max(-slope, 0.0)
    else:                      # too few usable frames to fit a decay -> no correction
        k = 0.0
    gains = np.exp(k * t)
    out = np.clip(stack * gains[:, None, None], 0, 1).astype(np.float32)
    return out, gains, levels


def denoise(stack, sigma=1.0):
    """Light Gaussian smoothing to suppress shot noise before registration / thresholding."""
    k = int(2 * round(3 * sigma) + 1)
    return np.stack([cv2.GaussianBlur(f, (k, k), sigma) for f in stack])


# ----------------------------------------------------------------------------- registration
def register_stack(stack, mode="euclidean", downsample=2, ecc_iters=200):
    """Stabilise the video so every frame is in the coordinate system of frame 0.

    1. Phase correlation gives a robust frame-to-frame translation (global drift).
    2. ECC (enhanced correlation coefficient) refines it, warm-started from step 1.
       mode="euclidean" (default) = rigid translation + rotation, which is what stage drift
       physically is. mode="affine" also allows scale/shear, but because cells themselves
       grow/move, a free affine slowly drifts in scale when chained over many frames
       (we measured ~13 % over 21 frames), so it is only offered as an option.
    3. Frame-to-frame transforms are chained so frame t is mapped onto frame 0.

    Returns (registered_stack, transforms) with transforms[t] a 3x3 matrix mapping
    frame-t pixel coordinates to frame-0 coordinates.
    """
    T, H, W = stack.shape
    motion = {"affine": cv2.MOTION_AFFINE, "euclidean": cv2.MOTION_EUCLIDEAN,
              "translation": cv2.MOTION_TRANSLATION}[mode]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iters, 1e-5)
    small = [cv2.resize(f, (W // downsample, H // downsample), interpolation=cv2.INTER_AREA)
             for f in stack]

    cum = [np.eye(3, dtype=np.float64)]
    for t in range(1, T):
        prev, cur = small[t - 1], small[t]
        shift, _, _ = phase_cross_correlation(prev, cur, upsample_factor=10, normalization=None)
        warp = np.array([[1, 0, -shift[1]], [0, 1, -shift[0]]], dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(prev, cur, warp, motion, criteria, None, 5)
        except cv2.error:
            pass  # ECC failed to converge - keep the phase-correlation translation
        # ECC warp maps prev coords -> cur coords; we want cur -> prev (inverse)
        M = np.vstack([warp.astype(np.float64), [0, 0, 1]])
        M[:2, 2] *= downsample
        cum.append(cum[-1] @ np.linalg.inv(M))

    reg = np.empty_like(stack)
    for t in range(T):
        reg[t] = cv2.warpAffine(stack[t], cum[t][:2].astype(np.float32), (W, H),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return reg, np.stack(cum)


def valid_region(transforms, shape):
    """Pixels that are inside the field of view in *every* registered frame."""
    H, W = shape
    mask = np.ones((H, W), np.uint8)
    for M in transforms:
        m = cv2.warpAffine(np.ones((H, W), np.uint8), M[:2].astype(np.float32), (W, H),
                           flags=cv2.INTER_NEAREST, borderValue=0)
        mask &= m
    return mask.astype(bool)


# ----------------------------------------------------------------------------- optical flow
def optical_flow_magnitude(stack):
    """Per-pixel optical displacement (px / frame) between consecutive frames (Farneback).

    Returns an array (T, H, W); frame 0 re-uses the 0->1 flow so shapes line up with `stack`.
    """
    u8 = (np.clip(stack, 0, 1) * 255).astype(np.uint8)
    mags = []
    for t in range(1, len(u8)):
        flow = cv2.calcOpticalFlowFarneback(u8[t - 1], u8[t], None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mags.append(np.linalg.norm(flow, axis=-1))
    mags.insert(0, mags[0])
    return np.stack(mags).astype(np.float32)
