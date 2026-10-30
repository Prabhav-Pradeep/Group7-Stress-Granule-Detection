"""Stage 2 - bivariate (intensity x optical displacement) histogram, cell masks and granule
pseudo-masks. These pseudo-masks are the training targets for YOLO and U-Net (no manual labels).
"""
import numpy as np
import cv2
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture
from skimage import measure, morphology, segmentation, feature


def _remove_small(x, min_size):
    """Remove objects with area < min_size (works on bool masks and label images, any skimage version)."""
    lab = measure.label(x) if x.dtype == bool else x
    counts = np.bincount(lab.ravel())
    keep = counts >= min_size
    keep[0] = False
    out = keep[lab]
    return out if x.dtype == bool else np.where(out, lab, 0)


# ----------------------------------------------------------------------------- features
def _features(I, D):
    """Feature transform used for the mixture model.
    sqrt(I) spreads out the dark end where background lives; log1p(D) compresses the long
    tail of optical-flow noise."""
    return np.stack([np.sqrt(np.clip(I, 0, 1)), np.log1p(D)], -1)


class BivariateCellClassifier:
    """Gaussian mixture on per-pixel (intensity, optical displacement).

    Why 3 components and not 2: optical flow is exactly ~0 in flat dark regions but noisy in
    textured dark regions, so with 2 components the "background" Gaussian collapses onto
    D ~ 0 and any dark pixel with flow noise is called "cell" (the model separates moving vs
    still, not cell vs background). With 3 components the mixture finds
    (i) empty background (I ~ 0, D ~ 0), (ii) dark, noisy pixels (low I, small random D) and
    (iii) cell pixels (high I, coherent D). Components (i)+(ii) = background.
    The `n_bg` lowest-intensity components are background, the rest are cell.
    `bounds()` gives P(cell) on an (intensity, displacement) grid for drawing the bound.
    """

    def __init__(self, n_components=3, n_bg=2, n_samples=200_000, seed=0):
        self.n_components, self.n_bg = n_components, n_bg
        self.n_samples = n_samples
        self.seed = seed

    def fit(self, I_stack, D_stack):
        rng = np.random.default_rng(self.seed)
        I, D = I_stack.ravel(), D_stack.ravel()
        idx = rng.choice(I.size, size=min(self.n_samples, I.size), replace=False)
        X = _features(I[idx], D[idx])
        self.gmm = GaussianMixture(self.n_components, covariance_type="full",
                                   random_state=self.seed).fit(X)
        order = np.argsort(self.gmm.means_[:, 0])
        self.cell_comps = order[self.n_bg:]
        self.sample = (I[idx], D[idx])
        # intensity floor = the bound for a pixel at rest (D = 0)
        grid = np.linspace(0, 1, 2001)
        p0 = self._gmm_prob(grid, np.zeros_like(grid))
        self.i_floor = float(grid[np.argmax(p0 > 0.5)]) if (p0 > 0.5).any() else 0.0
        return self

    def _gmm_prob(self, I, D):
        X = _features(np.ravel(I), np.ravel(D))
        return self.gmm.predict_proba(X)[:, self.cell_comps].sum(1).reshape(np.shape(I))

    def prob_cell(self, I, D):
        """P(cell) = GMM posterior x soft intensity floor. The floor encodes the physical
        constraint that displacement alone (optical-flow noise in dark areas) cannot make a
        pixel a cell pixel - it must also be brighter than resting background."""
        gate = 1 / (1 + np.exp(-(np.asarray(I) - self.i_floor) / 0.005))
        return self._gmm_prob(I, D) * gate

    def bounds(self, i_max=1.0, d_max=10.0, n=300):
        ii, dd = np.meshgrid(np.linspace(0, i_max, n), np.linspace(0, d_max, n))
        return ii, dd, self.prob_cell(ii, dd)


def cell_mask(prob, min_area=500, smooth=3):
    """Clean the per-pixel P(cell) map into a binary cell mask (semantic segmentation)."""
    m = cv2.GaussianBlur(prob.astype(np.float32), (0, 0), smooth) > 0.5
    m = _remove_small(m, min_area)
    m = ndi.binary_fill_holes(m)          # nuclei are dark holes in a cytoplasmic stain
    return m


def cell_instances(I, mask, sigma=10, min_distance=30, min_area=800):
    """Split the cell mask into individual cells (instance segmentation of cells).

    Touching cells form large clumps, so a distance-transform watershed under-segments them.
    Instead each cell's cytoplasm forms its own brightness "dome": we take log-intensity
    (so dim cells get a dome too), close the dark nucleus holes, smooth with a Gaussian
    (sigma ~ cell radius / 3), seed the watershed at its local maxima and flood inside `mask`.
    Returns an int label image (0 = background).
    """
    L = np.log(I.astype(np.float32) + 0.02)
    L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    S = cv2.GaussianBlur(L, (0, 0), sigma)
    peaks = feature.peak_local_max(S, min_distance=min_distance, labels=mask.astype(int),
                                   exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    lab = segmentation.watershed(-S, markers, mask=mask)
    lab = _remove_small(lab, min_area)
    lab, _, _ = segmentation.relabel_sequential(lab)
    return lab


def propagate_cells(I, mask, prev, sigma=10, core_radius=6, min_distance=30, min_area=800,
                    next_id=None, new_min_area=1500):
    """Temporally consistent cell instances: seed frame t's watershed with the eroded cores
    of frame t-1's cells (so a cell keeps its ID), plus new seeds for cell regions that no
    previous core covers (cells entering the field / newly bright); new cells smaller than
    `new_min_area` are discarded. Returns (labels, next_id)."""
    L = np.log(I.astype(np.float32) + 0.02)
    L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    S = cv2.GaussianBlur(L, (0, 0), sigma)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * core_radius + 1,) * 2)
    markers = np.zeros(mask.shape, np.int32)
    for l in np.unique(prev):
        if l == 0:
            continue
        core = cv2.erode((prev == l).astype(np.uint8), se).astype(bool) & mask
        if core.sum() >= 20:
            markers[core] = l
    next_id = int(prev.max()) + 1 if next_id is None else next_id
    uncovered = mask & ~ndi.binary_dilation(markers > 0, iterations=min_distance)
    if uncovered.any():
        peaks = feature.peak_local_max(np.where(uncovered, S, S.min()), min_distance=min_distance,
                                       labels=measure.label(uncovered), exclude_border=False)
        for py, px in peaks:
            markers[py, px] = next_id
            next_id += 1
    first_new = int(prev.max()) + 1 if prev.max() else 1
    lab = segmentation.watershed(-S, markers, mask=mask)
    lab = _remove_small(lab, min_area)
    # a *new* cell must be a substantial region, otherwise flickering dim patches keep
    # spawning short-lived IDs
    counts = np.bincount(lab.ravel())
    small_new = np.where((np.arange(len(counts)) >= first_new) & (counts < new_min_area))[0]
    small_new = small_new[np.isin(small_new, markers[markers > 0])]
    if len(small_new):
        lab[np.isin(lab, small_new)] = 0
    return lab, next_id


# ----------------------------------------------------------------------------- granules
def granule_score(I, tophat_radius=8, eps=0.05, smooth=1.0, cells=None, fill_sigma=15):
    """Local-contrast granule score = (I - local cytoplasm level) / local cytoplasm level.

    A single global threshold fails because cell brightness varies ~10x across the field
    (bright cells' texture passes, faint granules in dim cells are missed). Dividing the
    top-hat (granule excess brightness) by the morphological opening (local cytoplasm
    level) makes the score a *relative* brightness, comparable across cells.

    Edge fix: next to dark background the opening drops to ~0, which made cell rims look
    like granules. If `cells` is given, pixels outside cells are first filled with the
    nearby cytoplasm level (normalised convolution: G*(I*M) / G*M) so the background
    estimate is not dragged down at cell borders.
    """
    I = cv2.GaussianBlur(I.astype(np.float32), (0, 0), smooth)
    if cells is not None:
        M = cells.astype(np.float32)
        num = cv2.GaussianBlur(I * M, (0, 0), fill_sigma)
        den = cv2.GaussianBlur(M, (0, 0), fill_sigma)
        fill = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
        I = np.where(cells, I, fill).astype(np.float32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tophat_radius + 1,) * 2)
    bg = cv2.morphologyEx(I, cv2.MORPH_OPEN, se)
    bg = cv2.GaussianBlur(bg, (0, 0), tophat_radius / 2)
    return (I - bg) / (bg + eps)


def granule_mask(I, cells, tophat_radius=8, k=4.0, min_area=4, max_area=400, sat_area=400,
                 min_ratio=0.3, max_ecc=0.95, min_solidity=0.7):
    """Pseudo-mask of stress granules inside cells.

    Granules are small bright puncta on top of a diffuse cytoplasmic signal, so we
    1. compute a local-contrast score (see `granule_score`),
    2. threshold at median + k*MAD of the score *inside cells* (robust statistics, so the
       granules themselves do not inflate the threshold) and also require the granule to
       be at least `min_ratio` (25 %) brighter than its surroundings,
    3. keep components with min_area <= area <= max_area that are roughly round
       (eccentricity <= max_ecc and solidity >= min_solidity for objects > 20 px; this
       rejects thin bright streaks along cell edges),
    4. ignore large saturated blobs (whole over-exposed cells are not granules).
    """
    th = granule_score(I, tophat_radius, cells=cells)
    vals = th[cells]
    if vals.size == 0:
        return np.zeros_like(cells)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
    m = (th > max(med + k * mad, min_ratio)) & cells
    m = ndi.binary_opening(m, structure=morphology.disk(1))
    lab = measure.label(m)
    sat = (I >= 0.999)
    sat_lab = measure.label(_remove_small(sat, sat_area))
    keep = np.zeros(lab.max() + 1, bool)
    for r in measure.regionprops(lab):
        if not (min_area <= r.area <= max_area):
            continue
        if r.area > 20 and (r.eccentricity > max_ecc or r.solidity < min_solidity):
            continue
        keep[r.label] = True
    out = keep[lab]
    out &= ~(sat_lab > 0)
    return out


def blobness(I, sigmas=(1.5, 2.5, 3.5)):
    """Scale-normalised Hessian blob response.

    For a bright blob both Hessian eigenvalues are strongly negative; for a ridge (cell rim,
    membrane) one eigenvalue is ~0. Using B = max_sigma sigma^2 * max(0, -lambda_max), where
    lambda_max is the *larger* eigenvalue, therefore responds to round granules and ignores
    thin bright lines. Returns (B, best_sigma)."""
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    best = np.zeros(I.shape, np.float32)
    best_s = np.zeros(I.shape, np.float32)
    for s in sigmas:
        H = hessian_matrix(I, sigma=s, order="rc", use_gaussian_derivatives=False)
        l1, l2 = hessian_matrix_eigvals(H)          # l1 >= l2
        b = (s ** 2) * np.maximum(0, -l1)
        upd = b > best
        best[upd], best_s[upd] = b[upd], s
    return best, best_s


def granule_instances_blob(I, cells, k=4.0, min_ratio=0.5, rel_level=0.5, min_area=4,
                           max_area=400, min_distance=3, tophat_radius=8, sat_area=400,
                           max_ecc=0.95, min_solidity=0.7, max_radius=12):
    """Granule *instance* pseudo-labels via blob detection + half-max region growing.

    1. local background level `bg` (edge-filled morphological opening, see granule_score);
    2. relative blobness  B / (bg + eps)  -> seeds = local maxima above median + k*MAD (inside
       cells) whose local contrast (granule_score) is >= `min_ratio`;
    3. each seed is grown by watershed on the smoothed image, but only over pixels brighter than
       bg + rel_level * (peak - bg)  (half-maximum contour -> stable size across frames) and
       within 2.5*sigma+2 px of the seed (sigma = best blob scale), so it cannot leak along rims;
    4. size / shape / saturation filters as in `granule_mask`.
    Returns an int32 label image.
    """
    Is = cv2.GaussianBlur(I.astype(np.float32), (0, 0), 1.0)
    score = granule_score(I, tophat_radius, cells=cells)
    eps = 0.05
    bg = (Is - eps * score) / (1 + score)   # invert score = (I - bg) / (bg + eps)
    B, Bsig = blobness(Is)
    Brel = B / (np.maximum(bg, 0) + eps)
    vals = Brel[cells & (Brel > 0)]   # most pixels have zero blobness -> use positive part only
    if vals.size == 0:
        return np.zeros(I.shape, np.int32)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
    thr = med + k * mad
    peaks = feature.peak_local_max(Brel, min_distance=min_distance, threshold_abs=thr,
                                   labels=cells.astype(np.int32), exclude_border=False)
    peaks = np.array([p for p in peaks if score[p[0], p[1]] >= min_ratio]).reshape(-1, 2)
    if len(peaks) == 0:
        return np.zeros(I.shape, np.int32)
    markers = np.zeros(I.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    # per-seed half-max level
    pk_I = Is[tuple(peaks.T)]
    pk_bg = np.maximum(bg[tuple(peaks.T)], 0)
    level = np.concatenate([[np.inf], pk_bg + rel_level * (pk_I - pk_bg)])
    candidate = cells & (score > 0)
    lab = segmentation.watershed(-Is, markers, mask=candidate)
    lab = np.where(Is >= level[lab], lab, 0)
    # cap growth at a radius tied to the blob scale (stops leaking along dim membrane rims)
    radius = np.concatenate([[0], np.minimum(max_radius, 2.5 * Bsig[tuple(peaks.T)] + 2)])
    sy = np.concatenate([[0], peaks[:, 0]]); sx = np.concatenate([[0], peaks[:, 1]])
    yy, xx = np.indices(I.shape)
    lab = np.where(np.hypot(yy - sy[lab], xx - sx[lab]) <= radius[lab], lab, 0)
    # keep only the connected piece that contains each seed
    cc = measure.label(lab > 0, connectivity=1)
    seed_cc = np.zeros(len(peaks) + 1, np.int64)
    ok = lab[tuple(peaks.T)] == np.arange(1, len(peaks) + 1)
    seed_cc[1:] = np.where(ok, cc[tuple(peaks.T)], -1)
    out = np.where((lab > 0) & (cc == seed_cc[lab]), lab, 0)
    # filters
    sat_lab = measure.label(_remove_small(I >= 0.999, sat_area))
    final = np.zeros_like(out)
    n = 0
    for r in measure.regionprops(out):
        if not (min_area <= r.area <= max_area):
            continue
        if r.area > 20 and (r.eccentricity > max_ecc or r.solidity < min_solidity):
            continue
        cy, cx = map(int, r.centroid)
        if sat_lab[cy, cx]:
            continue
        n += 1
        final[tuple(r.coords.T)] = n
    return final


def granule_instances(gmask, split=True):
    """Label granules; touching granules are split with a small watershed on the distance map."""
    lab = measure.label(gmask)
    if not split:
        return lab
    dist = ndi.distance_transform_edt(gmask)
    peaks = feature.peak_local_max(dist, min_distance=2, labels=lab, exclude_border=False)
    markers = np.zeros(gmask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return segmentation.watershed(-dist, markers, mask=gmask)
