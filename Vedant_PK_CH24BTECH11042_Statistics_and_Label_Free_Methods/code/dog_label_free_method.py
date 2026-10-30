"""
DoG: a label-free granule segmenter (Vedant, Group 7).

No training and no labels. For each tile:
  1. Difference of Gaussians (sigma 1.0 minus sigma 5.0) keeps blobs a few pixels wide.
  2. Threshold at median + 4 robust sigmas (MAD x 1.4826) of the DoG response.
  3. Remove objects under 4 px, fill holes.
  4. Split touching granules with a watershed on the distance map (min_distance=3).

Run in Google Colab after unzipping Prabhav's ROBOFLOW_V3/images.zip to /content/tiles.
Writes one binary mask PNG per tile into masks_methodA/ (see results/masks_methodA/).
Copied unchanged from the Colab notebook Statistics_Run1_and_DoG_Method.ipynb (cell 9).
"""
import os, glob, numpy as np
from skimage.io import imread, imsave
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, feature


TILES = ["v3_t000_y0256_x0512", "v3_t037_y0000_x0256", "v3_t037_y0256_x0512",
         "v3_t037_y0512_x0256", "v3_t074_y0512_x0000", "v3_t074_y0512_x0256",
         "v3_t110_y0000_x0256", "v3_t110_y0256_x0512"]

def my_method(img):
    x = img.astype(np.float32)
    # 1. band-pass: keeps blobs a few pixels wide, removes smooth cell background
    dog = filters.gaussian(x, 1.0) - filters.gaussian(x, 5.0)
    # 2. threshold set from the image's own noise level (median + 4 robust sigmas)
    med = np.median(dog)
    mad = np.median(np.abs(dog - med)) * 1.4826 + 1e-6
    mask = dog > med + 4.0 * mad
    # 3. clean up
    mask = morphology.remove_small_objects(mask, 4)
    mask = ndi.binary_fill_holes(mask)
    # 4. split touching granules (watershed on the distance map)
    dist = ndi.distance_transform_edt(mask)
    cc = measure.label(mask)
    peaks = feature.peak_local_max(filters.gaussian(dist, 1), min_distance=3,
                                   labels=cc, exclude_border=False)
    markers = np.zeros(mask.shape, int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    ws = segmentation.watershed(-dist, markers, mask=mask, watershed_line=True)
    return ws > 0

OUT = "masks_methodA"          # use a different folder name for each method
os.makedirs(OUT, exist_ok=True)
for t in TILES:
    path = glob.glob(f"/content/tiles/**/{t}.png", recursive=True)[0]
    img = imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    mask = my_method(img)
    imsave(f"{OUT}/{t}.png", (mask * 255).astype("uint8"), check_contrast=False)
print(sorted(os.listdir(OUT)))
