"""Build the labelled-data package (Group7_Labelled_Data_V4_auto_and_hand.zip).

Inputs
  --drive    zip of the Drive output folder of the V4 notebook (SGTracking_5FOLD_v3_30-1500px)
  --tiles    images.zip: the 20 video-3 tiles uploaded to Roboflow
  --coco     Roboflow export of the hand labels (COCO segmentation, compressed RLE masks)
  --out      output zip

What it does
  1. copies the V4 automatic labels (5-fold dataset + full-frame labels of videos 1-3);
  2. cuts the V4 automatic mask of every Roboflow tile out of the full-frame video-3 labels
     (tile name v3_t<frame>_y<row>_x<col> = top-left corner of a 256 x 256 crop);
  3. decodes the hand labels (COCO compressed RLE) into 0/255 masks and checks every decoded
     granule against its COCO bounding box;
  4. writes tiles.csv (frame, position, hand-labelled?, granules hand / auto) and a README.
"""
import argparse, ast, io, json, os, re, zipfile

import numpy as np
from PIL import Image

S = "SGTracking_5FOLD_v3_30-1500px/"
TILE = 256
README = """Group 7 - Stress-granule detection - labelled data (output)
Original videos: see the Google Drive link.

automatic_labels/                 automatic (V4) labels, the training data
    sg_dataset_5fold.zip          videos 1+2: 1,487 tiles (256x256 px), 13,507 granules, 5 folds.
                                  Per tile: images/, masks/ (granule 0/255), instances/ (granule ids),
                                  cells/ (cell ids), labels_yolo/ (boxes). tiles.csv = fold of each tile.
    v1_pl.npz, v2_pl.npz,         full-frame labels per video (cells, granules = instance ids)
    v3_pl.npz

video3_test_tiles/                held-out video 3, never used for training
    images/                       the 20 tiles uploaded to Roboflow (name = frame t, top-left y, x)
    auto_labels_V4/               automatic V4 granule masks for the same 20 tiles (0/255)
    hand_labels/coco/             manual Roboflow export: 8 tiles, 102 granules (COCO segmentation)
    hand_labels/masks/            the same 8 hand labels as 0/255 masks
    tiles.csv                     per tile: frame, y, x, hand-labelled?, granules (hand / auto)
"""


# ----------------------------------------------------------------------------- COCO RLE decoding
def rle_decode(seg):
    """COCO compressed RLE (string counts, column-major) -> bool mask (as pycocotools rleFrString)."""
    s, (h, w) = seg["counts"], seg["size"]
    cnts, p = [], 0
    while p < len(s):
        x = k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    flat = np.zeros(h * w, bool)
    pos = 0
    for i, c in enumerate(cnts):
        if i % 2:
            flat[pos:pos + c] = True
        pos += c
    return flat.reshape((w, h)).T


def parse(v):
    """Roboflow writes some fields as strings ("[2, 167, 14, 13]")."""
    return ast.literal_eval(v) if isinstance(v, str) else v


# ----------------------------------------------------------------------------- hand labels
def hand_masks(coco):
    out = {}
    for im in coco["images"]:
        name = im["file_name"].split("_png")[0]
        m = np.zeros((int(im["height"]), int(im["width"])), bool)
        n = 0
        for a in coco["annotations"]:
            if str(a["image_id"]) != str(im["id"]):
                continue
            g = rle_decode(parse(a["segmentation"]))
            bx = parse(a["bbox"])                       # Roboflow 'area' is the bbox area, so check the bbox
            ys, xs = np.where(g)
            assert (abs(xs.min() - bx[0]) <= 1 and abs(ys.min() - bx[1]) <= 1
                    and abs(xs.max() + 1 - bx[0] - bx[2]) <= 1 and abs(ys.max() + 1 - bx[1] - bx[3]) <= 1), (name, a["id"])
            m |= g
            n += 1
        out[name] = ((m * 255).astype(np.uint8), n)
    return out


def png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--out", default="Group7_Labelled_Data_V4_auto_and_hand.zip")
    a = ap.parse_args()

    src = zipfile.ZipFile(a.drive)
    v3_name = next(n for n in src.namelist() if n.startswith(S + "cache/v3_pl_"))
    G = np.load(io.BytesIO(src.read(v3_name)))["granules"]          # (frames, 1024, 1024) granule ids
    tiles = zipfile.ZipFile(a.tiles)
    coco_zip = zipfile.ZipFile(a.coco)
    hand = hand_masks(json.loads(coco_zip.read("test/_annotations.coco.json")))

    root = "Group7_Labelled_Data/"
    store = (".zip", ".npz", ".png", ".jpg")
    rows = ["name,frame,y,x,hand_labelled,hand_granules,auto_granules"]
    with zipfile.ZipFile(a.out, "w", allowZip64=True) as zf:
        def add(arc, data):
            zf.writestr(root + arc, data,
                        compress_type=zipfile.ZIP_STORED if arc.endswith(store) else zipfile.ZIP_DEFLATED)

        add("README.txt", README)
        for n in ["sg_dataset_5fold.zip", "v1_pl.npz", "v2_pl.npz"]:
            add("automatic_labels/" + n, src.read(S + "DATA_FOR_TEAM_5FOLD/" + n))
        add("automatic_labels/v3_pl.npz", src.read(v3_name))

        for n in sorted(x for x in tiles.namelist() if x.endswith(".png")):
            name = os.path.basename(n)[:-4]
            t, y, x = map(int, re.findall(r"t(\d+)_y(\d+)_x(\d+)", name)[0])
            auto = G[t, y:y + TILE, x:x + TILE]
            add("video3_test_tiles/images/" + name + ".png", tiles.read(n))
            add("video3_test_tiles/auto_labels_V4/" + name + ".png", png(((auto > 0) * 255).astype(np.uint8)))
            h = hand.get(name)
            if h:
                add("video3_test_tiles/hand_labels/masks/" + name + ".png", png(h[0]))
            rows.append("%s,%d,%d,%d,%s,%s,%d" % (name, t, y, x, "yes" if h else "no",
                                                   h[1] if h else "", len(np.unique(auto[auto > 0]))))
        for n in coco_zip.namelist():
            if not n.endswith("/"):
                add("video3_test_tiles/hand_labels/coco/" + n, coco_zip.read(n))
        add("video3_test_tiles/tiles.csv", "\n".join(rows) + "\n")

    with zipfile.ZipFile(a.out) as zf:
        assert zf.testzip() is None
    print("wrote %s: %d hand-labelled tiles, %d hand granules" % (a.out, len(hand), sum(v[1] for v in hand.values())))


if __name__ == "__main__":
    main()
