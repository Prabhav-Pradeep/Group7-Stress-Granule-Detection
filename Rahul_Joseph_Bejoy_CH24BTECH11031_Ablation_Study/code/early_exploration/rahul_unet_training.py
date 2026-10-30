"""
Rahul's training script — Depthwise-Conv U-Net + Attention U-Net for stress-granule
segmentation (Group 7).

Run this in Google Colab. Either:
  (a) paste the whole file into one cell and run, or
  (b) split at the "# %%" markers into separate cells (VSCode/Jupytext-style,
      Colab will also just run top to bottom fine as a single script).

Before running:
  1. Set runtime to GPU (Runtime -> Change runtime type -> T4/A100).
  2. Upload DATA_FOR_TEAM.zip to Colab (or mount Drive) and unzip it so that
     DATA_FOR_TEAM/{split.json, v1_pre.npz, v2_pre.npz, v1_pl.npz, v2_pl.npz} exist.
     You do NOT need yolo_dataset.zip for this script — that's Prabhav's data.
  3. pip install any missing packages (cell 0 below does this).

What this script does (matches your team's training-instructions.txt):
  - Loads v2 (train/val) and v1 (test) frames + pseudo-label instance masks.
  - Tiles train/val into 256x256 (2x-upsampled to 512) with ~20% empty tiles.
  - Trains two models: DepthwiseUNet and AttentionUNet, each with Dice+0.5*BCE loss.
  - Runs a small hyperparameter sweep on VAL ONLY, logs every run to a CSV.
  - Trains a final run per architecture with the best config.
  - Evaluates ONCE on test (video 1, full 1024x1024 frames, never touched before
    this point): pixel IoU/Dice, instance AP@0.5, matched IoU/Dice, precision/
    recall/F1, AJI, localisation error, point-F1@3px, throughput (fps).
  - Saves a results table next to the YOLO11n+U-Net reference numbers.
  - Saves 3-5 qualitative overlay images.

Honesty notes baked in (see instructions.txt section 8): the granule masks you are
training against are PSEUDO-labels (classical bivariate-histogram + Hessian blob
detection), not human ground truth. High scores here mean "reproduces the classical
method", not "is correct". Say this explicitly in your report. Do not tune or
early-stop on test. When the 8 hand-labelled tiles arrive, re-evaluate on those too.
"""

# %% [0] Setup -----------------------------------------------------------------
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "torch", "torchvision", "scipy", "scikit-image", "numpy", "pandas",
                 "matplotlib"], check=False)

import os, json, time, random, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as cc_label, regionprops
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Device: cuda -", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")   # Apple Silicon GPU (M1/M2/M3)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # fall back to CPU for any op MPS doesn't support
    print("Device: mps (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Device: cpu (no GPU found — will be slow)")

DATA_DIR = "DATA_FOR_TEAM"   # change if your unzip path differs
OUT_DIR = "rahul_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

# %% [1] Inspect & load data -----------------------------------------------------
# DO NOT assume key names — print them first and adapt if different.
for f in ["v1_pre", "v2_pre", "v1_pl", "v2_pl"]:
    z = np.load(os.path.join(DATA_DIR, f"{f}.npz"))
    print(f, {k: (z[k].shape, z[k].dtype) for k in z.files})

split = json.load(open(os.path.join(DATA_DIR, "split.json")))
print(split.keys())

# Adjust these three key names if the printout above differs.
FRAME_KEY = "reg"
GRANULE_KEY = "granules"

v2_pre = np.load(os.path.join(DATA_DIR, "v2_pre.npz"))[FRAME_KEY]       # (T,1024,1024) float
v2_pl  = np.load(os.path.join(DATA_DIR, "v2_pl.npz"))[GRANULE_KEY]      # (T,1024,1024) int instance labels
v1_pre = np.load(os.path.join(DATA_DIR, "v1_pre.npz"))[FRAME_KEY]
v1_pl  = np.load(os.path.join(DATA_DIR, "v1_pl.npz"))[GRANULE_KEY]

train_idx = split["train"]["v2"]
val_idx   = split["val"]["v2"]
test_idx  = split["test"]["v1"]

print(f"train frames: {len(train_idx)}  val frames: {len(val_idx)}  test frames: {len(test_idx)}")

# NOTE on indexing: split.json frame numbers are 1-based in this project's convention
# (frame 0 of each video is excluded, per instructions.txt section 2). If your arrays
# are 0-indexed, subtract 1. Verify by checking array length vs max index below.
print("v2_pre length:", len(v2_pre), "max train/val idx:", max(train_idx), max(val_idx))
IDX_OFFSET = 0 if max(train_idx) < len(v2_pre) else -1  # auto-detect off-by-one

def fetch(arr, indices):
    return np.stack([arr[i + IDX_OFFSET] for i in indices])

train_frames = fetch(v2_pre, train_idx)
train_inst   = fetch(v2_pl, train_idx)
val_frames   = fetch(v2_pre, val_idx)
val_inst     = fetch(v2_pl, val_idx)
test_frames  = fetch(v1_pre, test_idx)
test_inst    = fetch(v1_pl, test_idx)   # kept as instance labels for metric computation

print("train_frames", train_frames.shape, "val_frames", val_frames.shape, "test_frames", test_frames.shape)

# %% [2] Tiling (train/val only — test stays as full frames) --------------------
TILE = 256
UPSCALE = 2          # 2x upsample -> effective 512px, per instructions.txt section 4
EMPTY_FRACTION = 0.2  # keep ~20% empty tiles

def make_tiles(frames, inst_labels, tile=TILE, upscale=UPSCALE, empty_frac=EMPTY_FRACTION, seed=0):
    rng = np.random.default_rng(seed)
    img_tiles, mask_tiles = [], []
    empty_pool = []
    H, W = frames.shape[1:]
    for f_idx in range(frames.shape[0]):
        frame = frames[f_idx]
        inst = inst_labels[f_idx]
        for y in range(0, H, tile):
            for x in range(0, W, tile):
                if y + tile > H or x + tile > W:
                    continue
                img_t = frame[y:y+tile, x:x+tile]
                inst_t = inst[y:y+tile, x:x+tile]
                mask_t = (inst_t > 0).astype(np.float32)
                if mask_t.sum() == 0:
                    empty_pool.append((img_t, mask_t))
                else:
                    img_tiles.append(img_t); mask_tiles.append(mask_t)
    n_keep_empty = int(len(img_tiles) * empty_frac / (1 - empty_frac))
    if empty_pool:
        chosen = rng.choice(len(empty_pool), size=min(n_keep_empty, len(empty_pool)), replace=False)
        for i in chosen:
            img_tiles.append(empty_pool[i][0]); mask_tiles.append(empty_pool[i][1])
    imgs = np.stack(img_tiles).astype(np.float32)
    masks = np.stack(mask_tiles).astype(np.float32)
    if upscale != 1:
        imgs_t = torch.from_numpy(imgs).unsqueeze(1)
        masks_t = torch.from_numpy(masks).unsqueeze(1)
        imgs_t = F.interpolate(imgs_t, scale_factor=upscale, mode="bilinear", align_corners=False)
        masks_t = F.interpolate(masks_t, scale_factor=upscale, mode="nearest")
        imgs, masks = imgs_t.squeeze(1).numpy(), masks_t.squeeze(1).numpy()
    return imgs, masks

train_imgs, train_masks = make_tiles(train_frames, train_inst, seed=0)
val_imgs, val_masks     = make_tiles(val_frames, val_inst, seed=1)
print("train tiles:", train_imgs.shape, "val tiles:", val_imgs.shape,
      f"(~{100*(train_masks.sum(axis=(1,2))==0).mean():.1f}% empty in train)")

# %% [3] Dataset + augmentation --------------------------------------------------
class GranuleTileDataset(Dataset):
    """flips, 90-degree rotations, brightness/contrast jitter only — NO hue/sat
    (grayscale fluorescence), NO heavy blur / large scale jumps, per instructions.txt."""
    def __init__(self, imgs, masks, augment=False):
        self.imgs, self.masks, self.augment = imgs, masks, augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, mask = self.imgs[idx].copy(), self.masks[idx].copy()
        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5:
                img, mask = img[::-1, :].copy(), mask[::-1, :].copy()
            k = random.choice([0, 1, 2, 3])
            img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()
            if random.random() < 0.5:
                brightness = random.uniform(-0.1, 0.1)
                contrast = random.uniform(0.85, 1.15)
                img = np.clip((img - 0.5) * contrast + 0.5 + brightness, 0, 1)
        img_t = torch.from_numpy(img).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return img_t, mask_t

def make_loaders(batch_size=16):
    train_ds = GranuleTileDataset(train_imgs, train_masks, augment=True)
    val_ds = GranuleTileDataset(val_imgs, val_masks, augment=False)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2))

# %% [4] Model definitions --------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class DWBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class DepthwiseUNet(nn.Module):
    """Lightweight U-Net built entirely from depthwise-separable convolutions."""
    def __init__(self, base=16):
        super().__init__()
        b = base
        self.enc1 = DWBlock(1, b)
        self.enc2 = DWBlock(b, b*2)
        self.enc3 = DWBlock(b*2, b*4)
        self.enc4 = DWBlock(b*4, b*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DWBlock(b*8, b*16)
        self.up4 = nn.ConvTranspose2d(b*16, b*8, 2, stride=2)
        self.dec4 = DWBlock(b*16, b*8)
        self.up3 = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.dec3 = DWBlock(b*8, b*4)
        self.up2 = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.dec2 = DWBlock(b*4, b*2)
        self.up1 = nn.ConvTranspose2d(b*2, b, 2, stride=2)
        self.dec1 = DWBlock(b*2, b)
        self.out_conv = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bott = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bott), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)   # logits


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class AttentionGate(nn.Module):
    """g = coarser decoder signal, x = finer encoder skip connection.
    Learns to suppress irrelevant background in the skip connection before
    it's merged into the decoder — meant to help small-object segmentation."""
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg = nn.Conv2d(g_ch, inter_ch, 1)
        self.Wx = nn.Conv2d(x_ch, inter_ch, 1)
        self.psi = nn.Conv2d(inter_ch, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        attn = self.sigmoid(self.psi(self.relu(self.Wg(g) + self.Wx(x))))
        return x * attn

class AttentionUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        b = base
        self.enc1 = ConvBlock(1, b)
        self.enc2 = ConvBlock(b, b*2)
        self.enc3 = ConvBlock(b*2, b*4)
        self.enc4 = ConvBlock(b*4, b*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(b*8, b*16)

        self.up4 = nn.ConvTranspose2d(b*16, b*8, 2, stride=2)
        self.att4 = AttentionGate(b*8, b*8, b*4)
        self.dec4 = ConvBlock(b*16, b*8)

        self.up3 = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.att3 = AttentionGate(b*4, b*4, b*2)
        self.dec3 = ConvBlock(b*8, b*4)

        self.up2 = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.att2 = AttentionGate(b*2, b*2, b)
        self.dec2 = ConvBlock(b*4, b*2)

        self.up1 = nn.ConvTranspose2d(b*2, b, 2, stride=2)
        self.att1 = AttentionGate(b, b, b//2)
        self.dec1 = ConvBlock(b*2, b)

        self.out_conv = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bott = self.bottleneck(self.pool(e4))

        d4 = self.up4(bott); e4a = self.att4(d4, e4); d4 = self.dec4(torch.cat([d4, e4a], dim=1))
        d3 = self.up3(d4);   e3a = self.att3(d3, e3); d3 = self.dec3(torch.cat([d3, e3a], dim=1))
        d2 = self.up2(d3);   e2a = self.att2(d2, e2); d2 = self.dec2(torch.cat([d2, e2a], dim=1))
        d1 = self.up1(d2);   e1a = self.att1(d1, e1); d1 = self.dec1(torch.cat([d1, e1a], dim=1))
        return self.out_conv(d1)

MODEL_REGISTRY = {"depthwise_unet": DepthwiseUNet, "attention_unet": AttentionUNet}

# %% [5] Loss ---------------------------------------------------------------------
def dice_bce_loss(logits, target, bce_weight=0.5, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1 - ((2 * inter + eps) / (denom + eps)).mean()
    return dice_loss + bce_weight * bce

@torch.no_grad()
def dice_score(logits, target, eps=1.0):
    probs = (torch.sigmoid(logits) > 0.5).float()
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean().item()

# %% [6] Training loop --------------------------------------------------------------
def train_model(model_name, base=16, lr=1e-3, bce_weight=0.5, batch_size=16,
                 max_epochs=40, patience=6, seed=0, tag="run", verbose=True):
    set_seed(seed)
    model = MODEL_REGISTRY[model_name](base=base).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader = make_loaders(batch_size=batch_size)

    best_val_dice, best_state, epochs_no_improve = -1, None, 0
    history = {"epoch": [], "train_loss": [], "val_dice": []}
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            logits = model(imgs)
            loss = dice_bce_loss(logits, masks, bce_weight=bce_weight)
            loss.backward()
            opt.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits = model(imgs)
                val_dices.append(dice_score(logits, masks))
        val_dice = float(np.mean(val_dices))

        history["epoch"].append(epoch); history["train_loss"].append(train_loss); history["val_dice"].append(val_dice)
        if verbose:
            print(f"[{tag}] epoch {epoch:02d}  train_loss {train_loss:.4f}  val_dice {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice, best_state, epochs_no_improve = val_dice, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"[{tag}] early stopping at epoch {epoch} (best val_dice={best_val_dice:.4f})")
                break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    return model, best_val_dice, history, elapsed

# %% [7] Hyperparameter sweep (VAL ONLY) ---------------------------------------------
sweep_configs = {
    "depthwise_unet": [
        {"base": 16, "lr": 1e-3}, {"base": 16, "lr": 3e-3},
        {"base": 32, "lr": 1e-3}, {"base": 32, "lr": 3e-3},
    ],
    "attention_unet": [
        {"base": 16, "lr": 1e-3}, {"base": 16, "lr": 3e-3},
        {"base": 32, "lr": 1e-3}, {"base": 32, "lr": 3e-3},
    ],
}

sweep_rows = []
for model_name, configs in sweep_configs.items():
    for cfg in configs:
        tag = f"{model_name}_sweep_base{cfg['base']}_lr{cfg['lr']}"
        _, val_dice, _, elapsed = train_model(
            model_name, base=cfg["base"], lr=cfg["lr"], bce_weight=0.5,
            max_epochs=12, patience=4, tag=tag, verbose=False,
        )
        sweep_rows.append({"model": model_name, **cfg, "bce_weight": 0.5,
                            "epochs": 12, "val_dice": val_dice, "train_time_s": round(elapsed, 1)})
        print(tag, "-> val_dice", round(val_dice, 4))

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(os.path.join(OUT_DIR, "hparam_sweep.csv"), index=False)
print(sweep_df)

# %% [8] Final training with best config per architecture ------------------------------
final_models, final_histories, final_results = {}, {}, {}
for model_name in MODEL_REGISTRY:
    best_row = sweep_df[sweep_df.model == model_name].sort_values("val_dice", ascending=False).iloc[0]
    print(f"Best config for {model_name}: base={best_row.base} lr={best_row.lr} (val_dice={best_row.val_dice:.4f})")
    model, val_dice, history, elapsed = train_model(
        model_name, base=int(best_row.base), lr=float(best_row.lr), bce_weight=0.5,
        max_epochs=40, patience=6, seed=0, tag=f"{model_name}_final",
    )
    final_models[model_name] = model
    final_histories[model_name] = history
    final_results[model_name] = {"val_dice": val_dice, "train_time_s": elapsed,
                                  "base": int(best_row.base), "lr": float(best_row.lr)}
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{model_name}_final.pt"))

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train loss")
    plt.plot(history["epoch"], history["val_dice"], label="val dice")
    plt.xlabel("epoch"); plt.legend(); plt.title(f"{model_name} training curve")
    plt.savefig(os.path.join(OUT_DIR, f"{model_name}_curve.png"), dpi=150)
    plt.close()

# %% [9] Test-set evaluation (full 1024x1024 frames, instance metrics) ------------------
def binary_to_instances(binary_mask):
    return cc_label(binary_mask > 0)

def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0

def compute_instance_metrics(pred_inst, gt_inst, iou_thresh=0.5, point_thresh=3.0):
    pred_ids = [i for i in np.unique(pred_inst) if i != 0]
    gt_ids = [i for i in np.unique(gt_inst) if i != 0]
    n_pred, n_gt = len(pred_ids), len(gt_ids)

    if n_pred == 0 or n_gt == 0:
        return dict(ap50=0.0, matched_iou=0.0, matched_dice=0.0, precision=0.0, recall=0.0,
                     f1=0.0, aji=0.0, loc_err=np.nan, point_f1=0.0,
                     pixel_iou=mask_iou(pred_inst > 0, gt_inst > 0),
                     pixel_dice=(2 * np.logical_and(pred_inst > 0, gt_inst > 0).sum() /
                                 max(1, (pred_inst > 0).sum() + (gt_inst > 0).sum())))

    pred_masks = [pred_inst == pid for pid in pred_ids]
    gt_masks = [gt_inst == gid for gid in gt_ids]

    iou_mat = np.zeros((n_pred, n_gt))
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            if np.logical_and(pm, gm).any():
                iou_mat[i, j] = mask_iou(pm, gm)

    cost = 1 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = [(r, c) for r, c in zip(row_ind, col_ind) if iou_mat[r, c] >= iou_thresh]

    tp = len(matched)
    precision = tp / n_pred
    recall = tp / n_gt
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    matched_ious = [iou_mat[r, c] for r, c in matched]
    matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    matched_dices = [2 * iv / (1 + iv) for iv in matched_ious]  # dice from iou: 2*iou/(1+iou)
    matched_dice = float(np.mean(matched_dices)) if matched_dices else 0.0

    # AP@0.5 via precision-recall sweep, ranking predictions by area (proxy for confidence
    # since the binary U-Net head has no per-instance score; larger/cleaner blobs = higher "confidence")
    scores = [pred_masks[i].sum() for i in range(n_pred)]
    order = np.argsort(-np.array(scores))
    tp_flags = np.zeros(n_pred)
    matched_pred_idx = {r for r, c in matched}
    for rank_pos, p_idx in enumerate(order):
        if p_idx in matched_pred_idx:
            tp_flags[rank_pos] = 1
    cum_tp = np.cumsum(tp_flags)
    cum_fp = np.cumsum(1 - tp_flags)
    rec = cum_tp / n_gt
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_at_t = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p_at_t / 11
    ap50 = ap

    # AJI (Aggregated Jaccard Index)
    used_gt = set()
    inter_sum, union_sum = 0, 0
    for i in range(n_pred):
        if iou_mat[i].max(initial=0) > 0:
            j = np.argmax(iou_mat[i])
            if iou_mat[i, j] > 0:
                inter = np.logical_and(pred_masks[i], gt_masks[j]).sum()
                union = np.logical_or(pred_masks[i], gt_masks[j]).sum()
                inter_sum += inter; union_sum += union
                used_gt.add(j)
    unused_gt_area = sum(gt_masks[j].sum() for j in range(n_gt) if j not in used_gt)
    union_sum += unused_gt_area
    aji = inter_sum / union_sum if union_sum > 0 else 0.0

    # centroid localisation error + point F1 @ 3px (separate 1-to-1 matching on centroids)
    pred_cent = np.array([regionprops(pm.astype(int))[0].centroid for pm in pred_masks])
    gt_cent = np.array([regionprops(gm.astype(int))[0].centroid for gm in gt_masks])
    dist_mat = np.linalg.norm(pred_cent[:, None, :] - gt_cent[None, :, :], axis=2)
    r2, c2 = linear_sum_assignment(dist_mat)
    point_matches = [(r, c) for r, c in zip(r2, c2) if dist_mat[r, c] <= point_thresh]
    point_tp = len(point_matches)
    point_prec = point_tp / n_pred
    point_rec = point_tp / n_gt
    point_f1 = 2 * point_prec * point_rec / (point_prec + point_rec) if (point_prec + point_rec) > 0 else 0.0
    loc_err = float(np.mean([dist_mat[r, c] for r, c in matched])) if matched else np.nan

    pixel_iou = mask_iou(pred_inst > 0, gt_inst > 0)
    pixel_dice = (2 * np.logical_and(pred_inst > 0, gt_inst > 0).sum() /
                  max(1, (pred_inst > 0).sum() + (gt_inst > 0).sum()))

    return dict(ap50=ap50, matched_iou=matched_iou, matched_dice=matched_dice,
                precision=precision, recall=recall, f1=f1, aji=aji, loc_err=loc_err,
                point_f1=point_f1, pixel_iou=pixel_iou, pixel_dice=pixel_dice)


@torch.no_grad()
def evaluate_on_test(model, frames, instance_masks, upscale=UPSCALE, tag=""):
    model.eval()
    per_frame_metrics = []
    t0 = time.time()
    for i in range(frames.shape[0]):
        img = torch.from_numpy(frames[i]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        if upscale != 1:
            img_up = F.interpolate(img, scale_factor=upscale, mode="bilinear", align_corners=False)
        else:
            img_up = img
        logits = model(img_up)
        probs = torch.sigmoid(logits)
        if upscale != 1:
            probs = F.interpolate(probs, size=frames.shape[1:], mode="bilinear", align_corners=False)
        pred_binary = (probs.squeeze().cpu().numpy() > 0.5)
        pred_inst = binary_to_instances(pred_binary)
        gt_inst = instance_masks[i]
        per_frame_metrics.append(compute_instance_metrics(pred_inst, gt_inst))
    elapsed = time.time() - t0
    fps = frames.shape[0] / elapsed

    df = pd.DataFrame(per_frame_metrics)
    summary = df.mean(numeric_only=True).to_dict()
    summary["throughput_fps"] = fps
    summary["tag"] = tag
    return summary, df

results_summary = []
for model_name, model in final_models.items():
    summary, per_frame_df = evaluate_on_test(model, test_frames, test_inst, tag=model_name)
    per_frame_df.to_csv(os.path.join(OUT_DIR, f"{model_name}_test_per_frame.csv"), index=False)
    results_summary.append(summary)
    print(model_name, "test results:", {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})

results_df = pd.DataFrame(results_summary)
results_df.to_csv(os.path.join(OUT_DIR, "test_results_summary.csv"), index=False)

# Reference row from instructions.txt section 6 (YOLO11n + U-Net, vs pseudo-labels)
reference_row = {
    "tag": "reference_YOLO11n_UNet", "ap50": 0.845, "matched_iou": 0.870, "matched_dice": 0.927,
    "pixel_iou": 0.600, "pixel_dice": 0.746, "precision": 0.693, "recall": 0.872, "f1": 0.771,
    "aji": 0.563, "loc_err": 0.43, "point_f1": 0.759, "throughput_fps": 5.5,
}
comparison_df = pd.concat([results_df, pd.DataFrame([reference_row])], ignore_index=True)
comparison_df.to_csv(os.path.join(OUT_DIR, "comparison_with_reference.csv"), index=False)
print(comparison_df)

# %% [10] Qualitative overlay examples --------------------------------------------------
def save_overlay_examples(model, frames, instance_masks, model_name, n=4, upscale=UPSCALE):
    model.eval()
    idxs = np.linspace(0, frames.shape[0] - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    with torch.no_grad():
        for ax, i in zip(axes, idxs):
            img = torch.from_numpy(frames[i]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            img_up = F.interpolate(img, scale_factor=upscale, mode="bilinear", align_corners=False)
            probs = torch.sigmoid(model(img_up))
            probs = F.interpolate(probs, size=frames.shape[1:], mode="bilinear", align_corners=False)
            pred_binary = (probs.squeeze().cpu().numpy() > 0.5)

            ax.imshow(frames[i], cmap="gray")
            gt_contour = (instance_masks[i] > 0).astype(float)
            ax.contour(gt_contour, colors="lime", linewidths=0.6)
            ax.contour(pred_binary.astype(float), colors="red", linewidths=0.6)
            ax.set_title(f"frame {i}"); ax.axis("off")
    fig.suptitle(f"{model_name}: green=pseudo-label, red=prediction")
    fig.savefig(os.path.join(OUT_DIR, f"{model_name}_overlay_examples.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

for model_name, model in final_models.items():
    save_overlay_examples(model, test_frames, test_inst, model_name)

print("\nAll outputs saved under:", os.path.abspath(OUT_DIR))
print("""
REMEMBER for your report (instructions.txt section 8):
- These masks are pseudo-labels (classical bivariate-histogram + Hessian blob method),
  not human ground truth. State this explicitly next to every number above.
- No tuning or early-stopping happened on test — only on val. State this too.
- When the 8 hand-labelled test tiles arrive, re-run evaluate_on_test() against them
  and report those numbers as the only true-ground-truth comparison.
""")
