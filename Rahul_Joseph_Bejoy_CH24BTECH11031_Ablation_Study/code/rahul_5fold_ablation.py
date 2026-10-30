"""
Rahul: 5-fold ablation study (Group 7).

Trains three models that differ by ONE change each, on Prabhav's 5-fold tile package:
    UNet      standard 3x3 convolutions (baseline)
    DWUNet    same network, convolutions replaced by depthwise-separable ones
    DWAttUNet same as DWUNet + attention gates on the skip connections

Everything else is identical (base width, depth, loss, optimiser, augmentation, early stopping).

Usage (Terminal, venv active, from the folder that contains sg_dataset_5fold/):

    # 1) quick check that everything runs (about 2-4 minutes):
    python rahul_5fold_ablation.py --folds 1 --seeds 0 --max-epochs 1 --out test_run

    # 2) the real run (leave it overnight; keeps the Mac awake):
    caffeinate -dimsu python rahul_5fold_ablation.py

The script is RESUMABLE: if it stops (crash, sleep, Ctrl+C), run the same command again and it
skips every run already saved in runs.csv. Seeds are the outer loop, so after seed 0 you already
have a complete 5-fold set for all three models; seeds 1 and 2 are a bonus if time allows.

Outputs (in --out, default rahul_5fold_outputs/):
    runs.csv        one row per run, EXACTLY Prabhav's format: model,fold,seed,dice,iou,f1,aji,ap50,tp,fp,fn
    runs_extra.csv  same runs + precision, recall, localisation error, parameters, speed, epochs, time
    history/        training curve (loss, val dice) per run
    ckpt/           best checkpoint per run (needed later to score the video-3 tiles)
    preds/          test-tile predictions per run (binary masks)

Metric definitions (tell Vedant, so everyone scores the same way):
    dice  pixel Dice over all test tiles of the fold (foreground vs background, pooled)
    iou   mean IoU of matched granules (match = IoU >= 0.5, one-to-one, Hungarian)
    f1, tp, fp, fn   granule counts pooled over the fold's test tiles
    aji   Aggregated Jaccard Index pooled over the test tiles
    ap50  AP@0.5, predictions ranked by blob area (a U-Net gives no confidence score)
"""
import argparse, glob, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.io import imread
from skimage.measure import label as cc_label

# ----------------------------------------------------------------------------- setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
else:
    DEVICE = torch.device("cpu")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync():
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    elif DEVICE.type == "mps": torch.mps.synchronize()


# ----------------------------------------------------------------------------- data
def read_gray(path):
    a = imread(path)
    if a.ndim == 3:
        a = a[..., 0]
    return a


def load_fold(root, j, role, with_instances=False):
    """Returns images uint8 (N,H,W), masks uint8 (N,H,W) in {0,1}, instances uint16 or None, names."""
    img_files = sorted(glob.glob(os.path.join(root, f"fold_{j}", role, "images", "*.png")))
    if not img_files:
        raise FileNotFoundError(f"No tiles found in {root}/fold_{j}/{role}/images - check --root and that the zip was unzipped")
    imgs = np.stack([read_gray(f).astype(np.uint8) for f in img_files])
    masks = np.stack([(read_gray(f.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep)) > 0).astype(np.uint8)
                      for f in img_files])
    inst = None
    if with_instances:
        inst = np.stack([read_gray(f.replace(os.sep + "images" + os.sep, os.sep + "instances" + os.sep)).astype(np.uint16)
                         for f in img_files])
    return imgs, masks, inst, [os.path.basename(f) for f in img_files]


class TileDataset(Dataset):
    """Same augmentation for every model: flips, 90-degree rotations, brightness/contrast jitter."""
    def __init__(self, imgs, masks, augment):
        self.imgs, self.masks, self.augment = imgs, masks, augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].astype(np.float32) / 255.0
        mask = self.masks[i].astype(np.float32)
        if self.augment:
            if random.random() < 0.5: img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5: img, mask = img[::-1, :], mask[::-1, :]
            k = random.choice([0, 1, 2, 3])
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if random.random() < 0.5:
                img = np.clip((img - 0.5) * random.uniform(0.85, 1.15) + 0.5 + random.uniform(-0.1, 0.1), 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0))


# ----------------------------------------------------------------------------- models
class DSConv(nn.Module):
    """Depthwise 3x3 (spatial, per channel) + pointwise 1x1 (mixes channels) + BN + ReLU."""
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin), nn.Conv2d(cin, cout, 1),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


def conv_block(cin, cout, kind):
    if kind == "std":
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    return nn.Sequential(DSConv(cin, cout), DSConv(cout, cout))


class AttentionGate(nn.Module):
    """g = coarser decoder signal, x = encoder skip. Learns to down-weight irrelevant skip regions."""
    def __init__(self, g_ch, x_ch, inter):
        super().__init__()
        self.Wg, self.Wx = nn.Conv2d(g_ch, inter, 1), nn.Conv2d(x_ch, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, g, x):
        return x * torch.sigmoid(self.psi(F.relu(self.Wg(g) + self.Wx(x))))


class UNet(nn.Module):
    def __init__(self, base=16, conv="std", attention=False):
        super().__init__()
        b = base
        self.enc = nn.ModuleList([conv_block(1, b, conv), conv_block(b, 2*b, conv),
                                  conv_block(2*b, 4*b, conv), conv_block(4*b, 8*b, conv)])
        self.pool = nn.MaxPool2d(2)
        self.bott = conv_block(8*b, 16*b, conv)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(16*b, 8*b, 2, stride=2), nn.ConvTranspose2d(8*b, 4*b, 2, stride=2),
                                  nn.ConvTranspose2d(4*b, 2*b, 2, stride=2), nn.ConvTranspose2d(2*b, b, 2, stride=2)])
        self.atts = (nn.ModuleList([AttentionGate(8*b, 8*b, 4*b), AttentionGate(4*b, 4*b, 2*b),
                                    AttentionGate(2*b, 2*b, b), AttentionGate(b, b, max(b // 2, 1))])
                     if attention else None)
        self.decs = nn.ModuleList([conv_block(16*b, 8*b, conv), conv_block(8*b, 4*b, conv),
                                   conv_block(4*b, 2*b, conv), conv_block(2*b, b, conv)])
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = self.pool(x)
        x = self.bott(x)
        for i, (up, dec) in enumerate(zip(self.ups, self.decs)):
            x = up(x)
            s = skips[-(i + 1)]
            if self.atts is not None:
                s = self.atts[i](x, s)
            x = dec(torch.cat([x, s], dim=1))
        return self.out(x)


MODELS = {
    "UNet":      dict(conv="std", attention=False),
    "DWUNet":    dict(conv="dw",  attention=False),
    "DWAttUNet": dict(conv="dw",  attention=True),
}


# ----------------------------------------------------------------------------- loss / training
def dice_bce_loss(logits, target, bce_weight=0.5, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - ((2 * inter + eps) / (denom + eps)).mean()) + bce_weight * bce


@torch.no_grad()
def batch_dice(logits, target, eps=1.0):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean().item()


def train_one(model_name, data, seed, args):
    set_seed(seed)
    model = UNet(base=args.base, **MODELS[model_name]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr = DataLoader(TileDataset(data["tr_x"], data["tr_y"], True), batch_size=args.batch,
                    shuffle=True, num_workers=0, drop_last=True)
    va = DataLoader(TileDataset(data["va_x"], data["va_y"], False), batch_size=args.batch,
                    shuffle=False, num_workers=0)
    best, best_state, bad, hist, t0 = -1.0, None, 0, [], time.time()
    for ep in range(1, args.max_epochs + 1):
        te = time.time()
        model.train(); run = 0.0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = dice_bce_loss(model(x), y, args.bce)
            loss.backward(); opt.step()
            run += loss.item() * x.size(0)
        tl = run / len(tr.dataset)
        model.eval(); vd = []
        with torch.no_grad():
            for x, y in va:
                vd.append(batch_dice(model(x.to(DEVICE)), y.to(DEVICE)))
        vd = float(np.mean(vd))
        hist.append(dict(epoch=ep, train_loss=tl, val_dice=vd, epoch_seconds=time.time() - te))
        print(f"    epoch {ep:02d}  loss {tl:.4f}  val_dice {vd:.4f}  ({time.time() - te:.0f}s)", flush=True)
        if vd > best:
            best, bad = vd, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"    early stop at epoch {ep} (best val_dice {best:.4f})", flush=True)
                break
    model.load_state_dict(best_state)
    return model, best, hist, time.time() - t0


# ----------------------------------------------------------------------------- evaluation
def compact_labels(lbl):
    """Relabel to 0 (background), 1..n. Returns (int64 labels, n_instances)."""
    lbl = lbl.astype(np.int64)
    ids, inv = np.unique(lbl, return_inverse=True)
    inv = inv.reshape(lbl.shape)
    if ids[0] != 0:
        inv = inv + 1
    return inv.astype(np.int64), int((ids != 0).sum())


@torch.no_grad()
def predict_tiles(model, imgs, batch=32):
    model.eval()
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(imgs[i:i + batch].astype(np.float32) / 255.0).unsqueeze(1).to(DEVICE)
        out.append((torch.sigmoid(model(x))[:, 0] > 0.5).cpu().numpy())
    return np.concatenate(out)


def evaluate_tiles(preds, gt_insts, iou_thresh=0.5):
    tp = fp = fn = 0
    ious, locs, ap_items = [], [], []          # ap_items: (blob area, is_true_positive)
    n_gt_total, C, U = 0, 0, 0
    inter_fg = pred_fg = gt_fg = 0
    for pred_bin, gt in zip(preds, gt_insts):
        p_lbl, n_p = cc_label(pred_bin, return_num=True)
        p_lbl = p_lbl.astype(np.int64)
        g_lbl, n_g = compact_labels(gt)
        pred_fg += int((p_lbl > 0).sum()); gt_fg += int((g_lbl > 0).sum())
        n_gt_total += n_g
        if n_p == 0 and n_g == 0:
            continue
        cont = np.bincount((p_lbl * (n_g + 1) + g_lbl).ravel(),
                           minlength=(n_p + 1) * (n_g + 1)).reshape(n_p + 1, n_g + 1)
        area_p, area_g, inter = cont.sum(1)[1:], cont.sum(0)[1:], cont[1:, 1:]
        inter_fg += int(inter.sum())
        if n_g == 0:
            fp += n_p; ap_items += [(a, False) for a in area_p]; U += int(area_p.sum()); continue
        if n_p == 0:
            fn += n_g; U += int(area_g.sum()); continue
        union = area_p[:, None] + area_g[None, :] - inter
        iou = inter / np.maximum(union, 1)
        r, c = linear_sum_assignment(1 - iou)
        matched = [(i, j) for i, j in zip(r, c) if iou[i, j] >= iou_thresh]
        t = len(matched); tp += t; fp += n_p - t; fn += n_g - t
        ious += [iou[i, j] for i, j in matched]
        mp = {i for i, _ in matched}
        ap_items += [(area_p[i], i in mp) for i in range(n_p)]
        if t:
            ones = np.ones(p_lbl.shape, np.uint8)
            pc = np.array(ndi.center_of_mass(ones, p_lbl, np.arange(1, n_p + 1)))
            gc = np.array(ndi.center_of_mass(ones, g_lbl, np.arange(1, n_g + 1)))
            locs += [float(np.linalg.norm(pc[i] - gc[j])) for i, j in matched]
        # AJI (for each GT granule: best-overlapping prediction; unused predictions add to the union)
        used = np.zeros(n_p, bool)
        for j in range(n_g):
            i = int(np.argmax(iou[:, j]))
            if iou[i, j] > 0:
                C += int(inter[i, j]); U += int(union[i, j]); used[i] = True
            else:
                U += int(area_g[j])
        U += int(area_p[~used].sum())
    # AP@0.5 pooled over all test tiles (11-point interpolation, ranked by blob area)
    ap50 = 0.0
    if ap_items and n_gt_total:
        order = np.argsort([-a for a, _ in ap_items])
        flags = np.array([1.0 if ap_items[k][1] else 0.0 for k in order])
        ctp, cfp = np.cumsum(flags), np.cumsum(1 - flags)
        rec, prec = ctp / n_gt_total, ctp / (ctp + cfp)
        ap50 = float(np.mean([prec[rec >= th].max() if (rec >= th).any() else 0.0 for th in np.linspace(0, 1, 11)]))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return dict(
        dice=2 * inter_fg / max(pred_fg + gt_fg, 1),
        iou=float(np.mean(ious)) if ious else 0.0,
        f1=2 * precision * recall / (precision + recall) if tp else 0.0,
        aji=C / max(U, 1), ap50=ap50, tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, loc_err=float(np.mean(locs)) if locs else float("nan"))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sg_dataset_5fold", help="folder that contains fold_1 ... fold_5")
    ap.add_argument("--out", default="rahul_5fold_outputs")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--base", type=int, default=16, help="base channel width, same for all three models")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bce", type=float, default=0.5, help="weight of the BCE term in Dice + w*BCE")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    for d in ("history", "ckpt", "preds"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    runs_path, extra_path = os.path.join(args.out, "runs.csv"), os.path.join(args.out, "runs_extra.csv")
    json.dump(vars(args) | {"device": str(DEVICE), "torch": torch.__version__},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    done = set()
    if os.path.exists(runs_path):
        prev = pd.read_csv(runs_path)
        done = set(zip(prev.model, prev.fold, prev.seed))
    print(f"Device: {DEVICE} | already finished runs: {len(done)}", flush=True)

    cache = {}
    for seed in args.seeds:                      # seeds outer: after seed 0 you already have a full 5-fold set
        for j in args.folds:
            if j not in cache:
                tr = load_fold(args.root, j, "train"); va = load_fold(args.root, j, "val")
                te = load_fold(args.root, j, "test", with_instances=True)
                cache[j] = dict(tr_x=tr[0], tr_y=tr[1], va_x=va[0], va_y=va[1], te_x=te[0], te_inst=te[2], te_names=te[3])
                print(f"fold {j}: train {len(tr[0])}  val {len(va[0])}  test {len(te[0])} tiles", flush=True)
            data = cache[j]
            for name in args.models:
                if (name, j, seed) in done:
                    continue
                print(f"\n=== {name} | fold {j} | seed {seed} ===", flush=True)
                model, best_val, hist, secs = train_one(name, data, seed, args)
                sync(); t0 = time.time()
                preds = predict_tiles(model, data["te_x"])
                sync(); tiles_per_s = len(preds) / max(time.time() - t0, 1e-9)
                m = evaluate_tiles(preds, data["te_inst"])
                n_params = sum(p.numel() for p in model.parameters())
                print(f"    TEST  dice {m['dice']:.3f}  iou {m['iou']:.3f}  f1 {m['f1']:.3f}  aji {m['aji']:.3f}  "
                      f"ap50 {m['ap50']:.3f}  tp/fp/fn {m['tp']}/{m['fp']}/{m['fn']}  ({secs / 60:.1f} min)", flush=True)
                tag = f"{name}_fold{j}_seed{seed}"
                torch.save(model.state_dict(), os.path.join(args.out, "ckpt", tag + ".pt"))
                np.savez_compressed(os.path.join(args.out, "preds", tag + ".npz"),
                                    pred=preds.astype(np.uint8), names=np.array(data["te_names"]))
                pd.DataFrame(hist).to_csv(os.path.join(args.out, "history", tag + ".csv"), index=False)
                row = dict(model=name, fold=j, seed=seed, dice=m["dice"], iou=m["iou"], f1=m["f1"], aji=m["aji"],
                           ap50=m["ap50"], tp=m["tp"], fp=m["fp"], fn=m["fn"])
                extra = row | dict(precision=m["precision"], recall=m["recall"], loc_err=m["loc_err"],
                                   val_dice=best_val, epochs=len(hist), train_minutes=secs / 60,
                                   params=n_params, tiles_per_s=tiles_per_s, device=str(DEVICE))
                for path, r in ((runs_path, row), (extra_path, extra)):   # append after EVERY run (crash-safe)
                    pd.DataFrame([r]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                done.add((name, j, seed))

    if os.path.exists(extra_path):
        e = pd.read_csv(extra_path)
        print("\n=== mean over finished runs ===")
        print(e.groupby("model")[["dice", "iou", "f1", "aji", "ap50", "tp", "fp", "fn", "params"]].mean().round(3).to_string())
    print("\nDone. Send runs.csv to Vedant.")


if __name__ == "__main__":
    main()
