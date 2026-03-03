import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
import segmentation_models_pytorch as smp


BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
VAL_RATIO = 0.2
SEED = 42

LANDSAT_DIR = "/Users/anoukwieczorek/Documents/Geographie/Projektseminar/Projekt/2017_patches_128"   
MASK_DIR    = "/Users/anoukwieczorek/Documents/Geographie/Projektseminar/Projekt/mask_patches_128_npy"   
MASK_EXT = ".npy"  

EXPECTED_H = 128
EXPECTED_W = 128


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Nutze Device: {device}")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

ID_RE = re.compile(r"(\d{6})")

def extract_id(filename: str) -> str:
    m = ID_RE.search(filename)
    if not m:
        raise ValueError(f"Keine 6-stellige ID in Dateiname gefunden: {filename}")
    return m.group(1)

def list_by_id(folder: str, ext: str) -> dict:
    files = [f for f in os.listdir(folder) if f.endswith(ext)]
    d = {}
    for f in files:
        pid = extract_id(f)
        d[pid] = f
    return d

def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    """
    logits: (B,1,H,W) raw outputs
    targets: (B,1,H,W) in {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds_f = preds.view(preds.shape[0], -1)
    t_f = targets.view(targets.shape[0], -1)

    inter = (preds_f * t_f).sum(dim=1)
    union = preds_f.sum(dim=1) + t_f.sum(dim=1)

    dice = (2 * inter + eps) / (union + eps)

    denom = (preds_f.sum(dim=1) + t_f.sum(dim=1) - inter)
    iou = (inter + eps) / (denom + eps)

    return dice.mean().item(), iou.mean().item()

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        probs_f = probs.view(probs.shape[0], -1)
        t_f = targets.view(targets.shape[0], -1)

        inter = (probs_f * t_f).sum(dim=1)
        union = probs_f.sum(dim=1) + t_f.sum(dim=1)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()

class GlacierDataset(Dataset):
    def __init__(self, landsat_dir: str, mask_dir: str, mask_ext: str = ".npy", augment: bool = False):
        self.landsat_dir = landsat_dir
        self.mask_dir = mask_dir
        self.mask_ext = mask_ext
        self.augment = augment

        self.img_by_id = list_by_id(landsat_dir, ".tif")
        self.msk_by_id = list_by_id(mask_dir, mask_ext)

        self.ids = sorted(set(self.img_by_id.keys()) & set(self.msk_by_id.keys()))
        if not self.ids:
            raise RuntimeError("Keine gemeinsamen IDs zwischen Image- und Mask-Ordner gefunden.")

        print(f"Gefundene gemeinsame Patch-IDs: {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def _read_image(self, img_path: str) -> np.ndarray:
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
        image = np.nan_to_num(image, nan=0.0)
        image = np.maximum(image, 0)
        image = image / 30000.0
        image = np.clip(image, 0, 1)
        return image

    def _read_mask(self, msk_path: str) -> np.ndarray:
        if self.mask_ext == ".npy":
            mask = np.load(msk_path).astype(np.float32)
        else:
            with rasterio.open(msk_path) as src:
                mask = src.read(1).astype(np.float32)
        mask = np.nan_to_num(mask, nan=0.0)
        mask = np.clip(mask, 0, 1)
        mask = (mask > 0.5).astype(np.float32)
        return mask

    def __getitem__(self, idx: int):
        pid = self.ids[idx]
        img_path = os.path.join(self.landsat_dir, self.img_by_id[pid])
        msk_path = os.path.join(self.mask_dir, self.msk_by_id[pid])

        image = self._read_image(img_path)  # (10,128,128)
        mask  = self._read_mask(msk_path)   # (128,128)

        #
        if image.shape[0] != 10:
            raise ValueError(f"ID {pid}: Erwartet 10 Bänder, bekommen {image.shape[0]}")
        if image.shape[1] != EXPECTED_H or image.shape[2] != EXPECTED_W:
            raise ValueError(
                f"ID {pid}: Erwartet {EXPECTED_H}x{EXPECTED_W}, "
                f"bekommen {image.shape[1]}x{image.shape[2]}"
            )
        if mask.shape != (EXPECTED_H, EXPECTED_W):
            raise ValueError(f"ID {pid}: Mask shape mismatch: {mask.shape}")

        # Augmentation
        if self.augment:
            k = random.randint(0, 3)  # 0°, 90°, 180°, 270°
            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
                mask  = np.rot90(mask,  k=k, axes=(0, 1)).copy()

            if random.random() < 0.5:
                image = image[:, :, ::-1].copy()
                mask  = mask[:, ::-1].copy()

            if random.random() < 0.5:
                image = image[:, ::-1, :].copy()
                mask  = mask[::-1, :].copy()

        x = torch.from_numpy(image)
        y = torch.from_numpy(mask).unsqueeze(0)
        return x, y, pid






# Erst Dataset ohne Augmentation erstellen, um IDs zu bekommen
base_ds = GlacierDataset(LANDSAT_DIR, MASK_DIR, mask_ext=MASK_EXT, augment=False)

all_ids = base_ds.ids.copy()
random.shuffle(all_ids)
n_val = int(len(all_ids) * VAL_RATIO)

val_ids = set(all_ids[:n_val])
train_ids = set(all_ids[n_val:])

#  Train mit Augmentierung, Val ohne
train_full = GlacierDataset(LANDSAT_DIR, MASK_DIR, mask_ext=MASK_EXT, augment=True)
val_full   = GlacierDataset(LANDSAT_DIR, MASK_DIR, mask_ext=MASK_EXT, augment=False)

train_indices = [i for i, pid in enumerate(base_ds.ids) if pid in train_ids]
val_indices   = [i for i, pid in enumerate(base_ds.ids) if pid in val_ids]

train_ds = torch.utils.data.Subset(train_full, train_indices)
val_ds   = torch.utils.data.Subset(val_full, val_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train IDs: {len(train_ids)} | Val IDs: {len(val_ids)}")
print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")


model = smp.Unet(
    encoder_name="resnet34",
    in_channels=10,
    classes=1
).to(device)

pos_weight = torch.tensor([50.0], device=device)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
dice_loss = DiceLoss()

def loss_fn(logits, targets):
    return bce(logits, targets) + dice_loss(logits, targets)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

best_val_dice = -1.0

print("Starte Training")

# Training
for epoch in range(1, EPOCHS + 1):
  
    model.train()
    train_loss = 0.0

    for imgs, msks, _ in train_loader:
        imgs = imgs.to(device)
        msks = msks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, msks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= max(1, len(train_loader))

   
    model.eval()
    val_loss = 0.0
    val_dice_accum = 0.0
    val_iou_accum = 0.0
    n_batches = 0

    with torch.no_grad():
        for imgs, msks, _ in val_loader:
            imgs = imgs.to(device)
            msks = msks.to(device)

            logits = model(imgs)
            loss = loss_fn(logits, msks)
            val_loss += loss.item()

            d, i = dice_iou_from_logits(logits, msks)
            val_dice_accum += d
            val_iou_accum += i
            n_batches += 1

    val_loss /= max(1, n_batches)
    val_dice = val_dice_accum / max(1, n_batches)
    val_iou = val_iou_accum / max(1, n_batches)

    scheduler.step(val_dice)
import os, math

if math.isnan(val_dice):
    print(">>> val_dice ist NaN – deshalb wird nie gespeichert.")
else:
    print(f">>> Check save condition: val_dice={val_dice:.6f} best_val_dice={best_val_dice:.6f}")

if val_dice > best_val_dice:
    print(">>> SAVE TRIGGERED")
    best_val_dice = val_dice

    save_path = "gletscher_unet_best.pth"
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, save_path)

    print(">>> CWD:", os.getcwd())
    print(">>> Saved to:", os.path.abspath(save_path))
    print(">>> Exists?", os.path.exists(save_path))

    # save best model
    #if val_dice > best_val_dice:
     #   best_val_dice = val_dice
      #  torch.save(model.state_dict(), "gletscher_unet_best.pth")

   
    #if epoch == 1 or epoch % 5 == 0:
     #   lr = optimizer.param_groups[0]["lr"]
      #  print(
       #     f"Epoch {epoch:02d} | lr={lr:.2e} | "
        #    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
         #   f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f} | best_dice={best_val_dice:.4f}"
       # )

print("\nTraining beendet.")
print("Bestes Modell gespeichert als: gletscher_unet_best.pth")

