import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
import rasterio

BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.0001

LANDSAT_DIR = "/Users/nicolasyukio/Downloads/Landsat_patches"
PANGEA_DIR = "/Users/nicolasyukio/Downloads/Pangea_patches"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Nutze Device: {device}")


class GlacierDataset(Dataset):
    def __init__(self, landsat_dir, pangea_dir):
        self.landsat_dir = landsat_dir
        self.pangea_dir = pangea_dir
        self.l_files = sorted([f for f in os.listdir(landsat_dir) if f.endswith(".tif")])
        self.p_files = sorted([f for f in os.listdir(pangea_dir) if f.endswith(".npy")])

        self.sub_patches = []
        for idx in range(len(self.l_files)):
            for q in range(4):
                self.sub_patches.append((idx, q))

    def __len__(self):
        return len(self.sub_patches)

    def __getitem__(self, idx):
        orig_idx, quadrant = self.sub_patches[idx]

        with rasterio.open(os.path.join(self.landsat_dir, self.l_files[orig_idx])) as src:
            image = src.read().astype(np.float32)
            image = np.nan_to_num(image, nan=0.0)
            image = np.maximum(image, 0)
            image = image / 30000.0
            image = np.clip(image, 0, 1)

        mask = np.load(os.path.join(self.pangea_dir, self.p_files[orig_idx])).astype(np.float32)
        mask = np.nan_to_num(mask, nan=0.0)
        mask = np.clip(mask, 0, 1)

        if quadrant == 0:
            img_c, mask_c = image[:, :128, :128], mask[:128, :128]
        elif quadrant == 1:
            img_c, mask_c = image[:, :128, 128:], mask[:128, 128:]
        elif quadrant == 2:
            img_c, mask_c = image[:, 128:, :128], mask[128:, :128]
        else:
            img_c, mask_c = image[:, 128:, 128:], mask[128:, 128:]

        return torch.from_numpy(img_c), torch.from_numpy(mask_c).unsqueeze(0)


full_dataset = GlacierDataset(LANDSAT_DIR, PANGEA_DIR)
train_set, val_set = random_split(full_dataset,
                                  [int(0.8 * len(full_dataset)), len(full_dataset) - int(0.8 * len(full_dataset))])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

model = smp.Unet(encoder_name="resnet34", in_channels=10, classes=1).to(device)

pos_weight = torch.tensor([50.0]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Starte Gletscher-Boost Training...")

for epoch in range(EPOCHS):
    model.train()
    t_loss = 0
    correct = 0
    total = 0

    for imgs, msks in train_loader:
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = loss_fn(outs, msks)
        loss.backward()
        optimizer.step()

        t_loss += loss.item()
        preds = (torch.sigmoid(outs) > 0.5).float()
        correct += (preds == msks).sum().item()
        total += torch.numel(preds)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:02d} | Loss: {t_loss / len(train_loader):.4f} | Acc: {correct / total:.2%}")

torch.save(model.state_dict(), "gletscher_unet_final.pth")
print("\nTraining beendet. Modell gespeichert!")