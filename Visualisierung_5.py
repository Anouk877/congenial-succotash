import os
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# -----------------------
# CONFIG (ANPASSEN)
# -----------------------
YEAR = 2013
INPUT_TIF = "/Users/anoukwieczorek/Documents/Geographie/Projektseminar/Projekt/Landsat_Input/2014/LC08_L2SP_193027_2014XXXX_20200911_02_T1_SR_stack_mosaic.TIF"
MODEL_PATH = "gletscher_unet_best.pth"

IN_CHANNELS = 10
TILE = 128
STRIDE = 128          # 128 = keine Überlappung; 64 = Überlappung (glatter, langsamer)
THRESH = 0.5

# RGB-Band-Indizes (0-indexed) – ggf. anpassen
# Beispiel wie zuvor: R=Band4, G=Band3, B=Band2 -> (3,2,1)
RGB_IDXS = (3, 2, 1)

# Für die Anzeige: wenn Szene sehr groß ist, wird downsampled (schneller, weniger RAM)
DISPLAY_MAX_EDGE = 1600  # None = keine Verkleinerung

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# -----------------------
# MODEL LADEN
# -----------------------
model = smp.Unet(encoder_name="resnet34", in_channels=IN_CHANNELS, classes=1)
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.to(device)
model.eval()

def normalize_like_training(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.maximum(x, 0)
    x = x / 30000.0
    x = np.clip(x, 0, 1)
    return x

def make_rgb(img10: np.ndarray, rgb_idxs=(3,2,1)) -> np.ndarray:
    r, g, b = img10[rgb_idxs[0]], img10[rgb_idxs[1]], img10[rgb_idxs[2]]
    rgb = np.stack([r, g, b], axis=-1)
    lo = np.percentile(rgb, 2)
    hi = np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb

def simple_downsample(arr, max_edge: int):
    """Downsample by integer step (quick and dependency-free)."""
    if max_edge is None:
        return arr
    h, w = arr.shape[:2]
    step = int(np.ceil(max(h, w) / max_edge))
    if step <= 1:
        return arr
    return arr[::step, ::step] if arr.ndim == 2 else arr[::step, ::step, :]

# -----------------------
# INPUT LESEN + INFERENZ (TILING)
# -----------------------
with rasterio.open(INPUT_TIF) as src:
    if src.count != IN_CHANNELS:
        raise ValueError(f"Erwartet {IN_CHANNELS} Bänder, aber Datei hat {src.count}")

    H, W = src.height, src.width
    crs = src.crs
    transform = src.transform

    print("Input size (H,W):", (H, W))
    print("CRS:", crs)

    # große Vorhersage-Arrays
    prob_big = np.zeros((H, W), dtype=np.float32)
    cnt_big  = np.zeros((H, W), dtype=np.uint16)

    # optional: wir lesen RGB später einmal (für Anzeige) – aber dafür brauchen wir 10 Bänder nur für ein Downsample
    # Für korrektes RGB-Quicklook aus exakt derselben Normalisierung:
    # Wir lesen die drei RGB-Bänder vollständig (kann groß sein, aber meist ok). Wenn zu groß: downsample per window.
    # -> robust: wir bauen RGB aus einem Downsample-Read

    # --- Inference loop ---
    with torch.no_grad():
        for top in range(0, H - TILE + 1, STRIDE):
            for left in range(0, W - TILE + 1, STRIDE):
                win = Window(left, top, TILE, TILE)

                # (bands, TILE, TILE)
                patch = src.read(window=win).astype(np.float32)

                patch = normalize_like_training(patch)

                inp = torch.from_numpy(patch).unsqueeze(0).to(device)  # (1,10,128,128)
                logits = model(inp)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)

                prob_big[top:top+TILE, left:left+TILE] += prob
                cnt_big[top:top+TILE, left:left+TILE]  += 1

    covered = cnt_big > 0
    prob_big[covered] = prob_big[covered] / cnt_big[covered]
    prob_big[~covered] = np.nan

    pred_big = (prob_big >= THRESH) & covered

    # -----------------------
    # FLÄCHE BERECHNEN (km²)
    # -----------------------
    px_w = float(transform.a)
    px_h = float(transform.e)
    pixel_area = abs(px_w * px_h)  # in CRS-Einheiten^2 (bei UTM: m²)

    glacier_area_m2 = pred_big.sum() * pixel_area
    glacier_area_km2 = glacier_area_m2 / 1e6

    if crs is not None and crs.is_geographic:
        print("WARNUNG: CRS ist geografisch (Grad). km²-Fläche ist so nicht korrekt. Reprojiziere in metrisches CRS (z.B. UTM).")

    # -----------------------
    # RGB QUICKLOOK (DOWN-SAMPLED, damit es schnell bleibt)
    # -----------------------
    # Wir lesen ein grobes RGB-Bild via Resampling durch Window-Stepping
    # Für Einfachheit: wir lesen die RGB-Bänder komplett und downsamplen danach.
    rgb_full = src.read(indexes=[RGB_IDXS[0]+1, RGB_IDXS[1]+1, RGB_IDXS[2]+1]).astype(np.float32)
    rgb_full = normalize_like_training(rgb_full)
    rgb_img = np.stack([rgb_full[0], rgb_full[1], rgb_full[2]], axis=-1)
    lo = np.percentile(rgb_img, 2)
    hi = np.percentile(rgb_img, 98)
    rgb_img = np.clip((rgb_img - lo) / (hi - lo + 1e-6), 0, 1)

# -----------------------
# DISPLAY (wie Screenshot)
# -----------------------
rgb_disp = simple_downsample(rgb_img, DISPLAY_MAX_EDGE)
pred_disp = simple_downsample(pred_big.astype(np.uint8), DISPLAY_MAX_EDGE)

fig = plt.figure(figsize=(12, 6))

ax1 = plt.subplot(1, 2, 1)
ax1.imshow(rgb_disp)
ax1.set_title(f"Landsat Satellitenbild {YEAR}")
ax1.axis("off")

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(rgb_disp)

# blaues Overlay (RGBA)
overlay = np.zeros((pred_disp.shape[0], pred_disp.shape[1], 4), dtype=np.float32)
overlay[..., 2] = 1.0
overlay[..., 3] = pred_disp.astype(np.float32) * 0.45
ax2.imshow(overlay)

ax2.set_title(
    f"U-Net Gletscherkartierung ({YEAR})\nBerechnete Fläche: {glacier_area_km2:.2f} km²",
    color="navy"
)
ax2.axis("off")

plt.tight_layout()
plt.show()

