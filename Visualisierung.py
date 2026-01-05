import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
MODEL_FILE = "gletscher_unet_best.pth"  # besser als "final"
BASE_DIR = "Landsat_Input"
OUTPUT_BASE = "Landsat_Analyse"

TILE = 128
STRIDE = 128        # 64 = Overlap (bessere Kachelränder, langsamer)
BATCH_TILES = 16
THRESH = 0.5
SCALE_DIV = 30000.0

# optional: Probability-TIF zusätzlich schreiben
WRITE_PROB_TIF = True

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Nutze Device: {device}")


# -----------------------
# MODEL
# -----------------------
def load_model():
    model = smp.Unet(encoder_name="resnet34", in_channels=10, classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    return model


# -----------------------
# PREPROCESS
# -----------------------
def preprocess(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.maximum(arr, 0)
    arr = arr / SCALE_DIV
    arr = np.clip(arr, 0, 1).astype(np.float32)
    return arr


def pad_to_tile(arr: np.ndarray, tile_size: int):
    c, h, w = arr.shape
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h == 0 and pad_w == 0:
        return arr, (0, 0)
    padded = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return padded, (pad_h, pad_w)


def unpad_2d(arr2d: np.ndarray, pad_h: int, pad_w: int):
    if pad_h == 0 and pad_w == 0:
        return arr2d
    return arr2d[: arr2d.shape[0] - pad_h, : arr2d.shape[1] - pad_w]


# -----------------------
# SLIDING WINDOW PREDICTION
# -----------------------
def predict_prob_map(model, src: rasterio.io.DatasetReader) -> np.ndarray:
    """
    Returns probability map (H,W) float32.
    Uses overlap-averaging if STRIDE < TILE.
    """
    H, W = src.height, src.width

    prob_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    xs = list(range(0, W, STRIDE))
    ys = list(range(0, H, STRIDE))

    batch_tiles = []
    batch_meta = []  # (x, y, win_w, win_h, pad_h, pad_w)

    def flush():
        if not batch_tiles:
            return

        x_np = np.stack(batch_tiles, axis=0)  # (B,C,TILE,TILE)
        x_t = torch.from_numpy(x_np).to(device)

        with torch.no_grad():
            logits = model(x_t)  # (B,1,TILE,TILE)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)  # (B,TILE,TILE)

        for i, (x0, y0, win_w, win_h, pad_h, pad_w) in enumerate(batch_meta):
            p = unpad_2d(probs[i], pad_h, pad_w)  # (win_h, win_w)
            y1, x1 = y0 + win_h, x0 + win_w
            prob_sum[y0:y1, x0:x1] += p
            weight_sum[y0:y1, x0:x1] += 1.0

        batch_tiles.clear()
        batch_meta.clear()

    for y in ys:
        for x in xs:
            win_w = min(TILE, W - x)
            win_h = min(TILE, H - y)
            window = Window(x, y, win_w, win_h)

            tile = src.read(window=window).astype(np.float32)  # (C,win_h,win_w)
            tile = preprocess(tile)
            tile, (pad_h, pad_w) = pad_to_tile(tile, TILE)

            batch_tiles.append(tile)
            batch_meta.append((x, y, win_w, win_h, pad_h, pad_w))

            if len(batch_tiles) >= BATCH_TILES:
                flush()

    flush()

    weight_sum = np.maximum(weight_sum, 1e-6)
    return prob_sum / weight_sum


# -----------------------
# OUTPUT WRITERS
# -----------------------
def write_geotiff_like(ref: rasterio.io.DatasetReader, out_path: str, arr2d: np.ndarray, dtype, nodata=None):
    profile = ref.profile.copy()
    profile.update(driver="GTiff", count=1, dtype=dtype, compress="lzw", nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr2d, 1)


def make_overlay_png(ref: rasterio.io.DatasetReader, pred_mask: np.ndarray, out_png: str, title: str):
    """
    For visualization: reads RGB bands and overlays prediction.
    NOTE: Band indices [3,2,1] assume a particular Landsat band order in your 10-band stack.
    Adjust if needed.
    """
    img = ref.read().astype(np.float32)
    img_n = preprocess(img)

    rgb = img_n[[3, 2, 1]].transpose(1, 2, 0)
    rgb = np.clip(rgb * 5, 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    ax1.imshow(rgb)
    ax1.set_title(title, fontsize=14)
    ax1.axis("off")

    ax2.imshow(rgb)
    overlay = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 4), dtype=np.float32)
    overlay[pred_mask == 1] = [0, 0.4, 1, 0.6]
    ax2.imshow(overlay)
    ax2.set_title("U-Net Vorhersage (Overlay)", fontsize=14, color="darkblue")
    ax2.axis("off")

    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)


# -----------------------
# MAIN
# -----------------------
def erstelle_visuelle_analyse():
    print("Starte visuelle Analyse (PNG + GeoTIFF Outputs)...")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    model = load_model()

    jahre = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

    for jahr in jahre:
        jahr_dir = os.path.join(BASE_DIR, jahr)
        out_dir = os.path.join(OUTPUT_BASE, str(jahr))
        os.makedirs(out_dir, exist_ok=True)

        tif_files = sorted([f for f in os.listdir(jahr_dir) if f.lower().endswith((".tif", ".tiff"))])
        if not tif_files:
            continue

        for fname in tif_files:
            path = os.path.join(jahr_dir, fname)
            scene = os.path.splitext(fname)[0]
            print(f"Verarbeite {jahr} / {scene} ...")

            with rasterio.open(path) as src:
                if src.count != 10:
                    raise ValueError(f"{path}: Erwartet 10 Bänder, gefunden {src.count}")

                # Predict probability map (robust for big scenes)
                prob_map = predict_prob_map(model, src)
                pred_mask = (prob_map > THRESH).astype(np.uint8)

                # Area in km² from transform (not hardcoded 900 m²)
                px_w = src.transform.a
                px_h = src.transform.e
                pixel_area_m2 = abs(px_w * px_h)
                area_km2 = (pred_mask.sum() * pixel_area_m2) / 1_000_000.0

                # Write GeoTIFF mask (+ optional prob map)
                out_mask_tif = os.path.join(out_dir, f"{scene}_glacier_mask.tif")
                write_geotiff_like(src, out_mask_tif, pred_mask.astype(np.uint8), dtype=rasterio.uint8, nodata=0)

                if WRITE_PROB_TIF:
                    out_prob_tif = os.path.join(out_dir, f"{scene}_glacier_prob.tif")
                    write_geotiff_like(src, out_prob_tif, prob_map.astype(np.float32), dtype=rasterio.float32, nodata=None)

                # PNG overlay (quicklook)
                out_png = os.path.join(out_dir, f"{scene}_overlay.png")
                make_overlay_png(src, pred_mask, out_png, title=f"{jahr} – {scene} | Fläche: {area_km2:.2f} km²")

                print(f"  -> Fläche: {area_km2:.2f} km²")
                print(f"  -> Maske:  {out_mask_tif}")
                if WRITE_PROB_TIF:
                    print(f"  -> Prob:   {out_prob_tif}")
                print(f"  -> PNG:    {out_png}")

    print(f"\nFertig. Outputs gespeichert in: {OUTPUT_BASE}")


if __name__ == "__main__":
    erstelle_visuelle_analyse()
