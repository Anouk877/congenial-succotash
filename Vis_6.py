import os
import glob
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import csv

# -----------------------
# CONFIG (ANPASSEN)
# -----------------------
BASE_DIR = "/Users/anoukwieczorek/Documents/Geographie/Projektseminar/Projekt/Landsat_Input"
MODEL_PATH = "gletscher_unet_best.pth"

# Falls du wirklich NUR mosaics mit bestimmtem Namen hast, nutze PATTERN enger.
# Ansonsten: "*.TIF" oder "*.tif"
TIF_PATTERN = "*.TIF"

OUT_DIR = os.path.join(BASE_DIR, "_outputs_glacier")
os.makedirs(OUT_DIR, exist_ok=True)

IN_CHANNELS = 10
TILE = 128
STRIDE = 128          # 128 = keine Überlappung; 64 = Überlappung (glatter, langsamer)
THRESH = 0.5

# RGB-Band-Indizes (0-indexed)
RGB_IDXS = (3, 2, 1)

# Quicklook-Größe (längste Kante in Pixeln)
DISPLAY_MAX_EDGE = 1600

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)


# -----------------------
# MODEL LADEN (einmal)
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


def compute_positions(length: int, tile: int, stride: int):
    """
    Liefert Startpositionen so, dass der letzte Tile den Rand abdeckt.
    Funktioniert auch, wenn length < tile.
    """
    if length <= tile:
        return [0]
    last = length - tile
    pos = list(range(0, last + 1, stride))
    if pos[-1] != last:
        pos.append(last)
    return pos


def read_rgb_quicklook(src: rasterio.DatasetReader, rgb_idxs, max_edge: int):
    """
    Liest RGB als downsampled Quicklook direkt aus Rasterio (RAM-schonend).
    """
    h, w = src.height, src.width
    step = int(np.ceil(max(h, w) / max_edge))
    step = max(step, 1)

    out_h = int(np.ceil(h / step))
    out_w = int(np.ceil(w / step))

    # Rasterio ist 1-indexed bei Band-Indexes
    band_indexes = [rgb_idxs[0] + 1, rgb_idxs[1] + 1, rgb_idxs[2] + 1]

    rgb = src.read(
        indexes=band_indexes,
        out_shape=(3, out_h, out_w),
        resampling=Resampling.bilinear
    ).astype(np.float32)

    rgb = normalize_like_training(rgb)  # gleiche Normalisierung wie Training
    rgb_img = np.stack([rgb[0], rgb[1], rgb[2]], axis=-1)

    # Kontraststretch wie bei dir
    lo = np.percentile(rgb_img, 2)
    hi = np.percentile(rgb_img, 98)
    rgb_img = np.clip((rgb_img - lo) / (hi - lo + 1e-6), 0, 1)

    return rgb_img, step


def downsample_mask(mask: np.ndarray, step: int):
    """
    Maske (H,W) auf Quicklook-Größe bringen, indem jedes step-te Pixel genommen wird.
    """
    if step <= 1:
        return mask
    return mask[::step, ::step]


def find_year_folders(base_dir: str):
    """
    Findet Unterordner, die wie Jahre aussehen (z.B. '2013', '2014', ...).
    """
    years = []
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and name.isdigit() and len(name) == 4:
            years.append(int(name))
    return sorted(years)


def find_tif_for_year(year_dir: str, pattern: str):
    """
    Sucht eine passende TIF im Jahresordner. Nimmt die erste, falls mehrere gefunden werden.
    Du kannst hier auch Logik einbauen (z.B. 'stack_mosaic' bevorzugen).
    """
    candidates = glob.glob(os.path.join(year_dir, pattern))
    if not candidates:
        # auch lowercase versuchen
        candidates = glob.glob(os.path.join(year_dir, pattern.lower()))
    if not candidates:
        return None

    # Optional: bevorzugt Dateien mit 'mosaic' oder 'stack'
    preferred = [c for c in candidates if ("mosaic" in os.path.basename(c).lower() or "stack" in os.path.basename(c).lower())]
    if preferred:
        return sorted(preferred)[0]
    return sorted(candidates)[0]


def process_one_year(year: int, tif_path: str):
    print(f"\n--- Year {year} ---")
    print("Input:", tif_path)

    with rasterio.open(tif_path) as src:
        if src.count != IN_CHANNELS:
            raise ValueError(f"[{year}] Erwartet {IN_CHANNELS} Bänder, aber Datei hat {src.count}")

        H, W = src.height, src.width
        crs = src.crs
        transform = src.transform

        print(f"[{year}] Size (H,W): {(H, W)}  CRS: {crs}")

        prob_big = np.zeros((H, W), dtype=np.float32)
        cnt_big  = np.zeros((H, W), dtype=np.uint16)

        tops = compute_positions(H, TILE, STRIDE)
        lefts = compute_positions(W, TILE, STRIDE)

        with torch.no_grad():
            for top in tops:
                for left in lefts:
                    win = Window(left, top, TILE, TILE)
                    patch = src.read(window=win).astype(np.float32)  # (bands, TILE, TILE)

                    patch = normalize_like_training(patch)
                    inp = torch.from_numpy(patch).unsqueeze(0).to(device)  # (1,10,128,128)

                    logits = model(inp)
                    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

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
        pixel_area = abs(px_w * px_h)  # bei UTM: m²

        glacier_area_m2 = float(pred_big.sum()) * pixel_area
        glacier_area_km2 = glacier_area_m2 / 1e6

        if crs is not None and crs.is_geographic:
            print(f"[{year}] WARNUNG: CRS ist geografisch (Grad). Fläche in km² ist so nicht korrekt. "
                  f"Reprojiziere in metrisches CRS (z.B. UTM).")

        # -----------------------
        # RGB QUICKLOOK + MASK DOWNSAMPLE
        # -----------------------
        rgb_img, step = read_rgb_quicklook(src, RGB_IDXS, DISPLAY_MAX_EDGE)
        pred_disp = downsample_mask(pred_big.astype(np.uint8), step)

    # -----------------------
    # FIGURE SPEICHERN
    # -----------------------
    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(rgb_img)
    ax1.set_title(f"Landsat Satellitenbild {year}")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(rgb_img)

    overlay = np.zeros((pred_disp.shape[0], pred_disp.shape[1], 4), dtype=np.float32)
    overlay[..., 2] = 1.0
    overlay[..., 3] = pred_disp.astype(np.float32) * 0.45
    ax2.imshow(overlay)

    ax2.set_title(
        f"U-Net Gletscherkartierung ({year})\nBerechnete Fläche: {glacier_area_km2:.2f} km²",
        color="navy"
    )
    ax2.axis("off")

    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"preview_{year}.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[{year}] Saved figure: {out_png}")
    print(f"[{year}] Glacier area: {glacier_area_km2:.4f} km²")

    return glacier_area_km2, out_png


def main():
    years = find_year_folders(BASE_DIR)
    if not years:
        raise RuntimeError(f"Keine Jahresordner in {BASE_DIR} gefunden (erwartet Ordnernamen wie '2013', '2014', ...).")

    results = []  # (year, area_km2, png_path)

    for year in years:
        year_dir = os.path.join(BASE_DIR, str(year))
        tif_path = find_tif_for_year(year_dir, TIF_PATTERN)
        if tif_path is None:
            print(f"\n--- Year {year} ---")
            print(f"[{year}] Keine TIF gefunden in {year_dir} mit Pattern '{TIF_PATTERN}'. Überspringe.")
            continue

        area_km2, png_path = process_one_year(year, tif_path)
        results.append((year, area_km2, png_path))

    if not results:
        raise RuntimeError("Es wurden keine Jahre verarbeitet (keine passenden TIFs gefunden).")

    # -----------------------
    # CSV SUMMARY
    # -----------------------
    csv_path = os.path.join(OUT_DIR, "glacier_areas.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "glacier_area_km2", "preview_png"])
        for year, area, png in results:
            w.writerow([year, f"{area:.6f}", png])

    print("\nSaved CSV:", csv_path)

    # -----------------------
    # OPTIONAL: Zeitreihe plotten
    # -----------------------
    years_sorted = [r[0] for r in results]
    areas_sorted = [r[1] for r in results]

    fig = plt.figure(figsize=(10, 5))
    plt.plot(years_sorted, areas_sorted, marker="o")
    plt.title("Gletscherfläche über die Jahre (aus U-Net Prediction)")
    plt.xlabel("Jahr")
    plt.ylabel("Fläche (km²)")
    plt.grid(True, alpha=0.3)
    out_ts = os.path.join(OUT_DIR, "glacier_area_timeseries.png")
    fig.savefig(out_ts, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved timeseries:", out_ts)
    print("\nFertig. Output-Ordner:", OUT_DIR)


if __name__ == "__main__":
    main()

