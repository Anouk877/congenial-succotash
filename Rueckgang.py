import glob
import csv
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from pathlib import Path

# =====================================================
# SCRIPT LOCATION (portabel)
# =====================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# =====================================================
# CONFIG (ANPASSEN)
# =====================================================
BASE_DIR = Path("/Users/anoukwieczorek/Documents/Geographie/Projektseminar/Projekt/Landsat_Input")

# Modell neben dem Skript (empfohlen)
MODEL_PATH = SCRIPT_DIR / "gletscher_unet_best.pth"

TIF_PATTERN = "*.TIF"          # ggf. enger: "*stack_mosaic*.TIF"
OUT_DIR = SCRIPT_DIR / "outputs_glacier_change"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CHANNELS = 10
TILE = 128
STRIDE = 128
THRESH = 0.5

# RGB-Band-Indizes (0-indexed) -> Rasterio indexes sind +1
RGB_IDXS = (3, 2, 1)

# Quicklook max Kante
DISPLAY_MAX_EDGE = 1800

# Overlay-Parameter
LOSS_COLOR = (1.0, 0.0, 0.0)   # Rot für Rückgang
LOSS_ALPHA = 0.55

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# =====================================================
# MODEL LADEN
# =====================================================
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
    """Startpositionen so, dass der letzte Tile den Rand abdeckt."""
    if length <= tile:
        return [0]
    last = length - tile
    pos = list(range(0, last + 1, stride))
    if pos[-1] != last:
        pos.append(last)
    return pos


def find_year_folders(base_dir: Path):
    years = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.isdigit() and len(p.name) == 4:
            years.append(int(p.name))
    return sorted(years)


def find_tif_for_year(year_dir: Path, pattern: str):
    candidates = glob.glob(str(year_dir / pattern))
    if not candidates:
        candidates = glob.glob(str(year_dir / pattern.lower()))
    if not candidates:
        return None
    preferred = [c for c in candidates if ("mosaic" in Path(c).name.lower() or "stack" in Path(c).name.lower())]
    return sorted(preferred)[0] if preferred else sorted(candidates)[0]


def read_rgb_quicklook(src: rasterio.DatasetReader, rgb_idxs, max_edge: int):
    h, w = src.height, src.width
    step = int(np.ceil(max(h, w) / max_edge))
    step = max(step, 1)

    out_h = int(np.ceil(h / step))
    out_w = int(np.ceil(w / step))

    band_indexes = [rgb_idxs[0] + 1, rgb_idxs[1] + 1, rgb_idxs[2] + 1]
    rgb = src.read(
        indexes=band_indexes,
        out_shape=(3, out_h, out_w),
        resampling=Resampling.bilinear
    ).astype(np.float32)

    rgb = normalize_like_training(rgb)
    rgb_img = np.stack([rgb[0], rgb[1], rgb[2]], axis=-1)

    lo = np.percentile(rgb_img, 2)
    hi = np.percentile(rgb_img, 98)
    rgb_img = np.clip((rgb_img - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb_img, step


def downsample_mask(mask: np.ndarray, step: int):
    if step <= 1:
        return mask
    return mask[::step, ::step]


def infer_glacier_mask(tif_path: Path):
    """Gibt binäre Maske (H,W), Fläche (km²), CRS/Transform zurück."""
    with rasterio.open(tif_path) as src:
        if src.count != IN_CHANNELS:
            raise ValueError(f"Erwartet {IN_CHANNELS} Bänder, aber Datei hat {src.count}: {tif_path}")

        H, W = src.height, src.width
        crs = src.crs
        transform = src.transform

        prob_big = np.zeros((H, W), dtype=np.float32)
        cnt_big  = np.zeros((H, W), dtype=np.uint16)

        tops = compute_positions(H, TILE, STRIDE)
        lefts = compute_positions(W, TILE, STRIDE)

        with torch.no_grad():
            for top in tops:
                for left in lefts:
                    win = Window(left, top, TILE, TILE)
                    patch = src.read(window=win).astype(np.float32)
                    patch = normalize_like_training(patch)

                    inp = torch.from_numpy(patch).unsqueeze(0).to(device)
                    logits = model(inp)
                    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

                    prob_big[top:top+TILE, left:left+TILE] += prob
                    cnt_big[top:top+TILE, left:left+TILE]  += 1

        covered = cnt_big > 0
        prob_big[covered] = prob_big[covered] / cnt_big[covered]
        prob_big[~covered] = np.nan

        mask = (prob_big >= THRESH) & covered

        # Fläche
        px_w = float(transform.a)
        px_h = float(transform.e)
        pixel_area = abs(px_w * px_h)  # bei UTM: m²
        area_km2 = (float(mask.sum()) * pixel_area) / 1e6

        return mask, area_km2, crs, transform


def main():
    years = find_year_folders(BASE_DIR)
    if len(years) < 2:
        raise RuntimeError(f"Mindestens 2 Jahresordner nötig in {BASE_DIR}, gefunden: {years}")

    year_old = years[0]
    year_new = years[-1]

    tif_old = find_tif_for_year(BASE_DIR / str(year_old), TIF_PATTERN)
    tif_new = find_tif_for_year(BASE_DIR / str(year_new), TIF_PATTERN)

    if tif_old is None:
        raise RuntimeError(f"Keine TIF gefunden für {year_old} in {BASE_DIR/str(year_old)} (Pattern: {TIF_PATTERN})")
    if tif_new is None:
        raise RuntimeError(f"Keine TIF gefunden für {year_new} in {BASE_DIR/str(year_new)} (Pattern: {TIF_PATTERN})")

    tif_old = Path(tif_old)
    tif_new = Path(tif_new)

    print(f"Vergleich: {year_old} (alt) vs. {year_new} (neu)")
    print("Alt:", tif_old)
    print("Neu:", tif_new)

    # Inferenz beider Jahre
    mask_old, area_old_km2, crs_old, transform_old = infer_glacier_mask(tif_old)
    mask_new, area_new_km2, crs_new, transform_new = infer_glacier_mask(tif_new)

    # Minimaler Konsistenzcheck
    if mask_old.shape != mask_new.shape:
        raise RuntimeError(
            "Die Raster haben unterschiedliche Ausdehnung/Shape. "
            "Für saubere Differenzen müssen beide Jahre auf identischem Grid liegen "
            "(gleiches Mosaik-Extent, gleiche Auflösung/Transform)."
        )

    # Change: Rückgang (Loss) = alt & ~neu
    loss = mask_old & (~mask_new)
    loss_px = int(loss.sum())

    # Flächenverlust in km² (über Transform des 'neu' – identisch angenommen)
    px_w = float(transform_new.a)
    px_h = float(transform_new.e)
    pixel_area = abs(px_w * px_h)  # m² (bei UTM)
    loss_km2 = (loss_px * pixel_area) / 1e6

    if crs_new is not None and crs_new.is_geographic:
        print("WARNUNG: CRS ist geografisch (Grad). km² ist so nicht korrekt. Reprojiziere in metrisches CRS (z.B. UTM).")

    # RGB als Hintergrund: i. d. R. jüngstes Jahr ist anschaulicher
    with rasterio.open(tif_new) as src_new:
        rgb_img, step = read_rgb_quicklook(src_new, RGB_IDXS, DISPLAY_MAX_EDGE)
    loss_disp = downsample_mask(loss.astype(np.uint8), step)

    # Figure
    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(rgb_img)
    ax1.set_title(f"RGB Hintergrund ({year_new})")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(rgb_img)

    overlay = np.zeros((loss_disp.shape[0], loss_disp.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = LOSS_COLOR[0]
    overlay[..., 1] = LOSS_COLOR[1]
    overlay[..., 2] = LOSS_COLOR[2]
    overlay[..., 3] = loss_disp.astype(np.float32) * LOSS_ALPHA
    ax2.imshow(overlay)

    ax2.set_title(
        f"Rückgang {year_old} → {year_new}\n"
        f"Fläche {year_old}: {area_old_km2:.2f} km² | {year_new}: {area_new_km2:.2f} km²\n"
        f"Verlust (alt ∧ ¬neu): {loss_km2:.2f} km²",
        color="darkred"
    )
    ax2.axis("off")

    plt.tight_layout()

    out_png = OUT_DIR / f"glacier_retreat_{year_old}_{year_new}.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # CSV summary
    out_csv = OUT_DIR / f"glacier_change_{year_old}_{year_new}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year_old", "year_new", "area_old_km2", "area_new_km2", "loss_km2", "output_png"])
        w.writerow([year_old, year_new, f"{area_old_km2:.6f}", f"{area_new_km2:.6f}", f"{loss_km2:.6f}", str(out_png)])

    print("\nSaved:", out_png)
    print("Saved:", out_csv)
    print("Done.")


if __name__ == "__main__":
    main()
