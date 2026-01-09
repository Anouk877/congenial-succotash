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
MODEL_PATH = SCRIPT_DIR / "gletscher_unet_best.pth"

TIF_PATTERN = "*.TIF"
OUT_DIR = SCRIPT_DIR / "outputs_glacier_change"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CHANNELS = 10
TILE = 128
STRIDE = 128
THRESH = 0.5

RGB_IDXS = (3, 2, 1)
DISPLAY_MAX_EDGE = 1800

LOSS_COLOR = (1.0, 0.0, 0.0)
LOSS_ALPHA = 0.55

# GeoTIFF Outputs
WRITE_PROB_TIF = True   # prob_<year>.tif schreiben (float32)
WRITE_MASK_TIF = True   # mask_<year>.tif schreiben (uint8 0/1)
WRITE_LOSS_TIF = True   # loss_<old>_<new>.tif schreiben (uint8 0/1)

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


def write_singleband_geotiff(out_path: Path, arr2d: np.ndarray, ref_src: rasterio.DatasetReader,
                             dtype=None, nodata=None, compress="deflate"):
    """
    Schreibt ein 2D-Array als georeferenziertes GeoTIFF mit CRS/Transform/Shape aus ref_src.
    """
    if dtype is None:
        dtype = arr2d.dtype

    profile = ref_src.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=nodata
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr2d.astype(dtype), 1)


def infer_glacier_mask_and_prob(tif_path: Path):
    """
    Gibt mask (bool), prob (float32, 0..1, nan außerhalb), area_km2, crs, transform zurück.
    """
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

        px_w = float(transform.a)
        px_h = float(transform.e)
        pixel_area = abs(px_w * px_h)
        area_km2 = (float(mask.sum()) * pixel_area) / 1e6

        if crs is not None and crs.is_geographic:
            print(f"WARNUNG: CRS ist geografisch (Grad). Fläche in km² ist so nicht korrekt. "
                  f"Reprojiziere in metrisches CRS (z.B. UTM).")

        # Wir müssen ref_src fürs Schreiben draußen erneut öffnen oder hier pflegen.
        # Darum geben wir nur arrays zurück; schreiben erfolgt später mit einem erneut geöffneten src.
        return mask, prob_big, area_km2, crs, transform


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
    mask_old, prob_old, area_old_km2, crs_old, transform_old = infer_glacier_mask_and_prob(tif_old)
    mask_new, prob_new, area_new_km2, crs_new, transform_new = infer_glacier_mask_and_prob(tif_new)

    # Konsistenzcheck
    if mask_old.shape != mask_new.shape:
        raise RuntimeError(
            "Die Raster haben unterschiedliche Ausdehnung/Shape. "
            "Für saubere Differenzen müssen beide Jahre auf identischem Grid liegen "
            "(gleiches Mosaik-Extent, gleiche Auflösung/Transform)."
        )

    # Change: Rückgang
    loss = mask_old & (~mask_new)
    loss_px = int(loss.sum())

    px_w = float(transform_new.a)
    px_h = float(transform_new.e)
    pixel_area = abs(px_w * px_h)
    loss_km2 = (loss_px * pixel_area) / 1e6

    if crs_new is not None and crs_new.is_geographic:
        print("WARNUNG: CRS ist geografisch (Grad). km² ist so nicht korrekt. Reprojiziere in metrisches CRS (z.B. UTM).")

    # -----------------------
    # GeoTIFF Outputs schreiben
    # -----------------------
    # Wir nutzen die Profile/Georeferenz vom jeweiligen Eingabetif beim Schreiben.
    if WRITE_MASK_TIF or WRITE_PROB_TIF:
        with rasterio.open(tif_old) as src_old:
            if WRITE_MASK_TIF:
                out_mask_old = OUT_DIR / f"mask_{year_old}.tif"
                write_singleband_geotiff(out_mask_old, mask_old.astype(np.uint8), src_old, dtype=np.uint8, nodata=0)
                print("Saved:", out_mask_old)

            if WRITE_PROB_TIF:
                out_prob_old = OUT_DIR / f"prob_{year_old}.tif"
                nodata_val = -9999.0
                prob_out = prob_old.copy()
                prob_out[np.isnan(prob_out)] = nodata_val
                write_singleband_geotiff(out_prob_old, prob_out.astype(np.float32), src_old, dtype=np.float32, nodata=nodata_val)
                print("Saved:", out_prob_old)

        with rasterio.open(tif_new) as src_new:
            if WRITE_MASK_TIF:
                out_mask_new = OUT_DIR / f"mask_{year_new}.tif"
                write_singleband_geotiff(out_mask_new, mask_new.astype(np.uint8), src_new, dtype=np.uint8, nodata=0)
                print("Saved:", out_mask_new)

            if WRITE_PROB_TIF:
                out_prob_new = OUT_DIR / f"prob_{year_new}.tif"
                nodata_val = -9999.0
                prob_out = prob_new.copy()
                prob_out[np.isnan(prob_out)] = nodata_val
                write_singleband_geotiff(out_prob_new, prob_out.astype(np.float32), src_new, dtype=np.float32, nodata=nodata_val)
                print("Saved:", out_prob_new)

    if WRITE_LOSS_TIF:
        # Loss hat dasselbe Grid wie die Eingaben; wir schreiben es mit Profil von "new"
        with rasterio.open(tif_new) as src_new:
            out_loss = OUT_DIR / f"loss_{year_old}_{year_new}.tif"
            write_singleband_geotiff(out_loss, loss.astype(np.uint8), src_new, dtype=np.uint8, nodata=0)
            print("Saved:", out_loss)

    # -----------------------
    # PNG Quicklook
    # -----------------------
    with rasterio.open(tif_new) as src_new:
        rgb_img, step = read_rgb_quicklook(src_new, RGB_IDXS, DISPLAY_MAX_EDGE)
    loss_disp = downsample_mask(loss.astype(np.uint8), step)

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
        w.writerow([
            "year_old", "year_new",
            "area_old_km2", "area_new_km2", "loss_km2",
            "png",
            "mask_old_tif", "mask_new_tif",
            "loss_tif",
            "prob_old_tif", "prob_new_tif"
        ])

        w.writerow([
            year_old, year_new,
            f"{area_old_km2:.6f}", f"{area_new_km2:.6f}", f"{loss_km2:.6f}",
            str(out_png),
            str(OUT_DIR / f"mask_{year_old}.tif") if WRITE_MASK_TIF else "",
            str(OUT_DIR / f"mask_{year_new}.tif") if WRITE_MASK_TIF else "",
            str(OUT_DIR / f"loss_{year_old}_{year_new}.tif") if WRITE_LOSS_TIF else "",
            str(OUT_DIR / f"prob_{year_old}.tif") if WRITE_PROB_TIF else "",
            str(OUT_DIR / f"prob_{year_new}.tif") if WRITE_PROB_TIF else ""
        ])

    print("\nSaved:", out_png)
    print("Saved:", out_csv)
    print("Done.")


if __name__ == "__main__":
    main()
