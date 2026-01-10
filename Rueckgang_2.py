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
OUT_DIR = SCRIPT_DIR / "outputs_glacier_change_majority_6y"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CHANNELS = 10
TILE = 128
STRIDE = 128
THRESH = 0.7

RGB_IDXS = (3, 2, 1)
DISPLAY_MAX_EDGE = 1800

LOSS_COLOR = (1.0, 0.0, 0.0)
LOSS_ALPHA = 0.55

# ==========================
# MAJORITY CONFIG (6 years)
# ==========================
WINDOW_YEARS = 6
# For 6 years, recommend a strict threshold to suppress snow/noise:
# 4/6 years must be glacier for pixel to be considered glacier.
MAJORITY_K = 4

# GeoTIFF Outputs
WRITE_PROB_TIF = False          # prob_<year>.tif would be per-year; usually not needed for majority
WRITE_YEAR_MASKS = True         # optional: write each yearly mask used in composites
WRITE_COMPOSITES = True         # write start/end majority composites
WRITE_LOSS_TIF = True           # write loss between composites

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
    return Path(sorted(preferred)[0] if preferred else sorted(candidates)[0])


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
    """Schreibt ein 2D-Array als georeferenziertes GeoTIFF mit CRS/Transform/Shape aus ref_src."""
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


def infer_glacier_mask(tif_path: Path):
    """Gibt mask (bool), area_km2, crs, transform zurück."""
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

        return mask, area_km2, crs, transform


def majority_mask(masks: list[np.ndarray], k: int) -> np.ndarray:
    """
    Pixelweise Mehrheitsentscheidung: True, wenn Pixel in >=k Masken True ist.
    masks: Liste von bool arrays gleicher Shape.
    """
    if len(masks) == 0:
        raise ValueError("majority_mask: empty masks list")
    stack = np.stack([m.astype(np.uint8) for m in masks], axis=0)  # (N,H,W)
    votes = stack.sum(axis=0)                                      # (H,W)
    return votes >= int(k)


def main():
    years_all = find_year_folders(BASE_DIR)
    if len(years_all) < 2 * WINDOW_YEARS:
        raise RuntimeError(
            f"Mindestens {2*WINDOW_YEARS} Jahresordner nötig, gefunden: {years_all}"
        )

    # First 6 and last 6 years (robust against missing intermediate years)
    years_first = years_all[:WINDOW_YEARS]
    years_last = years_all[-WINDOW_YEARS:]

    print("Years (first window):", years_first)
    print("Years (last window): ", years_last)
    print(f"Majority threshold: >= {MAJORITY_K}/{WINDOW_YEARS}")

    # ---- Infer masks for both windows ----
    masks_first = []
    masks_last = []
    areas_first = []
    areas_last = []

    # reference datasets for writing (use first year of each window)
    tif_ref_first = None
    tif_ref_last = None

    # Helper to infer a list of years
    def infer_years(years_list):
        masks = []
        areas = []
        tifs = []
        for y in years_list:
            tif_path = find_tif_for_year(BASE_DIR / str(y), TIF_PATTERN)
            if tif_path is None:
                raise RuntimeError(f"Keine TIF gefunden für {y} in {BASE_DIR/str(y)} (Pattern: {TIF_PATTERN})")
            tif_path = Path(tif_path)
            print(f"Infer year {y}: {tif_path.name}")
            mask, area_km2, _, _ = infer_glacier_mask(tif_path)
            masks.append(mask)
            areas.append(area_km2)
            tifs.append(tif_path)
        return masks, areas, tifs

    masks_first, areas_first, tifs_first = infer_years(years_first)
    masks_last,  areas_last,  tifs_last  = infer_years(years_last)

    # ---- Consistency check: shapes must match across all years used ----
    shape0 = masks_first[0].shape
    for m in masks_first + masks_last:
        if m.shape != shape0:
            raise RuntimeError(
                "Nicht alle Masken haben denselben Shape/Extent. "
                "Für Majority + Differenzen müssen alle Jahre auf identischem Grid liegen."
            )

    # ---- Majority composites ----
    comp_first = majority_mask(masks_first, MAJORITY_K)
    comp_last  = majority_mask(masks_last,  MAJORITY_K)

    # ---- Retreat between composites ----
    loss = comp_first & (~comp_last)

    # Areas for composites/loss (use transform from reference)
    with rasterio.open(tifs_last[-1]) as src_ref:
        transform = src_ref.transform
        crs = src_ref.crs
        px_w = float(transform.a)
        px_h = float(transform.e)
        pixel_area = abs(px_w * px_h)
        comp_first_km2 = (float(comp_first.sum()) * pixel_area) / 1e6
        comp_last_km2  = (float(comp_last.sum())  * pixel_area) / 1e6
        loss_km2       = (float(loss.sum())       * pixel_area) / 1e6

        if crs is not None and crs.is_geographic:
            print("WARNUNG: CRS ist geografisch (Grad). km² ist so nicht korrekt. Reprojiziere in metrisches CRS (z.B. UTM).")

    # ---- Write GeoTIFF outputs ----
    if WRITE_YEAR_MASKS:
        # write yearly masks used in each window
        for y, tif, m in zip(years_first, tifs_first, masks_first):
            with rasterio.open(tif) as src:
                out = OUT_DIR / f"mask_{y}.tif"
                write_singleband_geotiff(out, m.astype(np.uint8), src, dtype=np.uint8, nodata=0)
        for y, tif, m in zip(years_last, tifs_last, masks_last):
            with rasterio.open(tif) as src:
                out = OUT_DIR / f"mask_{y}.tif"
                write_singleband_geotiff(out, m.astype(np.uint8), src, dtype=np.uint8, nodata=0)

    if WRITE_COMPOSITES or WRITE_LOSS_TIF:
        # use last-year profile for composites/loss (same grid assumed)
        with rasterio.open(tifs_last[-1]) as src_ref:
            if WRITE_COMPOSITES:
                out_first = OUT_DIR / f"majority_{years_first[0]}_{years_first[-1]}_k{MAJORITY_K}of{WINDOW_YEARS}.tif"
                out_last  = OUT_DIR / f"majority_{years_last[0]}_{years_last[-1]}_k{MAJORITY_K}of{WINDOW_YEARS}.tif"
                write_singleband_geotiff(out_first, comp_first.astype(np.uint8), src_ref, dtype=np.uint8, nodata=0)
                write_singleband_geotiff(out_last,  comp_last.astype(np.uint8),  src_ref, dtype=np.uint8, nodata=0)
                print("Saved:", out_first)
                print("Saved:", out_last)

            if WRITE_LOSS_TIF:
                out_loss = OUT_DIR / f"loss_majority_{years_first[0]}_{years_first[-1]}_to_{years_last[0]}_{years_last[-1]}.tif"
                write_singleband_geotiff(out_loss, loss.astype(np.uint8), src_ref, dtype=np.uint8, nodata=0)
                print("Saved:", out_loss)

    # ---- PNG Quicklook (use RGB from last year of last window) ----
    with rasterio.open(tifs_last[-1]) as src_bg:
        rgb_img, step = read_rgb_quicklook(src_bg, RGB_IDXS, DISPLAY_MAX_EDGE)

    loss_disp = downsample_mask(loss.astype(np.uint8), step)

    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(rgb_img)
    ax1.set_title(f"RGB background ({years_last[-1]})")
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
        f"Retreat from Majority Masks\n"
        f"Start: {years_first[0]}–{years_first[-1]} (≥{MAJORITY_K}/{WINDOW_YEARS})  |  "
        f"End: {years_last[0]}–{years_last[-1]} (≥{MAJORITY_K}/{WINDOW_YEARS})\n"
        f"Area start: {comp_first_km2:.2f} km² | Area end: {comp_last_km2:.2f} km² | Loss: {loss_km2:.2f} km²",
        color="darkred"
    )
    ax2.axis("off")

    plt.tight_layout()

    out_png = OUT_DIR / f"retreat_majority_{years_first[0]}_{years_first[-1]}_to_{years_last[0]}_{years_last[-1]}.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- CSV summary ----
    out_csv = OUT_DIR / "majority_change_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "window_first_years", "window_last_years",
            "majority_k", "window_size",
            "area_majority_first_km2", "area_majority_last_km2", "loss_km2",
            "png_output"
        ])
        w.writerow([
            f"{years_first[0]}-{years_first[-1]}",
            f"{years_last[0]}-{years_last[-1]}",
            MAJORITY_K, WINDOW_YEARS,
            f"{comp_first_km2:.6f}",
            f"{comp_last_km2:.6f}",
            f"{loss_km2:.6f}",
            str(out_png)
        ])

    print("\nSaved PNG:", out_png)
    print("Saved CSV:", out_csv)
    print("Output folder:", OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
