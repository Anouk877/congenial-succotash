

import glob
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from pathlib import Path
from scipy.ndimage import distance_transform_edt

# =====================================================
# SCRIPT LOCATION (portable)
# =====================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# =====================================================
# CONFIG (adjust)
# =====================================================
BASE_DIR = Path("/Users/anoukwieczorek/Documents/Geographie/Projektseminar/Projekt/Landsat_Input")

YEAR_OLD = 2013
YEAR_NEW = 2024

MODEL_PATH = SCRIPT_DIR / "gletscher_unet_best.pth"
TIF_PATTERN = "*.TIF"  # optionally narrower: "*stack_mosaic*.TIF"

OUT_DIR = SCRIPT_DIR / "outputs_vulnerability_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CHANNELS = 10
TILE = 128
STRIDE = 128          # 128 no overlap, 64 overlap
THRESH = 0.5

# RGB (0-indexed) for preview
RGB_IDXS = (3, 2, 1)
DISPLAY_MAX_EDGE = 1800

# Vulnerability parameters (Option B: exponential)
MAX_BUFFER_M = 1000.0     # maximum influence distance (m)
LAMBDA_M = 250.0          # decay length scale (m); smaller -> faster decay
NODATA = -9999.0

# Preview style
VULN_COLOR = (1.0, 0.0, 0.0)   # red
VULN_ALPHA_MAX = 0.75         # max alpha for V=1

# Output toggles
WRITE_PROB_TIF = False         # optional probability maps
WRITE_DISTANCE_TIF = True

# =====================================================
# DEVICE
# =====================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# =====================================================
# MODEL
# =====================================================
model = smp.Unet(encoder_name="resnet34", in_channels=IN_CHANNELS, classes=1)
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.to(device)
model.eval()


# =====================================================
# HELPERS
# =====================================================
def normalize_like_training(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.maximum(x, 0)
    x = x / 30000.0
    x = np.clip(x, 0, 1)
    return x


def compute_positions(length: int, tile: int, stride: int):
    """Start positions ensuring last tile hits the edge."""
    if length <= tile:
        return [0]
    last = length - tile
    pos = list(range(0, last + 1, stride))
    if pos[-1] != last:
        pos.append(last)
    return pos


def find_tif(year: int) -> Path:
    year_dir = BASE_DIR / str(year)
    candidates = glob.glob(str(year_dir / TIF_PATTERN))
    if not candidates:
        candidates = glob.glob(str(year_dir / TIF_PATTERN.lower()))
    if not candidates:
        raise RuntimeError(f"Keine TIF gefunden für {year} in {year_dir} (Pattern: {TIF_PATTERN})")

    preferred = [c for c in candidates if ("mosaic" in Path(c).name.lower() or "stack" in Path(c).name.lower())]
    return Path(sorted(preferred)[0] if preferred else sorted(candidates)[0])


def read_rgb_quicklook(src: rasterio.DatasetReader, rgb_idxs, max_edge: int):
    """Read downsampled RGB quicklook via rasterio out_shape."""
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


def downsample(arr: np.ndarray, step: int):
    if step <= 1:
        return arr
    if arr.ndim == 2:
        return arr[::step, ::step]
    return arr[::step, ::step, :]


def write_singleband_geotiff(out_path: Path, arr2d: np.ndarray, ref_src: rasterio.DatasetReader,
                             dtype=None, nodata=None, compress="deflate"):
    """Write 2D array as GeoTIFF using CRS/transform/profile from ref_src."""
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


def infer_mask_and_prob(tif_path: Path):
    """
    Returns:
      mask (bool), prob (float32 0..1 with nan outside), crs, transform
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

        if crs is not None and crs.is_geographic:
            print("WARNUNG: CRS ist geografisch (Grad). Distanz/Buffer in Metern ist so nicht korrekt. "
                  "Bitte reprojiziere in metrisches CRS (z.B. UTM).")

        return mask, prob_big, crs, transform


def compute_loss(mask_old: np.ndarray, mask_new: np.ndarray):
    """Retreat/loss pixels: glacier before, not glacier now."""
    return mask_old & (~mask_new)


def vulnerability_exponential(loss_mask: np.ndarray,
                              current_glacier_mask: np.ndarray,
                              transform,
                              max_buffer_m: float,
                              lambda_m: float):
    """
    Continuous vulnerability V in [0,1] with exponential decay from loss pixels.

    - Evaluated in buffer around retreat
    - Outside buffer -> 0
    - Inside current glacier -> 0 (explicitly, NOT nodata)
    """
    px_w = float(transform.a)
    px_h = float(transform.e)
    sampling = (abs(px_h), abs(px_w))  # meters if CRS is metric

    # distance_transform_edt computes distance to zeros -> set loss pixels to 0
    inv = (~loss_mask).astype(np.uint8)
    dist = distance_transform_edt(inv, sampling=sampling).astype(np.float32)

    # initialize with zeros everywhere
    V = np.zeros(loss_mask.shape, dtype=np.float32)

    # valid zone: within buffer AND outside current glacier
    in_buffer = dist <= max_buffer_m
    outside_glacier = ~current_glacier_mask
    valid = in_buffer & outside_glacier

    # exponential decay
    V_valid = np.exp(-dist[valid] / float(lambda_m)).astype(np.float32)
    V_valid = np.clip(V_valid, 0.0, 1.0)

    V[valid] = V_valid

    # enforce: loss pixels themselves = 1
    V[loss_mask & outside_glacier] = 1.0

    # inside current glacier remains 0 by construction
    return V, dist



# =====================================================
# MAIN
# =====================================================
def main():
    tif_old = find_tif(YEAR_OLD)
    tif_new = find_tif(YEAR_NEW)

    print("Old:", tif_old)
    print("New:", tif_new)

    mask_old, prob_old, crs_old, transform_old = infer_mask_and_prob(tif_old)
    mask_new, prob_new, crs_new, transform_new = infer_mask_and_prob(tif_new)

    if mask_old.shape != mask_new.shape:
        raise RuntimeError(
            "Die Raster haben unterschiedliche Shapes. Für Differenzen müssen beide Jahre "
            "auf identischem Grid liegen (Extent/Resolution/Transform)."
        )

    # retreat
    loss = compute_loss(mask_old, mask_new)

    # vulnerability (Option B)
    V, dist_m = vulnerability_exponential(
        loss_mask=loss,
        current_glacier_mask=mask_new,     # evaluate vulnerability outside current glacier
        transform=transform_new,
        max_buffer_m=MAX_BUFFER_M,
        lambda_m=LAMBDA_M,
        
    )

    # ---------------------------------
    # WRITE GEOTIFF OUTPUTS
    # ---------------------------------
    with rasterio.open(tif_new) as src_new:
        out_loss = OUT_DIR / f"loss_{YEAR_OLD}_{YEAR_NEW}.tif"
        write_singleband_geotiff(out_loss, loss.astype(np.uint8), src_new, dtype=np.uint8, nodata=0)
        print("Saved:", out_loss)

        out_vuln = OUT_DIR / f"vulnerability_0to1_exp_{YEAR_OLD}_{YEAR_NEW}.tif"
        write_singleband_geotiff(out_vuln, V.astype(np.float32), src_new, dtype=np.float32, nodata=None)
        print("Saved:", out_vuln)

        if WRITE_DISTANCE_TIF:
            out_dist = OUT_DIR / f"distance_to_loss_m_{YEAR_OLD}_{YEAR_NEW}.tif"
            write_singleband_geotiff(out_dist, dist_m.astype(np.float32), src_new, dtype=np.float32, nodata=None)
            print("Saved:", out_dist)

        # write masks (use their own profiles for correctness)
    with rasterio.open(tif_old) as src_old:
        out_mask_old = OUT_DIR / f"mask_{YEAR_OLD}.tif"
        write_singleband_geotiff(out_mask_old, mask_old.astype(np.uint8), src_old, dtype=np.uint8, nodata=0)
        print("Saved:", out_mask_old)

        if WRITE_PROB_TIF:
            out_prob_old = OUT_DIR / f"prob_{YEAR_OLD}.tif"
            prob_out = prob_old.copy()
            prob_out[np.isnan(prob_out)] = NODATA
            write_singleband_geotiff(out_prob_old, prob_out.astype(np.float32), src_old, dtype=np.float32, nodata=NODATA)
            print("Saved:", out_prob_old)

    with rasterio.open(tif_new) as src_new:
        out_mask_new = OUT_DIR / f"mask_{YEAR_NEW}.tif"
        write_singleband_geotiff(out_mask_new, mask_new.astype(np.uint8), src_new, dtype=np.uint8, nodata=0)
        print("Saved:", out_mask_new)

        if WRITE_PROB_TIF:
            out_prob_new = OUT_DIR / f"prob_{YEAR_NEW}.tif"
            prob_out = prob_new.copy()
            prob_out[np.isnan(prob_out)] = NODATA
            write_singleband_geotiff(out_prob_new, prob_out.astype(np.float32), src_new, dtype=np.float32, nodata=NODATA)
            print("Saved:", out_prob_new)

        # ---------------------------------
        # PREVIEW PNG (RGB 2025 + vulnerability overlay)
        # ---------------------------------
        rgb_img, step = read_rgb_quicklook(src_new, RGB_IDXS, DISPLAY_MAX_EDGE)

    V_disp = downsample(V, step)
    loss_disp = downsample(loss.astype(np.uint8), step)

    # For visualization: nodata -> 0 alpha
    V_vis = V_disp.copy()
    V_vis[V_vis == NODATA] = 0.0
    V_vis = np.clip(V_vis, 0.0, 1.0)

    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(rgb_img)
    ax1.set_title(f"RGB ({YEAR_NEW})")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(rgb_img)

    overlay = np.zeros((V_vis.shape[0], V_vis.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = VULN_COLOR[0]
    overlay[..., 1] = VULN_COLOR[1]
    overlay[..., 2] = VULN_COLOR[2]
    overlay[..., 3] = (V_vis.astype(np.float32) * VULN_ALPHA_MAX)
    ax2.imshow(overlay)

    # emphasize loss core (optional)
    overlay2 = np.zeros((loss_disp.shape[0], loss_disp.shape[1], 4), dtype=np.float32)
    overlay2[..., 0] = 1.0
    overlay2[..., 3] = loss_disp.astype(np.float32) * 0.9
    ax2.imshow(overlay2)

    ax2.set_title(
        f"Vulnerability (0–1, exponential)\n{YEAR_OLD} → {YEAR_NEW} | Dmax={MAX_BUFFER_M:.0f} m | λ={LAMBDA_M:.0f} m",
        color="darkred"
    )
    ax2.axis("off")

    plt.tight_layout()
    out_png = OUT_DIR / f"preview_vulnerability_exp_{YEAR_OLD}_{YEAR_NEW}.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Done.")


if __name__ == "__main__":
    main()

