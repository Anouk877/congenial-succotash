import os
import numpy as np
import rasterio
from rasterio.windows import Window

mask_path = "gletscher_mask.tif"


output_dir_tif = "mask_patches_128_tif"
output_dir_npy = "mask_patches_128_npy"

patch_size = 128
stride = 128

os.makedirs(output_dir_tif, exist_ok=True)
os.makedirs(output_dir_npy, exist_ok=True)

with rasterio.open(mask_path) as src:
    height = src.height
    width = src.width
    transform = src.transform

    mask_preview = src.read(1, window=Window(0, 0, min(width, 512), min(height, 512)))
    print("Maske geladen:")
    print(f" - Größe: {height} x {width}")
    print(f" - CRS: {src.crs}")
    print(f" - dtype: {src.dtypes[0]}")
    print(f" - Beispielwerte (Preview): {np.unique(mask_preview)}")

    patch_id = 0
    id_width = 6  # 000000 – 999999

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):

            window = Window(x, y, patch_size, patch_size)
            patch = src.read(1, window=window)

            # Robust binarize (keep 0/1). Remove if you want original values.
            patch = (patch > 0).astype(np.uint8)

            pid = f"{patch_id:0{id_width}d}"

            # ---------- write NPY ----------
            npy_path = os.path.join(output_dir_npy, f"mask_patch_{pid}.npy")
            np.save(npy_path, patch)

            # ---------- write GeoTIFF ----------
            tif_path = os.path.join(output_dir_tif, f"mask_patch_{pid}.tif")

            out_meta = src.meta.copy()
            out_meta.update(
                driver="GTiff",
                height=patch_size,
                width=patch_size,
                count=1,
                dtype=rasterio.uint8,
                nodata=0,
                compress="lzw",
                transform=rasterio.windows.transform(window, transform),
            )

            with rasterio.open(tif_path, "w", **out_meta) as dst:
                dst.write(patch, 1)

            patch_id += 1

print(f"[FERTIG] {patch_id} Masken-Patches erzeugt:")
print(f" - GeoTIFF: {output_dir_tif}")
print(f" - NPY:     {output_dir_npy}")
