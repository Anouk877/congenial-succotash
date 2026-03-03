import os
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from tqdm import tqdm


mosaic_path = "/Users/anoukwieczorek/Documents/Geographie/Projektseminar/Landsat/2016/Mosaics/LC08_L2SP_193027_2016XXXX_20200906_02_T1_SR_stack_mosaic.TIF"


output_root = "/Users/anoukwieczorek/Documents/Geographie/Projektseminar/Projekt"

year = "2016"
patch_folder_name = "Mosaik_patches"
output_folder_name = "2016_Patches_128"

patch_size = 128
stride = 128 

output_dir = os.path.join(output_root, output_folder_name)
os.makedirs(output_dir, exist_ok=True)


def create_patches(mosaic_path):

    print(f"\nÖffne Mosaik: {mosaic_path}")

    with rasterio.open(mosaic_path) as src:

        width, height = src.width, src.height
        meta = src.meta.copy()
        transform = src.transform

        print("  Größe:", width, "x", height)
        print("  Bänder:", src.count)

        patch_id = 0

        for top in tqdm(range(0, height - patch_size + 1, stride)):
            for left in range(0, width - patch_size + 1, stride):

                window = Window(left, top, patch_size, patch_size)
                patch = src.read(window=window)

                out_meta = meta.copy()
                out_meta.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": window_transform(window, transform)
                })

                # Dateiname mit 2016 am Anfang
                out_path = os.path.join(
                    output_dir,
                    f"{year}_{patch_folder_name}_{patch_id:06d}.tif"
                )

                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(patch)

                patch_id += 1

        print(f"FERTIG: {patch_id} Patches erzeugt und gespeichert unter:")
        print(f"       {output_dir}")


create_patches(mosaic_path)
