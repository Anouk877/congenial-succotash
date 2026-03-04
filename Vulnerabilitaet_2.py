import numpy as np
import rasterio
from pathlib import Path
from scipy.ndimage import distance_transform_edt


SCRIPT_DIR = Path(__file__).resolve().parent


INPUT_TIF = Path("/Users/anoukwieczorek/Documents/Geographie/Projektseminar/Projekt/Unet_m5_change/mask_m5_change_1317-2125.tif")

OUT_DIR = SCRIPT_DIR / "outputs_vulnerability_change"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Raster classes
RETREAT_VALUE = 2  
OTHER_VALUE = 1     

# Vulnerability parameters
MAX_BUFFER_M = 1500.0   
LAMBDA_M = 500.0        #

WRITE_DISTANCE_TIF = True

NODATA_FLOAT = None     # keep None unless you want explicit nodata like -9999
COMPRESS = "deflate"

def write_singleband_geotiff(out_path: Path, arr2d: np.ndarray, ref_src: rasterio.DatasetReader,
                             dtype=None, nodata=None, compress="deflate"):
    
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


def vulnerability_exponential_from_value1(src_arr: np.ndarray,
                                         transform,
                                         retreat_value: int,
                                         max_buffer_m: float,
                                         lambda_m: float):
   
    retreat_mask = (src_arr == retreat_value)

    px_w = float(transform.a)
    px_h = float(transform.e)
    sampling = (abs(px_h), abs(px_w))

    
    inv = (~retreat_mask).astype(np.uint8)
    dist_m = distance_transform_edt(inv, sampling=sampling).astype(np.float32)

    V = np.zeros(src_arr.shape, dtype=np.float32)

    in_buffer = dist_m <= float(max_buffer_m)

    V[in_buffer] = np.exp(-dist_m[in_buffer] / float(lambda_m)).astype(np.float32)
    V = np.clip(V, 0.0, 1.0)

    V[retreat_mask] = 1.0


    return V, dist_m, retreat_mask

def main():
    if not INPUT_TIF.exists():
        raise FileNotFoundError(f"Input nicht gefunden: {INPUT_TIF}")

    with rasterio.open(INPUT_TIF) as src:
        arr = src.read(1)
        transform = src.transform

        # Basic sanity check of expected values
        unique_vals = np.unique(arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr)
        print("Unique values (sample):", unique_vals[:20])

        if RETREAT_VALUE not in unique_vals:
            raise ValueError(f"Wert {RETREAT_VALUE} kommt im Raster nicht vor. Gefunden: {unique_vals}")

        if src.crs is not None and src.crs.is_geographic:
            print(
                "WARNUNG: CRS ist geografisch (Grad). Distanz/Buffer in Metern ist so nicht korrekt. "
                "Bitte reprojiziere in ein metrisches CRS (z.B. UTM), sonst sind MAX_BUFFER_M/LAMBDA_M nicht in Metern."
            )

        V, dist_m, retreat_mask = vulnerability_exponential_from_value1(
            src_arr=arr,
            transform=transform,
            retreat_value=RETREAT_VALUE,
            max_buffer_m=MAX_BUFFER_M,
            lambda_m=LAMBDA_M
        )

        out_core = OUT_DIR / f"core_value{RETREAT_VALUE}.tif"
        write_singleband_geotiff(out_core, retreat_mask.astype(np.uint8), src, dtype=np.uint8, nodata=0, compress=COMPRESS)
        print("Saved:", out_core)

        out_vuln = OUT_DIR / f"vulnerability_0to1_exp_value{RETREAT_VALUE}_D{int(MAX_BUFFER_M)}m_L{int(LAMBDA_M)}m.tif"
        write_singleband_geotiff(out_vuln, V.astype(np.float32), src, dtype=np.float32, nodata=NODATA_FLOAT, compress=COMPRESS)
        print("Saved:", out_vuln)

        if WRITE_DISTANCE_TIF:
            out_dist = OUT_DIR / f"distance_to_value{RETREAT_VALUE}_m.tif"
            write_singleband_geotiff(out_dist, dist_m.astype(np.float32), src, dtype=np.float32, nodata=NODATA_FLOAT, compress=COMPRESS)
            print("Saved:", out_dist)

    print("Done.")


if __name__ == "__main__":
    main()
