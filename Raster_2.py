import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np


shapefile = "/Users/anoukwieczorek/Documents/Geographie/Projektseminar/Projekt/Gletscher_reproj"
reference_raster = "/Users/anoukwieczorek/Documents/Geographie/Projektseminar/Landsat/2017/Mosaics/LC08_L2SP_193027_2017XXXX_20200903_02_T1_SR_stack_mosaic.TIF"
output_mask = "gletscher_mask.tif"


with rasterio.open(reference_raster) as src:
    meta = src.meta.copy()
    raster_crs = src.crs
    transform = src.transform
    out_shape = (src.height, src.width)

print("[INFO] Raster CRS:", raster_crs)


gdf = gpd.read_file(shapefile)
print("[INFO] Shapefile CRS vor Reprojektion:", gdf.crs)

gdf = gdf.to_crs(raster_crs)
print("[INFO] Shapefile CRS nach Reprojektion:", gdf.crs)


shapes = [(geom, 1) for geom in gdf.geometry]


print("[INFO] Rasterisiere Maske...")
mask_array = rasterize(
    shapes=shapes,
    out_shape=out_shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)


meta.update({
    "count": 1,
    "dtype": "uint8",
    "nodata": 0
})

with rasterio.open(output_mask, "w", **meta) as dst:
    dst.write(mask_array, 1)

print("[FERTIG] Maske gespeichert als:", output_mask)
