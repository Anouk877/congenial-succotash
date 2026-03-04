"""
Landsat8 Batch-Prozessor
- erstellt Jahresordner 2013-2025 mit Unterordnern "Layerstacks" und "Indices"
- berechnet NDSI, NDVI, NDWI aus Bändern 1-7 (pro Datum)
- erstellt Layerstacks (B1..B7 + NDSI, NDVI, NDWI)
- clippt alle Ausgaben exakt auf ein AOI-Shapefile
- speichert GeoTIFFs unter dem Jahresordner
Hinweise:
- Indices werden als float32 gespeichert (Werte ~ -1..1).
- Layerstacks werden als float32 (Bänder 1-7 werden zunächst in float gelesen).
- Originaldateien bleiben unangetastet.
"""

import os
import glob
import re
from collections import defaultdict
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
import geopandas as gpd
from rasterio.transform import Affine
import tempfile
import warnings

# ---------------------------
# 1) Benutzer-variablen (einfach anpassbar)
# ---------------------------
# Pfad mit den heruntergeladenen Landsat8 SR-Dateien (Originalordner)
SRC_FOLDER = r"C:\Users\Anwender\Documents\Studium\PSeminar\Daten\Landsat8\bulk_download\LC08"

# Ziel-Basisordner, unter dem Jahresordner 2013-2025 angelegt werden
DEST_BASE = r"C:\Users\Anwender\Documents\Studium\PSeminar\Daten\Landsat8\Daten"

# AOI Shapefile (soll im gleichen CRS wie Raster sein)
AOI_SHP = r"C:\Users\Anwender\Documents\Studium\PSeminar\Daten\AOI\Untersuchungsgebiet_32632.shp"

# Ordner für temporäre oder sonstige Zwischenschritte (falls benötigt)
OTHER_FOLDER = r"C:\Users\Anwender\Documents\Studium\PSeminar\Daten\Landsat8\Sonstiges"

# Jahre, für die Ordner erstellt werden
YEARS = list(range(2013, 2026))

# Dateinamensmuster: wir erwarten z.B.
# LC08_L2SP_193027_20130709_20200912_02_T1_SR_B1.TIF
# Datum ist 8-stellig (YYYYMMDD), Band als _B<number>.TIF am Ende
FNAME_DATE_RE = re.compile(r"LC08_.*?_(\d{8})_.*?_SR_B(\d+)\.TIF$", flags=re.IGNORECASE)

# ---------------------------
# 2) Hilfsfunktionen
# ---------------------------

def make_year_dirs(base, years):
    """Erstellt Jahresordner mit Unterordnern Layerstacks und Indices."""
    for y in years:
        year_dir = os.path.join(base, str(y))
        os.makedirs(os.path.join(year_dir, "Layerstacks"), exist_ok=True)
        os.makedirs(os.path.join(year_dir, "Indices"), exist_ok=True)

def scan_files_by_date(src_folder):
    """
    Scannt die SRC_FOLDER und gruppiert Dateien nach Datum (YYYYMMDD).
    Liefert dict: date_str -> dict(bandnum->filepath)
    """
    files = glob.glob(os.path.join(src_folder, "*.TIF")) + glob.glob(os.path.join(src_folder, "*.tif"))
    by_date = defaultdict(dict)
    for fp in files:
        fn = os.path.basename(fp)
        m = FNAME_DATE_RE.match(fn)
        if not m:
            # Datei passt nicht ins erwartete Muster -> warnen und überspringen
            warnings.warn(f"Dateiname passt nicht zum erwarteten Muster und wird übersprungen: {fn}")
            continue
        date_str = m.group(1)  # '20130709'
        band = int(m.group(2))  # z.B. 1
        by_date[date_str][band] = fp
    return by_date

def read_band_as_float(fp):
    """Öffnet ein Band und gibt Array (float32), Profile zurück."""
    src = rasterio.open(fp)
    arr = src.read(1).astype(np.float32)
    profile = src.profile.copy()
    src.close()
    return arr, profile

def write_raster(out_fp, arr, profile, dtype="float32", compress="LZW", nodata=None):
    """Schreibt ein Raster (arr kann 2D oder 3D) unter Beibehaltung von CRS/Transform (profile)."""
    p = profile.copy()
    # arr kann shape (bands, rows, cols) oder (rows, cols)
    if arr.ndim == 2:
        count = 1
        write_arr = np.expand_dims(arr, 0)
    else:
        count = arr.shape[0]
        write_arr = arr
    p.update({
        "dtype": dtype,
        "count": count,
        "compress": compress,
    })
    if nodata is not None:
        p.update({"nodata": nodata})
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    with rasterio.open(out_fp, 'w', **p) as dst:
        dst.write(write_arr.astype(dtype))
    return out_fp

def clip_array_to_aoi(arr, profile, aoi_shapes):
    """
    Clipt ein Raster-Array (arr: 2D oder 3D) exakt auf die AOI-Shapes.
    Gibt (clipped_array, clipped_profile) zurück.
    """
    # Wenn arr 3D (bands, rows, cols) -> benutzen wir rasterio's in-memory dataset:
    if arr.ndim == 3:
        bands, rows, cols = arr.shape
        mem_profile = profile.copy()
        mem_profile.update({"count": bands, "dtype": arr.dtype})
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**mem_profile) as dataset:
                dataset.write(arr)
                out_image, out_transform = mask(dataset, aoi_shapes, crop=True, filled=True, nodata=np.nan)
                out_profile = mem_profile.copy()
                out_profile.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform, "nodata": np.nan})
                return out_image, out_profile
    else:
        # arr 2D
        mem_profile = profile.copy()
        mem_profile.update({"count": 1, "dtype": arr.dtype})
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**mem_profile) as dataset:
                dataset.write(arr, 1)
                out_image, out_transform = mask(dataset, aoi_shapes, crop=True, filled = True, nodata=np.nan)
                out_profile = mem_profile.copy()
                out_profile.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform, "nodata": np.nan})
                # out_image hat shape (1, h, w)
                return out_image[0], out_profile

# ---------------------------
# 3) Index-Formeln (Dokumentation im Code)
# ---------------------------
# Landsat 8 Bandzuordnung (OLI):
# B1 = Coastal
# B2 = Blue
# B3 = Green
# B4 = Red
# B5 = NIR
# B6 = SWIR1
# B7 = SWIR2
#
# Indices (mit Bandnummern):
# NDSI  = (Green - SWIR1) / (Green + SWIR1)  --> (B3 - B6) / (B3 + B6)
# NDVI  = (NIR - Red) / (NIR + Red)          --> (B5 - B4) / (B5 + B4)
# NDWI  = (Green - NIR) / (Green + NIR)      --> (B3 - B5) / (B3 + B5)
#
# Wir berechnen die Indices mit float32; wo Nenner 0 ist, setzen wir den Index auf np.nan.

def compute_indices(bands_dict):
    """
    Erwartet bands_dict: bandnummer->2D-array (float32)
    Liefert tuple (ndsi, ndvi, ndwi) als 2D float32-Arrays.
    """
    # Bänder extrahieren (als float32)
    B1 = bands_dict[1]
    B2 = bands_dict[2]
    B3 = bands_dict[3]  # Green
    B4 = bands_dict[4]  # Red
    B5 = bands_dict[5]  # NIR
    B6 = bands_dict[6]  # SWIR1
    B7 = bands_dict[7]

    # NDSI = (B3 - B6) / (B3 + B6)
    num = (B3 - B6)
    den = (B3 + B6)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndsi = num / den
        ndsi[den == 0] = np.nan

    # NDVI = (B5 - B4) / (B5 + B4)
    num = (B5 - B4)
    den = (B5 + B4)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = num / den
        ndvi[den == 0] = np.nan

    # NDWI = (B3 - B5) / (B3 + B5)
    num = (B3 - B5)
    den = (B3 + B5)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = num / den
        ndwi[den == 0] = np.nan

    return ndsi.astype(np.float32), ndvi.astype(np.float32), ndwi.astype(np.float32)

# ---------------------------
# 4) Hauptverarbeitung
# ---------------------------

def main():
    # 4.1 Erzeuge benötigte Jahresordner
    make_year_dirs(DEST_BASE, YEARS)
    os.makedirs(OTHER_FOLDER, exist_ok=True)

    # 4.2 Lade AOI-Shapefile (als GeoJSON-ähnliche Geometrien)
    aoi = gpd.read_file(AOI_SHP)
    aoi = aoi.to_crs(aoi.crs)  # sicherstellen
    aoi_shapes = [feature["geometry"] for feature in aoi.__geo_interface__['features']]

    # 4.3 Scanne Quelldateien und gruppiere nach Datum
    by_date = scan_files_by_date(SRC_FOLDER)
    print(f"Gefundene Datumsgruppen: {len(by_date)}")

    # 4.4 Pro Datum verarbeiten
    for date_str, bands_map in sorted(by_date.items()):
        # parse Jahr für Ausgabeordner
        year = int(date_str[:4])
        if year not in YEARS:
            print(f"Datum {date_str} (Jahr {year}) außerhalb der spezifizierten Jahre. Übersprungen.")
            continue

        # prüfen, ob alle Bänder 1..7 vorhanden sind
        missing = [b for b in range(1,8) if b not in bands_map]
        if missing:
            warnings.warn(f"Für Datum {date_str} fehlen Bänder: {missing} — diese Gruppe wird übersprungen.")
            continue

        print(f"\nVerarbeite Datum {date_str} (Jahr {year})")

        # 4.4.1 Namen-Template entnehmen (wir nehmen den Basisnamen einer Banddatei als Vorlage)
        example_fp = bands_map[1]
        example_fn = os.path.basename(example_fp)
        base_prefix = re.sub(r"_B\d+\.TIF$|_B\d+\.tif$", "", example_fn, flags=re.IGNORECASE)
        # Beispiel: LC08_L2SP_193027_20130709_20200912_02_T1_SR

        # 4.4.2 Bänder einlesen (float32)
        bands = {}
        profile = None
        for bnum in range(1,8):
            fp = bands_map[bnum]
            arr, prof = read_band_as_float(fp)
            bands[bnum] = arr
            if profile is None:
                profile = prof  # Profil eines Beispielbands als Vorlage
        # jetzt haben wir bands[1..7] als float32 arrays und profile

        # 4.4.3 Indices berechnen
        ndsi, ndvi, ndwi = compute_indices(bands)

        # 4.4.4 Clippen (exakt zuschneiden) auf AOI
        # Wir clippen Indices und Layerstack. Für Indices:
        # Verwende profile aus einem Band als Grundlage. Mask gibt zurück (arr, new_transform) mit arr shape (bands, h, w)
        # Indices sind 2D -> clip_array_to_aoi erwartet 2D oder 3D.
        ndsi_clipped, ndsi_prof = clip_array_to_aoi(ndsi, profile, aoi_shapes)
        ndvi_clipped, ndvi_prof = clip_array_to_aoi(ndvi, profile, aoi_shapes)
        ndwi_clipped, ndwi_prof = clip_array_to_aoi(ndwi, profile, aoi_shapes)

        # 4.4.5 Indices speichern (Dateinamen: ursprünglicher Name mit _<INDEX>.TIF)
        idx_out_dir = os.path.join(DEST_BASE, str(year), "Indices")
        # NDSI
        ndsi_name = f"{base_prefix}_NDSI.TIF"
        ndsi_out_fp = os.path.join(idx_out_dir, ndsi_name)
        ndsi_prof.update({"dtype": "float32", "count": 1})
        write_raster(ndsi_out_fp, ndsi_clipped, ndsi_prof, dtype="float32", nodata=np.nan)
        print(f"  NDSI gespeichert: {ndsi_out_fp}")

        # NDVI
        ndvi_name = f"{base_prefix}_NDVI.TIF"
        ndvi_out_fp = os.path.join(idx_out_dir, ndvi_name)
        ndvi_prof.update({"dtype": "float32", "count": 1})
        write_raster(ndvi_out_fp, ndvi_clipped, ndvi_prof, dtype="float32", nodata=np.nan)
        print(f"  NDVI gespeichert: {ndvi_out_fp}")

        # NDWI
        ndwi_name = f"{base_prefix}_NDWI.TIF"
        ndwi_out_fp = os.path.join(idx_out_dir, ndwi_name)
        ndwi_prof.update({"dtype": "float32", "count": 1})
        write_raster(ndwi_out_fp, ndwi_clipped, ndwi_prof, dtype="float32", nodata=np.nan)
        print(f"  NDWI gespeichert: {ndwi_out_fp}")

        # 4.4.6 Layerstack erstellen: B1..B7 (in dieser Reihenfolge) + NDSI, NDVI, NDWI
        # Wir clippen die Bänder alle einzeln und dann stacken wir die bereits geclippten Arrays.
        clipped_bands = []
        for bnum in range(1,8):
            arr = bands[bnum]
            arr_clipped, arr_prof = clip_array_to_aoi(arr, profile, aoi_shapes)
            # arr_clipped hat shape (h, w) - wir behalten sie
            clipped_bands.append(arr_clipped.astype(np.float32))  # convert to float32 for stack

        # füge Indices in Reihenfolge: NDSI, NDVI, NDWI als Bänder 8,9,10
        clipped_bands.append(ndsi_clipped.astype(np.float32))
        clipped_bands.append(ndvi_clipped.astype(np.float32))
        clipped_bands.append(ndwi_clipped.astype(np.float32))

        # Staple (bands, h, w)
        stack_arr = np.stack(clipped_bands, axis=0).astype(np.float32)

        # Profil für das stack (wir verwenden arr_prof von einem vorherigen Clip, z.B. arr_prof)
        stack_prof = arr_prof.copy()
        stack_prof.update({"count": stack_arr.shape[0], "dtype": "float32"})

        # Dateiname für Stack: ursprünglicher Name mit _stack.TIF
        stack_name = f"{base_prefix}_stack.TIF"
        stack_out_fp = os.path.join(DEST_BASE, str(year), "Layerstacks", stack_name)
        write_raster(stack_out_fp, stack_arr, stack_prof, dtype="float32", nodata=np.nan)
        print(f"  Layerstack gespeichert: {stack_out_fp}")

    print("\nFertig mit allen Datumsgruppen.")

if __name__ == "__main__":
    main()
