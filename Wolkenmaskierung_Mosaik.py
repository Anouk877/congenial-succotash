"""
Landsat Layerstack Maskierung + Jahres-Mosaike
Benötigte Libraries: numpy, rasterio (und Standardlib: os, re, glob, pathlib)
Python-Version: 3.8+

Hinweis:
- Alle Pfade und Einstellungen sind ganz oben als Variablen definiert.
- MASK_BITS kann angepasst werden (0-basierte Bit-Nummerierung).
- Skript verwendet Eingangs-nodata, fallback = -9999 wenn nicht gesetzt.
"""

import os
import re
import glob
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

# ======================
# ===  Benutzereinstellungen (einfach anpassbar) ===
# ======================
# Quellordner mit jährlichen Unterordnern und je "Layerstacks"
BASE_IN = Path(r"C:\Users\Anwender\Documents\Studium\PSeminar\Daten\Landsat8\Landsat_Daten_clean")
# Pfad zu den QA_PIXEL Dateien (bulk download)
QA_DIR = Path(r"C:\Users\Anwender\Documents\Studium\PSeminar\Daten\Landsat8\bulk_download\LC08")

# Zielordner für maskierte Layerstacks und Mosaike
BASE_OUT = Path(r"C:\Users\Anwender\Documents\Studium\PSeminar\Daten\Landsat8\Landsat_Daten_Maske_clean")

# Jahre, die bearbeitet werden sollen
YEARS = list(range(2013, 2026))  # inkl. 2025

# Bit-Nummern, die geprüft werden sollen (0-basierte Bit-Indexierung)
MASK_BITS = [0, 2, 3]

# Schwelle für Band 8 (bereits in 0..1 Skala, wie bestätigt)
BAND8_THRESHOLD = 0.3

# Namesteile: Layerstacks enden auf "_SR_stack.TIF"
STACK_SUFFIX = "_SR_stack.TIF"
QA_SUFFIX = "_QA_PIXEL.TIF"

# Fallback nodata falls die Eingabedatei kein nodata hat
NODATA_FALLBACK = -9999

# ======================
# ===  Hilfsfunktionen  ===
# ======================

def find_layerstacks_for_year(base_in: Path, year: int):
    """
    Findet alle Layerstack-Dateien im Ordner <base_in>/<year>/Layerstacks
    Liefert Liste von Pfaden (Path-Objekte).
    """
    folder = base_in / str(year) / "Layerstacks"
    if not folder.exists():
        return []
    # Suchen nach *.TIF, die auf _SR_stack.TIF enden
    pattern = str(folder / f"*{STACK_SUFFIX}")
    return [Path(p) for p in glob.glob(pattern)]

def find_matching_qa(qa_dir: Path, stack_name: str):
    """
    Sucht in qa_dir nach der passenden QA_PIXEL Datei, anhand des gemeinsamen Präfixes.
    Erwartet stack_name als Dateiname (z.B. 'LC08_L2SP_193027_20130709_20200912_02_T1_SR_stack.TIF')
    und gibt Path oder None zurück.
    Die QA-Datei hat das gleiche Präfix, endet aber auf _QA_PIXEL.TIF.
    """
    # Ersetze das _SR_stack.TIF Ende durch _QA_PIXEL.TIF
    candidate_name = stack_name.replace(STACK_SUFFIX, QA_SUFFIX)
    # Suchen nach exakt diesem Namen oder nach Dateien, die diesen Namen enthalten (sicherheitshalber)
    exact = qa_dir / candidate_name
    if exact.exists():
        return exact
    # Falls nicht exakt gefunden (z.B. Groß-/Kleinschreibung oder leicht anders), suche nach prefix
    # Bestimme prefix bis vor dem Datumsteil (oder bis vor _SR_stack)
    prefix = stack_name.replace(STACK_SUFFIX, "")
    # Suche alle QA_PIXEL Dateien und vergleiche start mit prefix (oder umgekehrt)
    pattern = str(qa_dir / f"*{QA_SUFFIX}")
    for p in glob.glob(pattern):
        pname = os.path.basename(p)
        if pname.startswith(prefix) or prefix.startswith(pname.replace(QA_SUFFIX, "")):
            return Path(p)
    # Als Fallback: versuche nur die Szene-ID-Teile (z.B. LC08_L2SP_193027_)
    # extrahiere 1. drei token bis Szene-Kennung
    scene_prefix = "_".join(stack_name.split("_")[:3]) + "_"  # z.B. "LC08_L2SP_193027_"
    for p in glob.glob(pattern):
        if os.path.basename(p).startswith(scene_prefix):
            return Path(p)
    return None

def window_from_bounds_intersection(src_ds, ref_ds):
    """
    Erzeugt ein Fenster in ref_ds, das genau den Bereich von src_ds abdeckt (sofern CRS gleich).
    Falls CRS unterschiedlich ist, wirft die Funktion eine Exception.
    """
    if src_ds.crs != ref_ds.crs:
        raise ValueError("CRS der Layerstack und der QA_PIXEL Datei stimmen nicht überein. "
                         "Reprojektion wird hier nicht durchgeführt.")
    bounds = src_ds.bounds
    # Erzeuge Fenster im referenz-dataset (z.B. QA) für die Bounds des src (layerstack)
    win = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top,
                      transform=ref_ds.transform)
    # Convert to integer window window - rasterio liest mit integer windows
    win = win.round_offsets().round_lengths()
    return win

def read_array_window(ds, window, out_shape=None, indexes=None):
    """
    Lies ein Fenster aus ds und liefere numpy-array.
    Wenn indexes=None -> alle Bänder.
    out_shape kann verwendet werden, falls Resampling benötigt wird (nicht standardmäßig verwendet).
    """
    if indexes is None:
        # alle Bänder
        return ds.read(window=window)
    else:
        return ds.read(indexes, window=window)

def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# ======================
# ===  Maskierung und Speichern  ===
# ======================

def mask_and_save_layerstack(stack_path: Path, qa_path: Path, out_path: Path):
    """
    Führt die Maskierung für eine einzelne Layerstack-Datei durch und speichert das Ergebnis als *_mask.TIF.
    Returns: Path zur gespeicherten Datei
    """
    with rasterio.open(stack_path) as src_stack:
        # Metadaten kopieren
        meta = src_stack.meta.copy()
        dtype = meta.get("dtype")
        src_nodata = src_stack.nodata
        if src_nodata is None:
            nodata = NODATA_FALLBACK
            meta.update({"nodata": nodata})
        else:
            nodata = src_nodata

        # Lese gesamten Layerstack (alle Bänder) — die Arrays haben Form (bands, h, w)
        stack_arr = src_stack.read()  # shape: (bands, h, w)
        bands, h, w = stack_arr.shape

        # Öffne QA und lese nur das Fenster, das Layerstack-Bereich überlappt
        with rasterio.open(qa_path) as src_qa:
            # Erzeuge Fenster im QA, welches den Layerstack abdeckt (nur Überschneidungsbereich)
            win = window_from_bounds_intersection(src_stack, src_qa)

            # Lese QA in Fenster
            qa_win = src_qa.read(1, window=win)  # 2D array

            # Lese ggf. die transform für das Fenster (um Positionen zu kennen)
            qa_transform_win = src_qa.window_transform(win)

        # Prüfe Formen: qa_win sollte h x w entsprechen
        if qa_win.shape != (h, w):
            # Falls die Formen nicht übereinstimmen, versuchen wir eine Resampling-Lösung:
            # (dieser Fall sollte selten auftreten, aber wir behandeln ihn robust)
            # Resample QA auf die Größe des Layerstacks (nearest)
            # Dafür öffnen wir QA erneut und read mit out_shape
            with rasterio.open(qa_path) as src_qa:
                qa_win = src_qa.read(1, window=win, out_shape=(h, w), resampling=Resampling.nearest)

        # Berechne Bit-Maske: TRUE, wenn eines der MASK_BITS gesetzt ist
        qa_uint = qa_win.astype(np.uint32)
        bit_masks = np.zeros_like(qa_uint, dtype=bool)
        for bit in MASK_BITS:
            bit_masks |= ((qa_uint >> bit) & 1).astype(bool)

        # Hole Band 8 (Index 8 -> arr index 7)
        band8 = stack_arr[7].astype(np.float32)  # float für Vergleich

        # Erstelle final_mask: True, wenn (bit gesetzt) UND (band8 < Schwelle)
        final_mask = (bit_masks) & (band8 < BAND8_THRESHOLD)

        # Wandle final_mask in Maske für alle Bänder -> setze auf nodata dort
        # Achtung: final_mask shape (h,w). Wir möchten die Pixel in allen Bändern auf nodata setzen.
        # Wende Maskierung an
        for b in range(bands):
            band = stack_arr[b]
            # Wenn nodata vorhanden: setze auf nodata dort
            band[final_mask] = nodata
            stack_arr[b] = band

        # Wenn Eingangs-Datei bereits nodata an anderen Stellen hatte (z.B. 0 oder nodata),
        # bleiben diese erhalten. Wir setzen nur die neu maskierten Pixel auf nodata.

        # Zum Speichern: sicherstellen, dass meta Count korrekt ist
        meta.update({"count": bands, "dtype": stack_arr.dtype})

        # Schreibe Ergebnis
        ensure_dir(out_path.parent)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(stack_arr)

    return out_path

# ======================
# ===  Jahres-Mosaik erstellen  ===
# ======================

def make_year_mosaic(masked_stack_paths, out_mosaic_path):
    """
    Aus einer Liste von maskierten Layerstack-Pfaden ein Mosaik erzeugen.
    Regeln:
      - Für jeden Pixel: nimm den Layerstack (alle Bänder), bei dem Band8 den geringsten (numerischen) Wert hat.
      - NoData wird beachtet: falls alle Layerstacks für einen Pixel NoData haben -> NoData im Mosaik.
    Rückgabe: Path zu gespeichertem Mosaik.
    """
    if not masked_stack_paths:
        return None

    # Öffne ersten Stack, um Form, Metadaten zu bekommen
    with rasterio.open(masked_stack_paths[0]) as ref:
        meta = ref.meta.copy()
        bands = ref.count
        h = ref.height
        w = ref.width
        dtype = ref.dtypes[0]
        nodata = ref.nodata
        if nodata is None:
            nodata = NODATA_FALLBACK
            meta.update({"nodata": nodata})

    n_files = len(masked_stack_paths)

    # Arrays vorbereiten: band8_stack (n_files, h, w), stacks für alle Bänder ev. geladen on demand
    band8_stack = np.full((n_files, h, w), np.nan, dtype=np.float32)  # nan für NoData
    stacks = []  # liste mit arrays (bands, h, w) pro file

    for i, p in enumerate(masked_stack_paths):
        with rasterio.open(p) as ds:
            arr = ds.read()  # (bands, h, w)
            stacks.append(arr)
            b8 = arr[7].astype(np.float32)
            # Ersetze nodata-Werte in b8 durch np.nan für Vergleich
            if ds.nodata is not None:
                mask_nodata = (b8 == ds.nodata)
            else:
                mask_nodata = np.zeros_like(b8, dtype=bool)
            b8[mask_nodata] = np.nan
            band8_stack[i] = b8

    # Falls alle Werte nan an einer Position -> Mosaik wird nodata dort.
    # Sonst: finde Index des min (nanignored)
    # np.nanargmin würde erratenes Verhalten bei Spalten nur-nan; wir guardieren:
    any_valid = ~np.isnan(band8_stack)
    any_valid_any = np.any(any_valid, axis=0)  # (h,w)

    # Für stabile argmin: setze nan auf large value; dann argmin liefert index mit minimalem validen Wert.
    large = 1e6
    band8_for_argmin = np.where(np.isnan(band8_stack), large, band8_stack)
    argmin_idx = np.argmin(band8_for_argmin, axis=0)  # (h,w)

    # Erzeuge leeres Mosaik-Array und fülle mit nodata
    mosaic = np.full((bands, h, w), nodata, dtype=stacks[0].dtype)

    # Für jedes Datei-Index, übertrage Pixel, die von argmin gewählt sind (und gültig sind)
    for i in range(n_files):
        sel = (argmin_idx == i) & any_valid_any  # True wo index i gewählt und mindestens ein gültiger
        if not np.any(sel):
            continue
        # Über alle Bänder
        arr = stacks[i]
        for b in range(bands):
            band = mosaic[b]
            band[sel] = arr[b][sel]
            mosaic[b] = band

    # Schreibe Mosaik-Datei
    ensure_dir(out_mosaic_path.parent)
    meta.update({"count": bands, "dtype": mosaic.dtype})
    with rasterio.open(out_mosaic_path, "w", **meta) as dst:
        dst.write(mosaic)

    return out_mosaic_path

# ======================
# ===  Index-Export (B8-B10 -> NDSI, NDVI, NDWI)  ===
# ======================

def export_indices_from_mosaic(mosaic_path: Path, out_dir: Path):
    """
    Exportiert Bänder 8..10 aus dem Mosaik als separate Einband-Dateien.
    Benennung: ersetze "_stack_mosaic.TIF" durch "_NDSI_mosaic.TIF", "_NDVI_mosaic.TIF", "_NDWI_mosaic.TIF".
    Band 8 = NDSI, 9 = NDVI, 10 = NDWI (gemäß Vorgabe).
    Zusätzlich: keine Veränderung am Mosaik selbst (dort bleiben alle Bänder erhalten).
    """
    with rasterio.open(mosaic_path) as src:
        meta = src.meta.copy()
        bands = src.count
        if bands < 10:
            raise ValueError("Mosaik hat weniger als 10 Bänder, kann Indices nicht exportieren.")
        # NDSI = Band 8, NDVI = Band 9, NDWI = Band 10
        mapping = {
            "NDSI": 8,
            "NDVI": 9,
            "NDWI": 10
        }
        for name, bidx in mapping.items():
            arr = src.read(bidx)  # 2D
            outname = mosaic_path.name
            # Ersetze _stack_mosaic.TIF -> _<INDEX>_mosaic.TIF
            outname = outname.replace("_stack_mosaic.TIF", f"_{name}_mosaic.TIF")
            outpath = out_dir / outname
            out_meta = meta.copy()
            out_meta.update({"count": 1, "dtype": arr.dtype})
            ensure_dir(outpath.parent)
            with rasterio.open(outpath, "w", **out_meta) as dst:
                dst.write(arr, 1)

# ======================
# ===  Main Processing Loop  ===
# ======================

def process_all():
    for year in YEARS:
        print(f"\n=== Bearbeite Jahr {year} ===")
        # Input & Output Ordner
        in_layer_dir = BASE_IN / str(year) / "Layerstacks"
        out_year_dir = BASE_OUT / str(year)
        out_layer_dir = out_year_dir / "Layerstacks"
        out_mosaic_dir = out_year_dir / "Mosaics"
        ensure_dir(out_layer_dir)
        ensure_dir(out_mosaic_dir)

        # Finde alle Layerstack-Dateien für dieses Jahr
        stacks = find_layerstacks_for_year(BASE_IN, year)
        print(f"Gefundene Layerstacks: {len(stacks)}")

        masked_paths = []

        for stack_path in stacks:
            print(f"-> Verarbeite: {stack_path.name}")
            qa_path = find_matching_qa(QA_DIR, stack_path.name)
            if qa_path is None:
                print(f"   WARNUNG: Keine passende QA_PIXEL Datei für {stack_path.name} gefunden. Übersprungen.")
                continue
            try:
                # Name für die maskierte Datei: füge _mask vor .TIF ein
                out_name = stack_path.name.replace(".TIF", "_mask.TIF")
                out_path = out_layer_dir / out_name

                # Nicht überschreiben, falls schon vorhanden
                if out_path.exists():
                    print(f"   Maskierte Datei existiert bereits -> wird verwendet: {out_path.name}")
                    masked_paths.append(out_path)
                    continue

                saved = mask_and_save_layerstack(stack_path, qa_path, out_path)
                print(f"   Gespeichert: {saved.name}")
                masked_paths.append(saved)
            except Exception as e:
                print(f"   FEHLER bei Verarbeitung {stack_path.name}: {e}")
                continue

        # Falls keine maskierten Stacks existieren: weiter zum nächsten Jahr
        if not masked_paths:
            print("   Keine maskierten Layerstacks für dieses Jahr vorhanden; Mosaikerstellung übersprungen.")
            continue

        # Erzeuge Mosaik für das Jahr
        # Bestimme Namen für Mosaik: wir nehmen den ersten Layerstacknamen und ersetzen Datum (MMDD) mit XXXX
        first_name = masked_paths[0].name  # example: LC08_..._20130709_..._SR_stack_mask.TIF
        # Entferne _mask.TIF temporär, ersetze Datum und hänge _stack_mosaic.TIF an
        base_no_mask = first_name.replace("_mask.TIF", "")
        # Suche Muster _YYYYMMDD_ und ersetze zu _YYYYXXXX_
        def replace_date_with_xxx(match):
            yyyy = match.group(1)
            return f"_{yyyy}XXXX_"
        new_base = re.sub(r'_(\d{4})(\d{2})(\d{2})_', replace_date_with_xxx, base_no_mask, count=1)
        mosaic_name = new_base + "_mosaic.TIF"
        mosaic_path = out_mosaic_dir / mosaic_name

        if mosaic_path.exists():
            print(f"   Mosaik existiert bereits: {mosaic_path.name}")
        else:
            print(f"   Erzeuge Mosaik: {mosaic_path.name}")
            try:
                saved_mosaic = make_year_mosaic(masked_paths, mosaic_path)
                if saved_mosaic:
                    print(f"   Mosaik gespeichert: {saved_mosaic.name}")
                else:
                    print("   Mosaik konnte nicht erstellt werden (keine Eingabedateien).")
            except Exception as e:
                print(f"   FEHLER beim Erstellen des Mosaiks: {e}")
                continue

        # Exportiere Indices B8-B10 aus dem Mosaik
        try:
            print("   Exportiere NDSI/NDVI/NDWI aus dem Mosaik...")
            export_indices_from_mosaic(mosaic_path, out_mosaic_dir)
            print("   Indices exportiert.")
        except Exception as e:
            print(f"   WARNUNG: Fehler beim Export der Indices: {e}")

    print("\n=== Verarbeitung abgeschlossen ===")

# ======================
# ===  Script Entrypoint  ===
# ======================
if __name__ == "__main__":
    process_all()



