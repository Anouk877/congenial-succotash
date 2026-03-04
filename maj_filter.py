import numpy as np
import rasterio
from scipy.ndimage import generic_filter

for jahr in range(2013, 2026):

    # ============================================================
    # USER INPUT
    # ============================================================

    input_raster = f"C:/Users/Anwender/Documents/Studium/PSeminar/Daten/Unet_Class/all_clean_08_results/years/mask_{jahr}.tif"
    output_raster = f"C:/Users/Anwender/Documents/Studium/PSeminar/Daten/Unet_Class/all_clean_08_results/years_m5/mask_{jahr}_m5.tif"

    # Fenstergröße (ungerade Zahl: 3, 5, 7, ...)
    window_size = 5

    # ============================================================
    # MODAL FILTER FUNCTION
    # ============================================================

    def modal_filter(values, nodata):
        """
        Ersetzt den Mittelpunkt eines Moving Windows durch den Modalwert.
        Falls NODATA die Mehrheit bildet, wird NODATA zurückgegeben.
        """
        values = np.array(values)

        # Anzahl der Elemente im Fenster
        n_total = values.size

        # NODATA-Maske
        nodata_mask = values == nodata
        n_nodata = np.sum(nodata_mask)

        # Falls NODATA dominiert → NODATA zurückgeben
        if n_nodata > n_total / 2:
            return nodata

        # Werte ohne NODATA
        valid_values = values[~nodata_mask]

        if valid_values.size == 0:
            return nodata

        # Modalwert bestimmen
        unique, counts = np.unique(valid_values, return_counts=True)
        return unique[np.argmax(counts)]

    # ============================================================
    # APPLY FILTER
    # ============================================================

    with rasterio.open(input_raster) as src:
        data = src.read(1)
        profile = src.profile
        nodata = src.nodata

        if nodata is None:
            raise ValueError("Das Eingaberaster besitzt keinen definierten NODATA-Wert.")

        filtered = generic_filter(
            data,
            function=modal_filter,
            size=window_size,
            mode="constant",
            cval=nodata,
            extra_arguments=(nodata,)
        )

    # ============================================================
    # WRITE OUTPUT
    # ============================================================

    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(filtered, 1)
