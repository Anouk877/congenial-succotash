import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
import rasterio
from tqdm import tqdm
import pandas as pd

MODEL_FILE = "gletscher_unet_final.pth"
# Hier liegen Ordner "2013", "2017", "2020" etc.
BASE_DIR = "/Users/nicolasyukio/Downloads/Landsat_Zeitserien"
OUTPUT_BASE = "/Users/nicolasyukio/Downloads/Analyse_Ergebnisse"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def analyse_zeitreihe():
    model = smp.Unet(encoder_name="resnet34", in_channels=10, classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    jahre = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
    ergebnisse = []

    for jahr in jahre:
        jahr_dir = os.path.join(BASE_DIR, jahr)
        out_dir = os.path.join(OUTPUT_BASE, jahr)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- Analysiere Jahr: {jahr} ---")
        files = [f for f in os.listdir(jahr_dir) if f.endswith(".tif")]
        jahr_pixel = 0

        for f_name in tqdm(files):
            path = os.path.join(jahr_dir, f_name)
            with rasterio.open(path) as src:
                img = src.read().astype(np.float32)
                meta = src.meta.copy()

                img = np.nan_to_num(img, nan=0.0)
                img = np.maximum(img, 0)
                img = img / 30000.0
                img = np.clip(img, 0, 1)

                input_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = (torch.sigmoid(output) > 0.5).cpu().squeeze().numpy().astype(np.uint8)

                jahr_pixel += np.sum(pred == 1)

            meta.update(count=1, dtype='uint8', nodata=0)
            with rasterio.open(os.path.join(out_dir, f"Pred_{f_name}"), 'w', **meta) as dst:
                dst.write(pred, 1)

        # Fläche berechnen (30m x 30m = 900m²)
        flaeche_km2 = (jahr_pixel * 900) / 1_000_000
        ergebnisse.append({"Jahr": jahr, "Flaeche_km2": flaeche_km2})
        print(f"Ergebnis {jahr}: {flaeche_km2:.2f} km²")

    df = pd.DataFrame(ergebnisse)
    df.to_csv(os.path.join(OUTPUT_BASE, "Gletscher_Statistik_2013_2020.csv"), index=False)
    print("\n✅ Gesamtanalyse fertig! CSV-Tabelle wurde erstellt.")


if __name__ == "__main__":
    analyse_zeitreihe()