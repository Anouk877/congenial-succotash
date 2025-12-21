import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
import rasterio
import matplotlib.pyplot as plt

MODEL_FILE = "gletscher_unet_final.pth"
BASE_DIR = "/Users/nicolasyukio/Landsat_Analyse"
OUTPUT_BASE = "/Users/nicolasyukio/Gletscher_Visualisierung"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def erstelle_visuelle_analyse():
    print("Starte visuelle Analyse...")
    model = smp.Unet(encoder_name="resnet34", in_channels=10, classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    if not os.path.exists(OUTPUT_BASE):
        os.makedirs(OUTPUT_BASE)

    jahre = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

    for jahr in jahre:
        jahr_dir = os.path.join(BASE_DIR, jahr)
        tif_files = [f for f in os.listdir(jahr_dir) if f.endswith(".TIF")]

        if not tif_files: continue

        path = os.path.join(jahr_dir, tif_files[0])
        print(f"Verarbeite {jahr}...")

        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
            img = np.nan_to_num(img, nan=0.0)
            img_normalized = np.clip(img / 30000.0, 0, 1)

            input_tensor = torch.from_numpy(img_normalized).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred = (torch.sigmoid(output) > 0.5).cpu().squeeze().numpy().astype(np.uint8)

            pixel_count = np.sum(pred == 1)
            flaeche_km2 = (pixel_count * 900) / 1_000_000

            rgb = img_normalized[[3, 2, 1]].transpose(1, 2, 0)
            rgb = np.clip(rgb * 5, 0, 1)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            ax1.imshow(rgb)
            ax1.set_title(f"Landsat Satellitenbild {jahr}", fontsize=16, pad=10)
            ax1.axis('off')

            ax2.imshow(rgb)
            mask_overlay = np.zeros((*pred.shape, 4))
            mask_overlay[pred == 1] = [0, 0.4, 1, 0.6]  # Blaues Overlay
            ax2.imshow(mask_overlay)

            ax2.set_title(f"U-Net Gletscherkartierung ({jahr})\nBerechnete Fläche: {flaeche_km2:.2f} km²",
                          fontsize=16, color='darkblue', pad=10)
            ax2.axis('off')

            save_path = os.path.join(OUTPUT_BASE, f"Analyse_{jahr}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()

            print(f" {jahr} erledigt: {flaeche_km2:.2f} km²")

    print(f"\n Alle Bilder wurden in {OUTPUT_BASE} gespeichert!")


if __name__ == "__main__":
    erstelle_visuelle_analyse()
