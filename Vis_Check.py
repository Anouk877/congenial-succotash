import os
import random
import numpy as np
import rasterio
import matplotlib.pyplot as plt


image_dir = "Mosaik_patches"       
mask_dir = "mask_patches"           


num_samples = 5


image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
mask_files  = sorted([f for f in os.listdir(mask_dir)  if f.endswith(".npy")])


assert len(image_files) == len(mask_files), "Image- und Maskenanzahl stimmen NICHT überein!"

print(f"Finde {len(image_files)} Patch-Paare. Wähle {num_samples} zufällig.\n")



valid_indices = []

for i, maskfile in enumerate(mask_files):
    patch = np.load(os.path.join(mask_dir, maskfile))
    if np.max(patch) == 1:
        valid_indices.append(i)

print(f"Finde {len(valid_indices)} Patches mit Gletscher.")


sample_indices = random.sample(valid_indices, min(num_samples, len(valid_indices)))


for idx in sample_indices:

    image_path = os.path.join(image_dir, image_files[idx])
    mask_path  = os.path.join(mask_dir, mask_files[idx])


    with rasterio.open(image_path) as src:
        img = src.read()   

       
        if img.shape[0] >= 3:
            rgb = np.stack([img[2], img[1], img[0]], axis=-1)
        else:
            rgb = img[0]    

        
        rgb_norm = np.clip(rgb / 10000, 0, 1)


   
    mask = np.load(mask_path)

  
    overlay = rgb_norm.copy()
    overlay_mask = mask == 1

    overlay[overlay_mask, 0] = 1.0   
    overlay[overlay_mask, 1] = 0.0
    overlay[overlay_mask, 2] = 0.0

    
    fig, ax = plt.subplots(1, 3, figsize=(14, 6))

    ax[0].imshow(rgb_norm)
    ax[0].set_title(f"Bild Patch {idx}")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Maske")
    ax[1].axis("off")

    ax[2].imshow(overlay)
    ax[2].set_title("Overlay (Maske Rot)")
    ax[2].axis("off")

    plt.show()
