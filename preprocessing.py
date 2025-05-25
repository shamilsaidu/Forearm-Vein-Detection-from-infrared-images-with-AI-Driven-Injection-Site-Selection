# Convert to Grayscale (if not already)

# Histogram Equalization (CLAHE)

# Gaussian Blur – to reduce noise

# Bilateral Filtering – edge-preserving smoothing

# Sharpening – to enhance details



import cv2
import pandas as pd
import os
import numpy as np

# Load dataset
df = pd.read_csv("dataset.csv")  # Replace with actual path

# Create output directory
output_dir = "preprocessed_images"
os.makedirs(output_dir, exist_ok=True)

# CLAHE and other processing
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for idx, row in df.iterrows():
    nir_path = row['nir_image']

    if not os.path.exists(nir_path):
        print(f"[Warning] File not found: {nir_path}")
        continue

    # Load in grayscale
    img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[Error] Failed to load: {nir_path}")
        continue

    # Step 1: CLAHE
    img_clahe = clahe.apply(img)

    # Step 2: Gaussian Blur
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), sigmaX=0)

    # Step 3: Bilateral Filter (for edge preservation)
    img_bilateral = cv2.bilateralFilter(img_blur, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 4: Sharpening
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    img_sharp = cv2.filter2D(img_bilateral, -1, kernel_sharp)

   

    # Save preprocessed image
    filename = f"preprocessed_{idx}.png"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, img_sharp)

    # Update DataFrame
    df.at[idx, 'preprocessed_image'] = save_path

# Save updated DataFrame
df.to_csv("updated_dataset.csv", index=False)
print("Advanced preprocessing complete. File saved as 'updated_dataset.csv'")
