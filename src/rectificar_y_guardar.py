import cv2
import os
import glob
import pickle
import numpy as np

with open("../data/calibracion_buddha/stereo_maps_new.pkl", "rb") as f:
    maps = pickle.load(f)

left_map_x = maps["left_map_x"]
left_map_y = maps["left_map_y"]
right_map_x = maps["right_map_x"]
right_map_y = maps["right_map_y"]


# === RUTAS ===
input_dir = "../datasets/stereo_budha_board/captures"
output_dir = "../data/fotos_buddha_rectificadas"
os.makedirs(output_dir, exist_ok=True)

# === PROCESAR TODAS LAS IMÁGENES ===
left_images = sorted(glob.glob(os.path.join(input_dir, "left_*.jpg")))
right_images = sorted(glob.glob(os.path.join(input_dir, "right_*.jpg")))

for left_path, right_path in zip(left_images, right_images):
    print(f"Procesando: {os.path.basename(left_path)} y {os.path.basename(right_path)}")

    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)

    # Rectificación
    left_rect  = cv2.remap(left_image,  left_map_x,  left_map_y,  cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_image, right_map_x, right_map_y, cv2.INTER_LINEAR)

    # Guardar
    left_name = os.path.basename(left_path).replace(".jpg", "_rect.jpg")
    right_name = os.path.basename(right_path).replace(".jpg", "_rect.jpg")

    cv2.imwrite(os.path.join(output_dir, left_name), left_rect)
    cv2.imwrite(os.path.join(output_dir, right_name), right_rect)

print(f"\n Imágenes rectificadas guardadas en: {output_dir}")
