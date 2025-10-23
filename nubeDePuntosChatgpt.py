import calib
import pickle
import numpy as np
import open3d as o3d
import cv2
import json
from pathlib import Path
import os

from disparity.method_cre_stereo import CREStereo
from disparity.method_opencv_bm import StereoBM, StereoSGBM
from disparity.methods import Calibration, InputPair, Config


# =====================
# 1. Carga de calibración estéreo
# =====================

img = cv2.imread("./fotos_objetos_rectificadas/left_1_rect.jpg")
w, h = img.shape[1], img.shape[0]

with open("./data/stereo_calibration.pkl", "rb") as f:
    c = pickle.load(f)

K1 = c["left_K"]
K2 = c["right_K"]
dist1 = c["left_dist"]
dist2 = c["right_dist"]
R = c["R"]
T = c["T"]
image_size = c["image_size"]

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, dist1, K2, dist2, image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

fx = K1[0, 0]
fy = K1[1, 1]
cx0 = K1[0, 2]
cy0 = K1[1, 2]
baseline = np.linalg.norm(T)

calibration = Calibration(
    width=w,
    height=h,
    baseline_meters=baseline / 1000,
    fx=fx,
    fy=fy,
    cx0=cx0,
    cx1=cx0,
    cy=cy0,
    depth_range=[0.05, 20.0],
    left_image_rect_normalized=[0, 0, 1, 1],
)


# =====================
# 2. Configuración del tablero
# =====================

checkerboard = (9, 6)
cuadradito_size_mm = 20
object_3dpoints = calib.board_points(checkerboard)
object_3dpoints_mm = object_3dpoints * cuadradito_size_mm

nubes = []

# Transformación del tablero de referencia (se definirá en la primera iteración)
o_T_c_ref = None


# =====================
# 3. Procesar pares de imágenes estéreo
# =====================

for i in range(1, 10):
    print(f"Procesando par {i}...")
    left_rectified = cv2.imread(f"./fotos_objetos_rectificadas/left_{i}_rect.jpg")
    right_rectified = cv2.imread(f"./fotos_objetos_rectificadas/right_{i}_rect.jpg")

    if left_rectified is None or right_rectified is None:
        print(f"[{i}] Imágenes no encontradas — skipping")
        continue

    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

    left_found, left_corners = calib.detect_board(checkerboard, left_gray)
    right_found, right_corners = calib.detect_board(checkerboard, right_gray)

    print(len(left_corners))
    print(len(right_corners))

    if not left_found or not right_found:
        print(f"[{i}] Tablero no detectado correctamente — skipping")
        continue

    # =====================
    # 3.1 Calcular pose cámara-objeto (izquierda)
    # =====================

    ret, rvec, tvec = cv2.solvePnP(
        object_3dpoints_mm, left_corners, K1, dist1, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    R_left, _ = cv2.Rodrigues(rvec)

    c_T_o_left = np.eye(4)
    c_T_o_left[:3, :3] = R_left
    c_T_o_left[:3, 3] = tvec.squeeze()

    o_T_c_left = np.linalg.inv(c_T_o_left)

    # Guardamos el primero como referencia global
    if o_T_c_ref is None:
        o_T_c_ref = o_T_c_left.copy()
        print("Se estableció el tablero de referencia global.")

    # =====================
    # 3.2 Calcular disparidad con CREStereo
    # =====================

    models_path = Path("models")
    models_path.mkdir(exist_ok=True)

    pair = InputPair(left_rectified, right_rectified, calibration)
    config = Config(models_path=models_path)
    method = CREStereo(config)
    method.parameters["Shape"].set_value("1280x720")

    disparity = method.compute_disparity(pair)
    disp = disparity.disparity_pixels.astype(np.float32)

    # =====================
    # 3.3 Reconstrucción 3D y transformación al sistema global
    # =====================

    points_3D = cv2.reprojectImageTo3D(disp, Q)
    mask = disp > 0
    points = points_3D[mask]
    colors = left_rectified[mask] / 255.0

    pts_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])

    # Transformación actual -> sistema del tablero de referencia
    o_ref_T_o_i = np.linalg.inv(o_T_c_ref) @ o_T_c_left
    pts_world = (o_ref_T_o_i @ pts_h.T).T[:, :3]

    # Crear nube Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Opcional: reducción de ruido
    pcd = pcd.voxel_down_sample(voxel_size=2.0)

    nubes.append((pcd, o_ref_T_o_i))


# =====================
# 4. Fusionar todas las nubes en el sistema global
# =====================

combined = o3d.geometry.PointCloud()
frames = []

frame_board = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0, origin=[0, 0, 0])
frames.append(frame_board)

for idx, (pcd, T) in enumerate(nubes):
    combined += pcd
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0)
    cam_frame.transform(T)
    frames.append(cam_frame)


# =====================
# 5. (Opcional) Refinar con ICP
# =====================

# Si querés ajustar finamente las nubes (útil si hay leve error de detección del checkerboard)
# habilitá este bloque:

# print("Ajustando nubes con ICP...")
# target = nubes[0][0]
# combined_icp = target
# for idx, (source, _) in enumerate(nubes[1:], start=1):
#     reg = o3d.pipelines.registration.registration_icp(
#         source, target, max_correspondence_distance=5.0,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
#     )
#     source.transform(reg.transformation)
#     combined_icp += source
# combined = combined_icp


# =====================
# 6. Visualización
# =====================

o3d.visualization.draw_geometries([combined] + frames)
