import calib
import pickle
import numpy as np
import open3d as o3d
import cv2

# Cre Stereo

import os
from pathlib import Path
from disparity.method_cre_stereo import CREStereo
from disparity.method_opencv_bm import StereoBM, StereoSGBM
from disparity.methods import Calibration, InputPair, Config

img=cv2.imread("./data/fotos_buddha_rectificadas/left_1_rect.jpg")

w, h = img.shape[1], img.shape[0]

with open("./data/calibracion_buddha/stereo_calibration_new.pkl", "rb") as f:
    c = pickle.load(f)

K1 = c['left_K']
K2 = c['right_K']

fx  = K1[0, 0]
fy  = K1[1, 1]
cx0 = K1[0, 2]
cy0 = K1[1, 2]

R = c['R']
T = c['T']
image_size = c['image_size']
dist1 = c['left_dist']
dist2 = c['right_dist']

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, dist1,
    K2, dist2,
    image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

if 'reprojection_error' in c:
    print(f"Calibration reprojection error: {c['reprojection_error']}")

K1_rect = P1[:3, :3]
K2_rect = P2[:3, :3]

cx1 = P2[0, 2]

baseline = np.linalg.norm(T)
calibration = Calibration(**{
    "width": w,
    "height": h,
    "baseline_meters": baseline / 1000,
    "fx": fx,
    "fy": fy,
    "cx0": cx0,
    "cx1": cx1,
    "cy": cy0,
    "depth_range": [0.05, 20.0],
    "left_image_rect_normalized": [0, 0, 1, 1]
})

# copypaste del pdf consigna, seccion 5, usando el modulo calib de la tutorial 6

checkerboard=(10,7)
cuadradito_size_mm=24.2
object_3dpoints=calib.board_points(checkerboard)
object_3dpoints_mm=object_3dpoints*cuadradito_size_mm
nubes=[]

for i in range(0,6):      #tenemos 21 imagenes de banana
    left_rectified=cv2.imread(f"./data/fotos_buddha_rectificadas/left_{i}_rect.jpg")
    right_rectified=cv2.imread(f"./data/fotos_buddha_rectificadas/right_{i}_rect.jpg")
    left_rectified=cv2.cvtColor(left_rectified,cv2.COLOR_BGR2RGB)
    right_rectified=cv2.cvtColor(right_rectified,cv2.COLOR_BGR2RGB)

    # correct color conversion (OpenCV uses BGR)
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

    # detect boards and guard
    left_found, left_corners = calib.detect_board(checkerboard, left_gray)
    right_found, right_corners = calib.detect_board(checkerboard, right_gray)

    if not left_found or left_corners is None or len(left_corners) < 4:
        print(f"[{i}] left board not found or too few corners ({None if left_corners is None else len(left_corners)}) — skipping")
        continue
    if not right_found or right_corners is None or len(right_corners) < 4:
        print(f"[{i}] right board not found or too few corners ({None if right_corners is None else len(right_corners)}) — skipping")
        continue

    #imagen izquierda
    left_rectified_gray=cv2.cvtColor(left_rectified,cv2.COLOR_RGB2GRAY)
    left_rectified_gray = left_rectified_gray.astype(np.uint8)

    left_found,left_corners=calib.detect_board(checkerboard,left_rectified_gray)

    ret,rvec,tvec=cv2.solvePnP(object_3dpoints_mm,left_corners,K1_rect,None,flags=cv2.SOLVEPNP_IPPE)

    c_R_o_left = cv2.Rodrigues(rvec)[0]
    c_T_o_left = np.column_stack((c_R_o_left, tvec))
    c_T_o_left = np.vstack((c_T_o_left, [0, 0, 0, 1]))
    o_T_c_left = np.linalg.inv(c_T_o_left)        # object to camera transformation

    models_path = "models"
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    models_path = Path(models_path)
    pair = InputPair(left_rectified, right_rectified, calibration)
    config = Config(models_path=models_path)

    method = CREStereo(config)

    method.parameters["Shape"].set_value("1280x720")

    disparity = method.compute_disparity(pair)

    disp = disparity.disparity_pixels.astype(np.float32)
    points_3D = cv2.reprojectImageTo3D(disp, Q)

    # Mask invalid points
    mask = disp > 0
    points = points_3D[mask]
    colors = left_rectified[mask] / 255.0  # normalize colors to [0, 1]

    # transform points from left-camera coordinates to object/checkerboard (world) coordinates
    pts_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])  # (N,4)
    pts_world = (o_T_c_left @ pts_h.T).T[:, :3]  # use the camera->object 4x4 matrix you computed

##### para la bounding box
    # Define bounding box in world coordinates (mm)
    x_min, x_max = -100, 200   # X range
    y_min, y_max = -350, -100   # Y range  
    z_min, z_max = -300, 20   # Z range (above checkerboard)

    # Filter points
    mask_box = (
        (pts_world[:, 0] >= x_min) & (pts_world[:, 0] <= x_max) &
        (pts_world[:, 1] >= y_min) & (pts_world[:, 1] <= y_max) &
        (pts_world[:, 2] >= z_min) & (pts_world[:, 2] <= z_max)
    )

    pts_world_filtered = pts_world[mask_box]
    colors_filtered = colors[mask_box]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world_filtered)
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
    nubes.append((pcd, o_T_c_left))

# o3d.visualization.draw_geometries([pcd])
# Merge all transformed clouds into a single cloud
combined = o3d.geometry.PointCloud()

for idx, (pc, T) in enumerate(nubes):
    combined += pc


all_points = np.asarray(combined.points)
z_coords = all_points[:, 2]
object_height = z_coords.max() - z_coords.min()

print(f"\n=== Combined Object Height ===")
print(f"{object_height:.1f} mm")
print(f"{object_height/10:.1f} cm")

o3d.visualization.draw_geometries([combined])
o3d.io.write_point_cloud("nubeDePuntosBuddha.ply", combined)