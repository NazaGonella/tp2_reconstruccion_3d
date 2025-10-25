import cv2
import numpy as np
import glob
import pickle

# === CONFIGURATION ===
checkerboard = (10, 7)  # Inner corners (columns, rows)
square_size_mm = 24.2  # Your measured size

# === PREPARE OBJECT POINTS ===
objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
objp *= square_size_mm

# === FIND CALIBRATION IMAGES ===
left_images = sorted(glob.glob('./datasets/stereo_budha_board/calib/calib_left_*.jpg'))  # Adjust path
right_images = sorted(glob.glob('./datasets/stereo_budha_board/calib/calib_right_*.jpg'))

if len(left_images) == 0:
    print("No calibration images found! Check your path.")
    exit()

print(f"Found {len(left_images)} image pairs for calibration")

# === DETECT CORNERS ===
objpoints = []  # 3D points in real world
imgpoints_left = []  # 2D points in left image
imgpoints_right = []  # 2D points in right image

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for left_path, right_path in zip(left_images, right_images):
    img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
    if img_left is None or img_right is None:
        print(f"Could not read: {left_path} or {right_path}")
        continue
    
    # Find corners
    ret_left, corners_left = cv2.findChessboardCorners(img_left, checkerboard, None)
    ret_right, corners_right = cv2.findChessboardCorners(img_right, checkerboard, None)
    
    if ret_left and ret_right:
        # Refine corners to sub-pixel
        corners_left = cv2.cornerSubPix(img_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(img_right, corners_right, (11, 11), (-1, -1), criteria)
        
        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
        print(f"✓ {left_path}")
    else:
        print(f"✗ Failed to detect checkerboard in {left_path}")

print(f"\nSuccessfully detected checkerboard in {len(objpoints)} image pairs")

if len(objpoints) < 10:
    print("WARNING: Too few good images! Need at least 10-15 for reliable calibration")

# === CALIBRATE INDIVIDUAL CAMERAS ===
image_size = img_left.shape[::-1]

print("\nCalibrating left camera...")
ret_left, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
    objpoints, imgpoints_left, image_size, None, None
)
print(f"Left camera RMS reprojection error: {ret_left:.4f} pixels")

print("Calibrating right camera...")
ret_right, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
    objpoints, imgpoints_right, image_size, None, None
)
print(f"Right camera RMS reprojection error: {ret_right:.4f} pixels")

# === STEREO CALIBRATION ===
print("\nPerforming stereo calibration...")
flags = cv2.CALIB_FIX_INTRINSIC  # Use individual camera calibrations
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret_stereo, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    K1, dist1, K2, dist2,
    image_size,
    criteria=criteria_stereo,
    flags=flags
)

print(f"Stereo calibration RMS reprojection error: {ret_stereo:.4f} pixels")
print(f"Baseline: {np.linalg.norm(T):.2f} mm")

if ret_stereo > 1.0:
    print("⚠️  WARNING: High reprojection error! Consider recapturing calibration images.")
elif ret_stereo > 0.5:
    print("⚠️  Reprojection error is acceptable but could be better.")
else:
    print("✓ Excellent calibration quality!")

# === STEREO RECTIFICATION ===
print("\nComputing rectification...")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, dist1, K2, dist2,
    image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

# === GENERATE RECTIFICATION MAPS ===
left_map_x, left_map_y = cv2.initUndistortRectifyMap(
    K1, dist1, R1, P1, image_size, cv2.CV_32FC1
)
right_map_x, right_map_y = cv2.initUndistortRectifyMap(
    K2, dist2, R2, P2, image_size, cv2.CV_32FC1
)

# === SAVE CALIBRATION DATA ===
calibration_data = {
    'left_K': K1,
    'left_dist': dist1,
    'right_K': K2,
    'right_dist': dist2,
    'R': R,
    'T': T,
    'E': E,
    'F': F,
    'image_size': image_size,
    'reprojection_error': ret_stereo,
    'R1': R1,
    'R2': R2,
    'P1': P1,
    'P2': P2,
    'Q': Q
}

with open('./data/calibracionBuddha/stereo_calibration_new.pkl', 'wb') as f:
    pickle.dump(calibration_data, f)

maps_data = {
    'left_map_x': left_map_x,
    'left_map_y': left_map_y,
    'right_map_x': right_map_x,
    'right_map_y': right_map_y
}

with open('./data/calibracionBuddha/stereo_maps_new.pkl', 'wb') as f:
    pickle.dump(maps_data, f)

print("\n✓ Calibration saved to:")
print("  - stereo_calibration_new.pkl")
print("  - stereo_maps_new.pkl")

# === TEST RECTIFICATION ===
print("\nTesting rectification on first image pair...")
test_left = cv2.imread(left_images[0])
test_right = cv2.imread(right_images[0])

rect_left = cv2.remap(test_left, left_map_x, left_map_y, cv2.INTER_LINEAR)
rect_right = cv2.remap(test_right, right_map_x, right_map_y, cv2.INTER_LINEAR)

# Draw horizontal lines to check alignment
combined = np.hstack([rect_left, rect_right])
for y in range(0, combined.shape[0], 50):
    cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

cv2.imwrite('./rectification_test.jpg', combined)
print("✓ Rectification test saved as 'rectification_test.jpg'")
print("  Check that horizontal lines pass through corresponding points in both images!")