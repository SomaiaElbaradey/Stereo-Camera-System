import cv2
import numpy as np
import glob
import os
 
def calibrate_camera(calibration_dir="cal", pattern_size=(9, 7), square_size=15.0):
    """
    Perform camera calibration using chessboard images.
    Args:
        calibration_dir: Directory containing calibration images
        pattern_size: Number of inner corners (width, height)
        square_size: Size of chessboard squares in mm
    Returns:
        K: Camera matrix
        dist: Distortion coefficients
    """
    # Prepare 3D object points (0,0,0), (1,0,0), (2,0,0) ..., (8,6,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
 
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    detected_images = []
 
    # Get calibration images
    images = glob.glob(os.path.join(calibration_dir, "cal*.jpg"))
    if not images:
        raise FileNotFoundError(f"No calibration images found in {calibration_dir}")
 
    # Process each image
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            # Refine corner locations
            corners_subpix = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            detected_images.append(fname)
            # Visualize detected corners
            cv2.drawChessboardCorners(img, pattern_size, corners_subpix, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(300)
    cv2.destroyAllWindows()
 
    if len(objpoints) < 5:
        raise ValueError(f"Only {len(objpoints)} checkerboards detected. Need at least 5 for good calibration.")
 
    # Perform camera calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    # Print calibration results
    print(f"Calibration successful with {len(objpoints)} images")
    print("Camera Matrix (K):\n", K)
    print("Distortion Coefficients:\n", dist)
    # Save calibration data
    np.savez(
        "camera_calibration.npz",
        K=K, dist=dist, rvecs=rvecs, tvecs=tvecs,
        detected_images=detected_images
    )
    return ret ,K, dist , rvecs, tvecs