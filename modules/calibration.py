import cv2
import numpy as np
import glob
import os
 
def calibrate_cameras(calibration_images_path, chessboard_size=(9, 7)):
    """
    Calibrate the cameras using chessboard images.
   
    Parameters:
        calibration_images_path (str): Path to the folder containing calibration images.
        chessboard_size (tuple): Number of inner corners per chessboard row and column.
   
    Returns:
        ret (bool): Success flag.
        mtx (ndarray): Camera matrix.
        dist (ndarray): Distortion coefficients.
        rvecs (list): Rotation vectors.
        tvecs (list): Translation vectors.
    """
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
 
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
 
    # Load calibration images
    images = glob.glob(os.path.join(calibration_images_path, '*.jpg'))
 
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
 
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
 
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
   
    return ret, mtx, dist, rvecs, tvecs
 
def print():
    print("just trail")