import cv2
import numpy as np
import glob
import os

def stereo_rectification(left_img_path, right_img_path, mtx1, dist1, mtx2, dist2):
    """
    Perform stereo rectification on the left and right images.
   
    Parameters:
        left_img_path (str): Path to the left image.
        right_img_path (str): Path to the right image.
        mtx1 (ndarray): Camera matrix for the left camera.
        dist1 (ndarray): Distortion coefficients for the left camera.
        mtx2 (ndarray): Camera matrix for the right camera.
        dist2 (ndarray): Distortion coefficients for the right camera.
   
    Returns:
        rectified_left (ndarray): Rectified left image.
        rectified_right (ndarray): Rectified right image.
    """
    # Load images
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
 
    # Find the size of the images
    h, w = left_img.shape[:2]
 
    # Stereo calibration (dummy values for R and T, replace with actual values)
    R = np.eye(3)  # Rotation matrix
    T = np.array([[1], [0], [0]])  # Translation vector
 
    # Compute the rectification transforms
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, (w, h), R, T)
 
    # Compute the undistort and rectify maps
    map1x, map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, (w, h), cv2.CV_32FC1)
 
    # Apply the rectification
    rectified_left = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
 
    return rectified_left, rectified_right
 