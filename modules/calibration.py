import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
 
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
 
def feature_detection_and_matching(left_img, right_img):
    """
    Detect and match features between the left and right images.
   
    Parameters:
        left_img (ndarray): Left image.
        right_img (ndarray): Right image.
   
    Returns:
        matched_image (ndarray): Image with matched features displayed.
        good_matches (list): List of good matches.
    """
    # Convert images to grayscale
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
 
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
 
    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray_left, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_right, None)
 
    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
 
    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
 
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
 
    # Draw matches
    matched_image = cv2.drawMatches(left_img, keypoints1, right_img, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
    return matched_image, matches
 
def main():
    # Paths
    calibration_images_path = 'images/cal'
    left_img_path = 'images/images/left/items_l.png'
    right_img_path = 'images/images/right/items_r.png'
 
    # Camera calibration
    ret, mtx1, dist1, rvecs1, tvecs1 = calibrate_cameras(calibration_images_path)
    ret, mtx2, dist2, rvecs2, tvecs2 = calibrate_cameras(calibration_images_path)  
 
    # Stereo rectification
    rectified_left, rectified_right = stereo_rectification(left_img_path, right_img_path, mtx1, dist1, mtx2, dist2)
 
    # Feature detection and matching
    matched_image, good_matches = feature_detection_and_matching(rectified_left, rectified_right)
 
    # Display matched features
    plt.figure(figsize=(15, 10))
    plt.imshow(matched_image)
    plt.title(f'Matched Features: {len(good_matches)} matches')
    plt.axis('off')
    plt.show()
 
    # Save rectified images
    cv2.imwrite('rectified_left.png', rectified_left)
    cv2.imwrite('rectified_right.png', rectified_right)
 
 
    left = 'images/images/left/items2_l.png'
    right = 'images/images/right/items2_r.png'
 
        # Stereo rectification
    rectified_l, rectified_r = stereo_rectification(left, right, mtx1, dist1, mtx2, dist2)
 
    # Feature detection and matching
    matched, matches = feature_detection_and_matching(rectified_l, rectified_r)
 
    # Display matched features
    plt.figure(figsize=(15, 10))
    plt.imshow(matched)
    plt.title(f'Matched Features: {len(matches)} matches')
    plt.axis('off')
    plt.show()
 
    # Save rectified images
    cv2.imwrite('rectified_left.png', rectified_left)
    cv2.imwrite('rectified_right.png', rectified_right)
 
 
if __name__ == "__main__":
    main()