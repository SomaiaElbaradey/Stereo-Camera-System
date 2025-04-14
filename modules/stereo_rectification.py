import cv2
import numpy as np
def rectificate_images(left_img, right_img, K, dist, R, T):
    """
    Perform stereo rectification on the input images using the given camera parameters.

    Args:
        left_img (numpy.ndarray): Left image.
        right_img (numpy.ndarray): Right image.
        K (numpy.ndarray): Camera intrinsic matrix.
        dist (numpy.ndarray): Distortion coefficients.
        R (numpy.ndarray): Rotation matrix.
        T (numpy.ndarray): Translation vector.

    Returns:
        tuple: Rectified left and right images.
    """
    h, w = left_img.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, dist, K, dist, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w, h), cv2.CV_32FC1)
    rect_left = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
    return rect_left, rect_right
