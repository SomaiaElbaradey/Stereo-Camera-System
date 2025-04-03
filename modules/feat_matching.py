import cv2
import numpy as np
import matplotlib.pyplot as plt

def feature_detection_and_matching(left_img, right_img):
    """
    Detect and match features between the left and right images with optimized performance.
    
    Parameters:
        left_img (ndarray): Left image.
        right_img (ndarray): Right image.
        
    Returns:
        matched_image (ndarray): Image with matched features displayed.
        good_matches (list): List of good matches.
    """
    
    # Convert images to grayscale only if they're not already
    if len(left_img.shape) == 3:
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_left = left_img
        
    if len(right_img.shape) == 3:
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_right = right_img
    
    # 1: Use SIFT which works better with FLANN
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray_left, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_right, None)
    
    # Check if any features were found
    if descriptors1 is None or descriptors2 is None or len(keypoints1) < 2 or len(keypoints2) < 2:
        return None, []
    
    # Configure FLANN for SIFT (which uses floating-point descriptors)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Higher values = more accurate but slower
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Perform KNN matching
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test to filter good matches
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Sort matches by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    
    # Only draw top matches to avoid clutter
    num_matches_to_draw = min(50, len(good_matches))
    
    # Draw matches
    matched_image = cv2.drawMatches(
        left_img, keypoints1,
        right_img, keypoints2,
        good_matches[:num_matches_to_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Display matched features
    plt.figure(figsize=(15, 10))
    plt.imshow(matched_image)
    plt.title(f'Matched Features: {len(good_matches)} matches')
    plt.axis('off')
    plt.show()

    return matched_image, good_matches