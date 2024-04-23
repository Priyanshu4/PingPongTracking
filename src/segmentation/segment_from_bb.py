""" Segmenting a blurred ball from its bounding box.
"""

import cv2
import numpy as np

def adaptive_threshold_ball_from_bb(image, bb):
    """ Segment a blurred ball from its bounding box using adaptive thresholding.
    
    Args:
        image: A grayscale image.
        bb: A bounding box [x, y, x2, y2]
    
    Returns:
        A binary image.
    """
    # Extract the bounding box.
    x, y, x2, y2 = bb
    ball_crop = image[y:y2, x:x2]
    
    hsv_ball_crop = cv2.cvtColor(ball_crop, cv2.COLOR_BGR2HSV)

    # Isolating the Hue channel
    hue_channel = hsv_ball_crop[:,:,0]

    # Get color of center of the ball
    center_hue = hue_channel[hsv_ball_crop.shape[1]//2, hsv_ball_crop.shape[0]//2]

    # Apply adaptive thresholding.
    binary = cv2.adaptiveThreshold(hue_channel, center_hue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Put binary mask on full original image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y:y2, x:x2] = binary
    
    return mask