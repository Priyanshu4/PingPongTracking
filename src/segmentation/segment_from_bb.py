""" Segmenting a blurred ball from its bounding box.
"""

import cv2
import numpy as np

def apply_bb_mask_to_full_image(image, box, box_mask):
    """ Apply a binary mask from a bounding box to the full image.
    
    Args:
        image: A grayscale image.
        box: A bounding box [x, y, x2, y2].
        mask: A binary mask covering the bounding box region.
    
    Returns:
        A binary image.
    """
    x, y, x2, y2 = box
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y:y2, x:x2] = box_mask
    return mask

def color_threshold_ball_from_bb(image, bb, lower_hsv, upper_hsv):
    """ Segment a blurred ball from its bounding box using color thresholding.
    
    Args:
        image: A BGR image.
        bb: A bounding box [x, y, x2, y2]
        lower_hsv: Lower HSV values for color thresholding.
        upper_hsv: Upper HSV values for color thresholding.
    
    Returns:
        A binary image.
    """
    # Extract the bounding box.
    x, y, x2, y2 = bb
    ball_crop = image[y:y2, x:x2]
    
    # Convert to HSV.
    hsv_ball_crop = cv2.cvtColor(ball_crop, cv2.COLOR_BGR2HSV)
    
    # Apply the color threshold.
    mask = cv2.inRange(hsv_ball_crop, lower_hsv, upper_hsv)
    
    return apply_bb_mask_to_full_image(image, bb, mask)

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