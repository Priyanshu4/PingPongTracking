import numpy as np
import cv2
from typing import Tuple

class CircleDetector:
    """ Detects the ping pong ball assuming it is a perfect circle (no blur).
    """

    def __init__(self):
        pass

    def hough_circles(self, img: np.ndarray, minRadius: int, maxRadius: int, minDist: int, param1: int, param2: int) -> np.ndarray:
        """ Detects circles in an image using the Hough Circle Transform.
            Args:
                img: OpenCV image.
                minRadius: Minimum radius of the circles to detect.
                maxRadius: Maximum radius of the circles to detect.
                minDist: Minimum distance between the centers of the detected circles.
                param1: First method-specific parameter. In case of CV_HOUGH_GRADIENT, it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
                param2: Second method-specific parameter. In case of CV_HOUGH_GRADIENT, it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.
            Returns:
                np.ndarray: Detected circles as numpy array of (x, y, r) tuples.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        if circles is None:
            return np.array([])
        circles = np.uint16(np.around(circles))[0]
        return circles
    
    def draw_circles(self, img: np.ndarray, circles: np.ndarray, color: Tuple[int, int, int], thickness: int) -> np.ndarray:
        """ Draws circles on an image.
            Args:
                img: OpenCV image.
                circles: Detected circles as numpy array of (x, y, r) tuples.
                color: Color of the circles in BGR format.
                thickness: Thickness of the circles.
            Returns:
                augmented_img: OpenCV image with circles drawn on it.
        """
        augmented_img = img.copy()

        for circle in circles:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(augmented_img, center, radius, color, thickness)

        return augmented_img
    
    def filter_circles_by_color_threshold(self, img_bgr: np.ndarray, circles: np.ndarray, 
                                          color_low_hsv: Tuple[int, int, int], color_high_hsv: Tuple[int, int, int], 
                                          threshold: float) -> np.ndarray:
        """ Filters circles by color threshold.

            Args:
                img: OpenCV image.
                circles: Detected circles as numpy array of (x, y, r) tuples.
                color_low: Lower bound of the color threshold in HSV format.
                color_high: Upper bound of the color threshold in HSV format.
                threshold: Portion of the circle that must pass the color filter (0-1).
            Returns:
                filtered_circles: Circles that pass the color filter.
        """
        color_low = np.array(color_low_hsv)
        color_high = np.array(color_high_hsv)

        filtered_circles = []
        for circle in circles:
            x, y, r = circle
            mask = np.zeros_like(img_bgr)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
            masked_img = cv2.bitwise_and(img_bgr, mask)
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(masked_img, color_low, color_high)
            color_portion = cv2.countNonZero(color_mask) / (np.pi * r * r)
            if color_portion >= threshold:
                filtered_circles.append(circle)

        return np.array(filtered_circles)
    
    def filter_to_region(self, img: np.ndarray, circles: np.ndarray, lower: np.array, upper: np.array) -> np.ndarray:
        """ Filters circles to a region.

            Args:
                img: OpenCV image.
                circles: Detected circles as numpy array of (x, y, r) tuples.
                lower: Lower bound of the region.
                upper: Upper bound of the region.
            Returns:
                filtered_circles: Circles that are within the region.
        """
        x_min, y_min, x_max, y_max = lower[0], lower[1], upper[0], upper[1]
        d_min, d_max = lower[2], upper[2]

        filtered_circles = []
        for circle in circles:
            x, y, r = circle
            if x >= x_min and x <= x_max and y >= y_min and y <= y_max and r*2 >= d_min and r*2 <= d_max:
                filtered_circles.append(circle)

        return np.array(filtered_circles)

    def circle_to_mask(self, img: np.ndarray, circle: Tuple[int, int, int]) -> np.ndarray:
        """ Converts a Circle object to a mask with white circle on black background.

            Args:
                img: OpenCV image.
                circle: Circle object.
            Returns:
                np.ndarray: Mask of the circle.
        """
        mask = np.zeros_like(img)
        cv2.circle(mask, (circle[0], circle[1]), circle[2], (255, 255, 255), -1)
        return mask