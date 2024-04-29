import cv2
import numpy as np

class ColorThreshold:

    def __init__(self, lower_hsv: tuple, upper_hsv: tuple):
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv

    def apply(self, image: np.array, bounding_box: tuple = None) -> np.array:
        """
        Apply color thresholding to the image.

        Arguments:
            image: the image to apply color thresholding to
            bounding_box: the bounding box of the object in the image (x, y, x2, y2)
        
        Returns:
            mask: the mask of the object in the image
        """
        if bounding_box is None:
            bounding_box = (0, 0, image.shape[1], image.shape[0])
            
        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image as binary mask
        hsv_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Get binary mask for the bounding box
        x, y, x2, y2 = bounding_box
        mask = np.zeros_like(hsv_mask)
        mask[y:y2, x:x2] = 255

        # Find intersection of both masks
        mask = cv2.bitwise_and(hsv_mask, mask)

        ret, mask = cv2.threshold(mask, 0, 200, cv2.THRESH_BINARY)


        return mask