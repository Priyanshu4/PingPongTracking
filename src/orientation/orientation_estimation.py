""" Provides class to use DOE (Dot Orientation Estimation) to estimate the orientation of a table tennis ball.
    Serves as the bridge between our vision modules and the DOE implementation from https://github.com/cogsys-tuebingen/spindoe.
    The code in this file is developed by us, but the DOE code is credited to:
    T. Gossard, J. Tebbe, A. Ziegler, and A. Zell, 'SpinDOE: A ball spin estimation method for table tennis robot', arXiv [cs.CV]. 2023.
"""

from pathlib import Path
from .doe.doe import DOE 
from scipy.spatial.transform import Rotation
from typing import Optional
import cv2
import numpy as np

class OrientationEstimation:
    """ Estimates the orientation of a table tennis ball using the DOE (Dot Orientation Estimation) method.
        Serves as the bridge between vision modules and the DOE implementation.
        DOE expects a 60 by 60 image. 
        This class handles resizing functionality.
    """

    def __init__(self, dot_detector_model_path: Path, use_gpu=False) -> None:
        """ Initializes the OrientationEstimation object.
            Args:
                dot_detector_model_path (Path): Path to the trained PyTorch model for dot detection.
                use_gpu (bool): Whether to use the GPU for inference.
        """
        self.use_gpu = use_gpu
        self.doe = DOE(dot_detector_model_path, use_gpu)

    def estimate_orientation_with_doe(self, image, ball_mask, 
                                        ball_center_x: int, ball_center_y: int, 
                                        ball_radius: int) -> Rotation:
        """ Estimates the orientation of a table tennis ball using DOE.
            Args:
                image (np.ndarray): Image of the table tennis ball.
                ball_mask (np.ndarray): Mask of the table tennis ball.
                ball_center_x (int): pixel x-coordinate of the center of the table tennis ball.
                ball_center_y (int): pixel y-coordinate of the center of the table tennis ball.
                ball_radius (int): Radius of the table tennis ball in pixels.
            Returns:
                Rotation (scipy.spatial.transform.Rotation): Estimated orientation of the ball.
        """
        # Crop and resize the image
        resized_image = self._crop_and_resize(image, ball_mask, ball_center_x, ball_center_y, ball_diameter)

        # Estimate the orientation of the ball using DOE
        rotation = self.doe.estimate_single(resized_image)

        return rotation


    def _crop_and_resize(self, image: np.ndarray, ball_mask: Optional[np.ndarray], 
                             ball_center_x: int, ball_center_y: int, ball_radius: int) -> Rotation:
        """ Estimates the orientation of a table tennis ball in the image.
            Args:
                image (np.ndarray): Image of the table tennis ball.
                ball_mask Optional(np.ndarray): Mask of the table tennis ball.
                    If ball_mask is set to None, we use the center and diameter to crop the image.
                ball_center_x (int): pixel x-coordinate of the center of the table tennis ball.
                ball_center_y (int): pixel y-coordinate of the center of the table tennis ball.
                ball_radius (int): Radius of the table tennis ball in pixels.
            Returns:
                resized_image (np.ndarray): The cropped and resized image of the ball.
        """  
        # Mask the ball in the image
        if ball_mask is not None:
            masked_image = cv2.bitwise_and(image, image, mask=ball_mask)
        else:
            # If ball mask is not provided, use the center and diameter to mask the image
            mask = np.zeros_like(image)
            cv2.circle(mask, (ball_center_x, ball_center_y), ball_radius, (255, 255, 255), -1)
            masked_image = cv2.bitwise_and(image, mask)

        # Crop the image around the ball
        x1 = ball_center_x - ball_radius
        x2 = ball_center_x + ball_radius
        y1 = ball_center_y - ball_radius
        y2 = ball_center_y + ball_radius
        cropped_image = masked_image[y1:y2, x1:x2]

        # Resize the cropped image to 60x60 using opencv
        resized_image = cv2.resize(cropped_image, (self.doe.size, self.doe.size))

        return resized_image
