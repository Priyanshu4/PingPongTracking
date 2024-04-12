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
        This class handles interfacing with the DOE model and preprocessing the image for inference.
    """

    def __init__(self, dot_detector_model_path: Path, use_gpu=False, 
                 doe_expected_brightness: Optional[float] = 100) -> None:
        """ Initializes the OrientationEstimation object.
            Args:
                dot_detector_model_path (Path): Path to the trained PyTorch model for dot detection.
                use_gpu (bool): Whether to use the GPU for inference.
        """
        self.use_gpu = use_gpu
        self.doe = DOE(dot_detector_model_path, use_gpu)
        self.doe_expected_brightness = doe_expected_brightness

    def estimate_orientation_with_doe(self, image, ball_mask: Optional[np.ndarray], 
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
        # Preprocess the image for DOE inference
        image = self.preprocess_image(image, ball_mask, ball_center_x, ball_center_y, ball_radius)

        # Estimate the orientation of the ball using DOE
        rotation = self.doe.estimate_single(image)

        return rotation
    
    def estimate_orientation_with_doe_debug(self, image, ball_mask: Optional[np.ndarray], 
                                        ball_center_x: int, ball_center_y: int, 
                                        ball_radius: int) -> Rotation:
        """ Estimates the orientation of a table tennis ball using DOE.
            This is the debug version, that also returns preprocessed image, heatmap, augmented image and rotation.
            Args:
                image (np.ndarray): Image of the table tennis ball.
                ball_mask (np.ndarray): Mask of the table tennis ball.
                ball_center_x (int): pixel x-coordinate of the center of the table tennis ball.
                ball_center_y (int): pixel y-coordinate of the center of the table tennis ball.
                ball_radius (int): Radius of the table tennis ball in pixels.
            Returns:
                Rotation (scipy.spatial.transform.Rotation): Estimated orientation of the ball.
                preprocessed_image (np.ndarray): Preprocessed image for DOE inference.
                heatmap (np.ndarray): Heatmap of the ball returned by the DOE model.
                aug_img (np.ndarray): Augmented image for DOE inference.
        """
        # Preprocess the image for DOE inference
        preprocessed_image = self.preprocess_image(image, ball_mask, ball_center_x, ball_center_y, ball_radius)

        # Estimate the orientation of the ball using DOE
        rot, aug_img, heatmap = self.doe.debug(preprocessed_image)

        return rot, preprocessed_image, aug_img, heatmap
    
    
    def preprocess_image(self, image, ball_mask: Optional[np.ndarray], 
                                        ball_center_x: int, ball_center_y: int, 
                                        ball_radius: int):
        """ Preprocesses the image for DOE inference.
            Includes cropping, masking and brightness adjustment.
        """
        image = self._mask_and_crop(image, ball_mask, ball_center_x, ball_center_y, ball_radius)
        if self.doe_expected_brightness is not None:
            image = self._adjust_brightness(image, self.doe_expected_brightness)
        return image

    def _adjust_brightness(self, image: np.ndarray, target_brightness: float) -> np.ndarray:
        """ Reduces the brightness of the image by a given value.
            The pretrained DOE works well on dark images, so we reduce the brightness of the image.
            Args:
                image (np.ndarray): Image to reduce the brightness of.
                target_brightness (int): Target average brightness value.
            Returns:
                np.ndarray: Image with reduced brightness.
        """
        # Calculate the average brightness of the image
        average_brightness = np.mean(image)

        # Calculate the factor to multiply the image by
        factor = target_brightness / average_brightness

        # Multiply the image by the factor and clip the values to the range of 0-255
        reduced_image = np.clip(image * factor, 0, 255).astype(np.uint8)

        return reduced_image
    def _mask_and_crop(self, image: np.ndarray, ball_mask: Optional[np.ndarray], 
                             ball_center_x: int, ball_center_y: int, ball_radius: int) -> Rotation:
        """ Masks the ball and crops the image around it.
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

        return cropped_image