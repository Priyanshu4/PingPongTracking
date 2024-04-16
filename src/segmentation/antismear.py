import cv2
from typing import Optional
import numpy as np
from .ellipse import Ellipse
from typing import Tuple

class AntiSmear:
    """ Class for finding the original ball using the ball smear's segmented mask.
        This finds the original ball using an ellipse fitting algorithm.
        Uses the approach proposed in https://www.tandfonline.com/doi/full/10.1080/21642583.2018.1450167.
    """

    def __init__(self):
        pass
     

    def _find_ellipses(self, ball_mask) -> Optional[Ellipse]:
        
        # Find contours
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Check if contours are found
        if len(contours) != 0:
            # Iterate through contours
            for cont in contours:
                if len(cont) < 5:
                    break
                # Fit ellipse to contour
                ellipse = cv2.fitEllipse(cont)
                ellipse = Ellipse.from_rotated_rect(ellipse)
                return ellipse  # Return the first fitted ellipse
            
        return None
    
    def _sign(self, x) -> int:
        """ Returns the sign of a scalar.
            1 if positive, -1 if negative or zero.
        """
        if x == 0:
            return 0
        if x > 0:
            return 1
        return -1
    
    def find_original_ball(self, ball_mask, ball_speed_x, ball_speed_y) -> Optional[Tuple[Ellipse, Ellipse]]:
        """ Finds the original ball using the segmented mask of the ball smear.
            Considers an estimate of the sign of ball's speed in the x and y pixel directions.
            Be careful to consider that y-direction is inverted in images.
            Returns an tuple of the ball's Ellipse and the smear's ellipse.
            Args:
                ball_mask: Segmented mask of the ball smear.
                ball_speed_x: Estimated speed of the ball in the x direction (pixels, only sign matters).
                ball_speed_y: Estimated speed of the ball in the y direction (pixels, only sign matters).
            Returns:
                Tuple of the ball's Ellipse and the smear's ellipse.
                If no ellipse is found, returns None.
        """
        ellipse = self._find_ellipses(ball_mask)
        if ellipse is None:
            return None

        xc = ellipse.center_x               
        yc = ellipse.center_y               
        alpha = ellipse.angle_rad          
        s = ellipse.short_radius
        l = ellipse.long_radius

        # Snap velocity vector to long axis of ellipse
        velocity_vector = np.array([ball_speed_x, ball_speed_y])
        long_axis_vector = ellipse.long_radius_vector
        projection_right = np.dot(velocity_vector, long_axis_vector)
        projection_left = np.dot(velocity_vector, -long_axis_vector)
        if projection_right > projection_left:
            velocity_vector = long_axis_vector
        else:
            velocity_vector = -long_axis_vector

        x = xc + (l-s)*np.abs(np.cos(alpha)) * self._sign(velocity_vector[0])
        y = yc + (l-s)*np.abs(np.sin(alpha)) * self._sign(velocity_vector[1])
        x, y = int(x), int(y)

        ball = Ellipse(x, y, s, s, ellipse.angle)

        return ball, ellipse

