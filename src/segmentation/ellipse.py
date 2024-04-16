import cv2
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class Ellipse:
    center_x: float       # Center point x-coordinate
    center_y: float       # Center point y-coordinate
    long_radius: float    # Long radius
    short_radius: float   # Short radius
    angle: float          # Angle of rotation in degrees

    def __post_init__(self):
        if self.long_radius < self.short_radius:
            raise ValueError("Long radius must be greater than short radius.")

    @classmethod
    def from_rotated_rect(cls, ellipse: cv2.RotatedRect) -> 'Ellipse':
        """ Creates an Ellipse object from a RotatedRect object.
            Args:
                ellipse: RotatedRect object representing the ellipse.
            Returns:
                Ellipse object.
        """
        width = ellipse[1][0]
        height = ellipse[1][1]
        if width < height:
            return cls(ellipse[0][0], ellipse[0][1], height/2, width/2, ellipse[2] + 90)
        else:
            return cls(ellipse[0][0], ellipse[0][1], width/2, height/2, ellipse[2])
    
    @property
    def angle_rad(self) -> float:
        """ Angle of rotation in radians.
        """
        return np.radians(self.angle)

    @property
    def long_radius_vector(self) -> np.ndarray:
        x = self.long_radius * np.cos(self.angle_rad)
        y = self.short_radius * np.sin(self.angle_rad)
        v = [x, y]
        return np.array(v)
    
    def __str__(self) -> str:
        return f"Ellipse(center_x={self.center_x}, center_y={self.center_y}, short_radius={self.short_radius}, long_radius={self.long_radius}, angle={self.angle})"    


def draw_ellipse_on_image(img: np.ndarray, ellipse: Ellipse, color: Tuple[int, int, int], thickness: int) -> np.ndarray:
    """ Draws an ellipse (represented by Ellipse) on an image.
        Remember that OpenCV coordinate system starts with (0, 0) on the top-left corner. 
        It may be necessary to negate your angle to make it look like a typical cartesian angle.
        Returns the image with the ellipse drawn.
        Args:
            img: Image on which to draw the ellipse.
            ellipse: Ellipse object representing the ellipse.
            color: Color of the ellipse.
            thickness: Thickness of the ellipse.
        Returns:
            Image with the ellipse drawn.
    """
    # Convert ellipse parameters to OpenCV format
    center = (int(ellipse.center_x), int(ellipse.center_y))
    axes = (int(ellipse.long_radius), int(ellipse.short_radius))
    angle = int(ellipse.angle)
    
    img_with_ellipse = cv2.ellipse(img.copy(), center, axes, angle, 0, 360, color, thickness)
    
    return img_with_ellipse