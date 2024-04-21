import numpy as np
from dataclasses import dataclass
from src.pose.position_estimation import PositionEstimation


@dataclass
class ImageRegion:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    d_min: int
    d_max: int

    def __repr__(self):
        return f"ImageRegion(x_min={self.x_min}, x_max={self.x_max}, y_min={self.y_min}, y_max={self.y_max}, d_min={self.d_min}, d_max={self.d_max})"

def get_high_probability_region(
    predicted_ball_state: np.ndarray,
    predicted_ball_state_covariance: np.ndarray,
    position_estimator: PositionEstimation,
    image_width: int,
    image_height: int,
    standard_deviations: float = 3.0,
) -> ImageRegion:
    """
    Using a predicted ball state and its covariance, this function estimates the region in the image where the ball is likely to be.
    The region is defined by a bounding box and a range of diameters.

    This function uses a position estimator to project the ball state back to the image plane.
    By default, standard deviations = 3.0, which corresponds to a 99.7% confidence interval.

    Args:
        predicted_ball_state (np.ndarray): The predicted ball state.
        predicted_ball_state_covariance (np.ndarray): The predicted ball state covariance.
        position_estimator (PositionEstimation): The position estimator.
        standard_deviations (float): The number of standard deviations to use for the confidence interval.

    Returns:
        ImageRegion: The region where the ball is likely to be.
    """
    x, y, z = predicted_ball_state[:3]
    x_var, y_var, z_var = np.diag(predicted_ball_state_covariance)[:3]
    x_std, y_std, z_std = np.sqrt(x_var), np.sqrt(y_var), np.sqrt(z_var)

    x_pos = [x - standard_deviations * x_std, x + standard_deviations * x_std]
    y_pos = [y - standard_deviations * y_std, y + standard_deviations * y_std]
    z_pos = [z - standard_deviations * z_std, z + standard_deviations * z_std]

    # Calculate all possible projections
    min_x = image_width
    max_x = 0
    min_y = image_height
    max_y = 0
    min_d = max(image_width, image_height)
    max_d = 0
    for xp in x_pos:
        for yp in y_pos:
            for zp in z_pos:
                x, y, d = position_estimator.project_ball_position_table_reference_frame_to_camera_plane(np.array([xp, yp, zp]))
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_d = min(min_d, d)
                max_d = max(max_d, d)

    min_x = max(0, min_x)
    max_x = min(image_width, max_x)
    min_y = max(0, min_y)
    max_y = min(image_height, max_y)
    min_d = max(1, min_d)
    max_d = min(max_d, image_width, image_height)

    return ImageRegion(min_x, max_x, min_y, max_y, min_d, max_d)
