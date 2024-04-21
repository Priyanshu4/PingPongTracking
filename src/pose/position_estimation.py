from src.camera import Camera, CameraPose
from src.pingpong.ball import BallConstants
from src.pingpong.table import TableConstants
import numpy as np
from typing import Tuple

class PositionEstimation:

    def __init__(self, camera: Camera, ball: BallConstants, table: TableConstants):
        self.camera = camera
        self.ball = ball
        self.table = table

    def ball_position_camera_reference_frame(self, x: int, y: int, diameter_pix: int):
        """ Estimates the ball's position in the camera's frame of reference.
            Note that in camera's frame of reference, the z-axis is depth from the camera.
            In the table's frame of reference, the z-axis is up and down.
        Arguments:
            x: the x pixel coordinate of the ball's center
            y: the y pixel coordinate of the ball's center
            diameter_pix: the diameter of the ball in pixels
        Returns:
            position: Ball's position in meters in camera's reference frame as np array of [x, y, z] 
        """
        fx = self.camera.calibration.fx
        fy = self.camera.calibration.fy
        cx = self.camera.calibration.cx
        cy = self.camera.calibration.cy
        d = self.ball.diameter
  
        z_cam = (fx * d) / diameter_pix
        x_cam = ((x-cx) * z_cam) / fx
        y_cam = ((y-cy) * z_cam) / fy

        return np.array([x_cam, y_cam, z_cam])
    
    def ball_position_table_reference_frame(self, x: int, y: int, diameter_pix: int) -> np.array:
        """ Estimates the ball's position in the table's frame of reference.
        Arguments:
            x: the x pixel coordinate of the ball's center
            y: the y pixel coordinate of the ball's center
            diameter_pix: the diameter of the ball in pixels
        Returns:
            position: Ball's position in meters in table's reference frame as np array of [x, y, z] 
        """
        position_camera = self.ball_position_camera_reference_frame(x, y, diameter_pix)
        position_table = self.camera.pose.transform_to_table_reference_frame(position_camera)
        return position_table
    
    def project_ball_position_camera_reference_frame_to_camera_plane(self, position: np.array) -> Tuple[int, int, int]:
        """ 
        Projects the ball's position in the camera's reference frame to the camera's plane.
        Basically, this calculates the inverse of ball_position_camera_reference_frame.
        If the z position is less than or equal to 0, it is set to a tenth of ball diameter to avoid division by zero and negatives.

        Arguments:
            position: Ball's position in meters in camera's reference frame as np array of [x, y, z]

        Returns:
            x: the x pixel coordinate of the ball's center
            y: the y pixel coordinate of the ball's center
            diameter_pix: the diameter of the ball in pixels
        """
        fx = self.camera.calibration.fx
        fy = self.camera.calibration.fy
        cx = self.camera.calibration.cx
        cy = self.camera.calibration.cy
        d = self.ball.diameter

        x_cam, y_cam, z_cam = position
        if (z_cam <= 0):
            z_cam = d/10

        diameter_pix = round((fx * d) / z_cam)
        x = round((x_cam * fx) / z_cam + cx)
        y = round((y_cam * fy) / z_cam + cy)

        return x, y, diameter_pix

    def project_ball_position_table_reference_frame_to_camera_plane(self, position: np.array) -> Tuple[int, int, int]:
        """ 
        Projects the ball's position in the table's reference frame to the camera's plane.
        Basically, this calculates the inverse of ball_position_table_reference_frame.

        Arguments:
            position: Ball's position in meters in table's reference frame as np array of [x, y, z]

        Returns:
            x: the x pixel coordinate of the ball's center
            y: the y pixel coordinate of the ball's center
            diameter_pix: the diameter of the ball in pixels
        """
        position_camera = self.camera.pose.transform_to_camera_reference_frame(position)
        return self.project_ball_position_camera_reference_frame_to_camera_plane(position_camera)


