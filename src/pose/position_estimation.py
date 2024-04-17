from src.camera import Camera
from src.pingpong.ball import BallConstants
from src.pingpong.table import TableConstants
import numpy as np
from scipy.spatial.transform import Rotation

class PositionEstimation:

    def __init__(self, camera: Camera, ball: BallConstants, table: TableConstants):
        self.camera = camera
        self.ball = ball
        self.table = table

    def ball_position_camera_reference_frame(self, x, y, diameter_pix):
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
        d = self.ball.diameter * 1000
  
        z_cam = (fx * d) / diameter_pix
        x_cam = ((x-cx) * z_cam) / fx
        y_cam = ((y-cy) * z_cam) / fy

        return np.array([x_cam, y_cam, z_cam])/1000
    
    def ball_position_table_reference_frame(self, x, y, diameter_pix):
        """ Estimates the ball's position in the table's frame of reference.
        Arguments:
            x: the x pixel coordinate of the ball's center
            y: the y pixel coordinate of the ball's center
            diameter_pix: the diameter of the ball in pixels
        Returns:
            position: Ball's position in meters in table's reference frame as np array of [x, y, z] 
        """
        position_camera = self.ball_position_camera_reference_frame(x, y, diameter_pix)
        position_table = self.camera.transform_to_table_reference_frame(position_camera)
        return position_table




