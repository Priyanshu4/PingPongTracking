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
            Note that in camera's frame of reference, the z-axis is forward or backward from the camera.
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
        y_cam = ((cy-y) * z_cam) / fy

        return np.array([x_cam, y_cam, z_cam])
    
    def _to_table_reference_frame(self, position: np.ndarray) -> np.ndarray:
        """ Converts the ball's reference frame to the table reference frame.
            Note that in camera's frame of reference, the z-axis is forward or backward from the camera.
            In the table's frame of reference, the z-axis is up and down. 
        """

        # Transform to reference frame of edge of table
        camera_rot = Rotation.from_euler('xyz', self.camera.orientation)
        position = camera_rot.inv().apply(position)
        position = position - self.camera.position

        # Swap y and z for table frame
        new_y = position[2]
        new_z = position[1]
        position[1] = new_y
        position[2] = new_z

        # Transform to center of table frame
        position[1] -= self.table.width / 2

        return position




