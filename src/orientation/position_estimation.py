from src.camera import Camera
from src.pingpong.ball import BallConstants
import numpy as np

class PositionEstimation:

    def __init__(self, camera: Camera, ball: BallConstants):
        self.camera = camera
        self.ball = ball

    def ball_position_camera_reference_frame(self, x, y, diameter_pix):
        """
        Estimates the ball's position in the camera's frame of reference.
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
  
        zreal = (fx * d) / diameter_pix
        xreal = ((x-cx) * zreal) / fx
        yreal = ((cy-y) * zreal) / fy

        return np.array([xreal, yreal, zreal])

