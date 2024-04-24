# Add the parent directory to the path so that we can import the src module
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.camera import CameraCalibration, Camera
from src.camera.video import VideoStream
from src.segmentation.circledetector import CircleDetector
from src.pose.position_estimation import PositionEstimation
from src.pingpong.ball import BallConstants
from src.pingpong.table import TableConstants
from src.fileutils import DATA_DIR
from src.camera.intrinsic_matrix import find_single_intrinsic_matrix

import src.visualization.plot3D as plot3D
from src.ukf import StateVector, BallUKF, PoseMeasurementMode
from src.camera import CameraPose
from scipy.spatial.transform import Rotation

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO



actual_ball_positions = np.array([
    np.array([-0.5, 0, 1]),
    np.array([-0.5, 0, 2]),
    np.array([0.5, 0, 1]),
    np.array([0.5, 0, 2]),
    np.array([0, 0, 0.5]),
    np.array([1, 0, 2]),
    np.array([0, 0, 1]),
    np.array([0, 0, 2]),
    np.array([0, 0, 3])
])

ball = BallConstants(radius=20e-2 ) # Initialize ball with defaults

# Define the camera pose, we will plot it
camera_position = np.array([0, 0, 0])
camera_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
camera = CameraPose(camera_position, camera_orientation, mirror_y=False)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-1, 4)
ax.set_ylim3d(-1, 4)
ax.set_zlim3d(-1, 4)

plot3D.plot_camera(ax, camera)
plot3D.plot_balls(ax, ball, actual_ball_positions, color='red')
plot3D.view_from_camera_angle(ax, camera)
plt.show()