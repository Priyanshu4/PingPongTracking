from dataclasses import dataclass
from typing import Optional
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .intrinsic_matrix import find_average_intrinsic_matrix
from .video import VideoStream

@dataclass
class CameraSpecs:
    name: Optional[str]
    width_pixels: int
    height_pixels: int
    sensor_width_mm: float
    sensor_height_mm: float
    focal_length_mm: float
    frames_per_second: float
        
    @classmethod
    def load_from_yaml(cls, yaml_path: str, camera_name: str) -> 'CameraSpecs':
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            if camera_name not in data:
                raise ValueError(f'Camera with name {camera_name} not found in {yaml_path}.\nAvailable cameras: {list(data.keys())}')
            else:
                data = data[camera_name]
            return cls(
                name=camera_name,
                width_pixels=data['width_pixels'],
                height_pixels=data['height_pixels'],
                sensor_width_mm=data['sensor_width_mm'],
                sensor_height_mm=data['sensor_height_mm'],
                focal_length_mm=data['focal_length_mm'],
                frames_per_second=data['frames_per_second']
            )
        
@dataclass
class CameraCalibration:
    intrinsic_matrix: np.ndarray

    @property
    def fx(self) -> float:
        return self.intrinsic_matrix[0, 0]
    
    @property
    def fy(self) -> float:
        return self.intrinsic_matrix[1, 1]
    
    @property
    def cx(self) -> float:
        return self.intrinsic_matrix[0, 2]
    
    @property
    def cy(self) -> float:
        return self.intrinsic_matrix[1, 2]
    
    @classmethod
    def from_camera_specs(cls, camera_specs: CameraSpecs) -> 'CameraCalibration':
        fx = camera_specs.focal_length_mm * camera_specs.sensor_width_mm / camera_specs.width_pixels
        fy = camera_specs.focal_length_mm * camera_specs.sensor_height_mm / camera_specs.height_pixels
        cx = camera_specs.width_pixels / 2
        cy = camera_specs.height_pixels / 2
        return cls(intrinsic_matrix=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    
    @classmethod
    def from_calibration_video(cls, video: VideoStream) -> 'CameraCalibration':
        return cls(intrinsic_matrix=find_average_intrinsic_matrix(video))

class CameraPose:
    """
    CameraPose represents the position and orientation of a camera.
    It handles transformations between the camera reference frame and the table reference frame.
    """

    def __init__(self, position: np.ndarray, orientation: Rotation, mirror_x: bool = False, mirror_y: bool = False):
        """ Initializes a CameraPose object.
            Arguments:
                position: np array with [x, y, z] coordinates in meters relative to center of the table.
                    Position should be specified in table reference frame.
                     _____________ 
                    |             |
                    |             |
                    |             |
                   -----------------            
                    |             |
                    |             |
                    |_____________|

                    x corresponds to left (-) and right (+) of the net.
                    y corresponds to left (-) and right (+) from the player's perspective.
                    z corresponds to down (-) and up (+). 

                orientation: scipy.spatial.transform.Rotation object representing the camera's orientation.
                             This should be the rotation transform from table reference frame to camera reference frame.
                             Consider that first the position transformation will be applied, then this rotation.
                             In your camera reference frame, x is left (-) and right (+), y is up (-) and down (+), and z is depth.

                mirror_x: bool, whether to mirror the x-axis of the camera (after other transformations).
                mirror_y: bool, whether to mirror the y-axis of the camera (after other transformations).
                    After orientation is applied, the camera will be mirrored.
        """
        self.position = position
        self.orientation = orientation
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y

    def transform_to_camera_reference_frame(self, position: np.ndarray) -> np.ndarray:
        """ Transforms a position from table reference frame to camera reference frame.
        """
        position = position - self.position
        position = self.orientation.inv().apply(position)
        if self.mirror_x:
            position[0] = -position[0]
        if self.mirror_y:
            position[1] = -position[1]
        return position
    
    def transform_to_table_reference_frame(self, position: np.ndarray) -> np.ndarray:
        """ Transforms a position from camera reference frame to table reference frame.
        """
        if self.mirror_x:
            position[0] = -position[0]
        if self.mirror_y:
            position[1] = -position[1]
        position = self.orientation.apply(position)
        position = position + self.position
        return position

class Camera:

    def __init__(self, pose: CameraPose, 
                       calibration: Optional[CameraCalibration] = None, 
                       specs: Optional[CameraSpecs] = None):
        """ Initializes a Camera object.
            Arguments:
                pose: CameraPose object representing the camera's position and orientation.
                calibration: CameraCalibration object representing the camera's calibration.
                specs: CameraSpecs object representing the camera's specifications.
            If calibration is None, it is calculated from specs.
            If both calibration and specs are None, an error is raised.
        """
  
        if specs is None and calibration is None:
            raise ValueError("Both camera specs and calibration cannot be None.")
        
        if calibration is None:
            self.calibration = CameraCalibration.from_camera_specs(specs)
        else:
            self.calibration = calibration

        self.pose = pose
        self.specs = specs

    def __getattr__(self, attr):
        """ If attributes are not found in Camera, they are searched in CameraPose and CameraCalibration.
        """
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif attr in self.pose.__dict__:
            return getattr(self.pose, attr)
        elif attr in self.calibration.__dict__:
            return getattr(self.calibration, attr)
        else:
            raise AttributeError(f"'Camera' object has no attribute '{attr}'")
