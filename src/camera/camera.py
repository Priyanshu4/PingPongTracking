from dataclasses import dataclass
from typing import Optional
import numpy as np
import yaml

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
        fx = camera_specs.focal_length_mm * camera_specs.width_pixels / camera_specs.sensor_width_mm
        fy = camera_specs.focal_length_mm * camera_specs.height_pixels / camera_specs.sensor_height_mm
        cx = camera_specs.width_pixels / 2
        cy = camera_specs.height_pixels / 2
        return cls(intrinsic_matrix=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))

@dataclass
class Camera:
    specs: CameraSpecs

    # Position is relative to the closest center edge of the table
    position: np.ndarray    

    # Orientation is relative to the upright position where camera faces table from side.
    orientation: np.ndarray
