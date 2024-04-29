from src.camera import CameraCalibration, Camera, CameraPose
from src.camera.video import VideoStream
from src.pingpong.ball import BallConstants
from src.pingpong.table import TableConstants
from src.fileutils import PROJECT_ROOT

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import yaml
from itertools import islice
from typing import Optional

@dataclass
class Config:

    video_path: Path
    fps: Optional[int]
    analysis_start_frame: int
    analysis_end_frame: int
    setup_frames: int

    _background_image: int | Path | None

    camera_calibration_video_path: Path

    camera_pose: CameraPose
    ball_constants: BallConstants
    table_constants: TableConstants

    @property
    def video(self):
        return VideoStream(self.video_path)
    
    def get_analysis_video_iter(self, video: Optional[VideoStream] = None):
        if video is None:
            video = self.video
        return islice(video, self.analysis_start_frame, self.analysis_end_frame, 1)
    
    @property
    def calibration_video(self):
        return VideoStream(self.camera_calibration_video_path)

    def load_images(self):

        if isinstance(self._background_image, int):
            frames = list(self.video)
            self.background_image = frames[self._background_image]
        elif self._background_image is not None:
            self.background_image = cv2.imread(str(self._background_image))
        else:
            self.background_image = None
        

    def parse_path(self, path: Path) -> Path:
        return PROJECT_ROOT / path


    def __post_init__(self):
        for attr in ['video_path', 'camera_calibration_video_path', '_background_image']:
            if isinstance(getattr(self, attr), str):
                setattr(self, attr, self.parse_path(getattr(self, attr)))

        if self.fps is None:
            self.fps = self.video.fps

    @staticmethod
    def from_yaml(config_file: Path) -> 'Config':
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

            if 'ball' in config:
                ball = BallConstants(**config['ball'])
            else:
                ball = BallConstants()

            if 'table' in config:
                table = TableConstants(**config['table'])
            else:
                table = TableConstants()

            if 'camera' not in config:
                raise ValueError('Camera configuration not found in config file')

            camera_position = np.array(config['camera']['pose']['position'])
            camera_orientation = Rotation.from_euler('xyz', config['camera']['pose']['orientation'], degrees=True)
            mirror_x = config['camera']['pose'].get('mirror_x', False)    
            mirror_y = config['camera']['pose'].get('mirror_x', False)
            camera_pose = CameraPose(camera_position, camera_orientation, mirror_x, mirror_y)

            return Config(
                video_path=config['video']['path'],
                fps=config['video'].get('fps'),
                analysis_start_frame=config['video']['analysis_start_frame'],
                analysis_end_frame=config['video']['analysis_end_frame'],
                setup_frames=config['video']['setup_frames'],
                _background_image=config['images']['background'],
                camera_calibration_video_path=config['camera']['calibration'],
                camera_pose=camera_pose,
                ball_constants=ball,
                table_constants=table
            )





