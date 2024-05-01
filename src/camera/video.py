import cv2
from pathlib import Path
from typing import Optional

class VideoStream:
    """
    A class to read video frames from a video file or camera.
    Can be used as an iterator. 

    Example
    ---------
    video = VideoStream('video.mp4')

    for frame in video:
        do_something(frame)
    """

    def __init__(self, source: str | Path | int, frames_to_read: Optional[int] = None):
        self.source = source
        self.frames_to_read = frames_to_read
        self.frame_counter = 0
        if isinstance(source, Path):
            self.cap = cv2.VideoCapture(str(source))
        else:
            self.cap = cv2.VideoCapture(source)

        self._try_read_metadata()

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.frames_to_read is not None and self.frame_counter == self.frames_to_read:
            raise StopIteration
        
        ret, frame = self.cap.read()
        if not ret:
            if self.frame_counter == 0:
                raise ValueError(f"Video {self.source} could not be read.")
            raise StopIteration
        self.frame_counter += 1

        return frame

    def __del__(self):
        self.cap.release()

    def _try_read_metadata(self):
        """
        Try to read basic video metadata. 
        """
        try:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        except Exception as e:
            self.fps = None

        try:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception as e:
            self.width = None
            self.height = None

        try:
            self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            self.frames = None

    def get_length_if_known(self) -> Optional[int]:
        """
        Get the total number of frames in the VideoStream, if known.
        This is known if the source is a file and the metadata can be read or if self.frames_to_read is set.
        If self.frames_to_read is less than the number of frames in the video, it will be returned.
        """
        if self.frames is not None:

            if self.frames_to_read is not None:
                return min(self.frames, self.frames_to_read)
            else:
                return self.frames
        
        return self.frames_to_read
    
    def __repr__(self) -> str:
        unknown = "unknown"
        s = f"VideoStream:\n"
        s += f"\tSource: {self.source}\n" 
        s += f"\tFPS: {self.fps or unknown}\n"
        s += f"\tFrames: {self.frames or unknown}\n"

        if self.width is not None and self.height is not None:
            s += f"\tResolution: {self.width}x{self.height}\n"
        else:
            s += f"\tResolution: unknown\n"

        if self.frames_to_read is not None:
            s += f"\tFrames to read: {self.frames_to_read}\n"
        
        return s
        
        