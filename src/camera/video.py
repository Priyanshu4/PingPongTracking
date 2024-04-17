import cv2
from pathlib import Path

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

    def __init__(self, source: str | Path | int):
        self.source = source
        if isinstance(source, Path):
            self.cap = cv2.VideoCapture(str(source))
        else:
            self.cap = cv2.VideoCapture(source)

    def __iter__(self):
        return self
    
    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        return frame
    
    def __del__(self):
        self.cap.release()