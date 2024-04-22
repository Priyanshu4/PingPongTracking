from .split_image import ImageSplitter
import numpy as np
from typing import Tuple, Optional
import cv2

class SplitYoloDetector:
    """ Yolo based detector for the ping pong ball.
        Uses an ImageSplitter to split the image into smaller parts and run the Yolo model on each part.
    """

    def __init__(self, yolo_model, image_splitter: ImageSplitter):
        self.yolo_model = yolo_model
        self.yolo_model.set_classes(['person', 'ping pong ball', 'ball', 'sports ball'])
        self.ball_classes = [1, 2, 3]
        self.image_splitter = image_splitter

    def is_ball(self, yolo_cls: int) -> bool:
        return yolo_cls == 1 or yolo_cls == 2 or yolo_cls == 3

    def detect(self, img: np.ndarray, yolo_verbose: bool = False, debug_plots: bool = False) -> Tuple[Optional[list[int]], float]:
        """ Detects the ping pong ball in an image.
            This will split the image into smaller parts and run the Yolo model on each part.
            Then this finds the bounding box with the highest confidence.

            Args:
                img: OpenCV image.
            Returns:
                highest_confidence_box: Bounding box of the detected ping pong ball as list of (x, y, x2, y2) or None if no ball was detected.
                highest_confidence: Confidence of the detected ping pong ball.
        """
        self.image_splitter.split_image(img)
        highest_confidence = 0
        highest_confidence_box = None
        for i, split_image in enumerate(self.image_splitter.split_images):  
            start_y, start_x = self.image_splitter.start_points[i]
            box, confidence = self.detect_in_split(split_image, start_x, start_y, yolo_verbose)

            if box is not None and debug_plots:
                cv2.rectangle(split_image, (box[0]-start_x, box[1]-start_y), (box[2]-start_x, box[3]-start_y), (0, 255, 0), 2)
                cv2.putText(split_image, f'{confidence:.2f}', (box[0]-start_x, box[1]-start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                self.image_splitter.split_images[i] = split_image


            if confidence > highest_confidence:
                highest_confidence_box = box
                highest_confidence = confidence


        if debug_plots:
            self.image_splitter.plot_split_images()

        return highest_confidence_box, highest_confidence
    
    def detect_in_split(self, split_image: np.ndarray, split_image_x: int, split_image_y: int, yolo_verbose: bool = False) -> Tuple[Optional[list[int]], float]:
        """ Detects the ping pong ball in a split image.
            This will run the Yolo model on one split of an image and find the bounding box with the highest confidence.
            Note that the bounding box coordinates are relative to the original image.

            Args:
                split_image: OpenCV image.
                split_image_x: x coordinate of the split image in the original image.
                split_image_y: y coordinate of the split image in the original image.
            Returns:
                highest_confidence_box: Bounding box of the detected ping pong ball as list of (x, y, x2, y2) or None if no ball was detected.
                highest_confidence: Confidence of the detected ping pong ball.
        """
        detections = self.yolo_model.predict(split_image, verbose = yolo_verbose)
        detection = detections[0]

        highest_confidence = 0
        highest_confidence_box = None
        for box in detection.boxes:
            x, y, x2, y2 = box.xyxy[0, 0], box[0].xyxy[0, 1], box.xyxy[0, 2], box[0].xyxy[0, 3]
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)

            if self.is_ball(box.cls) and box.conf[0] > highest_confidence:
                highest_confidence_box = [x + split_image_x, y + split_image_y, x2 + split_image_x, y2 + split_image_y]
                highest_confidence = box.conf[0]
            
        return highest_confidence_box, highest_confidence
    
    @staticmethod
    def draw_bounding_box( frame: np.ndarray, box: list[int], conf: float, color=(0, 255, 0)) -> None:
        """ Draws a bounding box on an image with the confidence score.
            Args:
                frame: OpenCV image.
                box: Bounding box as list of (x, y, x2, y2).
                conf: Confidence score.
                color: Bounding box color.
        """
        x, y, x2, y2 = box
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{conf:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
 