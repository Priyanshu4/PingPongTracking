from .split_image import ImageSplitter
import numpy as np
from typing import Tuple, Optional
import cv2

class SplitYoloDetector:
    """ Yolo based detector for the ping pong ball.
        Uses an ImageSplitter to split the image into smaller parts and run the Yolo model on each part.
    """

    def __init__(self, yolo_model, image_splitter: ImageSplitter, ball_classes: set, 
                 fixed_background: np.ndarray = None, background_diff_threshold: float = 0):
        """
        Args:
            yolo_model: Yolo model to use for detection.
            image_splitter: ImageSplitter to split the image into smaller parts.
            ball_classes: set of classes that are considered as ping pong balls.
            fixed_background: Fixed background image. Splits are only processed if the difference to the background is above a certain threshold.
            background_diff_threshold: Threshold (0-1) for the difference between the split image and the background image.
        """
        self.yolo_model = yolo_model
        self.ball_classes = ball_classes
        self.image_splitter = image_splitter
        self.fixed_background = fixed_background
        self.background_diff_threshold = background_diff_threshold

    def is_ball(self, yolo_cls: int) -> bool:
        return yolo_cls in self.ball_classes
    
    def _is_above_background_diff_threshold(self, split_image: np.ndarray, start_x: int, start_y: int) -> bool:
        if self.fixed_background is None:
            return True
        
        background_diff = np.abs(self.fixed_background[start_y:start_y + self.image_splitter.split_height, start_x:start_x + self.image_splitter.split_width] - split_image)
        return np.mean(np.abs(background_diff)) >= self.background_diff_threshold * 255

    def detect(self, img: np.ndarray, yolo_verbose: bool = False, debug_plots: bool = False) -> Tuple[Optional[list[int]], float]:
        """ Detects the ping pong ball in an image.
            This will split the image into smaller parts and run the Yolo model on each part.
            Then this finds the bounding box with the highest confidence.

            Args:
                img: OpenCV image.
                background_diff: Raw array difference between this image and the background image.
            Returns:
                highest_confidence_box: Bounding box of the detected ping pong ball as list of (x, y, x2, y2) or None if no ball was detected.
                highest_confidence: Confidence of the detected ping pong ball.
        """
        self.image_splitter.split_image(img)
        highest_confidence = 0
        highest_confidence_box = None
        for i, split_image in enumerate(self.image_splitter.split_images):  
            start_y, start_x = self.image_splitter.start_points[i]

            if not self._is_above_background_diff_threshold(split_image, start_x, start_y):
                if debug_plots:
                    self.draw_x_on_split(i, color=(255, 0, 0))
                if yolo_verbose:
                    print(f'Split {i} is below background diff threshold')
                continue

            box, confidence = self.detect_in_split(split_image, start_x, start_y, yolo_verbose)

            if box is not None and debug_plots:
                self.draw_bounding_box(split_image, [box[0] - start_x, box[1] - start_y, box[2] - start_x, box[3] - start_y], confidence)
                self.image_splitter.split_images[i] = split_image

            if confidence > highest_confidence:
                highest_confidence_box = box
                highest_confidence = confidence

        if debug_plots:
            self.image_splitter.plot_split_images()

        return highest_confidence_box, highest_confidence
    
    def detect_in_subregion(self, img: np.ndarray, 
                            min_x: int, min_y: int, max_x: int, max_y: int,
                            yolo_verbose: bool = False, debug_plots: bool = False) -> Tuple[Optional[list[int]], float]:
        """ Detects the ping pong ball in a subregion of the image.
            This will split the image into smaller parts and run the Yolo model on each part.
            Then this finds the bounding box with the highest confidence.

            Args:
                img: OpenCV image.
                min_x: lower x coordinate of the subregion.
                min_y: lower y coordinate of the subregion.
                max_x: upper x coordinate of the subregion.
                max_y: upper y coordinate of the subregion.
                
            Returns:
                highest_confidence_box: Bounding box of the detected ping pong ball as list of (x, y, x2, y2) or None if no ball was detected.
                highest_confidence: Confidence of the detected ping pong ball.
        """
        self.image_splitter.split_image(img)
        split_indices = set(self.image_splitter.get_splits_in_subregion(min_x, min_y, max_x, max_y))


        highest_confidence = 0
        highest_confidence_box = None

        for i, split_image in enumerate(self.image_splitter.split_images):  
            start_y, start_x = self.image_splitter.start_points[i]

            if i not in split_indices:
                if debug_plots:
                    self.draw_x_on_split(i, color=(0, 0, 255))
                continue

            if not self._is_above_background_diff_threshold(split_image, start_x, start_y):
                if debug_plots:
                    self.draw_x_on_split(i, color=(255, 0, 0))
                if yolo_verbose:
                    print(f'Split {i} is below background diff threshold')
                continue

            box, confidence = self.detect_in_split(split_image, start_x, start_y, yolo_verbose)

            box, confidence = self.detect_in_split(split_image, start_x, start_y, yolo_verbose)

            if box is not None and debug_plots:
                self.draw_bounding_box(split_image, [box[0] - start_x, box[1] - start_y, box[2] - start_x, box[3] - start_y], confidence)
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

            if self.is_ball(box.cls.item()) and box.conf[0] > highest_confidence:
                highest_confidence_box = [x + split_image_x, y + split_image_y, x2 + split_image_x, y2 + split_image_y]
                highest_confidence = box.conf[0]

        if highest_confidence_box is not None:
            highest_confidence = highest_confidence.item()

        return highest_confidence_box, highest_confidence
    
    def draw_x_on_split(self, split_index: int, color = (0, 0, 255)):
        """
        Draws a red x on the split image at the given index.
        """
        split_image = self.image_splitter.split_images[split_index]
        width = self.image_splitter.split_width
        height = self.image_splitter.split_height
        cv2.line(split_image, (0, 0), (width, height), color, 2)
        cv2.line(split_image, (width, 0), (0, height), color, 2)
        self.image_splitter.split_images[split_index] = split_image
    
    @staticmethod
    def draw_bounding_box(frame: np.ndarray, box: list[int], conf: float, color=(0, 255, 0)) -> np.ndarray:
        """ Draws a bounding box on an image with the confidence score.
            Args:
                frame: OpenCV image.
                box: Bounding box as list of (x, y, x2, y2).
                conf: Confidence score.
                color: Bounding box color in BGR.
        """
        x, y, x2, y2 = box
        frame = cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        frame = cv2.putText(frame, f'{conf:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame
 