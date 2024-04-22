from typing import Tuple
import numpy as np
from math import ceil
from matplotlib import pyplot as plt

class ImageSplitter:

    def __init__(self, split_width: int, split_height: int, overlap: float = 0):
        """
        Args:
            split_width: width of the split images
            split_height: height of the split images
            overlap: portion of overlap between the split images
        """
        self.split_width = split_width
        self.split_height = split_height
        self.overlap = overlap
        self.split_images = None
        self.start_points = None
        self.images_per_col = 0
        self.images_per_row = 0

    
    def split_image(self, image) -> Tuple[list[np.ndarray], list[Tuple[int, int]]]:
        """
        Splits an image into multiple images with overlap
        
        Args:
            image: image to split

        Returns:
            split_images: list of split images
            start_points: list of start indices of the split images

        The relevant values are stored in the following attributes until the next call:
            split_images: list of split images
            start_points: list of start indices of the split images
            images_per_col: number of split images per column
            images_per_row: number of split images per row
        """
        self.image = image
        split_images = []
        start_points = []
        height, width = self.image.shape[:2]
        overlap_height = int(self.split_height * self.overlap)
        overlap_width = int(self.split_width * self.overlap)
        for i in range(0, height, self.split_height - overlap_height):
            for j in range(0, width, self.split_width - overlap_width):
                if i + self.split_height > height:
                    i = height - self.split_height
                if j + self.split_width > width:
                    j = width - self.split_width
                split_images.append(self.image[i:i + self.split_height, j:j + self.split_width].copy())
                start_points.append((i, j))

        self.split_images = split_images
        self.start_points = start_points
        self.images_per_col = ceil(height / (self.split_height - overlap_height))
        self.images_per_row = ceil(width / (self.split_width - overlap_width))

        return split_images, start_points
    
    def plot_split_images(self):
        """
        Plots the split images on a new figure.
        You may need to call plt.show() to display the figure.
        """
        fig, axes = plt.subplots(self.images_per_col, self.images_per_row, figsize=(15, 15))
        for i, ax in enumerate(axes.flat):
            ax.imshow(self.split_images[i])
            ax.axis('off')


def crop_image(image, lower: np.array, upper: np.array):
    """
    Crops an image to a bounding box defined by the lower and upper bounds.

    Args:
        image (np.ndarray): The image to crop.
        lower (np.array): The lower bound of the bounding box.
        upper (np.array): The upper bound of the bounding box.

    Returns:
        np.ndarray: The cropped image.
    """
    low_x = max(0, lower[0])
    low_y = max(0, lower[1])
    up_x = min(image.shape[1], upper[0])
    up_y = min(image.shape[0], upper[1])
    return image[low_y:up_y, low_x:up_x]