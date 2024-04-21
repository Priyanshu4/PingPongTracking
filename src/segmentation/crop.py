import numpy as np

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