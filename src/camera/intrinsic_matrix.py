import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import Optional
from tqdm import tqdm

def find_single_intrinsic_matrix(image, chessboard_size=(7,7), show_image = False) -> Optional[np.array]:
    """ 
    Finds the intrinsic matrix of the camera using a single OpenCV image of a chessboard pattern.
    The chessboard size is defined by the number of internal corners in each row and column.
    It should one less than the actual number of squares in each row and column, as we want to use inner corners.
    
    Args:
        image (np.array): The OpenCV image of the chessboard pattern.
        chessboard_size (tuple): The number of internal corners in each row and column.
        show_image (bool): Whether to display the image with the chessboard corners.

    Returns:
        np.array: The intrinsic matrix of the camera.
    """

    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

    # If corners are found, add object points and image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        if show_image:
            image_with_corners = cv.drawChessboardCorners(image.copy(), chessboard_size, corners, ret)
            plt.imshow(cv.cvtColor(image_with_corners, cv.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return mtx 
    
    print("Chessboard corners not found in the image.")
    return None

def find_average_intrinsic_matrix(images, chessboard_size = (7,7)) -> Optional[np.array]:
    """
    Finds the average intrinsic matrix of the camera using multiple OpenCV images of a chessboard pattern.
    The chessboard size is defined by the number of internal corners in each row and column.
    It should one less than the actual number of squares in each row and column, as we want to use inner corners.

    Args:
        images (list): A list of OpenCV images of the chessboard pattern.
        chessboard_size (tuple): The number of internal corners in each row and column.

    Returns:
        np.array: The average intrinsic matrix of the camera.
    """
    matrices = list()
    for image in images:
        mtx = find_single_intrinsic_matrix(image, chessboard_size, show_image = False)
        if mtx is not None:
            matrices.append(mtx)
    matrices_np = np.array(matrices)
    if len(matrices_np) == 0:
        return None
    return np.mean(matrices_np, axis=0)
