import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from .camera import CameraCalibration

# Define the dimensions of the chessboard (inner corners)
def find_intrinsic_matrix(image, chessboard_size=(7,7)):
    """
    Takes opencv image and optional chessboardsize of internal corners to return the intrinsic matrix of the camera
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

        # Draw chessboard corners on the image
        image_with_corners = cv.drawChessboardCorners(image.copy(), chessboard_size, corners, ret)
        
        # Display the image with corners using Matplotlib
        plt.imshow(cv.cvtColor(image_with_corners, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Print the intrinsic matrix
        print("Intrinsic Matrix (Camera Matrix):\n", mtx)
        return CameraCalibration(mtx)
    
    print("Chessboard corners not found in the image.")
    return None
