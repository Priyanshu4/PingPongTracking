import numpy as np

def velocity_regression(times: np.array, positions: np.array) -> np.array:
    """
    Computes the velocity vector for a series of positions at given times using linear regression.

    Args:
    times (numpy.ndarray): An array of time points.
    positions (numpy.ndarray): An n x 3 array where each row contains the x, y, z positions at each time point.

    Returns:
    numpy.ndarray: A 1 x 3 array containing the velocity components in the x, y, and z directions.
    """ 
    # Initialize an array to store the velocity components
    velocity_components = np.zeros(3)
    
    # Perform linear regression for each dimension (x, y, z)
    for i in range(3):
        # Solve the least squares problem
        result, residuals, rank, s = np.linalg.lstsq(times, positions[:, i], rcond=None)
        velocity_components[i] = result[0]  # Store the velocity component (slope)
    
    return velocity_components