from abc import ABC, abstractmethod
from typing import TypeAlias
from .state import StateVector, StateComponent, StateVectorUtilities
import numpy as np

MeasurementVector: TypeAlias = np.typing.NDArray[np.float64]

class MeasurementMode(ABC):
    """ Parent class for a mode of measurement for the Kalman Filter.
    """

    @abstractmethod
    def __init__(self, z_dim: int, noise_matrix: np.ndarray):
        """ Initializes the measurement mode.
            Arguments:
                z_dim (int): The dimension of the measurement vector.
                noise_matrix (np.ndarray): The measurement noise matrix.
        """
        self.z_dim = z_dim
        self.noise = noise_matrix

        if self.noise.shape != (z_dim, z_dim):
            raise ValueError(f"Noise matrix must be of shape ({z_dim}, {z_dim}) for mode {self.__class__}.")

    def init_measurement_vector(self) -> MeasurementVector:
        """ Initializes a measurement vector with zeros.
        """
        return np.zeros(self.z_dim)

    @abstractmethod
    def hx(self, x: StateVector) -> MeasurementVector:
        """ Measurement function for the mode.
        """
        pass

    def residual(self, a: MeasurementVector, b: MeasurementVector) -> MeasurementVector:
        """ Computes the difference between two measurement vectors a and b.
            This function should ensure that the angular states are wrapped to the range [-pi, pi].
            It differs from normal subtraction (since 359 deg - 1 deg = 2 deg).
            A residual function is required for the Unscented Kalman Filter.

        Args:
            a (MeasurementVector): The first state vector.
            b (MeasurementVector): The second state vector.
        
        Returns:
            residual (MeasurementVector): The residual vector.
        """
        residual = self.init_measurement_vector()
        return a - b

    
    def mean(self, measurement_vectors: list[MeasurementVector], weights: np.array) -> MeasurementVector:
        """ Computes the weighted mean of a series of measurement vectors.
            This function uses arithmetic mean for linear states and circular mean for angular states.
            This mean function is required for the UKF.

        Arguments:
            measurements (list[MeasurementVector]): The list of measurement vectors.
            weights (np.typing.NDArray[np.float64]): The weights for the measurements.
        
        Returns:
            mean (MeasurementVector): The mean of the measurement vectors.
        """
        return np.dot(weights, np.array(measurement_vectors, axis=0))
    

class PositionMeasurementMode(MeasurementMode):
    """ Measurement mode when we only measure position of the ball.
    """

    def __init__(self, noise_matrix: np.ndarray):
        """ Initializes the measurement mode.
            Arguments:
                noise_matrix (np.ndarray): 3 by 3 measurement noise matrix.
        """
        super().__init__(z_dim=3, noise_matrix=noise_matrix)

    def hx(self, x: StateVector) -> MeasurementVector:
        """ Measurement function for the position mode.
        """
        return x[StateComponent.X:StateComponent.Z+1]
    
class OrientationMeasurementMode(MeasurementMode):
    """ Measurement mode when we only measure orientation of the ball.
    """

    def __init__(self, noise_matrix: np.ndarray):
        """ Initializes the measurement mode.
            Arguments:
                noise_matrix (np.ndarray): 3 by 3 measurement noise matrix.
        """
        super().__init__(z_dim=3, noise_matrix=noise_matrix)

    def hx(self, x: StateVector) -> MeasurementVector:
        """ Measurement function for the orientation mode.
        """
        return x[StateComponent.R_X:StateComponent.R_Z+1]
    
    def mean(self, measurement_vectors: list[MeasurementVector], weights: np.array) -> MeasurementVector:
        return StateVectorUtilities.angular_vector_means(measurement_vectors, weights)
    
    def residual(self, a: MeasurementVector, b: MeasurementVector) -> MeasurementVector:
        residual = StateVectorUtilities.angular_residuals(a, b)
        return residual
    
class PoseMeasurementMode(MeasurementMode):
    """ Measurement mode when we measure both position and orientation of the ball.
    """

    def __init__(self, noise_matrix: np.ndarray):
        """ Initializes the measurement mode.
            Arguments:
                noise_matrix (np.ndarray): 6 by 6 measurement noise matrix.
        """
        super().__init__(z_dim=6, noise_matrix=noise_matrix)

    def hx(self, x: StateVector) -> MeasurementVector:
        """ Measurement function for the pose mode.
        """
        return x[StateComponent.X:StateComponent.R_Z+1]
    
    def mean(self, measurement_vectors: list[MeasurementVector], weights: np.array) -> MeasurementVector:
        measurement_vectors = np.array(measurement_vectors)
        angular_vectors = measurement_vectors[:, StateComponent.R_X:]
        linear_vectors = measurement_vectors[:, StateComponent.X:StateComponent.Z+1]
        linear_avg = np.dot(weights, linear_vectors)
        angular_avg = StateVectorUtilities.angular_vector_means(angular_vectors, weights)
        avg = np.hstack((linear_avg, angular_avg))
        return avg

    def residual(self, a: MeasurementVector, b: MeasurementVector) -> MeasurementVector:
        residual = self.init_measurement_vector()
        residual[StateComponent.X:StateComponent.Z+1] = a[StateComponent.X:StateComponent.Z+1] - b[StateComponent.X:StateComponent.Z+1]
        residual[StateComponent.R_X:] = StateVectorUtilities.angular_residuals(a[StateComponent.R_X:], b[StateComponent.R_X:])
        return residual  



