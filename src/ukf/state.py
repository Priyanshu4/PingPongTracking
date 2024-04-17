import numpy as np
from typing import Annotated, Literal, TypeAlias

StateVector: TypeAlias = Annotated[np.typing.NDArray[np.float64], Literal[12]]

class StateComponent:
    # Position
    X = 0
    Y = 1
    Z = 2

    # Rotation
    R_X = 3
    R_Y = 4
    R_Z = 5

    # Velocity
    V_X = 6
    V_Y = 7
    V_Z = 8

    # Angular Velocity
    W_X = 9
    W_Y = 10
    W_Z = 11

    LinearStates = [X, Y, Z, V_X, V_Y, V_Z]
    AngularStates = [R_X, R_Y, R_Z, W_X, W_Y, W_Z]

class StateVectorUtilities:

    @staticmethod
    def init_state_vector() -> StateVector:
        """ Initializes a state vector with zeros.
        """
        return np.zeros(12)

    @staticmethod
    def wrap_angular_vector(x: np.array) -> np.array:
        """ Wraps a vectors of angles to the range [-pi, pi].

        Args:
            x: The array of angles.
            
        Returns:
            x: The vector with the angular states wrapped.
        """
        x = (x + np.pi) % (2 * np.pi) - np.pi
        return x

    @staticmethod
    def wrap_angles(x: StateVector) -> StateVector:
        """ Wraps the angular states of the state vector x to the range [-pi, pi].

        Args:
            x (StateVector): The state vector to wrap.
        
        Returns:
            x (StateVector): The state vector with the angular states wrapped.
        """
        x[StateComponent.AngularStates] = StateVectorUtilities.wrap_angular_vector(x[StateComponent.AngularStates])
        return x

    @staticmethod
    def angular_mean(angles: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        """ Computes the circular mean of a series of angles.
            The angles should be in range [-pi, pi].

        Args:
            angles (np.typing.NDArray[np.float64]): The array of angles.
                   Could also be an array of vector of angles.

        Returns:
            mean (float): The circular mean of the angles.
        """
        sin_sum = np.sum(np.sin(angles), axis=0)
        cos_sum = np.sum(np.cos(angles), axis=0)
        return np.arctan2(sin_sum, cos_sum)
    
    @staticmethod
    def angular_residuals(x, y) -> np.typing.NDArray[np.float64]:
        """ Computes the angular residuals of two vectors of angles x and y.
            The angles should be in range [-pi, pi].

        Args:
            x (np.typing.NDArray[np.float64]): The first vector of angles.
            y (np.typing.NDArray[np.float64]): The second vector of angles.

        Returns:
            residuals (np.typing.NDArray[np.float64]): The angular residuals of the angles.
        """
        return np.arctan2(np.sin(x - y), np.cos(x - y))

    @staticmethod
    def sum(states: list[StateVector]) -> StateVector:
        """ Computes the sum of a list of state vectors.

        Args:
            states (list[StateVector]): The list of state vectors to sum.
        
        Returns:
            sum (StateVector): The sum of the state vectors.
        """
        return StateVectorUtilities.wrap_angles(np.sum(states, axis=0))
          
    @staticmethod
    def residual(a: StateVector, b: StateVector) -> StateVector:
        """ Computes the difference between two state vectors a and b.
            This function ensures that the angular states are wrapped to the range [-pi, pi].
            It differs from normal subtraction (since 359 deg - 1 deg = 2 deg).
            A residual function is required for the Unscented Kalman Filter.

        Args:
            a (StateVector): The first state vector.
            b (StateVector): The second state vector.
        
        Returns:
            residual (StateVector): The residual vector.
        """
        residual = StateVectorUtilities.init_state_vector()
        residual[StateComponent.LinearStates] = a[StateComponent.LinearStates] - b[StateComponent.LinearStates]
        residual[StateComponent.AngularStates] = StateVectorUtilities.angular_residuals(a[StateComponent.AngularStates], b[StateComponent.AngularStates])
        return residual

    @staticmethod
    def mean(states: list[StateVector]) -> StateVector:
        """ Computes the mean of a series of state vectors.
            This function uses arithmetic mean for linear states and circular mean for angular states.
            This mean function is required for the UKF.

        Arguments:
            states (list[StateVector]): The list of state vectors.
        
        Returns:
            mean (StateVector): The mean of the state vectors.
        """
        mean = StateVectorUtilities.init_state_vector()
        mean[StateComponent.LinearStates] = np.mean([state[StateComponent.LinearStates] for state in states], axis=0)
        mean[StateComponent.AngularStates] = StateVectorUtilities.angular_mean([state[StateComponent.AngularStates] for state in states])
        return mean

