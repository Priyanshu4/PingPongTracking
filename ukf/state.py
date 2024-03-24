import numpy as np
from typing import Annotated, Literal, TypeAlias
from enum import IntEnum

StateVector: TypeAlias = Annotated[np.typing.NDArray[np.float64], Literal[12]]
MeasurementVector: TypeAlias = Annotated[np.typing.NDArray[np.float64], Literal[6]]

class StateComponent(IntEnum):
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


def wrap_angles(x: StateVector) -> StateVector:
    """ Wraps the angular states of the state vector x to the range [0, 2*pi].

    Args:
        x (StateVector): The state vector to wrap.
    
    Returns:
        x (StateVector): The state vector with the angular states wrapped.
    """
    x[StateComponent.AngularStates] = x[StateComponent.AngularStates] % (2 * np.pi)
    return x

def residual(a: StateVector, b: StateVector) -> StateVector:
    """ Computes the difference between two state vectors a and b.
        This function ensures that the angular states are wrapped to the range [0, 2*pi].
        A residual function is required for the Unscented Kalman Filter.

    Args:
        a (StateVector): The first state vector.
        b (StateVector): The second state vector.
    
    Returns:
        residual (StateVector): The residual vector.
    """
    return wrap_angles(a - b)
    

