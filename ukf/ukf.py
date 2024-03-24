from filterpy.kalman import UnscentedKalmanFilter as UKF
from state import StateVector, MeasurementVector, StateComponent, wrap_angles, residual
from ballconstants import BallConstants, DefaultConstants
import numpy as np

def fx(x: StateVector, dt: float, ball_constants: BallConstants = DefaultConstants) -> StateVector:
    """ State transition function for the ball. 
        Adapted from https://ieeexplore.ieee.org/abstract/document/6917514
    """

    # Transition matrix that does not include external forces
    transition_matrix = np.array([
        [1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        )
    
    x_new = np.dot(transition_matrix, x)
    
    # Velocity adjustments for drag forces, magnus forces and gravity
    k1 = ball_constants.air_drag_factor
    k2 = ball_constants.magnus_factor
    g = ball_constants.gravity

    mag_v = np.linalg.norm(x[StateComponent.V_X:StateComponent.V_Z+1])
    drag = -k1 * mag_v * x[StateComponent.V_X:StateComponent.V_Z+1]
    
    w_cross_v = np.cross(x[StateComponent.W_X:StateComponent.W_Z+1], x[StateComponent.V_X:StateComponent.V_Z+1])
    magnus = k2 * w_cross_v

    gravity = np.array([0, 0, -g])

    x_new[StateComponent.V_X:StateComponent.V_Z+1] += (drag + magnus + gravity) * dt
    x_new = wrap_angles(x_new)
    return x_new

  
def hx(x: StateVector) -> MeasurementVector:
    """ Measurement function for UKF.
        Defines the measurement in terms of the state.
    """
    return x[StateComponent.X:StateComponent.R_Z+1]