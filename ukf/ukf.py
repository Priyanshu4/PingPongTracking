from filterpy.kalman import UnscentedKalmanFilter as UKF
from state import StateVector, MeasurementVector, StateComponent, StateVectorUtilities, MeasurementVectorUtilities
from ballconstants import BallConstants
import numpy as np
from functools import partial

def fx(x: StateVector, dt: float, ball_constants: BallConstants) -> StateVector:
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
    x_new = StateVectorUtilities.wrap_angles(x_new)
    return x_new

  
def hx(x: StateVector) -> MeasurementVector:
    """ Measurement function for UKF.
        Defines the measurement in terms of the state.
    """
    return x[StateComponent.X:StateComponent.R_Z+1]

def init_UKF(ball_constants: BallConstants) -> UKF:
    """ Initializes the Unscented Kalman Filter with the state transition and measurement functions.
    """
    points = None
    ukf = UKF(dim_x=12, 
              dim_z=6, 
              fx=partial(fx, ball_constants=ball_constants), 
              hx=hx, 
              dt=0.01, 
              points=points, 
              x_mean_fn=StateVectorUtilities.mean, 
              z_mean_fn=MeasurementVectorUtilities.mean, 
              residual_x=StateVectorUtilities.residual, 
              residual_z=MeasurementVectorUtilities.residual)
    pass