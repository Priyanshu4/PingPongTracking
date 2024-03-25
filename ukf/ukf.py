from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from state import StateVector, MeasurementVector, StateComponent, StateVectorUtilities, MeasurementVectorUtilities
from ballconstants import BallConstants
import numpy as np
from functools import partial

def fx(x: StateVector, dt: float, ball_constants: BallConstants) -> StateVector:
    """ State transition function for the ball. 
        Adapted from https://ieeexplore.ieee.org/abstract/document/6917514

        Arguments:
            x (StateVector): The state vector.
            dt (float): The time step.
            ball_constants (BallConstants): The constants of the ball.

        Returns:
            x_new (StateVector): The new state vector.
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

        Arguments:
            x (StateVector): The state vector.
        
        Returns:
            z (MeasurementVector): The measurement vector.
    """
    return x[StateComponent.X:StateComponent.R_Z+1]

def init_UKF(ball_constants: BallConstants, dt: float, initial_state: StateVector, initial_state_covariance, measurement_noise, process_noise) -> UKF:
    """ Initializes the Unscented Kalman Filter with the state transition and measurement functions.
    """
    sigma_points = MerweScaledSigmaPoints(n=12, alpha=0.1, beta=2.0, kappa=0.0, subtract=StateVectorUtilities.residual)
    ukf = UKF(dim_x=12, 
              dim_z=6, 
              fx=partial(fx, ball_constants=ball_constants), 
              hx=hx, 
              dt=dt, 
              points=sigma_points,
              x_mean_fn=StateVectorUtilities.mean, 
              z_mean_fn=MeasurementVectorUtilities.mean, 
              residual_x=StateVectorUtilities.residual, 
              residual_z=MeasurementVectorUtilities.residual)
    
    ukf.x = initial_state               # Initialize the state
    ukf.P = initial_state_covariance    # Initialize the state covariance matrix
    ukf.R = measurement_noise           # Initialize the measurement noise covariance matrix
    ukf.Q = process_noise               # Initialize the process noise covariance matrix

    return ukf