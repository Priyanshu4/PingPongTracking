from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from .state import StateVector, StateComponent, StateVectorUtilities
from .measurement import MeasurementMode, MeasurementVector
from src.pingpong.ball import BallConstants
import numpy as np
from functools import partial

class BallUKF:
    """ Unscented Kalman Filter for the ping pong ball.
    """

    def __init__(self, ball_constants: BallConstants, 
                 initial_state: StateVector, initial_state_covariance: np.ndarray, 
                 process_noise: np.ndarray, measurement_mode: MeasurementMode):
        """ Initializes the UKF for the ball.
        """
        self.ball_constants = ball_constants
        self.sigma_points = MerweScaledSigmaPoints(n=12, alpha=0.1, beta=2.0, kappa=0.0, subtract=StateVectorUtilities.residual)
        self.ukf = UKF(dim_x=12, 
                dim_z=6, 
                fx=partial(self.fx, ball_constants=ball_constants), 
                points=self.sigma_points,
                x_mean_fn=StateVectorUtilities.mean, 
                residual_x=StateVectorUtilities.residual)
            
        self.state = initial_state                          # Initialize the state
        self.state_covariance = initial_state_covariance    # Initialize the state covariance matrix
        self.process_noise = process_noise                  # Initialize the process noise matrix

        self.set_measurement_mode(measurement_mode)         # Set the measurement mode for the UKF  
        
        self.ukf.predict(0)

    def set_measurement_mode(self, measurement_mode: MeasurementMode):
        """ Sets the measurement mode for the UKF.
        """
        self.ukf.z_dim = measurement_mode.z_dim
        self.ukf.hx = measurement_mode.hx
        self.ukf.z_mean = measurement_mode.mean
        self.ukf.measurement_noise = measurement_mode.noise
        self.ukf.residual_z = measurement_mode.residual

    def predict(self, dt: float):
        """ 
        Predicts the next state of the ball.
        Should be called prior to each update step.

        Args:
            dt (float): The time step.
        """
        self.ukf.predict(dt)

    def update(self, z: MeasurementVector, R: np.ndarray = None):
        """
        Updates the state of the ball using the measurement z.
        If the measurement noise R is provided, it overrides the default measurement noise for this call only.

        Args:
            z (MeasurementVector): The measurement vector.
            R (np.ndarray): The measurement noise matrix.
        """
        if R is not None:
            self.ukf.update(z, R)
        else:
            self.ukf.update(z)

    def fx(self, x: StateVector, dt: float) -> StateVector:
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
        k1 = self.ball_constants.air_drag_factor
        k2 = self.ball_constants.magnus_factor
        g = self.ball_constants.gravity

        mag_v = np.linalg.norm(x[StateComponent.V_X:StateComponent.V_Z+1])
        drag = -k1 * mag_v * x[StateComponent.V_X:StateComponent.V_Z+1]
        
        w_cross_v = np.cross(x[StateComponent.W_X:StateComponent.W_Z+1], x[StateComponent.V_X:StateComponent.V_Z+1])
        magnus = k2 * w_cross_v

        gravity = np.array([0, 0, -g])

        x_new[StateComponent.V_X:StateComponent.V_Z+1] += (drag + magnus + gravity) * dt
        x_new = StateVectorUtilities.wrap_angles(x_new)
        return x_new

    @property
    def state(self) -> StateVector:
        return self.ukf.x
    
    @property
    def state_covariance(self) -> np.ndarray:
        return self.ukf.P

    @property
    def measurement_noise(self) -> np.ndarray:
        return self.ukf.R
    
    @property
    def process_noise(self) -> np.ndarray:
        return self.ukf.Q
    
