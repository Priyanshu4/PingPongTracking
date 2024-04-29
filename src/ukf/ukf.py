from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints, unscented_transform
from .state import StateVector, StateComponent, StateVectorUtilities
from .measurement import MeasurementMode, MeasurementVector
from src.pingpong.ball import BallConstants
import numpy as np
import scipy.linalg
from typing import Tuple

class BallUKF:
    """ Unscented Kalman Filter for the ping pong ball.
    """

    def __init__(self, ball: BallConstants, 
                 initial_state: StateVector, initial_state_covariance: np.ndarray, 
                 process_noise: np.ndarray, measurement_mode: MeasurementMode, default_dt: float = 0.01):
        """ Initializes the UKF for the ball.
        """
        self.state_dim = 12
        self.ball_constants = ball
        self.sigma_points = MerweScaledSigmaPoints(
            n=self.state_dim, 
            alpha=0.1, 
            beta=2.0, 
            kappa=3-self.state_dim, 
            sqrt_method=self._sqrt_func,
            subtract=StateVectorUtilities.residual)
        self.ukf = UKF(
            dim_x=self.state_dim,
            dim_z=measurement_mode.z_dim,
            fx=self.fx, 
            dt=default_dt,
            hx=measurement_mode.hx,
            points=self.sigma_points,
            x_mean_fn=StateVectorUtilities.mean, 
            residual_x=StateVectorUtilities.residual)
            
        self.state = initial_state                          # Initialize the state
        self.state_covariance = initial_state_covariance    # Initialize the state covariance matrix
        self.intial_state_covariance = initial_state_covariance
        self.process_noise = process_noise                  # Initialize the process noise matrix

        self.set_measurement_mode(measurement_mode)         # Set the measurement mode for the UKF  
        
    def _sqrt_func(self, x):
        try:
            result = scipy.linalg.cholesky(x)
        except scipy.linalg.LinAlgError as e:

            try:
                x = (x + x.T)/2 # Force the matrix to be symmetric
                result = scipy.linalg.cholesky(x)
            except:

                try:
                    x  = x + np.eye(x.shape[0]) * 1e-6
                    result = scipy.linalg.cholesky(x)
                except:
                    x = self.intial_state_covariance
                    print("WARNING: Resetting the state covariance matrix to the initial value due to negative eigenvalues.")
                    result = scipy.linalg.cholesky(x)
        return result

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
    
    def get_high_likelihood_measurement_region(self, n_std_devs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Estimates a bounding box around the mean of the measurement distribution.
            The bounding box is a region that includes n standard deviations around the mean.
            It is defined by two points, the lower and upper bounds of the region.
            The mean and covariance of the transformed points are also returned.

        Args:
            n_std_devs (float): The number of standard deviations to include in the region.

        Returns:
            lower_bound (np.array): The lower bound of the bounding box.
            upper_bound (np.array): The upper bound of the bounding box.
            mu_transformed (np.array): The mean of the transformed points.
            cov_transformed (np.ndarray): The covariance of the transformed points.
        """
        sigmas = self.sigma_points.sigma_points(self.state, self.state_covariance)

        # Transform sigma points through the non-linear measurement function
        transformed_sigmas = np.array([self.ukf.hx(sigma) for sigma in sigmas])

        # Calculate the mean and covariance of the transformed points
        mu_transformed, cov_transformed = unscented_transform(
                                            transformed_sigmas, 
                                            self.sigma_points.Wm, 
                                            self.sigma_points.Wc,
                                            mean_fn=self.ukf.z_mean)
        
        # Estimating a bounding box (95% confidence interval) around the mean
        std_dev = np.sqrt(np.diag(cov_transformed))

        lower = np.array(mu_transformed - n_std_devs * std_dev)
        upper = np.array(mu_transformed + n_std_devs * std_dev)

        return lower, upper, mu_transformed, cov_transformed

    @property
    def state(self) -> StateVector:
        return self.ukf.x
    
    @state.setter
    def state(self, state: StateVector):
        self.ukf.x = state

    @property
    def measurement_dim(self) -> int:
        return self.ukf.z_dim
    
    @property
    def state_covariance(self) -> np.ndarray:
        return self.ukf.P
    
    @state_covariance.setter
    def state_covariance(self, state_covariance: np.ndarray):
        self._check_positive_definite(state_covariance)
        self.ukf.P = state_covariance

    @property
    def measurement_noise(self) -> np.ndarray:
        return self.ukf.R
    
    @property
    def process_noise(self) -> np.ndarray:
        return self.ukf.Q
    
    @process_noise.setter
    def process_noise(self, process_noise: np.ndarray):
        self._check_positive_definite(process_noise)
        self.ukf.Q = process_noise
    
    def _check_positive_definite(self, m):
        """ Raises an exception if the matrix is not positive definite.
        """
        scipy.linalg.cholesky(m)

    
