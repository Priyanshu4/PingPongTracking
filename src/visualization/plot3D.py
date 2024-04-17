from src.pingpong.table import TableConstants
from src.pingpong.ball import BallConstants
from src.ukf.state import StateVector, StateComponent
from src.camera import Camera, CameraPose

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_table(ax: Axes3D, table: TableConstants, table_color='blue', net_color='grey', set_limits=True):
    """ Given an Axes3D, this plots the table and net onto it.
        If set_limits is True, the limits of the plot are set based on the table's length.
    
        Arguments:
            ax (Axes3D): The 3D axes.
            table (TableConstants): The table constants.
            table_color: Matplotlib color for the table.
            net_color: Matplotlib color for the net.
            set_limits: Whether to set the limits of the plot.
    """

    # Plot the table centered at 0, 0
    table_mid_x = table.length / 2
    table_mid_y = table.width / 2
    corners = [
        [-table_mid_x, -table_mid_y, table.height],
        [-table_mid_x, table_mid_y, table.height],
        [table_mid_x, table_mid_y, table.height],
        [table_mid_x, -table_mid_y, table.height],
    ]
    table_poly = Poly3DCollection([corners], facecolors=table_color, edgecolors='black', alpha=0.9)
    ax.add_collection3d(table_poly)
    
    # Plot the net
    net_corners = [
        [0, -table.net_width/2, table.height],
        [0, -table.net_width/2, table.height + table.net_height],
        [0, table.net_width/2, table.height + table.net_height],
        [0, table.net_width/2, table.height],
    ]

    net_poly = Poly3DCollection([net_corners], facecolors=net_color, edgecolors='black', alpha=1)
    ax.add_collection3d(net_poly)

    if set_limits:
        ax.set_xlim3d(-table.length * 0.55, table.length * 0.55)
        ax.set_ylim3d(-table.length * 0.55, table.length * 0.55)
        ax.set_zlim3d(0, table.length * 0.55 * 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

def plot_camera(ax: Axes3D, camera: Camera | CameraPose, length = 0.5, 
                color_x = 'red', color_y = 'green', color_z = 'blue'):
    """ Given an Axes3D, this plots the camera position onto it.
        The camera is represented as an arrow for each axis.
        The table is needed to determine the center of the table.

        Arguments:
            ax (Axes3D): The 3D axes.
            camera (Camera): The camera.
            table (TableConstants): The table constants.
            length: The length of the arrows representing the camera.
            color_x: Matplotlib color for the x-axis arrow.
            color_y: Matplotlib color for the y-axis arrow.
            color_z: Matplotlib color for the z-axis arrow.
    """
    camera_position = camera.position
    x_flip = 1 if camera.mirror_x else -1
    y_flip = 1 if camera.mirror_y else -1
    ax.quiver3D(camera_position[0], camera_position[1], camera_position[2],
                camera.orientation.apply([0.1, 0, 0])[0] * x_flip,
                camera.orientation.apply([0.1, 0, 0])[1] * x_flip,
                camera.orientation.apply([0.1, 0, 0])[2] * x_flip,
                pivot='tail', length=length, normalize=True, color=color_x)
    ax.quiver3D(camera_position[0], camera_position[1], camera_position[2],
                camera.orientation.apply([0, 0.1, 0])[0] * y_flip,
                camera.orientation.apply([0, 0.1, 0])[1] * y_flip,
                camera.orientation.apply([0, 0.1, 0])[2] * y_flip,
                pivot='tail', length=length, normalize=True, color=color_y)
    ax.quiver3D(camera_position[0], camera_position[1], camera_position[2],
                camera.orientation.apply([0, 0, 0.1])[0],
                camera.orientation.apply([0, 0, 0.1])[1],
                camera.orientation.apply([0, 0, 0.1])[2],
                pivot='tail', length=length, normalize=True, color=color_z)

def plot_sphere(ax, center, radius, color='orange', alpha=0.7):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def plot_balls(ax: Axes3D, ball: BallConstants, ball_positions: np.ndarray):
    """ Given an Axes3D and a list of ball positions, this plots the balls onto the Axes3D.
    
        Arguments:
            ax (Axes3D): The 3D axes.
            ball_positions: np.ndarray of shape (n_balls, 3) containing the x, y, z positions of the balls.
                            Can also be a list of ball vectors.
    """                     
    for ball_position in ball_positions:
        plot_sphere(ax, ball_position, ball.radius)

def plot_ball_states(ax: Axes3D, ball: BallConstants, ball_states: list[StateVector]):
    """ Given an Axes3D and a list of ball state vectors, this plots the balls onto the Axes3D.
    
        Arguments:
            ax (Axes3D): The 3D axes.
            ball_states: list of StateVector containing the state vectors of the balls.
    """
    ball_positions = np.array([state[StateComponent.X:StateComponent.Z+1] for state in ball_states])
    plot_balls(ax, ball, ball_positions)

def plot_trajectory(ax: Axes3D, trajectory: np.ndarray, color='red'):
    """ Given an Axes3D and a trajectory, this plots the trajectory onto the Axes3D.
    
        Arguments:
            ax (Axes3D): The 3D axes.
            trajectory: np.ndarray of shape (n_points, 3) containing the x, y, z positions of the trajectory.
    """
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color)

def view_from_camera_angle(ax: Axes3D, camera: Camera | CameraPose):
    """ Shows the 3D plot from the angle and elevation of the camera. 
        Matplotlib only allows us to rotate around the z-axis, so x and y rotations will not be reflected.
    
        Arguments:
            ax (Axes3D): The 3D axes.
            camera (Camera): The camera.
    """
    elevation = camera.position[2]
    azimuth = camera.orientation.as_euler('xyz', degrees=True)[2] - 90
    ax.view_init(elev=elevation, azim=azimuth)

