from src.pingpong.table import TableConstants
from src.pingpong.ball import BallConstants
from src.ukf.state import StateVector, StateComponent

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_table(ax: Axes3D, table: TableConstants, ):
    """ Given an Axes3D, this plots the table and net onto it.
    
        Arguments:
            ax (Axes3D): The 3D axes.
            table (TableConstants): The table constants.
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
    table_poly = Poly3DCollection([corners], facecolors='blue', edgecolors='black', alpha=0.9)
    ax.add_collection3d(table_poly)
    
    # Plot the net
    net_corners = [
        [0, -table.net_width/2, table.height],
        [0, -table.net_width/2, table.height + table.net_height],
        [0, table.net_width/2, table.height + table.net_height],
        [0, table.net_width/2, table.height],
    ]

    net_poly = Poly3DCollection([net_corners], facecolors='grey', edgecolors='black', alpha=1)
    ax.add_collection3d(net_poly)

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
