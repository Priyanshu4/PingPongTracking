from src.pingpong.table import TableConstants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_table(table: TableConstants, ax: Axes3D):
    """ Given an Axes3D, this plots the table and net onto it.
    
        Arguments:
            table (TableConstants): The table constants.
            ax (Axes3D): The 3D axes.
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

    print(corners)
    print(net_corners)