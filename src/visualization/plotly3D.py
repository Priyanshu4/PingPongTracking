import plotly.graph_objects as go
import numpy as np

def plot_table(fig, table, table_color='blue', net_color='grey', table_alpha = 0.9, net_alpha = 0.7, set_limits = True):
    table_mid_x = table.length / 2
    table_mid_y = table.width / 2
    
    x = [-table_mid_x, -table_mid_x, table_mid_x, table_mid_x]
    y = [-table_mid_y, table_mid_y, table_mid_y, -table_mid_y]
    z = [table.height] * 4
    
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        color=table_color,
        opacity=table_alpha
    ))

    plot_net(fig, table, net_color=net_color, net_alpha=net_alpha)

    fig.update_layout(
        scene=dict(
            aspectmode="manual",  # Ensure manual aspect ratio setting
            aspectratio=dict(x=1, y=1, z=1),  # Equal aspect ratio
            xaxis=dict(range=[-table.length, table.length]),
            yaxis=dict(range=[-table.length, table.length]),
            zaxis=dict(range=[0, table.height * 5])
        )
    )

def plot_net(fig, table, net_color='grey', net_alpha=0.7):
    net_x = [0, 0, 0, 0]
    net_y = [-table.net_width / 2, table.net_width / 2, table.net_width / 2, -table.net_width / 2]
    net_z = [table.height, table.height, table.height + table.net_height, table.height + table.net_height]
    
    fig.add_trace(go.Mesh3d(
        x=net_x, y=net_y, z=net_z,
        i=[0, 1, 2, 2, 3, 0], 
        j=[1, 2, 3, 3, 0, 1],
        k=[2, 3, 0, 0, 1, 2],
        color=net_color,
        opacity=net_alpha
    ))

def plot_sphere(fig, center, radius, color='orange', alpha=0.7):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], opacity=alpha))

def plot_trajectory(fig, trajectory, color='red'):
    fig.add_trace(go.Scatter3d(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        z=trajectory[:, 2],
        mode='lines',
        line=dict(color=color, width=3)
    ))

def animate_trajectory(trajectory, dt=0.1, ball_radius=0.02, ball_color='orange', ball_alpha=1):
    frames = []
    for i in range(len(trajectory)):
        frame_fig = go.Figure()
        plot_sphere(frame_fig, trajectory[i], ball_radius, color=ball_color, alpha=ball_alpha)
        frames.append(go.Frame(data=frame_fig.data))
    
    fig = go.Figure(
        frames=frames,
        layout=go.Layout(
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=dt*1000, redraw=True))])]
            )]
        )
    )
    fig.show()

def plot_camera(fig, camera, length=0.5, color_x='red', color_y='green', color_z='blue'):
    camera_position = camera.position
    orientation = camera.orientation.as_matrix()
    
    fig.add_trace(go.Scatter3d(
        x=[camera_position[0], camera_position[0] + length * orientation[0, 0]],
        y=[camera_position[1], camera_position[1] + length * orientation[1, 0]],
        z=[camera_position[2], camera_position[2] + length * orientation[2, 0]],
        mode='lines',
        line=dict(color=color_x, width=3)
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[camera_position[0], camera_position[0] + length * orientation[0, 1]],
        y=[camera_position[1], camera_position[1] + length * orientation[1, 1]],
        z=[camera_position[2], camera_position[2] + length * orientation[2, 1]],
        mode='lines',
        line=dict(color=color_y, width=3)
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[camera_position[0], camera_position[0] + length * orientation[0, 2]],
        y=[camera_position[1], camera_position[1] + length * orientation[1, 2]],
        z=[camera_position[2], camera_position[2] + length * orientation[2, 2]],
        mode='lines',
        line=dict(color=color_z, width=3)
    ))

def plot_camera_as_ball(fig, camera, radius=0.1, arrow_length = 0.2, color='black', alpha=0.7):
   
    plot_sphere(fig, camera.position, radius, color=color, alpha=alpha)
    
    orientation = camera.orientation.as_matrix()
    fig.add_trace(go.Scatter3d(
        x=[camera.position[0], camera.position[0] + arrow_length * orientation[0, 2]],
        y=[camera.position[1], camera.position[1] + arrow_length * orientation[1, 2]],
        z=[camera.position[2], camera.position[2] + arrow_length * orientation[2, 2]],
        mode='lines',
        line=dict(color=color, width=3)
    ))