import plotly.graph_objects as go

def plot_mesh_from_faces(vertices, faces, title="3D Mesh", show_edges=False, opacity=0.7, color='lightblue'):
    """
    Plot a 3D mesh using Plotly from vertices and faces.
    
    Args:
        vertices (np.ndarray): Array of shape (N, 3) containing vertex coordinates
        faces (np.ndarray): Array of shape (M, 3) containing face indices  
        title (str): Title for the plot
        show_edges (bool): Whether to show mesh edges
        opacity (float): Mesh opacity (0-1)
        color (str): Mesh color
    """
    import plotly.graph_objects as go
    
    # Create the mesh
    mesh3d = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1], 
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=opacity,
        color=color,
        showscale=False
    )
    
    traces = [mesh3d]
    
    # Create figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='data'
        )
    )
    
    fig.show()
    return fig