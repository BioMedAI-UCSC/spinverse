import torch
import plotly.graph_objects as go 

def plot_mesh_with_centroids(vertices, faces, centroids, title="Mesh with Face Centroids"):
    """
    Plot mesh faces with centroids as scatter points.
    
    Args:
        vertices: (3, V) or (V, 3) tensor of vertex positions
        faces: (3, F) or (F, 3) tensor of face indices
        centroids: (C, 3) or (3, C) tensor of centroid positions
        title: plot title
    """
    # Convert to numpy and ensure correct shape
    if torch.is_tensor(vertices):
        vertices = vertices.cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.cpu().numpy()
    if torch.is_tensor(centroids):
        centroids = centroids.cpu().numpy()
    
    # Ensure vertices are (V, 3)
    if vertices.shape[0] == 3:
        vertices = vertices.T
    
    # Ensure faces are (F, 3)
    if faces.shape[0] == 3:
        faces = faces.T
    
    # Ensure centroids are (C, 3)
    if centroids.shape[1] != 3:
        centroids = centroids.T
    
    # Create mesh plot
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=0.5,
        flatshading=True,
        name='Mesh'
    )
    
    # Create centroid scatter plot
    scatter = go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            symbol='circle'
        ),
        name='Centroids'
    )
    
    # Create figure
    fig = go.Figure(data=[mesh, scatter])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1000,
        height=800,
    )
    
    fig.show()
    return fig

def plot_two_centroid_sets(centroids1, centroids2, 
                           label1="Reference", label2="Optimized",
                           title="Mesh with Centroids"):
    """Plot mesh with two sets of centroids in different colors."""
    import plotly.graph_objects as go
    
    # Convert and reshape as before
    if torch.is_tensor(centroids1):
        centroids1 = centroids1.cpu().numpy()
    if torch.is_tensor(centroids2):
        centroids2 = centroids2.cpu().numpy()

    if centroids1.shape[1] != 3:
        centroids1 = centroids1.T
    if centroids2.shape[1] != 3:
        centroids2 = centroids2.T
    
    # Centroid set 1
    scatter1 = go.Scatter3d(
        x=centroids1[:, 0], y=centroids1[:, 1], z=centroids1[:, 2],
        mode='markers',
        marker=dict(size=4, color='red'),
        name=label1
    )
    
    # Centroid set 2
    scatter2 = go.Scatter3d(
        x=centroids2[:, 0], y=centroids2[:, 1], z=centroids2[:, 2],
        mode='markers',
        marker=dict(size=4, color='blue'),
        name=label2
    )
    
    fig = go.Figure(data=[scatter1, scatter2])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1000, height=800
    )
    fig.show()
    return fig