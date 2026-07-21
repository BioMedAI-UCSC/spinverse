import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np

def plot_mesh(femesh, target_indices=None, output_file=None, show_plot=False):
    """
    Plot femesh using Plotly for 3D visualization
    
    Args:
        femesh: Dictionary containing 'points', 'facets', 'elements'
        output_file: Optional path to save HTML file
        show_plot: Whether to display the plot
    """
    
    points = femesh["points"].cpu().detach().numpy()  # Shape: (3, N)
    facets = femesh["facets"].cpu().detach().numpy()  # Shape: (3, M) - triangular faces

    if target_indices is None:
        target_indices = np.full(facets.shape[1], True)
    else:
        target_indices = target_indices.cpu().detach().numpy()

    # Create 3D mesh plot
    fig = go.Figure(data=[
        go.Mesh3d(
            x=points[0, :],  # X coordinates
            y=points[1, :],  # Y coordinates  
            z=points[2, :],  # Z coordinates
            i=facets[0, target_indices],  # Indices for triangle vertices
            j=facets[1, target_indices],
            k=facets[2, target_indices],
            opacity=0.7,
            color='lightblue',
            showscale=False,
            hovertemplate='<b>Point %{pointNumber}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<extra></extra>'
        )
    ])
    
    # Update layout for better visualization
    fig.update_layout(
        title='3D Finite Element Mesh',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'  # Equal aspect ratio
        ),
        width=800,
        height=600
    )
    
    # Save to HTML if specified
    if output_file:
        fig.write_html(output_file)
        print(f"Plot saved to: {output_file}")
    
    # Show plot if requested
    if show_plot:
        fig.show()
    
    return fig