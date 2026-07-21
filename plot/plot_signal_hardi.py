import numpy as np
import torch
from typing import Union
from scipy.spatial import ConvexHull
import plotly.graph_objects as go

ArrayLike = Union[torch.Tensor, np.ndarray]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x))

def plot_dirs_scatter(points: ArrayLike):
    P = _to_numpy(points)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    pts = go.Scatter3d(x=x, y=y, z=z, mode="markers",
                    marker=dict(size=2, color="black"), name="samples")

    fig = go.Figure(pts)

    fig.update_layout(
        title=f"Gradient in {P.shape[0]} directions.",
        scene=dict(aspectmode="data",
                   xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def plot_hardi(points: ArrayLike, sig: ArrayLike, title: str = "HARDI"):
    P = _to_numpy(points).T  # (N,3)
    S = _to_numpy(sig).reshape(-1)  # (N,)
    assert P.ndim == 2 and P.shape[1] == 3 and S.shape[0] == P.shape[0]

    hull = ConvexHull(P)
    tri = hull.simplices  # (M,3)

    # Deformed positions: signal * direction
    XYZ = (P.T * S).T
    x, y, z = XYZ[:,0], XYZ[:,1], XYZ[:,2]

    # Mesh colored by vertex signal (Plotly handles per-vertex colors)
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=tri[:,0], j=tri[:,1], k=tri[:,2],
        intensity=S, 
        colorscale="Viridis", 
        colorbar = dict(
            title="S/S0"
        ),
        showscale=True,
        opacity=1.0, 
        flatshading=True,
        hoverinfo='skip'
    )
    # Original sampled points as small black dots
    pts = go.Scatter3d(x=x, y=y, z=z, mode="markers",
                       marker=dict(size=2, color="black"), name="samples")

    fig = go.Figure(data=[mesh, pts])
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data",
                   xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# ---------------------------
# Example (commented out):
# import torch
# N = 200
# # random directions on the unit sphere
# v = torch.randn(N, 3)
# v = v / v.norm(dim=1, keepdim=True)
# # pretend signal in [0,1]
# sig = (0.5 * (1 + (v[:, 2]))).clamp(0, 1)  # e.g., larger near +Z
# ax, surf = plot_hardi(v, sig, fig_title="HARDI Plot", caxis=(0, 1))
# plt.show()