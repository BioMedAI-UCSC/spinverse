import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import plotly.graph_objects as go
import torch

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def faces_prob_history_to_video(
    mesh,
    history,
    video_path="faces_prob_video.mp4",
    fps=10,
    target_value=1.0
):
    """
    Create a video of faces_prob evolution directly from `history` without saving PNGs.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    vertices = mesh["points"].T.cpu().numpy()
    faces = mesh["facets"].T.cpu().numpy()

    with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
        for it, (_, opt_prob) in enumerate(history):
            p = opt_prob.detach().cpu().numpy()
            tol = 0.1 * target_value
            lb, ub = target_value - tol, target_value + tol
            colours = np.where((p >= lb) & (p <= ub), "blue", "lightblue")

            fig = go.Figure(data=[
                go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    facecolor=colours, opacity=0.8,
                ),
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode="markers", marker=dict(size=10, color="blue"),
                    name=f"≈{target_value:.1e}"),
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode="markers", marker=dict(size=10, color="lightblue"),
                    name="far"),
            ])
            fig.update_layout(
                title=f"Iteration {it}",
                scene=dict(aspectmode="cube"),
                width=1000,
                height=800,
                margin=dict(l=0, r=0, b=0, t=30)
            )

            # Render to static image using kaleido
            fig_bytes = fig.to_image(format="png", width=1000, height=800, engine="kaleido")
            img = imageio.imread(fig_bytes)
            writer.append_data(img)

    print(f"Faces prob video saved to {video_path}")