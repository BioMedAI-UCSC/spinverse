import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import plotly.graph_objects as go
import torch

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def eigen_history_to_video(
    eigen_history, video_path="eigenvalues_video.mp4", fps=10
):
    with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
        for it, (ref_vals, opt_vals) in enumerate(eigen_history):
            ref_vals = ref_vals.detach().cpu().numpy()
            opt_vals = opt_vals.detach().cpu().numpy()
            x = np.arange(len(ref_vals))

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, ref_vals, "--", linewidth=2, label="ref")
            ax.plot(x, opt_vals, linewidth=2, label="opt")
            ax.set_yscale("log", base=10)
            ax.set_title(f"Eigenvalues – Iteration {it}")
            ax.set_xlabel("Eigenvalue index")
            ax.set_ylabel("Value (log10 scale)")
            ax.legend()
            plt.tight_layout()

            # Render to in-memory canvas
            canvas = FigureCanvas(fig)
            canvas.draw()
            image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(canvas.get_width_height()[::-1] + (4,))  # HWC

            writer.append_data(image)
            plt.close(fig)

    print(f"Eigenvalue video saved to {video_path}")


