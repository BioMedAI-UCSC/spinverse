import os
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def faces_prob_history_to_video_plot(
    history,
    video_path="faces_prob_comparison_video.mp4",
    fps=10
):
    """
    Render line plots comparing ref vs opt face probabilities over iterations directly to video.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
        for it, (ref_p, opt_p) in enumerate(history):
            ref = ref_p.numpy() if torch.is_tensor(ref_p) else np.asarray(ref_p)
            opt = opt_p.numpy() if torch.is_tensor(opt_p) else np.asarray(opt_p)
            x = np.arange(ref.shape[0])

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, ref, "--", linewidth=2, label="ref")
            ax.plot(x, opt, linewidth=2, label="opt")
            ax.set_yscale("log", base=10)
            ax.set_title(f"Iteration {it}")
            ax.set_xlabel("face index")
            ax.set_ylabel("probability (log10 scale)")
            ax.legend()
            plt.tight_layout()

            canvas = FigureCanvas(fig)
            canvas.draw()
            image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(canvas.get_width_height()[::-1] + (4,))
            writer.append_data(image)
            plt.close(fig)

    print(f"Faces prob lineplot video saved to {video_path}")
