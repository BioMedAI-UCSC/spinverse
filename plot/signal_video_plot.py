import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import plotly.graph_objects as go
import torch

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def signal_history_to_videos(
    signal_history,
    time_max=10000,
    out_dir="signal_videos_direct",
    fps=5
):
    """
    Directly render signal videos from signal_history without saving PNGs.
    One video per direction.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if len(signal_history) == 0:
        print("No signal history provided.")
        return

    sr0, _ = signal_history[0]
    ntime, n_tau, n_dir = sr0.shape
    times = np.linspace(0, time_max, ntime)
    colors = ["blue", "green", "red"]

    for d in range(n_dir):
        video_path = os.path.join(out_dir, f"signal_direction_{d:02d}.mp4")
        with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
            for it, (ref_signal, opt_signal) in enumerate(signal_history):
                ref_signal = ref_signal.numpy() if torch.is_tensor(ref_signal) else ref_signal
                opt_signal = opt_signal.numpy() if torch.is_tensor(opt_signal) else opt_signal

                fig, ax = plt.subplots(figsize=(8, 4))
                for t in range(n_tau):
                    ax.plot(times, ref_signal[:, t, d], "-",  label=f"ref τ{t}", color=colors[t])
                    ax.plot(times, opt_signal[:, t, d], "--", label=f"opt τ{t}", color=colors[t])
                ax.set_title(f"Iteration {it} – Direction {d}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Signal")
                ax.legend()
                plt.tight_layout()

                # Render figure to image in memory
                canvas = FigureCanvas(fig)
                canvas.draw()
                image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                image = image.reshape(canvas.get_width_height()[::-1] + (4,))  # HWC format
                writer.append_data(image)
                plt.close(fig)
        print(f"Saved direct video for direction {d} → {video_path}")


