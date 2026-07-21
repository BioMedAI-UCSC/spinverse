import os
import importlib
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterSciNotation
import numpy as np
import plotly.graph_objects as go
import torch
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import json
from PIL import Image

from mesh_setup.mesh_utils import get_split_indices

# ---------- Fixed rendering sizes ----------
PLOTLY_W, PLOTLY_H = 1000, 800      # each mesh panel
H_W, H_H = 2000, 1200               # histogram panel
FRAME_W, FRAME_H = 2000, 2000       # final frame (top: 1000x800 + 1000x800; bottom: 2000x1200) -> 2000x2000
H_DPI = 200                         # histogram dpi for crisp text

def create_mesh_split_visualization(mesh,
                                    faces_prob,
                                    iteration,
                                    boundary_face_indices,
                                    interior_face_indices,
                                    save_html=True):
    # Convert torch to numpy
    vertices = mesh["points"].T.cpu().numpy()
    faces = mesh["facets"].T.cpu().numpy()
    fp = faces_prob.detach().cpu().numpy()
    bound_indices = boundary_face_indices.cpu().numpy()
    inter_indices = interior_face_indices.cpu().numpy()

    # Define custom tick values and labels for the color bar
    log_fp = np.log10(np.clip(fp, 1e-30, None))
    min_log = np.floor(log_fp.min())
    max_log = np.ceil(log_fp.max())
    if max_log == min_log:
        max_log += 1

    tickvals = np.linspace(min_log, max_log, 10)
    ticktext = [f"{10**tv:.1e}" for tv in tickvals]

    # Boundary mesh
    boundary_mesh = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[bound_indices, 0], j=faces[bound_indices, 1], k=faces[bound_indices, 2],
        intensity=log_fp[bound_indices], intensitymode="cell",
        colorscale='Viridis', cmin=min_log, cmax=max_log,
        colorbar=dict(title="Face Permeability", tickmode="array", tickvals=tickvals, ticktext=ticktext),
        opacity=0.5, showscale=True
    )
    boundary_fig = go.Figure(boundary_mesh)
    boundary_fig.update_layout(
        title=f"Boundary Face Permeability: Iteration {iteration}",
        scene=dict(aspectmode="cube"),
        width=PLOTLY_W, height=PLOTLY_H,
    )

    # Interior mesh
    interior_mesh = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[inter_indices, 0], j=faces[inter_indices, 1], k=faces[inter_indices, 2],
        intensity=log_fp[inter_indices], intensitymode='cell',
        colorscale='Viridis', cmin=min_log, cmax=max_log,
        colorbar=dict(title="Face Permeability", tickmode="array", tickvals=tickvals, ticktext=ticktext),
        opacity=0.5, showscale=True
    )
    interior_fig = go.Figure(interior_mesh)
    interior_fig.update_layout(
        title=f"Interior Face Permeability: Iteration {iteration}",
        scene=dict(aspectmode="cube"),
        width=PLOTLY_W, height=PLOTLY_H,
    )

    return boundary_fig, interior_fig

def create_full_mesh_visualization(mesh, faces_prob, iteration):
    vertices = mesh["points"].T.cpu().numpy()
    faces = mesh["facets"].T.cpu().numpy()
    fp = faces_prob.detach().cpu().numpy()

    log_fp = np.log10(np.clip(fp, 1e-30, None))
    min_log = np.floor(log_fp.min())
    max_log = np.ceil(log_fp.max())
    if max_log == min_log:
        max_log += 1

    tickvals = np.linspace(min_log, max_log, 10)
    ticktext = [f"{10**tv:.1e}" for tv in tickvals]

    full_mesh = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=log_fp, intensitymode="cell",
        colorscale='Viridis', cmin=min_log, cmax=max_log,
        colorbar=dict(title="Face Permeability", tickmode="array", tickvals=tickvals, ticktext=ticktext),
        opacity=0.5, showscale=True
    )
    full_mesh_fig = go.Figure(full_mesh)
    full_mesh_fig.update_layout(
        title=f"Mesh Face Permeability: Iteration {iteration}",
        scene=dict(aspectmode="cube"),
        width=PLOTLY_W, height=PLOTLY_H,
    )
    return full_mesh_fig

def plot_face_prob_hist(faces_prob, bound_indices, inter_indices, iteration,
                        px_width=H_W, px_height=H_H, dpi=H_DPI):
    f_prob = faces_prob.detach().cpu().numpy()
    b_indices = bound_indices.cpu().numpy()
    i_indices = inter_indices.cpu().numpy()

    boundary_fp = f_prob[b_indices]
    interior_fp = f_prob[i_indices]
    bins = np.logspace(-10, 0, 21)

    b_counts, _ = np.histogram(boundary_fp, bins=bins)
    i_counts, _ = np.histogram(interior_fp, bins=bins)

    fig_w_in = px_width / dpi
    fig_h_in = px_height / dpi
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    bin_widths = np.diff(bins)
    ax.bar(bins[:-1], b_counts, width=bin_widths, align="edge", edgecolor='black',
           alpha=0.6, label="Boundary Faces")
    ax.bar(bins[:-1], i_counts, width=bin_widths, align="edge", edgecolor='black',
           alpha=0.6, label="Interior Faces")

    ax.set_xscale("log")
    xticks = np.logspace(-10, 0, 21)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{xt:.1e}' for xt in xticks], rotation=45)

    ax.set_xlabel("Face Permeability")
    ax.set_ylabel("Count")
    ax.set_title(f"Face Permeability Distribution (Iteration {iteration})")
    ax.legend()
    return fig

def _ensure_size(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize numpy image to exact width x height using PIL if needed."""
    if img.shape[1] == width and img.shape[0] == height:
        return img
    return np.array(Image.fromarray(img).resize((width, height), Image.LANCZOS))

def _pad_to_multiple(img: np.ndarray, mult: int = 16) -> np.ndarray:
    """Pad image to have width/height divisible by `mult` (zeros)."""
    h, w = img.shape[:2]
    new_w = ((w + mult - 1) // mult) * mult
    new_h = ((h + mult - 1) // mult) * mult
    if new_w == w and new_h == h:
        return img
    pad_w = new_w - w
    pad_h = new_h - h
    if img.ndim == 2:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    else:
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    return padded

class MeshLogger:
    '''
    Class handles logging of mesh data and plots/figures
    '''
    def __init__(self, fixed_mesh=None, log_dir=None):
        if not log_dir:
            log_dir = os.path.join("logs", datetime.now().strftime("mesh_data_%Y%m%d_%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # Persistent subdirs used by both logging and writer
        self.image_dir = os.path.join(self.log_dir, "mesh_images")
        self.plot_dir  = os.path.join(self.log_dir, "mesh_plots")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.plot_dir,  exist_ok=True)

        # Temp dir to write hist images per-iteration (so we can close figures immediately)
        self.tmp_plot_dir = os.path.join(self.log_dir, "_tmp_plots")
        os.makedirs(self.tmp_plot_dir, exist_ok=True)

        # Create log file for text
        self.text_log_file = os.path.join(log_dir, "log_file.txt")
        with open(self.text_log_file, 'w') as f:
            f.write("LOG FILE START")

        self.plot_buffer = []
        self.data_buffer = []
        self.train_data_buffer = []
        self.split_indices = {}
        self.text_log = ""

    def log_mesh(self, mesh_data, faces_prob, loss, iteration):
        '''
        Log mesh data and figures in buffer. We save & close the matplotlib figure NOW
        to avoid having many open figures.
        '''
        # Log faces_prob
        f_prob = faces_prob.detach().cpu().tolist()
        mesh_log_data = {'faces_prob': f_prob}
        mesh_log_data['loss'] = float(getattr(loss, "item", lambda: loss)())

        # Unpack mesh data
        vertices = mesh_data["points"].T.cpu().tolist()
        faces = mesh_data["facets"].T.cpu().tolist()
        mesh_log_data['data'] = {"points": vertices, "facets": faces}

        # Get boundary indices
        bound_indices = mesh_data['boundary_indices']
        inter_indices = mesh_data['interior_indices']

        # Plotly figures (kept in memory; these are not matplotlib)
        mesh_boundary_vis, mesh_interior_vis = create_mesh_split_visualization(
            mesh_data, faces_prob, iteration, bound_indices, inter_indices
        )
        mesh_log_data['boundary_vis'] = mesh_boundary_vis
        mesh_log_data['interior_vis'] = mesh_interior_vis
        mesh_log_data['full_mesh_vis'] = create_full_mesh_visualization(
            mesh_data, faces_prob, iteration
        )

        # Matplotlib histogram: SAVE & CLOSE NOW at a fixed size; keep only the path
        hfig = plot_face_prob_hist(faces_prob, bound_indices, inter_indices, iteration,
                                   px_width=H_W, px_height=H_H, dpi=H_DPI)
        hist_path = os.path.join(self.tmp_plot_dir, f"face_prob_hist_{iteration}.png")
        hfig.savefig(hist_path)  # no bbox_inches="tight" to keep size stable
        plt.close(hfig)
        mesh_log_data['face_prob_hist_path'] = hist_path

        # Add mesh to buffer
        self.mesh_buffer.append(mesh_log_data)

    def log_text(self, text):
        self.text_log += "\n" + text

    def write_log(self, fps=10):
        '''
        Write all mesh data including figures and plots to persistent memory
        '''
        # Log mesh data (jsonl)
        with open(os.path.join(self.log_dir, "mesh_data_log.jsonl"), "a+") as f:
            for m in self.mesh_buffer:
                mesh_data = m['data'].copy()
                mesh_data['faces_prob'] = m['faces_prob']
                f.write(json.dumps(mesh_data) + "\n")

        # Create combined video
        mesh_video_file = os.path.join(self.log_dir, "mesh_vis.mp4")

        with imageio.get_writer(mesh_video_file, fps=fps, codec='libx264') as writer:
            for i, m in enumerate(self.mesh_buffer):
                bfig = m['boundary_vis']
                ifig = m['interior_vis']
                mfig = m['full_mesh_vis']
                hist_path = m['face_prob_hist_path']

                # Save Plotly images at fixed size (guaranteed)
                boundary_path  = os.path.join(self.image_dir, f"boundary_mesh_{i}.png")
                interior_path  = os.path.join(self.image_dir, f"interior_mesh_{i}.png")
                mesh_html_path = os.path.join(self.image_dir, f"full_mesh_{i}.html")

                bfig.write_image(boundary_path, width=PLOTLY_W, height=PLOTLY_H, scale=1)
                ifig.write_image(interior_path, width=PLOTLY_W, height=PLOTLY_H, scale=1)
                mfig.write_html(mesh_html_path)

                # Read images
                boundary_img = imageio.imread(boundary_path)
                interior_img = imageio.imread(interior_path)
                plot_img     = imageio.imread(hist_path)

                # Force exact sizes
                boundary_img = _ensure_size(boundary_img, PLOTLY_W, PLOTLY_H)
                interior_img = _ensure_size(interior_img, PLOTLY_W, PLOTLY_H)
                plot_img     = _ensure_size(plot_img,     H_W,       H_H)

                # Combine: top row (2 panels), bottom histogram
                top_row = np.hstack((boundary_img, interior_img))  # -> (PLOTLY_H, 2*PLOTLY_W, C)
                # If histogram not same width, ensure it (we forced above)
                combined_img = np.vstack((top_row, plot_img))      # -> (PLOTLY_H + H_H, 2*PLOTLY_W, C)

                # Final sanity: pad to 16-multiples (should already be 2000x2000)
                combined_img = _pad_to_multiple(combined_img, mult=16)

                # Write frame
                writer.append_data(combined_img)

        # Optionally, move hist PNGs from tmp to persistent plot dir
        for p in os.listdir(self.tmp_plot_dir):
            src = os.path.join(self.tmp_plot_dir, p)
            dst = os.path.join(self.plot_dir, p)
            try:
                os.replace(src, dst)
            except Exception:
                pass
        
        # Write text log if any
        if len(self.text_log) > 0:
            with open(self.text_log_file, 'a') as f:
                f.write(self.text_log)

        # Clean up tmp dir if desired
        # import shutil; shutil.rmtree(self.tmp_plot_dir, ignore_errors=True)


if __name__ == '__main__':
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Test mesh
    mesh = torch.load("simple_mesh_6v_random.pth", weights_only=False, map_location='cpu')

    vis_mesh = {
        'points': mesh['points'],
        'facets': mesh['facets'],
        'elements': mesh['elements']
    }

    faces_prob = torch.full((mesh['facets'].shape[1],), 1e-10, device=DEVICE)

    # Initialize logger
    mesh_logger = MeshLogger()

    # Call log mesh on each iteration of training
    for i in range(10):
        print(f"Current iteration: {i}")
        mesh_logger.log_mesh(vis_mesh, faces_prob, i, i)

    # Write log to memory and create plots at end of training
    mesh_logger.write_log()