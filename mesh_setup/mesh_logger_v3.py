import os
import importlib
import imageio
import matplotlib
import io

from sklearn import metrics
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterSciNotation
import numpy as np
import plotly.graph_objects as go
import torch
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import pytz
import json
from PIL import Image
from itertools import combinations

from mesh_setup.mesh_utils import get_split_indices
from plot.plot_signal_hardi import plot_hardi

import csv
import math
from typing import Dict, Optional, List

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

LARGE_ARRAY_THRESHOLD = 10_000

# ---------- Fixed rendering sizes ----------
PLOTLY_W, PLOTLY_H = 1000, 800      # each mesh panel
H_W, H_H = 2000, 1200               # histogram panel
FRAME_W, FRAME_H = 2000, 2000       # final frame (top: 1000x800 + 1000x800; bottom: 2000x1200) -> 2000x2000
H_DPI = 200                         # histogram dpi for crisp text

def _iter_camera(iteration: int, total_iters: int,
                 radius: float = 2.2,
                 z: float = 1.2) -> dict:
    """
    Generate a camera dict that orbits around Z axis.
    """
    if total_iters <= 0:
        total_iters = 1
    angle = 2.0 * np.pi * (iteration / total_iters)
    eye = dict(x=radius * np.cos(angle),
               y=radius * np.sin(angle),
               z=z)
    up = dict(x=0, y=0, z=1)
    center = dict(x=0, y=0, z=0)
    return dict(eye=eye, up=up, center=center)

#TODO: move these to a plotter file for more general usage
def create_mesh_threshold_visualization(mesh,
                                        face_perm_ref,
                                        data,
                                        log_threshold=1e-3,
                                        save_html=True):
    # Convert torch to numpy
    vertices = mesh["points"].T.cpu().numpy()
    faces = mesh["facets"].T.cpu().numpy()
    fp = data['face_perm'].detach().cpu().numpy()
    fp_ref = face_perm_ref.detach().cpu().numpy()
    non_perm_filter_opt = fp < log_threshold
    non_perm_filter_ref = fp_ref < log_threshold

    # Update camera angle per iteration
    camera = _iter_camera(data['iteration'], total_iters=100)
    
    # Boundary mesh
    opt_mesh = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[non_perm_filter_opt, 0], j=faces[non_perm_filter_opt, 1], k=faces[non_perm_filter_opt, 2],
        color='blue',
        opacity=0.5, showscale=True
    )
    opt_fig = go.Figure(opt_mesh)
    opt_fig.update_layout(
        title=f"Target Mesh Non-permeable Faces: Iteration {data['iteration']}",
        scene=dict(aspectmode="cube", camera=camera),
        width=PLOTLY_W, height=PLOTLY_H,
    )

    # Interior mesh
    ref_mesh = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[non_perm_filter_ref, 0], j=faces[non_perm_filter_ref, 1], k=faces[non_perm_filter_ref, 2],
        color='blue',
        opacity=0.5, showscale=True
    )
    ref_fig = go.Figure(ref_mesh)
    ref_fig.update_layout(
        title=f"Reference Mesh Non-permeable Faces: Iteration {data['iteration']}",
        scene=dict(aspectmode="cube", camera=camera),
        width=PLOTLY_W, height=PLOTLY_H,
    )

    return opt_fig, ref_fig

def create_mesh_split_visualization(mesh,
                                    data,
                                    boundary_face_indices,
                                    interior_face_indices,
                                    save_html=True):
    # Convert torch to numpy
    vertices = mesh["points"].T.cpu().numpy()
    faces = mesh["facets"].T.cpu().numpy()
    fp = data['face_perm'].detach().cpu().numpy()
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
    
    # rotate camera per iteration
    camera = _iter_camera(data['iteration'], total_iters=100)

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
        title=f"Boundary Face Permeability: Iteration {data['iteration']}",
        scene=dict(aspectmode="cube", camera=camera),
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
        title=f"Interior Face Permeability: Iteration {data['iteration']}",
        scene=dict(aspectmode="cube", camera=camera),
        width=PLOTLY_W, height=PLOTLY_H,
    )

    return boundary_fig, interior_fig

def create_mesh_hardi_visualization(config,
                                    sig_ref_normed,
                                    sig_opt_normed,
                                    save_html=True):
    # Calculate MSE for given b value and sequence
    sig_norm_mse_b1 = (sig_ref_normed[1, :, :] - sig_opt_normed[1, :, :]) ** 2
    sig_norm_mse_bf = (sig_ref_normed[-1, :, :] - sig_opt_normed[-1, :, :]) ** 2

    # Plot hardi for MSE values
    # hardi_fig_s0_b1 = plot_hardi(config.gradient['directions'], sig_norm_mse_b1[0])

    #TODO setup plotting for final b values
    # return hardi_fig
    return True

def create_full_mesh_visualization(mesh, data):
    vertices = mesh["points"].T.cpu().numpy()
    faces = mesh["facets"].T.cpu().numpy()
    fp = data['face_perm'].detach().cpu().numpy()

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
        title=f"Mesh Face Permeability: Iteration {data['iteration']}",
        scene=dict(aspectmode="cube"),
        width=PLOTLY_W, height=PLOTLY_H,
    )
    return full_mesh_fig

def plot_face_prob_hist(data, bound_indices, inter_indices,
                        px_width=H_W, px_height=H_H, dpi=H_DPI):
    f_prob = data['face_perm'].detach().cpu().numpy()
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
    ax.set_title(f"Face Permeability Distribution (Iteration {data['iteration']})")
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

def _serialize_value(value, raw_data_dir, _path_prefix=""):
    """
    Recursively convert value to JSON-serializable form.
    Tensors / ndarrays larger than LARGE_ARRAY_THRESHOLD are saved separately.
    """
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu()
        numel = arr.numel()
        if numel > LARGE_ARRAY_THRESHOLD:
            fname = f"{_path_prefix}tensor_{hash(arr.storage())}_{numel}.pt"
            fpath = os.path.join(raw_data_dir, fname)
            if not os.path.exists(fpath):
                torch.save(arr, fpath)
            return {"__type__": "tensor_ref", "path": fpath, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        else:
            return arr.tolist()
    if isinstance(value, np.ndarray):
        numel = value.size
        if numel > LARGE_ARRAY_THRESHOLD:
            fname = f"{_path_prefix}ndarray_{hash(value.data.tobytes())}_{numel}.npy"
            fpath = os.path.join(raw_data_dir, fname)
            if not os.path.exists(fpath):
                np.save(fpath, value)
            return {"__type__": "ndarray_ref", "path": fpath, "shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            return value.tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v, raw_data_dir, _path_prefix + "l_") for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v, raw_data_dir, _path_prefix + f"{k}_") for k, v in value.items()}
    # Fallback
    return str(value)

#
# Tensorboard logging functionality
#

def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return None
        return float(x.detach().float().mean().item())
    return None


def perm_stats(face_perm_log: torch.Tensor,
               non_perm_bound: float = -3.0) -> Dict[str, float]:
    p = face_perm_log.detach().float()
    return {
        "perm/mean": p.mean().item(),
        "perm/std":  p.std(unbiased=False).item(),
        "perm/min":  p.min().item(),
        "perm/max":  p.max().item(),
        "perm/pct_non_perm": (p <= non_perm_bound).float().mean().item(),
        "perm/pct_perm": (p >= non_perm_bound).float().mean().item(),
    }


def spectrum_stats(lambdas: torch.Tensor,
                   keep_mask: Optional[torch.Tensor] = None,
                   K_eff: Optional[float] = None,
                   prefix: str = "spec") -> Dict[str, float]:
    lam = lambdas.detach().float().view(-1)
    K = lam.numel()
    out = {
        f"{prefix}/K_total": float(K),
        f"{prefix}/lambda0": lam[0].item() if K > 0 else math.nan,
        f"{prefix}/lambda1": lam[1].item() if K > 1 else math.nan,
        f"{prefix}/lambda_last": lam[-1].item() if K > 0 else math.nan,
    }
    if K > 1:
        out[f"{prefix}/gap_1_0"] = (lam[1] - lam[0]).item()
        out[f"{prefix}/ratio_last_over_0"] = (lam[-1] / (lam[0] + 1e-12)).item()

    if keep_mask is not None:
        m = keep_mask.detach().bool().view(-1)
        out[f"{prefix}/K_kept"] = float(m.sum().item())
        out[f"{prefix}/K_kept_frac"] = float(m.float().mean().item())
        kept = lam[m]
        out[f"{prefix}/lambda0_kept"] = kept[0].item() if kept.numel() > 0 else math.nan
        out[f"{prefix}/lambda_last_kept"] = kept[-1].item() if kept.numel() > 0 else math.nan

    if K_eff is not None:
        out[f"{prefix}/K_eff"] = float(K_eff)
    return out


def mode_weight_stats(lambdas: torch.Tensor,
                      D: float,
                      t_eff: float,
                      a_k: Optional[torch.Tensor] = None,
                      keep_mask: Optional[torch.Tensor] = None,
                      prefix: str = "modes") -> Dict[str, float]:
    lam = lambdas.detach().float().view(-1)
    if lam.numel() == 0:
        return {}

    decay = torch.exp(-(D * t_eff) * lam)  # (K,)

    if a_k is not None:
        amp = torch.abs(a_k.detach()).view(-1)
        w = amp * decay
    else:
        w = decay

    if keep_mask is not None:
        m = keep_mask.detach().bool().view(-1)
        w = w[m]
        if w.numel() == 0:
            return {
                f"{prefix}/entropy": 0.0,
                f"{prefix}/effective_rank": 0.0,
                f"{prefix}/concentration_l2": 1.0,
                f"{prefix}/top1_frac": 0.0,
                f"{prefix}/top5_frac": 0.0,
            }

    w = w.clamp_min(1e-30)
    p = w / w.sum()

    entropy = -(p * torch.log(p)).sum()
    eff_rank = torch.exp(entropy)
    concentration = (p * p).sum()

    p_sorted, _ = torch.sort(p, descending=True)
    top1 = p_sorted[0].item()
    top5 = p_sorted[:5].sum().item() if p_sorted.numel() >= 5 else p_sorted.sum().item()

    return {
        f"{prefix}/entropy": entropy.item(),
        f"{prefix}/effective_rank": eff_rank.item(),
        f"{prefix}/concentration_l2": concentration.item(),
        f"{prefix}/top1_frac": top1,
        f"{prefix}/top5_frac": top5,
    }


def grad_group_stats(optimizer: torch.optim.Optimizer,
                     group_names: Optional[List[str]] = None,
                     prefix: str = "grad") -> Dict[str, float]:
    out = {}
    for gi, group in enumerate(optimizer.param_groups):
        name = group_names[gi] if (group_names is not None and gi < len(group_names)) else f"group{gi}"
        sqsum = 0.0
        absmean_acc = 0.0
        n = 0

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad.detach()
            sqsum += float((g * g).sum().item())
            absmean_acc += float(g.abs().mean().item())
            n += 1

        out[f"{prefix}/{name}_norm"] = (sqsum ** 0.5) if n > 0 else 0.0
        out[f"{prefix}/{name}_absmean"] = (absmean_acc / max(n, 1)) if n > 0 else 0.0
        out[f"{prefix}/{name}_nparams_with_grad"] = float(n)

    return out


class MeshLogger:
    '''
    Class handles logging of mesh data and plots/figures
    '''
    def __init__(self, mesh, exp_config, ref_data, log_dir=None):
        if not log_dir:
            pst = pytz.timezone('America/Los_Angeles')
            log_dir = os.path.join("logs", datetime.now(pst).strftime("mesh_data_%Y%m%d_%H%M%S"))
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.raw_data_dir = os.path.join(self.log_dir, "raw_data")
        os.makedirs(self.raw_data_dir, exist_ok=True)

        # Fixed mesh data
        self.mesh = mesh
        self.ref_data = ref_data
        self.exp_config = exp_config

        # Write config to json config file
        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.exp_config, f, indent=4)

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
        self.split_indices = {}
        self.text_log = ""

        # ---- Scalar diagnostics logging ----
        self.scalar_csv_path = os.path.join(self.log_dir, "scalars.csv")
        self._scalar_header_written = os.path.exists(self.scalar_csv_path)
        self.scalar_flush_every = int(os.getenv("SCALAR_FLUSH_EVERY", "50"))
        self._scalar_rows_buffer = []
        self._scalar_buffered_keys = None
        self._scalar_fieldnames = None

        self.tb = None
        if SummaryWriter is not None:
            self.tb = SummaryWriter(log_dir=os.path.join(self.log_dir, "tb"))

    def _log_scalars(self, step: int, metrics: Dict[str, float]):
        # --- write to TensorBoard event files ---
        if self.tb is not None:
            for k, v in metrics.items():
                if v is None:
                    continue
                self.tb.add_scalar(k, float(v), step)

        row = {"iteration": int(step)}
        for k, v in metrics.items():
            row[k] = "" if (v is None) else float(v)

        self._scalar_rows_buffer.append(row)

        if len(self._scalar_rows_buffer) >= self.scalar_flush_every:
            self._flush_scalar_buffer()

    def _log_histograms(self, step: int, data: dict):
        if self.tb is None:
            return

        # ---------------- gradients (CAN be negative -> keep linear) ----------------
        g = data.get("face_perm_grad", None)
        if isinstance(g, torch.Tensor):
            self.tb.add_histogram("grad/face_perm_grad", g.detach().to(torch.float32).cpu(), step)  
            self.tb.add_histogram("grad/face_perm_grad_abs_log10",
                                  torch.log10(g.detach().abs().clamp_min(1e-30)).to(torch.float32).cpu(), step)

        # ---------------- face_perm (positive -> log10) ----------------
        fp = data.get("face_perm", None)
        if isinstance(fp, torch.Tensor):
            self.tb.add_histogram(
                "perm/face_perm_log10",
                torch.log10(fp.detach().clamp_min(1e-30)).to(dtype=torch.float32, device="cpu"),
                step
            )

        # ---------------- eigenvalues (assumed >=0 -> log10) ----------------
        ev = data.get("eigenvalues", None)
        if isinstance(ev, torch.Tensor):
            self.tb.add_histogram(
                "spec/eigenvalues_log10",
                torch.log10(ev.detach().clamp_min(1e-30)).to(dtype=torch.float32, device="cpu"),
                step
            )

        # ---------------- length scales (NO log; keep linear) ----------------
        ls = data.get("length_scales", None)
        if isinstance(ls, torch.Tensor):
            self.tb.add_histogram(
                "spec/length_scales",
                ls.detach().to(dtype=torch.float32, device="cpu"),
                step
            )

        # # -------- mode power spectrum histograms for ALL b values (positive -> log10) --------
        # mp = data.get("mode_power_mean", None)  # expected [n_amp, n_eig]
        # if isinstance(mp, torch.Tensor) and mp.ndim == 2:
        #     mp_cpu = torch.log10(mp.detach().clamp_min(1e-30)).to(dtype=torch.float32, device="cpu")
        #     n_amp = mp_cpu.shape[0]
        #     for bi in range(n_amp):
        #         self.tb.add_histogram(
        #             f"modes/power_mean_log10/b{bi:02d}",
        #             mp_cpu[bi],
        #             step
        #         )

        # mpv = data.get("mode_power_var", None)  # expected [n_amp, n_eig]
        # if isinstance(mpv, torch.Tensor) and mpv.ndim == 2:
        #     mpv_cpu = torch.log10(mpv.detach().clamp_min(1e-30)).to(dtype=torch.float32, device="cpu")
        #     n_amp = mpv_cpu.shape[0]
        #     for bi in range(n_amp):
        #         self.tb.add_histogram(
        #             f"modes/power_var_log10/b{bi:02d}",
        #             mpv_cpu[bi],
        #             step
        #         )

    # def _log_histograms(self, step: int, data: dict):
    #     if self.tb is None:
    #         return

    #     # gradients (vector)
    #     g = data.get("face_perm_grad", None)
    #     if isinstance(g, torch.Tensor):
    #         self.tb.add_histogram("grad/face_perm_grad_log10", torch.log10(g.detach().clamp_min(1e-30)).float().cpu(), step)

    #     # eigenvalues (vector)
    #     ev = data.get("eigenvalues", None)
    #     if isinstance(ev, torch.Tensor):
    #         self.tb.add_histogram("spec/eigenvalues", ev.detach().float().cpu(), step)

    #     # optional: permeabilities too
    #     fp = data.get("face_perm", None)
    #     if isinstance(fp, torch.Tensor):
    #         self.tb.add_histogram("perm/face_perm", fp.detach().float().cpu(), step)
        
    #     # -------- mode power spectrum histograms for ALL b values --------
    #     mp = data.get("mode_power_mean", None)  # expected [n_amp, n_eig]
    #     if isinstance(mp, torch.Tensor) and mp.ndim == 2:
    #         mp_cpu = mp.detach().float().cpu()
    #         n_amp = mp_cpu.shape[0]
    #         for bi in range(n_amp):
    #             # histogram over eigenmodes for this b-index
    #             self.tb.add_histogram(f"modes/power_mean/b{bi:02d}", mp_cpu[bi], step)

    #     mpv = data.get("mode_power_var", None)  # expected [n_amp, n_eig]
    #     if isinstance(mpv, torch.Tensor) and mpv.ndim == 2:
    #         mpv_cpu = mpv.detach().float().cpu()
    #         n_amp = mpv_cpu.shape[0]
    #         for bi in range(n_amp):
    #             self.tb.add_histogram(f"modes/power_var/b{bi:02d}", mpv_cpu[bi], step)

    def _flush_scalar_buffer(self):
        if not self._scalar_rows_buffer:
            return

        # Determine schema once
        if self._scalar_fieldnames is None:
            all_keys = set()
            for r in self._scalar_rows_buffer:
                all_keys.update(r.keys())
            self._scalar_fieldnames = ["iteration"] + sorted([k for k in all_keys if k != "iteration"])

        # If new keys appear later, either ignore or warn
        # (recommended: ignore for CSV stability)
        fieldnames = self._scalar_fieldnames

        write_header = not os.path.exists(self.scalar_csv_path)
        with open(self.scalar_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for r in self._scalar_rows_buffer:
                out = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(out)

        if self.tb is not None:
            self.tb.flush()

        self._scalar_rows_buffer.clear()

    def log_iteration_diagnostics(
        self,
        data: dict,
        optimizer: Optional[torch.optim.Optimizer] = None,
        group_names: Optional[List[str]] = None,
    ):
        """
        Call once per iteration (typically right after forward, and optionally after backward).

        Expected optional keys in `data`:
          - 'iteration' (required)
          - 'face_perm' (torch.Tensor)
          - 'eigenvalues' or 'lambdas' (torch.Tensor)
          - 'keep_mask' (torch.BoolTensor)  [your cutoff/filter mask]
          - 'D' (float) and 't_eff' (float)  [for mode dominance]
          - 'a_k' (torch.Tensor) optional modal amplitudes
        """
        it = int(data["iteration"])

        metrics = {}
        logged_keys = set()

        # Track iteration key as handled
        logged_keys.add("iteration")

        # Face permeability stats
        if "face_perm_log" in data and isinstance(data["face_perm_log"], torch.Tensor):
            metrics.update(perm_stats(data["face_perm_log"]))
            logged_keys.add("face_perm_log")

        # Spectrum stats
        lambdas = None
        if "eigenvalues" in data:
            lambdas = data["eigenvalues"]
            logged_keys.add("eigenvalues")
        elif "lambdas" in data:
            lambdas = data["lambdas"]
            logged_keys.add("lambdas")

        keep_mask = data.get("keep_mask", None)
        K_eff = data.get("K_eff", None)
        if keep_mask is not None:
            logged_keys.add("keep_mask")
        if K_eff is not None:
            logged_keys.add("K_eff")

        if lambdas is not None and isinstance(lambdas, torch.Tensor):
            metrics.update(spectrum_stats(lambdas, keep_mask=keep_mask, K_eff=K_eff, prefix="spec"))

            # Mode dominance if D, t_eff are available
            if ("D" in data) and ("t_eff" in data):
                D = float(data["D"])
                t_eff = float(data["t_eff"])
                logged_keys.update(["D", "t_eff"])
                
                a_k = data.get("a_k", None)
                if a_k is not None:
                    logged_keys.add("a_k")
                
                metrics.update(mode_weight_stats(
                    lambdas=lambdas,
                    D=D,
                    t_eff=t_eff,
                    a_k=a_k if isinstance(a_k, torch.Tensor) else None,
                    keep_mask=keep_mask if isinstance(keep_mask, torch.Tensor) else None,
                    prefix="modes"
                ))

        # Optional: grad stats (call this after loss.backward())
        if optimizer is not None:
            metrics.update(grad_group_stats(optimizer, group_names=group_names, prefix="grad"))

        for k in ("sim_grad_norm", "edge_grad_norm", "cont_grad_norm"):
            if k in data:
                metrics[f"grad/{k}"] = float(data[k])
                logged_keys.add(k)

        # Also log any scalar losses already in `data`
        for k, v in data.items():
            if "loss" not in k.lower():
                continue
            if isinstance(v, (int, float)):
                metrics[f"loss/{k}"] = float(v)
                logged_keys.add(k)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                metrics[f"loss/{k}"] = float(v.detach().item())
                logged_keys.add(k)

        for k in ("cos_sim_edge", "cos_sim_cont", "cos_edge_cont"):
            if k in data:
                metrics[f"grad_cos/{k}"] = float(data[k])
                logged_keys.add(k)

        if "clip_ratio" in data:
            metrics["grad/clip_ratio"] = float(data["clip_ratio"])
            logged_keys.add("clip_ratio")

        # Catch any remaining scalar keys and log them to misc/ group
        for k, v in data.items():
            if k in logged_keys:
                continue
            # Try to log as scalar if possible
            if isinstance(v, (int, float)):
                metrics[f"misc/{k}"] = float(v)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                metrics[f"misc/{k}"] = float(v.detach().item())

        if len(metrics) > 0:
            self._log_scalars(it, metrics)

        self._log_histograms(it, data)

    def log_data(self, data):
        self.data_buffer.append(data)

    def log_text(self, text):
        self.text_log += "\n" + text

    def write_log(self):
        # Log training data if exists (jsonl)
        if len(self.data_buffer) > 0:
            with open(os.path.join(self.log_dir, "mesh_data_log.jsonl"), "a+") as f:
                for data in self.data_buffer:
                    json_rdy_data = _serialize_value(data, raw_data_dir=self.raw_data_dir)
                    f.write(json.dumps(json_rdy_data) + "\n")

        # Write text log if any
        if len(self.text_log) > 0:
            with open(self.text_log_file, 'a') as f:
                f.write(self.text_log)

        self._flush_scalar_buffer()

        if self.tb is not None:
            self.tb.flush()

    def create_boundary_interior_video(self, fps=10, delete_tmp_plots=False):
        '''
        Write all mesh data including figures and plots to persistent memory
        '''

        # Create combined video
        mesh_video_file = os.path.join(self.log_dir, "mesh_vis.mp4")

        with imageio.get_writer(mesh_video_file, fps=fps, codec='libx264') as writer:
            for i, data in enumerate(self.data_buffer):
                # Get boundary indices
                bound_indices = self.mesh['boundary_indices']
                inter_indices = self.mesh['interior_indices']

                # Plotly figures (kept in memory; these are not matplotlib)
                mesh_boundary_vis, mesh_interior_vis = create_mesh_split_visualization(
                    self.mesh, data, bound_indices, inter_indices
                )

                full_mesh_vis = create_full_mesh_visualization(
                    self.mesh, data
                )

                # Matplotlib histogram: SAVE & CLOSE NOW at a fixed size; keep only the path
                hfig = plot_face_prob_hist(data, bound_indices, inter_indices,
                                        px_width=H_W, px_height=H_H, dpi=H_DPI)

                # Save Plotly images at fixed size (guaranteed)
                boundary_path  = os.path.join(self.image_dir, f"boundary_mesh_{i}.png")
                interior_path  = os.path.join(self.image_dir, f"interior_mesh_{i}.png")
                mesh_html_path = os.path.join(self.image_dir, f"full_mesh_{i}.html")
                hist_path = os.path.join(self.tmp_plot_dir, f"face_prob_hist_{data['iteration']}.png")

                mesh_boundary_vis.write_image(boundary_path, width=PLOTLY_W, height=PLOTLY_H, scale=1)
                mesh_interior_vis.write_image(interior_path, width=PLOTLY_W, height=PLOTLY_H, scale=1)
                full_mesh_vis.write_html(mesh_html_path)
                hfig.savefig(hist_path)  # no bbox_inches="tight" to keep size stable
                plt.close(hfig)
                
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

        #TODO: add some logic for deleting temporary plots

        # Optionally, move hist PNGs from tmp to persistent plot dir
        for p in os.listdir(self.tmp_plot_dir):
            src = os.path.join(self.tmp_plot_dir, p)
            dst = os.path.join(self.plot_dir, p)
            try:
                os.replace(src, dst)
            except Exception:
                pass

        # Clean up tmp dir if desired
        # import shutil; shutil.rmtree(self.tmp_plot_dir, ignore_errors=True)
    def create_boundary_interior_video_fast(self, fps=10, delete_tmp_plots=False):
        '''
        Optimized version: renders directly to memory without disk I/O
        '''
        mesh_video_file = os.path.join(self.log_dir, "mesh_vis.mp4")
        
        # Cache numpy conversions (done once)
        vertices = self.mesh["points"].T.cpu().numpy()
        faces = self.mesh["facets"].T.cpu().numpy()
        bound_indices = self.mesh['boundary_indices']
        inter_indices = self.mesh['interior_indices']
        
        with imageio.get_writer(mesh_video_file, fps=fps, codec='libx264') as writer:
            for i, data in enumerate(self.data_buffer):
                # Create Plotly figures
                mesh_boundary_vis, mesh_interior_vis = create_mesh_split_visualization(
                    self.mesh, data, bound_indices, inter_indices
                )
                
                # Convert directly to numpy (no disk I/O)
                boundary_img = np.array(Image.open(io.BytesIO(
                    mesh_boundary_vis.to_image(format="png", width=PLOTLY_W, height=PLOTLY_H)
                )))
                interior_img = np.array(Image.open(io.BytesIO(
                    mesh_interior_vis.to_image(format="png", width=PLOTLY_W, height=PLOTLY_H)
                )))
                
                # Create matplotlib histogram in memory
                hfig = plot_face_prob_hist(data, bound_indices, inter_indices,
                                        px_width=H_W, px_height=H_H, dpi=H_DPI)
                
                # Convert matplotlib to numpy array
                buf = io.BytesIO()
                hfig.savefig(buf, format='png')
                buf.seek(0)
                plot_img = np.array(Image.open(buf))
                plt.close(hfig)
                buf.close()
                
                # Ensure exact sizes
                boundary_img = _ensure_size(boundary_img, PLOTLY_W, PLOTLY_H)
                interior_img = _ensure_size(interior_img, PLOTLY_W, PLOTLY_H)
                plot_img = _ensure_size(plot_img, H_W, H_H)
                
                # Combine: top row (2 panels), bottom histogram
                top_row = np.hstack((boundary_img, interior_img))
                combined_img = np.vstack((top_row, plot_img))
                
                # Pad to multiple of 16
                combined_img = _pad_to_multiple(combined_img, mult=16)
                
                # Write frame
                writer.append_data(combined_img)    

    def create_loss_plot(self):
        '''
        Create loss plot from logged data
        '''
        data_keys = self.data_buffer[0].keys()
        plot_keys = ['total_loss', 'sim_loss', 'chamfer_loss', 'weighted_sim_loss', 'continuity_loss', 
                     'manifold_reg_loss','edge_reg_loss', 'sparsity_loss', 'binarity_loss']
        
        # Create figure
        plt.figure(figsize=(10, 6), dpi=100)
        
        iterations = [data['iteration'] for data in self.data_buffer]
        for pk in plot_keys:
            if pk in data_keys:
                plot_data = [data[pk] for data in self.data_buffer]
                plt.plot(iterations, plot_data, label=pk)
        
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curves Over Iterations')
        plt.legend()
        plt.grid(True, which="both", ls="--", lw=0.5)

        loss_plot_path = os.path.join(self.log_dir, "loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.close()

    def create_pct_correct_plot(self):
        '''
        Create percentage correct plot from logged data
        '''
        iterations = [data['iteration'] for data in self.data_buffer]
        pct_correct_perm = [data['pct_correct_perm_faces'] for data in self.data_buffer]
        pct_correct_nonperm = [data['pct_correct_nonperm_faces'] for data in self.data_buffer]

        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(iterations, pct_correct_perm, label='Pct Correct Permissible Faces')
        plt.plot(iterations, pct_correct_nonperm, label='Pct Correct Non-permissible Faces')
        plt.ylim(0, 100)
        plt.xlabel('Iteration')
        plt.ylabel('Percentage Correct (%)')
        plt.title('Percentage of Correctly Classified Faces Over Iterations')
        plt.legend()
        plt.grid(True, which="both", ls="--", lw=0.5)

        pct_correct_plot_path = os.path.join(self.log_dir, "pct_correct_plot.png")
        plt.savefig(pct_correct_plot_path)
        plt.close()

    def create_mesh_threshold_video_fast(self, fps=10, delete_tmp_plots=False):
        mesh_video_file = os.path.join(self.log_dir, "mesh_threshold_vis.mp4")
        
        # Cache numpy conversions (done once)
        vertices = self.mesh["points"].T.cpu().numpy()
        faces = self.mesh["facets"].T.cpu().numpy()
        fp_ref = self.ref_data['face_perm'].detach().cpu().numpy()
        
        with imageio.get_writer(mesh_video_file, fps=fps, codec='libx264') as writer:
            for i, data in enumerate(self.data_buffer):
                fp = data['face_perm'].detach().cpu().numpy()
                
                # Create figures (pass pre-computed arrays if you modify the function)
                kwargs = {}
                if "log_threshold" in self.exp_config:
                    kwargs["log_threshold"] = self.exp_config["log_threshold"]

                opt_mesh_vis, ref_mesh_vis = create_mesh_threshold_visualization(
                    self.mesh, self.ref_data['face_perm'], data, **kwargs
                )
                
                # Convert directly to numpy (no disk I/O)
                opt_mesh_img = np.array(Image.open(io.BytesIO(opt_mesh_vis.to_image(format="png", width=PLOTLY_W, height=PLOTLY_H))))
                ref_mesh_img = np.array(Image.open(io.BytesIO(ref_mesh_vis.to_image(format="png", width=PLOTLY_W, height=PLOTLY_H))))
                
                # Combine and write
                combined_img = _pad_to_multiple(np.hstack((opt_mesh_img, ref_mesh_img)), mult=16)
                writer.append_data(combined_img)
            
    def create_mesh_threshold_video(self, fps=10, delete_tmp_plots=False):
        '''
        Write all mesh data including figures and plots to persistent memory
        '''

        # Create combined video
        mesh_video_file = os.path.join(self.log_dir, "mesh_threshold_vis.mp4")

        with imageio.get_writer(mesh_video_file, fps=fps, codec='libx264') as writer:
            for i, data in enumerate(self.data_buffer):
                # Get boundary indices
                bound_indices = self.mesh['boundary_indices']
                inter_indices = self.mesh['interior_indices']

                # Plotly figures (kept in memory; these are not matplotlib)
                opt_mesh_vis, ref_mesh_vis = create_mesh_threshold_visualization(
                    self.mesh, self.ref_data['face_perm'], data
                )

                # Save Plotly images at fixed size (guaranteed)
                opt_mesh_path  = os.path.join(self.image_dir, f"opt_mesh_threshold_{i}.png")
                ref_mesh_path  = os.path.join(self.image_dir, f"ref_mesh_threshold_{i}.png")

                opt_mesh_vis.write_image(opt_mesh_path, width=PLOTLY_W, height=PLOTLY_H, scale=1)
                ref_mesh_vis.write_image(ref_mesh_path, width=PLOTLY_W, height=PLOTLY_H, scale=1)
                
                # Read images
                opt_mesh_img = imageio.imread(opt_mesh_path)
                ref_mesh_img = imageio.imread(ref_mesh_path)

                # Force exact sizes
                opt_mesh_img = _ensure_size(opt_mesh_img, PLOTLY_W, PLOTLY_H)
                ref_mesh_img = _ensure_size(ref_mesh_img, PLOTLY_W, PLOTLY_H)

                # Combine: top row (2 panels), bottom histogram
                combined_img = np.hstack((opt_mesh_img, ref_mesh_img))  # -> (PLOTLY_H, 2*PLOTLY_W, C)

                # Final sanity: pad to 16-multiples (should already be 2000x2000)
                combined_img = _pad_to_multiple(combined_img, mult=16)

                # Write frame
                writer.append_data(combined_img)

        #TODO: add some logic for deleting temporary plots

        # Optionally, move hist PNGs from tmp to persistent plot dir
        for p in os.listdir(self.tmp_plot_dir):
            src = os.path.join(self.tmp_plot_dir, p)
            dst = os.path.join(self.plot_dir, p)
            try:
                os.replace(src, dst)
            except Exception:
                pass

    def close(self):
        self._flush_scalar_buffer()
        if self.tb is not None:
            self.tb.close()
            self.tb = None

    # def create_signal_hardi_video(self, fps=10, delete_tmp_plots=False):
    #     '''
    #     Write all mesh data including figures and plots to persistent memory
    #     '''

    #     # Create combined video
    #     mesh_video_file = os.path.join(self.log_dir, "mesh_threshold_vis.mp4")

    #     with imageio.get_writer(mesh_video_file, fps=fps, codec='libx264') as writer:
    #         for i, data in enumerate(self.data_buffer):
    #             # Plotly figures (kept in memory; these are not matplotlib)
    #             it = data['iteration']

    #             for seq in range(data['signal_normed'].shape[1]):
    #                 hardi_b_1 = create_mesh_hardi_visualization(
    #                     self.exp_config, 
    #                     self.ref_data['signal_normed'][1, seq, :], 
    #                     data['signal_normed'][1, seq, :]
    #                 )

    #                 hardi_b_last = create_mesh_hardi_visualization(
    #                     self.exp_config,
    #                     self.ref_data['signal_normed'][-1, seq, :],
    #                     data['signal_normed'][-1, seq, :]
    #                 )

    #                 # Save Plotly images at fixed size (guaranteed)
    #                 hardi_b_1_path  = os.path.join(self.image_dir, f"hardi_b1_seq{seq}_{it}.png")
    #                 hardi_b_last_path  = os.path.join(self.image_dir, f"hardi_bf_seq{seq}_{it}.png")

    #                 hardi_b_1.write_image(hardi_b_1_path, width=PLOTLY_W, height=PLOTLY_H, scale=1)
    #                 hardi_b_last.write_image(hardi_b_last_path, width=PLOTLY_W, height=PLOTLY_H, scale=1)
                
    #                 # Read images
    #                 hardi_b_1_img = imageio.imread(hardi_b_1_path)
    #                 hardi_b_last_img = imageio.imread(hardi_b_last_path)

    #                 # Force exact sizes
    #                 hardi_b_1_img = _ensure_size(hardi_b_1_img, PLOTLY_W, PLOTLY_H)
    #                 hardi_b_last_img = _ensure_size(hardi_b_last_img, PLOTLY_W, PLOTLY_H)

    #                 # Combine: top row (2 panels), bottom histogram
    #                 combined_img = np.hstack((hardi_b_1_img, hardi_b_last_img))  # -> (PLOTLY_H, 2*PLOTLY_W, C)

    #                 # Final sanity: pad to 16-multiples (should already be 2000x2000)
    #                 combined_img = _pad_to_multiple(combined_img, mult=16)

    #                 # Write frame
    #                 writer.append_data(combined_img)

    #     #TODO: add some logic for deleting temporary plots

    #     # Optionally, move hist PNGs from tmp to persistent plot dir
    #     for p in os.listdir(self.tmp_plot_dir):
    #         src = os.path.join(self.tmp_plot_dir, p)
    #         dst = os.path.join(self.plot_dir, p)
    #         try:
    #             os.replace(src, dst)
    #         except Exception:
                # pass

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