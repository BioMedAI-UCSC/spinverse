#!/usr/bin/env python3
"""
Reconstruction with cosine-annealing blending v3 – multi-cylinder fix.

Builds on v2 with targeted changes that fix the "outline only" problem
on multi-cylinder meshes (cylk2+).  Single-cylinder meshes still work.

Root cause of "outline only"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Interior boundary faces (between two neighbouring cylinders) produce weak
gradient signal compared to the strong outer-boundary faces.  With constant
binarisation and aggressive LR decay, the optimiser commits to the outer
boundary early and has no budget left to carve the interior.

Changes from v2
~~~~~~~~~~~~~~~~
1. **Gradient fill-in boost** (``gradient_fill_boost``, default 1.5)
   After the warmup phase, uncommitted faces get their gradient scaled up.
   This creates a "flood fill" effect: once the outer boundary is found,
   nearby uncommitted faces (including interior cylinder boundaries)
   receive boosted gradients so the carving propagates inward.

2. **Scheduled binarisation** (``lambda_binarize: {start, end}``)
   Instead of a constant weight, binarisation ramps from low -> high.
   Early iterations explore freely; late iterations push to crisp 0/1.
   Prevents premature commitment to the outline-only solution.

3. **Scheduled simulation lambda** (``simulation_lambda: {start, end}``)
   Starts moderate so regularisation helps find the rough shape, then
   increases so the data term can overpower regularisation and carve
   fine interior boundaries.

4. **Adaptive edge coherence** (``adaptive_edge_coherence: true``)
   Edge regularisation weight is modulated by the max commitment of
   the two faces sharing each edge.  If both faces are still undecided,
   the coherence penalty is reduced, allowing the optimiser to explore
   interior boundary patterns without being smoothed out.

5. **No double LR decay** – StepLR gamma defaults to 1.0 (disabled).
   The cosine LR schedule alone provides sufficient smooth decay,
   avoiding the compound decay that starves late-stage optimisation.

6. **Fewer cosine cycles (default 2)** – each cycle gets more
   iterations, making transitions smoother.

Config example (YAML):

  iterations: 1000
  gamma_decay: 1.0
  simulation_lambda:
    start: 50.0
    end: 200.0
  lambda_binarize:
    start: 0.05
    end: 1.5
  gradient_fill_boost: 1.5
  adaptive_edge_coherence: true
  dt_cosine_anneal:
    warmup_iters: 80
    n_cycles: 2
    alpha_warmup_floor: 0.2
"""

import os
import argparse
import importlib
import imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import io
import json
import math
import time
import yaml
import inspect
from itertools import combinations
from pytorch3d.ops import knn_points

from src.get_vol_sa import get_vol_sa
from src.calculate_generalized_mean_diffusivity import calculate_generalized_mean_diffusivity
from src.length2eig import length2eig
from src.compute_laplace_eig_diff_v5_fast import (
    precompute_laplace_matrices,
    compute_laplace_eig_diff,
    precompute_reduced_constants,
    compute_laplace_reduced_projections,
)
from src.eig2length import eig2length
from src.solve_mf_v5_e import solve_mf, solve_mf_reduced
from src.split_mesh import split_mesh, split_mesh_per_element, split_mesh_fast_per_element
from src.ellipsoidal_scale_mesh import ellipsoidal_scale_mesh
from mindiffdt.tetra import TetraSet
from mesh_setup import microstructure
importlib.reload(microstructure)
from mesh_setup.microstructure import microstructure
from mesh_setup.mesh_logger_v3 import MeshLogger
from mesh_setup.mesh_utils import get_split_indices

from plot.signal_video_plot import signal_history_to_videos
from plot.plot_faces_prob_history import faces_prob_history_to_video_plot
from plot.eigen_value_plot import eigen_history_to_video

from mesh_setup.experiment import ExperimentSetup
from plot.plot_mesh_from_faces import plot_mesh_from_faces
from src.edge_regularizer import edge_to_faces
from mesh_setup.debug_plotters import *


DT_SHORT = 0
DT_MID = 1
DT_LONG = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset_adam_state_(optimizer: torch.optim.Optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state[p]
            st["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
            st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if "max_exp_avg_sq" in st:
                st["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)


def _as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _deep_update(base: dict, upd: dict) -> dict:
    if upd is None:
        return base
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping/dict at top-level. Got: {type(cfg)}")
    return cfg


def _resolve_device(device_str):
    if device_str is None or str(device_str).strip() == "":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    ds = str(device_str)
    if ds.startswith("cuda") and (not torch.cuda.is_available()):
        return "cpu"
    return ds


def _default_config() -> dict:
    return {
        # Mesh and experiment conig
        "file": None,
        "experiment": "mesh_setup/setup_files/cylinder_config_ls2.yaml",
        "target_mesh": "mesh_setup/mesh_files/torus_majr10_minr6.pth",
        "mesh_scale": 1.0,
        "iterations": None,
        "log_iter": 5,
        "clamp_vals": [1e-5, 0.1],
        "init_val": 1e-3,
        # Learning rate config
        "learning_rate": 0.6,
        "gamma_decay": 1.0,          # v3: default 1.0 = no StepLR decay
        "step_size": 400,
        # LR scheduler config: supports 'linear' (StepLR) or 'lambda' (LambdaLR)
        "lr_schedule": {
            "type": "lambda",           # 'linear' or 'lambda'
            # Linear (StepLR) params:
            "step_size": 400,           # for StepLR
            "gamma": 1.0,               # for StepLR
            # Lambda (LambdaLR) params:
            "cycle_iters": None,        # cycle length (None = use iterations)
            "warmup_iters": 500,        # warmup steps per cycle
            "min_lr_factor": 0.05,      # min LR multiplier (at end of cycle)
        },
        # Tau decades
        "log_sigmoid_tau": 1.0,
        "edge_tau_decades": 0.5,
        # Soft manifold parameters
        "soft_manifold": {
            "enabled": False,          # Use soft-min manifold regularizer
            "softmin_tau": 0.1,        # Temperature for soft-min (0.05-0.2)
            "degree_norm": True,       # Normalize by edge degree
        },
        # Regularization lambdas
        "lambda_continuity": {"start": 1.0, "end": 1.0, "block_len": None},
        "lambda_manifold_reg": {"start": 0.0, "end": 0.0},
        "lambda_edge_reg": {"start": 0.01, "end": 0.67, "block_len": None},
        "lambda_binarize": {"start": 0.05, "end": 1.5},
        # v3: simulation_lambda can be a schedule {start, end}
        "simulation_lambda": {"start": 50.0, "end": 200.0},
        "dt_schedule": [
            {"range": [0, 200], "dt_id": 2},
            {"range": [200, 400], "dt_id": 1},
        ],
        # ---- cosine-annealing blend ----
        "dt_cosine_anneal": {
            "dt_old": 2,
            "dt_new": 1,
            "warmup_iters": 80,       # v3: longer warmup
            "n_cycles": 2,            # v3: fewer, longer cycles
            "cycle_mult": 1.0,
            "restart_fraction": 0.2,
            "reset_adam_per_cycle": False,
            "post_switch_lr": 0.12,
            "lr_anneal_cosine": True,
            "grad_clip_norm_post": 0.25,
            "alpha_warmup_floor": 0.2, # v3: higher floor
        },
        # legacy dt_blend kept for backward compat
        "dt_blend": None,
        # ---- v2/v3: regularisation ----
        # v3: lambda_binarize can be a schedule {start, end}
        "edge_reg_anneal_with_alpha": 0.5,
        # ---- v3: new params ----
        "gradient_fill_boost": 1.5,          # boost for uncommitted faces
        "adaptive_edge_coherence": True,      # modulate edge reg by commitment
        "adaptive_edge_floor": 0.2,           # min edge weight for uncommitted pairs
        # ----
        "scalar_flush_every": 5,
        "device": None,
        "notes": "cosine-annealing v3 (multi-cylinder fix)",
        "tet_grid_cache_dir": None,
        "plot_mesh": True,
        "grad_clip_norm": 1.0,
        "chamfer_n_samples": 10,  # number of points to sample per face for chamfer distance
        # ---------------- merged toggles ----------------
        "loss_blend_anneal": True,
        "grad_clip_anneal": True,
        "gradient_fill": {"enabled": True, "boost": 0.0, "start": "warmup"},
        "rom": {
            "enabled": False,
            "start_iteration": 0,
            "eig_refresh_every": 10,
        },

    }


def _normalize_dt_schedule(dt_schedule):
    if dt_schedule is None:
        return []
    if not isinstance(dt_schedule, list):
        raise ValueError("dt_schedule must be a list of {range:[start,end], dt_id:int} entries")
    out = []
    for s in dt_schedule:
        if not isinstance(s, dict):
            raise ValueError("dt_schedule entries must be dicts")
        if "range" not in s or "dt_id" not in s:
            raise ValueError("dt_schedule entries must have keys: range, dt_id")
        r = s["range"]
        if not (isinstance(r, (list, tuple)) and len(r) == 2):
            raise ValueError("dt_schedule.range must be [start,end]")
        a, b = int(r[0]), int(r[1])
        out.append({"range": (a, b), "dt_id": int(s["dt_id"])})
    out.sort(key=lambda x: x["range"][0])
    return out


def get_current_dt_id(epoch, dt_schedule):
    for stage in dt_schedule:
        a, b = stage["range"]
        if a <= epoch < b:
            return int(stage["dt_id"])
    return None


def _ensure_tet_grid_cache_dir(outdir: str, cfg: dict) -> str:
    cache_dir = cfg.get("tet_grid_cache_dir", None)
    if cache_dir is None or str(cache_dir).strip() == "":
        cache_dir = os.path.join(outdir, "fem_tet_grids")
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _patch_split_mesh_cache_dir(cache_dir: str):
    import src.split_mesh as _sm
    for name in ["TET_GRID_DIR", "TETGRID_DIR", "TET_GRID_CACHE_DIR", "FEM_TET_GRID_DIR", "FEM_TET_GRIDS_DIR"]:
        if hasattr(_sm, name):
            setattr(_sm, name, cache_dir)
    if hasattr(_sm, "DEFAULT_TET_GRID_DIR"):
        setattr(_sm, "DEFAULT_TET_GRID_DIR", cache_dir)
    if hasattr(_sm, "DEFAULT_CACHE_DIR"):
        setattr(_sm, "DEFAULT_CACHE_DIR", cache_dir)


def _guard_default_relative_cache_paths():
    os.makedirs(os.path.join("mesh_setup", "fem_tet_grids"), exist_ok=True)


def _make_mesh_logger(mesh_like, cfg: dict, ref_data: dict, outdir: str):
    sig = inspect.signature(MeshLogger.__init__)
    kwargs = {}
    if "log_dir" in sig.parameters:
        kwargs["log_dir"] = outdir
    return MeshLogger(mesh_like, cfg, ref_data, **kwargs)


def find_face_indices(all_faces: torch.Tensor,
                      query_faces: torch.Tensor,
                      oriented: bool = False) -> torch.Tensor:
    assert all_faces.ndim == 2 and all_faces.shape[0] == 3
    assert query_faces.ndim == 2 and query_faces.shape[0] == 3

    af = all_faces.long().contiguous()
    qf = query_faces.long().contiguous()

    if not oriented:
        af = torch.sort(af, dim=0)[0]
        qf = torch.sort(qf, dim=0)[0]

    base = max(int(af.max().item()), int(qf.max().item())) + 1
    hash_af = af[0] + base * af[1] + (base ** 2) * af[2]
    hash_qf = qf[0] + base * qf[1] + (base ** 2) * qf[2]

    order = torch.argsort(hash_af)
    haf_sorted = hash_af[order]

    pos = torch.searchsorted(haf_sorted, hash_qf)
    in_range = pos < haf_sorted.numel()
    same = torch.zeros_like(pos, dtype=torch.bool)
    same[in_range] = (haf_sorted[pos[in_range]] == hash_qf[in_range])

    idx_sorted = torch.where(same, pos, torch.full_like(pos, -1))
    idx = torch.full_like(idx_sorted, -1)
    valid = idx_sorted != -1
    idx[valid] = order[idx_sorted[valid]]
    return idx


def get_face_centroids(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    V: (V, 3) float tensor (cuda)
    F: (F, 3) long tensor (cuda)
    Returns: (F, 3) centroids
    """
    V = V.transpose(0, 1).contiguous()
    F = F.transpose(0, 1).contiguous()
    return (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0


def get_face_points_3p(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Returns (4F, 3): [centroid, m01, m12, m20] per face.
    V_3xV: (3,V) float
    F_3xF: (3,F) long/int indices
    """
    assert V.shape[0] == 3 and F.shape[0] == 3
    V = V.transpose(0, 1).contiguous()          # (V,3)
    F = F.transpose(0, 1).contiguous().long()   # (F,3)

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    c   = (v0 + v1 + v2) / 3.0
    m01 = 0.5 * (v0 + v1)
    m12 = 0.5 * (v1 + v2)
    m20 = 0.5 * (v2 + v0)

    return torch.cat([c, m01, m12, m20], dim=0)     # (4F,3)


def get_face_points_sampled(V: torch.Tensor, F: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
    """
    Sample n_samples points per face using deterministic barycentric coordinates.
    V: (3, V) float tensor
    F: (3, F) long tensor
    n_samples: number of points to sample per face
    Returns: (n_samples*F, 3) sampled points
    """
    assert V.shape[0] == 3 and F.shape[0] == 3
    Vt = V.transpose(0, 1).contiguous()  # (V, 3)
    Ft = F.transpose(0, 1).contiguous().long()  # (F, 3)
    
    v0 = Vt[Ft[:, 0]]  # (F, 3)
    v1 = Vt[Ft[:, 1]]  # (F, 3)
    v2 = Vt[Ft[:, 2]]  # (F, 3)
    
    # Generate deterministic barycentric coordinates
    # Use a regular grid pattern for consistent sampling
    samples = []
    step = 1.0 / (int(n_samples**0.5) + 1)
    count = 0
    
    u_vals = []
    v_vals = []
    for i in range(int(n_samples**0.5) + 2):
        for j in range(int(n_samples**0.5) + 2):
            u = i * step
            v = j * step
            if u + v <= 1.0 and count < n_samples:
                u_vals.append(u)
                v_vals.append(v)
                count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break
    
    # Fill remaining samples if needed (edge/corner points)
    while len(u_vals) < n_samples:
        u_vals.append(0.333)
        v_vals.append(0.333)
    
    # Truncate if we got too many
    u_vals = u_vals[:n_samples]
    v_vals = v_vals[:n_samples]
    
    # Sample points using barycentric coordinates
    all_points = []
    for u, v in zip(u_vals, v_vals):
        w = 1.0 - u - v
        point = w * v0 + u * v1 + v * v2  # (F, 3)
        all_points.append(point)
    
    return torch.cat(all_points, dim=0)  # (n_samples*F, 3)

@torch.no_grad()
def chamfer_centroid_loss(
    Vopt: torch.Tensor,
    Fopt: torch.Tensor,
    Cref: torch.Tensor,
    squared: bool = True,
    n_samples: int = 10,
) -> torch.Tensor:
    if Fopt.numel() == 0 or Cref.numel() == 0:
        return torch.zeros((), device=Vopt.device)
    Copt = get_face_points_sampled(Vopt, Fopt, n_samples=n_samples)
    if Copt.numel() == 0:
        return torch.zeros((), device=Vopt.device)
    Copt_b = Copt.unsqueeze(0)
    Cref_b = Cref.unsqueeze(0)
    d_opt_to_ref = knn_points(Copt_b, Cref_b, K=1).dists[..., 0]
    d_ref_to_opt = knn_points(Cref_b, Copt_b, K=1).dists[..., 0]
    if squared:
        return d_opt_to_ref.mean() + d_ref_to_opt.mean()
    return torch.sqrt(torch.clamp(d_opt_to_ref, min=0)).mean() + torch.sqrt(torch.clamp(d_ref_to_opt, min=0)).mean()

@torch.no_grad()
def chamfer_centroid_loss_centered(
    Vopt: torch.Tensor,
    Fopt: torch.Tensor,
    Cref: torch.Tensor,
    squared: bool = True,
    eps: float = 1e-12,
    n_samples: int = 10,
) -> torch.Tensor:
    """Translation-invariant chamfer using centering."""
    if Fopt.numel() == 0 or Cref.numel() == 0:
        return torch.zeros((), device=Vopt.device)
    Copt = get_face_points_sampled(Vopt, Fopt, n_samples=n_samples)
    if Copt.numel() == 0:
        return torch.zeros((), device=Vopt.device)
    Copt_c = Copt - Copt.mean(dim=0, keepdim=True)
    Cref_c = Cref - Cref.mean(dim=0, keepdim=True)
    d_opt_to_ref2 = knn_points(Copt_c.unsqueeze(0), Cref_c.unsqueeze(0), K=1).dists[..., 0]
    d_ref_to_opt2 = knn_points(Cref_c.unsqueeze(0), Copt_c.unsqueeze(0), K=1).dists[..., 0]
    if squared:
        return d_opt_to_ref2.mean() + d_ref_to_opt2.mean()
    return d_opt_to_ref2.clamp_min(eps).sqrt().mean() + d_ref_to_opt2.clamp_min(eps).sqrt().mean()


@torch.no_grad()
def compute_edge_structure_metrics(faces: torch.Tensor, nonperm_mask: torch.Tensor) -> dict:
    """
    Compute structural metrics for non-manifold edges.
    
    Args:
        faces: (3, F) tensor of face vertex indices
        nonperm_mask: (F,) boolean mask indicating non-permeable faces
    
    Returns:
        dict with 'open_edge_pct', 'branch_edge_pct', 'bad_edge_pct'
    """
    device = faces.device
    
    # Get non-permeable faces only
    nonperm_faces = faces[:, nonperm_mask]  # (3, F_nonperm)
    
    if nonperm_faces.numel() == 0:
        return {
            'open_edge_pct': 0.0,
            'branch_edge_pct': 0.0,
            'bad_edge_pct': 0.0,
            'num_edges': 0,
            'num_open_edges': 0,
            'num_branch_edges': 0,
            'num_bad_edges': 0,
        }
    
    # Extract all edges from non-permeable faces
    # Each triangle has 3 edges: (v0,v1), (v1,v2), (v2,v0)
    F_t = nonperm_faces.T  # (F_nonperm, 3)
    edges_list = []
    edges_list.append(torch.stack([F_t[:, 0], F_t[:, 1]], dim=1))  # edge (v0, v1)
    edges_list.append(torch.stack([F_t[:, 1], F_t[:, 2]], dim=1))  # edge (v1, v2)
    edges_list.append(torch.stack([F_t[:, 2], F_t[:, 0]], dim=1))  # edge (v2, v0)
    all_edges = torch.cat(edges_list, dim=0)  # (3*F_nonperm, 2)
    
    # Sort each edge so (v0, v1) and (v1, v0) are the same
    all_edges_sorted = torch.sort(all_edges, dim=1)[0]  # (3*F_nonperm, 2)
    
    # Count edge occurrences using unique
    # Create a hash for each edge
    max_vertex = all_edges_sorted.max().item() + 1
    edge_hash = all_edges_sorted[:, 0] * max_vertex + all_edges_sorted[:, 1]
    
    unique_hashes, counts = torch.unique(edge_hash, return_counts=True)
    
    total_edges = len(unique_hashes)
    if total_edges == 0:
        return {
            'open_edge_pct': 0.0,
            'branch_edge_pct': 0.0,
            'bad_edge_pct': 0.0,
            'num_edges': 0,
            'num_open_edges': 0,
            'num_branch_edges': 0,
            'num_bad_edges': 0,
        }
    
    # Calculate metrics
    # d(e) = counts for each edge
    open_edges = (counts == 1).sum().item()      # degree 1
    branch_edges = (counts > 2).sum().item()     # degree > 2
    bad_edges = (counts != 2).sum().item()       # degree != 2
    
    return {
        'open_edge_pct': 100.0 * open_edges / total_edges,
        'branch_edge_pct': 100.0 * branch_edges / total_edges,
        'bad_edge_pct': 100.0 * bad_edges / total_edges,
        'num_edges': total_edges,
        'num_open_edges': open_edges,
        'num_branch_edges': branch_edges,
        'num_bad_edges': bad_edges,
    }




# ---------------------------------------------------------------------------
# Cosine-annealing warm-restart schedule
# ---------------------------------------------------------------------------

def _cosine_ramp(u: float) -> float:
    """Simple half-cosine: 0->1 as u goes 0->1."""
    u = max(0.0, min(1.0, u))
    return 0.5 - 0.5 * math.cos(math.pi * u)


def cosine_anneal_alpha(
    it: int,
    warmup: int,
    total_iters: int,
    n_cycles: int = 1,
    cycle_mult: float = 1.0,
    restart_fraction: float = 0.25,
) -> float:
    """
    Compute blend weight alpha in [0, 1] using cosine annealing with warm
    restarts.

    * During warmup (it < warmup): alpha = 0  (pure dt_old).
    * After all cycles (it >= total_iters): alpha = 1  (pure dt_new).
    * Between cycles the blend "restarts" -- alpha drops by restart_fraction
      of the previous cycle's range -- then climbs higher via a cosine curve.

    When n_cycles=1 this reduces to a smooth S-curve from 0->1.
    """
    if it < warmup:
        return 0.0
    if it >= total_iters:
        return 1.0

    t = float(it - warmup)
    T_eff = float(total_iters - warmup)

    # ---- single cycle: clean S-curve ----
    if n_cycles <= 1:
        return _cosine_ramp(t / T_eff)

    # ---- multiple cycles: warm restarts ----
    # Compute cycle boundaries.
    if abs(cycle_mult - 1.0) < 1e-9:
        T_0 = T_eff / n_cycles
    else:
        T_0 = T_eff * (1.0 - cycle_mult) / (1.0 - cycle_mult ** n_cycles)

    cum = 0.0
    cycle_idx = n_cycles - 1
    T_c = T_0
    for c in range(n_cycles):
        T_c = T_0 * (cycle_mult ** c)
        if cum + T_c > t + 1e-9:
            cycle_idx = c
            break
        cum += T_c

    t_in_cycle = t - cum

    # Target alpha at end of this cycle ramps linearly 0->1 across cycles.
    alpha_end = (cycle_idx + 1) / n_cycles
    # Alpha at start of this cycle.
    if cycle_idx == 0:
        alpha_start = 0.0
    else:
        alpha_prev_end = cycle_idx / n_cycles
        # On restart drop by restart_fraction of the range we just covered.
        drop = restart_fraction * (alpha_prev_end - ((cycle_idx - 1) / n_cycles))
        alpha_start = alpha_prev_end - drop

    # Cosine within cycle.
    u = t_in_cycle / max(T_c, 1e-9)
    cos_val = _cosine_ramp(u)
    alpha = alpha_start + (alpha_end - alpha_start) * cos_val
    return max(0.0, min(1.0, alpha))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False, type=str, help="Path to YAML config file")
    parser.add_argument("--outdir", required=False, type=str, help="Output directory for logs/artifacts")
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    cfg = _default_config()
    user_cfg = _load_yaml_config(args.config)
    cfg = _deep_update(cfg, user_cfg)

    cfg["config_path"] = os.path.abspath(args.config)
    cfg["outdir"] = outdir
    cfg["file"] = os.path.basename(__file__)

    if cfg.get("scalar_flush_every", None) is not None:
        os.environ["SCALAR_FLUSH_EVERY"] = str(int(cfg["scalar_flush_every"]))

    device_str = _resolve_device(cfg.get("device", None))
    DEVICE = torch.device(device_str)

    dt_schedule = _normalize_dt_schedule(cfg.get("dt_schedule", None))
    if cfg.get("iterations", None) is None:
        if len(dt_schedule) == 0:
            raise ValueError("iterations is None and dt_schedule is empty; set iterations or provide dt_schedule.")
        NUM_ITERS = int(max(s["range"][1] for s in dt_schedule))
    else:
        NUM_ITERS = int(cfg["iterations"])

    LOG_ITER = int(cfg.get("log_iter", 5))
    CLAMP_MIN = float(cfg["clamp_vals"][0])
    CLAMP_MAX = float(cfg["clamp_vals"][1])
    TARGET_MESH = str(cfg["target_mesh"])
    MESH_SCALE = float(cfg.get("mesh_scale", 1.0))
    EXPERIMENT = str(cfg["experiment"])

    cache_dir = _ensure_tet_grid_cache_dir(outdir, cfg)
    _patch_split_mesh_cache_dir(cache_dir)
    _guard_default_relative_cache_paths()

    # ------------------------------------------------------------------
    # Parse cosine-annealing blend parameters (fall back to dt_blend)
    # ------------------------------------------------------------------
    # ---------------- merged toggles ----------------
    loss_blend_anneal = bool(cfg.get("loss_blend_anneal", True))

    grad_clip_anneal = bool(cfg.get("grad_clip_anneal", True))

    binarize_on = str(cfg.get("binarize_on", "p_non"))
    binarize_sched = cfg.get("lambda_binarize", {"start": 0.0, "end": 0.0})

    grad_fill_cfg = cfg.get("gradient_fill", {}) if isinstance(cfg.get("gradient_fill", {}), dict) else {}
    gradient_fill_boost = float(grad_fill_cfg.get("boost", cfg.get("gradient_fill_boost", 0.0)))
    _gstart = grad_fill_cfg.get("start", "warmup")

    ca_cfg = cfg.get("dt_cosine_anneal", None) or cfg.get("dt_blend", {}) or {}

    dt_old = int(ca_cfg.get("dt_old", DT_LONG))
    dt_new = int(ca_cfg.get("dt_new", DT_MID))
    warmup_iters = int(ca_cfg.get("warmup_iters", ca_cfg.get("switch_iter", 50)))
    gradient_fill_start = warmup_iters if (isinstance(_gstart, str) and str(_gstart).lower() == "warmup") else int(_gstart)
    n_cycles = int(ca_cfg.get("n_cycles", 1))
    cycle_mult = float(ca_cfg.get("cycle_mult", 1.0))
    restart_fraction = float(ca_cfg.get("restart_fraction", 0.25))
    reset_adam_per_cycle = bool(ca_cfg.get("reset_adam_per_cycle",
                                            ca_cfg.get("reset_adam_on_switch", False)))
    lr_anneal_cosine = bool(ca_cfg.get("lr_anneal_cosine", True))

    alpha_warmup_floor = float(ca_cfg.get("alpha_warmup_floor", 0.2))

    base_lr = float(cfg["learning_rate"])
    post_lr = _as_float(ca_cfg.get("post_switch_lr", base_lr * 0.2), base_lr * 0.2)

    base_clip = _as_float(cfg.get("grad_clip_norm", 1.0), 1.0)
    post_clip = _as_float(ca_cfg.get("grad_clip_norm_post", min(base_clip, 0.25)),
                           min(base_clip, 0.25))

    # v2 params (carried over)
    edge_reg_anneal_rate = float(cfg.get("edge_reg_anneal_with_alpha", 0.5))

    # v3 params
    gradient_fill_boost = float(cfg.get("gradient_fill_boost", 1.5))
    use_adaptive_edge = bool(cfg.get("adaptive_edge_coherence", True))
    adaptive_edge_floor = float(cfg.get("adaptive_edge_floor", 0.2))
    rom_cfg = cfg.get("rom", {}) if isinstance(cfg.get("rom", {}), dict) else {}
    rom_enabled = bool(rom_cfg.get("enabled", False))
    rom_start_iteration = int(rom_cfg.get("start_iteration", 0))
    rom_eig_refresh_every = int(rom_cfg.get("eig_refresh_every", 10))

    # v3: resolve scheduled params (accept scalar OR {start, end} dict)
    def _resolve_schedule(cfg_val, default_start, default_end=None):
        if default_end is None:
            default_end = default_start
        if isinstance(cfg_val, dict):
            return {
                "start": _as_float(cfg_val.get("start", default_start), default_start),
                "end": _as_float(cfg_val.get("end", default_end), default_end),
                "block_len": cfg_val.get("block_len", None),
            }
        v = _as_float(cfg_val, default_start)
        return {"start": v, "end": v, "block_len": None}

    sim_lambda_sched = _resolve_schedule(cfg.get("simulation_lambda", 100.0), 50.0, 200.0)
    binarize_sched = _resolve_schedule(cfg.get("lambda_binarize", 0.5), 0.05, 1.5)

    print("=" * 72)
    print("Cosine-Annealing Blend v3 (multi-cylinder fix)")
    print("=" * 72)
    print(f"Config YAML : {args.config}")
    print(f"Outdir      : {outdir}")
    print(f"Device      : {device_str}")
    print(f"Experiment  : {EXPERIMENT}")
    print(f"Target mesh : {TARGET_MESH}")
    print(f"Iterations  : {NUM_ITERS}")
    print(f"DT schedule : {dt_schedule}")
    print(f"Cosine anneal: warmup={warmup_iters}, n_cycles={n_cycles}, "
          f"cycle_mult={cycle_mult}, restart_frac={restart_fraction}")
    print(f"  dt_old={dt_old}, dt_new={dt_new}")
    print(f"  base_lr={base_lr}, post_lr={post_lr}, lr_anneal_cosine={lr_anneal_cosine}")
    print(f"  base_clip={base_clip}, post_clip={post_clip}")
    print(f"  alpha_warmup_floor={alpha_warmup_floor}")
    print(f"v3 params:")
    print(f"  gradient_fill_boost={gradient_fill_boost}")
    print(f"  adaptive_edge_coherence={use_adaptive_edge} (floor={adaptive_edge_floor})")
    print(f"  simulation_lambda schedule: {sim_lambda_sched}")
    print(f"  lambda_binarize schedule: {binarize_sched}")
    print(f"  edge_reg_anneal_with_alpha={edge_reg_anneal_rate}")
    print(f"  gamma_decay={cfg.get('gamma_decay', 1.0)}")
    print(f"ROM: enabled={rom_enabled}, start_iteration={rom_start_iteration}, eig_refresh_every={rom_eig_refresh_every}")
    print(f"Tet-grid cache dir: {cache_dir}")

    # ------------------------------------------------------------------
    # Reference mesh
    # ------------------------------------------------------------------
    mesh_data = torch.load(TARGET_MESH, map_location=DEVICE, weights_only=False)
    mesh = {
        "points": mesh_data["points"],
        "facets": mesh_data["facets"],
        "boundary_faces": mesh_data.get("boundary_faces", None),
        "elements": mesh_data["elements"],
    }
    if mesh["boundary_faces"] is None:
        raise ValueError("target_mesh .pth must contain key 'boundary_faces'.")

    print(
        f"Mesh: {mesh['points'].shape[1]} points, "
        f"{mesh['facets'].shape[1]} facets, "
        f"{mesh['elements'].shape[1]} elements"
    )
    cfg["num_points"] = int(mesh["points"].shape[1])
    cfg["num_faces"] = int(mesh["facets"].shape[1])
    cfg["num_elements"] = int(mesh["elements"].shape[1])

    setup = ExperimentSetup(EXPERIMENT)
    cfg["length_scale"] = float(setup.mf["length_scale"])

    Fm = mesh["facets"].shape[1]
    Em = mesh["elements"].shape[1]
    elementmarkers = torch.arange(Em, dtype=torch.long, device=DEVICE)
    facetmarkers = torch.arange(Fm, dtype=torch.long, device=DEVICE)

    femesh = {
        "points": mesh["points"],
        "facets": mesh["facets"],
        "boundary_faces": mesh["boundary_faces"],
        "elements": mesh["elements"],
        "facetmarkers": facetmarkers,
        "elementmarkers": elementmarkers,
    }

    print("FeMesh initialized")

    femesh_scaled = ellipsoidal_scale_mesh(
        femesh, torch.tensor([MESH_SCALE, MESH_SCALE, MESH_SCALE], device=DEVICE)
    )

    verts_ref = femesh_scaled["points"]
    faces_ref = femesh_scaled["facets"]

    if bool(cfg.get("plot_mesh", True)):
        plot_mesh_from_faces(femesh_scaled["points"].T.cpu().numpy(),
                             femesh_scaled["boundary_faces"].T.cpu().numpy())
        plot_mesh_from_faces(femesh_scaled["points"].T.cpu().numpy(),
                             femesh_scaled["facets"].T.cpu().numpy())

    print("FeMesh scaled")

    femesh_split = split_mesh_fast_per_element(femesh_scaled, tet_grid=True)
    ncomp_ref = femesh_split["ncompartment"]

    D0 = setup.pde["diffusivity"].float()
    setup.pde["diffusivity"] = D0.repeat(ncomp_ref, 1, 1)
    setup.pde["relaxation"] = setup.pde["relaxation_out"] * torch.ones(ncomp_ref, device=DEVICE)
    setup.pde["initial_density"] = setup.pde["initial_density_out"] * torch.ones(ncomp_ref, device=DEVICE)

    print("FeMesh Split")
    surface_area_contrib = torch.ones(Fm, device=DEVICE)
    volumes, _ = get_vol_sa(femesh_split, surface_area_contrib)
    print("FeMesh Volume computed")

    face_perm_ref = torch.full((Fm,), CLAMP_MAX, device=DEVICE)
    bound_face_indices = find_face_indices(mesh["facets"], mesh["boundary_faces"])
    all_face_indices = torch.arange(Fm, device=DEVICE)
    mask = ~torch.isin(all_face_indices, bound_face_indices)
    inter_face_indices = all_face_indices[mask]
    mesh["boundary_indices"] = bound_face_indices
    mesh["interior_indices"] = inter_face_indices

    face_perm_ref[bound_face_indices] = CLAMP_MIN
    print(f"Reference: Set {bound_face_indices.numel()} boundary faces to CLAMP_MIN.")

    C_ref = get_face_points_sampled(verts_ref, faces_ref[:, bound_face_indices], n_samples=int(cfg.get("chamfer_n_samples", 10)))

    # ------------------------------------------------------------------
    # Optimisation mesh (same geometry; separate tensors)
    # ------------------------------------------------------------------
    opt_mesh_data = torch.load(TARGET_MESH, map_location=DEVICE, weights_only=False)
    opt_mesh = {
        "points": opt_mesh_data["points"],
        "facets": opt_mesh_data["facets"],
        "boundary_faces": opt_mesh_data.get("boundary_faces", None),
        "elements": opt_mesh_data["elements"],
    }
    if opt_mesh["boundary_faces"] is None:
        opt_mesh["boundary_faces"] = mesh["boundary_faces"]

    print(
        f"Optimization Mesh: {opt_mesh['points'].shape[1]} points, "
        f"{opt_mesh['facets'].shape[1]} facets, "
        f"{opt_mesh['elements'].shape[1]} elements"
    )

    F_opt = opt_mesh["facets"].shape[1]
    E_opt = opt_mesh["elements"].shape[1]
    elementmarkers_opt = torch.arange(E_opt, dtype=torch.long, device=DEVICE)
    facetmarkers_opt = torch.arange(F_opt, dtype=torch.long, device=DEVICE)

    femesh_opt = {
        "points": opt_mesh["points"],
        "facets": opt_mesh["facets"],
        "boundary_faces": opt_mesh["boundary_faces"],
        "elements": opt_mesh["elements"],
        "facetmarkers": facetmarkers_opt,
        "elementmarkers": elementmarkers_opt,
    }

    print("Optimization FeMesh initialized")

    femesh_scaled_opt = ellipsoidal_scale_mesh(
        femesh_opt, torch.tensor([MESH_SCALE, MESH_SCALE, MESH_SCALE], device=DEVICE)
    )
    print("Optimization FeMesh scaled")

    femesh_split_opt = split_mesh_fast_per_element(femesh_scaled_opt, tet_grid=True)
    print("Optimization FeMesh split")

    surface_area_contrib_opt = torch.ones(F_opt, device=DEVICE)
    volumes_opt, _ = get_vol_sa(femesh_split_opt, surface_area_contrib_opt)
    print("Optimization FeMesh Volume computed")

    # ------------------------------------------------------------------
    # Learnable face permeability
    # ------------------------------------------------------------------
    k_min, k_max = CLAMP_MIN, CLAMP_MAX
    k_init = float(cfg["init_val"])
    k_tau = float(cfg.get("log_sigmoid_tau", 1.0))

    log_k_min = math.log10(k_min)
    log_k_max = math.log10(k_max)
    alpha_init = (math.log10(k_init) - log_k_min) / (log_k_max - log_k_min)
    alpha_init = min(max(alpha_init, 1e-6), 1 - 1e-6)

    face_perm_v = torch.full(
        (F_opt,),
        torch.logit(torch.tensor(alpha_init, device=DEVICE)),
        device=DEVICE,
        requires_grad=True,
    )

    def hook_stat(name):
        def _h(grad):
            print(f"{name} grad: mean={grad.mean():.2e}, absmean={grad.abs().mean():.2e}, norm={grad.norm():.2e}")
        return _h

    face_perm_v.register_hook(hook_stat("face_perm_v"))

    def current_log_perm():
        return log_k_min + (log_k_max - log_k_min) * torch.sigmoid(face_perm_v / k_tau)

    def current_perm():
        return 10.0 ** current_log_perm()

    optimizer = optim.Adam([face_perm_v], lr=base_lr, foreach=False, fused=False)
    
    # ------------------------------------------------------------------
    # Learning rate scheduler (configurable: linear or lambda)
    # ------------------------------------------------------------------
    lr_sched_cfg = cfg.get("lr_schedule", {"type": "linear"})
    lr_sched_type = lr_sched_cfg.get("type", "linear").lower()
    
    if lr_sched_type == "linear":
        # StepLR: linear decay by gamma every step_size iterations
        step_size = int(lr_sched_cfg.get("step_size", cfg.get("step_size", 400)))
        gamma = float(lr_sched_cfg.get("gamma", cfg.get("gamma_decay", 1.0)))
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        print(f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}")
    
    elif lr_sched_type == "lambda":
        # LambdaLR: warmup + cosine decay with restarts
        cycle_iters = lr_sched_cfg.get("cycle_iters")
        if cycle_iters is None:
            cycle_iters = NUM_ITERS  # default: one cycle = full training
        else:
            cycle_iters = int(cycle_iters)
        
        warmup_iters = int(lr_sched_cfg.get("warmup_iters", 500))
        min_lr_factor = float(lr_sched_cfg.get("min_lr_factor", 0.05))
        
        def lr_lambda(step: int):
            # position within the current cycle
            u = step % cycle_iters  # 0..cycle_iters-1
            
            # warmup each cycle
            if u < warmup_iters:
                return (u + 1) / max(1, warmup_iters)  # 0->1 warmup
            
            # cosine decay within cycle
            progress = (u - warmup_iters) / max(1, (cycle_iters - warmup_iters))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1->0
            return min_lr_factor + (1.0 - min_lr_factor) * cosine
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        print(f"Using LambdaLR scheduler: cycle={cycle_iters}, warmup={warmup_iters}, min_factor={min_lr_factor}")
    
    else:
        raise ValueError(f"Unknown lr_schedule type: {lr_sched_type}. Use 'linear' or 'lambda'.")
    
    # ------------------------------------------------------------------
    # Reference signal
    # ------------------------------------------------------------------
    mean_diff = calculate_generalized_mean_diffusivity(setup.pde["diffusivity"].to(DEVICE), volumes)
    eiglim = length2eig(setup.mf["length_scale"], mean_diff)

    precomputed_ref = precompute_laplace_matrices(
        femesh_split, setup, setup.pde, eiglim, setup.mf["neig_max"],
    )
    lap_ref = compute_laplace_eig_diff(
        femesh_split, setup, precomputed_ref, faces_prob=face_perm_ref,
    )
    lap_ref["length_scales"] = eig2length(lap_ref["values"], mean_diff)

    mf_ref = solve_mf(femesh_split, setup, lap_ref, faces_prob=face_perm_ref)
    signal_ref = torch.abs(mf_ref["signal_allcmpts"])
    S0_ref = torch.abs(signal_ref[0, :, :])
    signal_ref_normed = torch.abs(signal_ref / S0_ref.clamp(min=1e-6).unsqueeze(0))

    n_seq_total = int(signal_ref_normed.shape[1])
    if not (0 <= dt_old < n_seq_total and 0 <= dt_new < n_seq_total):
        raise ValueError(f"dt_old/dt_new out of range: old={dt_old}, new={dt_new}, n_seq={n_seq_total}")

    # ------------------------------------------------------------------
    # Edge regularisation bookkeeping
    # ------------------------------------------------------------------
    verts_opt = opt_mesh["points"]
    faces_opt = opt_mesh["facets"]
    edges, edge_face_map = edge_to_faces(faces_opt)

    pairs = []
    for face_ids in edge_face_map:
        if len(face_ids) >= 2:
            pairs.extend(list(combinations(face_ids, 2)))
    if len(pairs) == 0:
        edges_ij = torch.zeros((0, 2), device=DEVICE, dtype=torch.long)
    else:
        edges_ij = torch.tensor(pairs, device=DEVICE, dtype=torch.long)

    mean_diff_opt = calculate_generalized_mean_diffusivity(setup.pde["diffusivity"].to(DEVICE), volumes_opt)
    eiglim_opt = length2eig(setup.mf["length_scale"], mean_diff_opt)
    precomputed = precompute_laplace_matrices(
        femesh_split_opt, setup, setup.pde, eiglim_opt, setup.mf["neig_max"],
    )

    ref_data = {
        "face_perm": face_perm_ref,
        "signal_normed": signal_ref_normed.detach().cpu(),
    }

    femesh_scaled_opt["boundary_indices"] = mesh["boundary_indices"]
    femesh_scaled_opt["interior_indices"] = mesh["interior_indices"]

    mesh_logger = _make_mesh_logger(femesh_scaled_opt, cfg, ref_data, outdir)

    # ------------------------------------------------------------------
    # Cosine-annealing schedule helpers
    # ------------------------------------------------------------------
    # Track cycle boundaries for per-cycle Adam resets.
    _cycle_reset_done = set()

    def _get_cycle_idx(it: int) -> int:
        """Return which cycle `it` belongs to (-1 during warmup)."""
        if it < warmup_iters:
            return -1
        t = float(it - warmup_iters)
        T_eff = float(NUM_ITERS - warmup_iters)
        if n_cycles <= 1:
            return 0
        if abs(cycle_mult - 1.0) < 1e-9:
            T_0 = T_eff / n_cycles
        else:
            T_0 = T_eff * (1.0 - cycle_mult) / (1.0 - cycle_mult ** n_cycles)
        cum = 0.0
        for c in range(n_cycles):
            T_c = T_0 * (cycle_mult ** c)
            if cum + T_c > t + 1e-9:
                return c
            cum += T_c
        return n_cycles - 1

    def current_alpha(it: int) -> float:
        """Calculate alpha schedule (independent of whether it's used for blending)."""
        raw = cosine_anneal_alpha(
            it, warmup_iters, NUM_ITERS,
            n_cycles=n_cycles,
            cycle_mult=cycle_mult,
            restart_fraction=restart_fraction,
        )
        # v2: remap to [floor, 1] so dt_new always has a small contribution
        return alpha_warmup_floor + (1.0 - alpha_warmup_floor) * raw

    def current_lr(it: int) -> float:
        if not lr_anneal_cosine:
            return base_lr
        alpha = current_alpha(it)
        # Interpolate LR from base to post using the same alpha.
        return base_lr + (post_lr - base_lr) * alpha

    def current_clip(it: int) -> float:
        if not grad_clip_anneal:
            return base_clip
        alpha = current_alpha(it)
        return base_clip + (post_clip - base_clip) * alpha

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    iteration_times = []
    prev_nmodes = None
    lap_basis_last = None
    reduced_const = None

    def update_regularizer(reg_cfg: dict, it: int):
        start = _as_float(reg_cfg.get("start", 0.0), 0.0)
        end = _as_float(reg_cfg.get("end", start), start)
        start_iter = reg_cfg.get("start_iter", 0)
        block_len = reg_cfg.get("block_len", None)
        
        # If we haven't reached the start iteration yet, return the start value
        if it < start_iter:
            return start
        
        # Adjust iteration number relative to start_iter for scheduling
        it_adjusted = it - start_iter
        num_iters_adjusted = NUM_ITERS - start_iter
        
        if block_len is None or int(block_len) <= 1:
            frac = min(max(it_adjusted / float(max(1, num_iters_adjusted - 1)), 0.0), 1.0)
            return start + (end - start) * frac
        block_len = int(block_len)
        frac = (it_adjusted % block_len) / float(max(1, block_len - 1))
        return start + (end - start) * frac

    # ------------------------------------------------------------------
    # Loss components helpers (toggable)
    # ------------------------------------------------------------------
    # def compute_manifold_reg(edge_face_map, face_perm_log, log_threshold, tau_decades):
    #     if edge_face_map is None or len(edge_face_map) == 0:
    #         return torch.zeros((), device=DEVICE)
    #     edge_penalty_sum = torch.zeros((), device=DEVICE)
    #     for face_ids in edge_face_map:
    #         if len(face_ids) < 0:
    #             continue
    #         idx = torch.as_tensor(face_ids, device=DEVICE, dtype=torch.long)
    #         local = face_perm_log.index_select(0, idx)
    #         p_non_local = torch.sigmoid((log_threshold - local) / tau_decades)
    #         k = p_non_local.sum()
    #         if idx.numel() < 2:
    #             penalty = k * k
    #         else:
    #             penalty = (k * (2.0 - k)) ** 2
    #         edge_penalty_sum = edge_penalty_sum + penalty
    #     return edge_penalty_sum / float(max(1, len(edge_face_map)))

    def compute_manifold_reg(edge_face_map, face_perm_log, log_threshold, tau_decades):
        if edge_face_map is None or len(edge_face_map) == 0:
            return torch.zeros((), device=DEVICE)

        edge_penalty_sum = torch.zeros((), device=DEVICE)
        count = 0

        for face_ids in edge_face_map:
            if len(face_ids) < 2:
                # QUICK FIX: skip boundary/single-face groups (prevents bias to high perm)
                continue

            idx = torch.as_tensor(face_ids, device=DEVICE, dtype=torch.long)
            local = face_perm_log.index_select(0, idx)
            p_non_local = torch.sigmoid((log_threshold - local) / tau_decades)
            k = p_non_local.sum()

            penalty = (k * (2.0 - k)) ** 2
            edge_penalty_sum = edge_penalty_sum + penalty
            count += 1

        return edge_penalty_sum / float(max(1, count))

    def compute_manifold_reg_softmin(
        edge_face_map,
        face_perm_log,
        log_threshold,
        tau_decades,
        softmin_tau=0.1,          # smaller -> closer to hard min; try 0.05..0.2
        degree_norm=False,         # normalize by edge degree m
        eps=1e-12,
    ):
        """
        Soft manifold regularizer that prefers k in {0, 2}, where
          k = sum_i p_non_i over faces incident to a mesh edge,
          p_non_i = sigmoid((log_threshold - face_perm_log_i) / tau_decades).

        Uses a smooth "soft-min" of the two quadratic basins k^2 and (k-2)^2:
          penalty(k) = -t * log(exp(-k^2/t) + exp(-(k-2)^2/t))

        This keeps gradients smooth and avoids the stiff barrier peak at k=1.
        """
        if edge_face_map is None or len(edge_face_map) == 0:
            return torch.zeros((), device=face_perm_log.device)

        device = face_perm_log.device
        dtype = face_perm_log.dtype

        edge_penalty_sum = torch.zeros((), device=device, dtype=dtype)
        count = 0

        t = float(softmin_tau)

        for face_ids in edge_face_map:
            if len(face_ids) == 0:
                continue

            idx = torch.as_tensor(face_ids, device=device, dtype=torch.long)
            local = face_perm_log.index_select(0, idx)

            # "non-permeable probability" per face (soft threshold in log-space)
            p_non_local = torch.sigmoid((log_threshold - local) / tau_decades)
            k = p_non_local.sum()

            # Two basins: k -> 0 or k -> 2
            a = k * k
            b = (k - 2.0) * (k - 2.0)

            # Soft-min (stable via logsumexp)
            # penalty = softmin(a,b) = -t * log(exp(-a/t) + exp(-b/t))
            # Note: this can be negative; shifting doesn't change gradients.
            penalty = -t * torch.logsumexp(torch.stack((-a / t, -b / t)), dim=0)

            # Optional: shift so penalty is ~0 at minima (purely cosmetic)
            # At k=0 or 2: softmin(0,4) = -t*log(1+exp(-4/t))
            # shift makes minima ~0 and keeps gradients identical.
            penalty = penalty + t * torch.log1p(torch.exp(torch.tensor(-4.0 / t, device=device, dtype=dtype)))

            if degree_norm:
                m = float(len(face_ids))
                penalty = penalty / max(1.0, m)

            edge_penalty_sum = edge_penalty_sum + penalty
            count += 1

        return edge_penalty_sum / float(max(1, count))

    def compute_pairwise_edge_reg(edges_ij, p_non, adaptive_edge_coherence, adaptive_edge_floor, commitment,
                                  gate_power=2.0, detach_gate=True):
        if edges_ij.numel() == 0:
            return torch.zeros((), device=DEVICE)

        pi = p_non[edges_ij[:, 0]]
        pj = p_non[edges_ij[:, 1]]

        # base disagreement
        base = (pi - pj) ** 2

        # NEW: downweight near boundaries (where pi and pj differ)
        boundary = (pi - pj).abs()                      # 0 inside, 1 at strong boundary
        w_gate = (1.0 - boundary).clamp(0.0, 1.0).pow(gate_power)
        if detach_gate:
            w_gate = w_gate.detach()
        base = w_gate * base

        # keep your commitment weighting if you like
        if adaptive_edge_coherence:
            ci = commitment[edges_ij[:, 0]]
            cj = commitment[edges_ij[:, 1]]
            w = torch.maximum(ci, cj)
            w = float(adaptive_edge_floor) + (1.0 - float(adaptive_edge_floor)) * w
            base = w * base

        return base.mean()

    def compute_binarize_loss(binarize_on: str, p_non: torch.Tensor, face_perm_v: torch.Tensor, k_tau: float):
        if str(binarize_on).lower() == "alpha":
            alpha = torch.sigmoid(face_perm_v / float(k_tau))
            return (alpha * (1.0 - alpha)).mean()
        return (p_non * (1.0 - p_non)).mean()

    def cos_sim(a, b, eps=1e-12):
        return (a.dot(b) / (a.norm().clamp_min(eps) * b.norm().clamp_min(eps))).item()

    for it in range(NUM_ITERS):
        iter_start_time = time.time()

        # Per-cycle Adam reset (optional).
        cyc = _get_cycle_idx(it)
        if reset_adam_per_cycle and cyc >= 0 and cyc not in _cycle_reset_done:
            reset_adam_state_(optimizer)
            _cycle_reset_done.add(cyc)
            print(f"  [cosine anneal] Adam state reset at cycle {cyc}, iter {it}")

        if lr_anneal_cosine:
            lr_now = current_lr(it)
            for pg in optimizer.param_groups:
                pg["lr"] = float(lr_now)
        else:
            lr_now = float(optimizer.param_groups[0].get("lr", base_lr))

        alpha_blend = current_alpha(it)

        # v3: scheduled simulation_lambda and binarize
        sim_lambda = update_regularizer(sim_lambda_sched, it)
        lambda_binarize_val = update_regularizer(binarize_sched, it)

        print(f"\nIteration {it}")
        optimizer.zero_grad()
        train_data = {"iteration": it}

        face_perm = current_perm()
        face_perm_snapshot = face_perm.detach().clone()
        face_perm_log = current_log_perm()
        face_perm_log_snapshot = face_perm_log.detach().clone()

        laplace_start_time = time.time()
        rom_active = rom_enabled and (it >= rom_start_iteration)
        do_rom_refresh = (
            (lap_basis_last is None)
            or (rom_eig_refresh_every <= 0)
            or (it % rom_eig_refresh_every == 0)
            or (not rom_active)
        )

        if do_rom_refresh:
            lap_opt = compute_laplace_eig_diff(
                femesh_split_opt, setup, precomputed, faces_prob=face_perm,
            )
            lap_opt["length_scales"] = eig2length(lap_opt["values"], mean_diff_opt)
            lap_basis_last = lap_opt
            reduced_const = precompute_reduced_constants(precomputed, lap_opt["funcs"].detach())
            lap_used = lap_opt
            use_reduced = False
        else:
            lap_red = compute_laplace_reduced_projections(
                precomputed, reduced_const, faces_prob=face_perm,
            )
            lap_red["funcs"] = reduced_const["Phi"]
            lap_used = lap_red
            use_reduced = True

        lap_stats = lap_basis_last if lap_basis_last is not None else lap_used
        laplace_time = time.time() - laplace_start_time

        with torch.no_grad():
            L_cut = float(setup.mf["length_scale"])
            Ls = lap_stats["length_scales"].detach()
            nmodes = int(Ls.numel())
            train_data["nmodes"] = nmodes
            train_data["nmodes_delta"] = 0 if prev_nmodes is None else (nmodes - prev_nmodes)
            prev_nmodes = nmodes
            margin = (Ls - L_cut).abs()
            train_data["cutoff_min_margin"] = float(margin.min().item())
            train_data["cutoff_p50_margin"] = float(margin.median().item())
            band = 0.5
            train_data["pct_near_cutoff"] = float((margin < band).float().mean().item())

                # --- Eigenvalue diagnostics (minimal) ---
            lam = lap_stats["values"].detach().flatten()

            # Always useful: min/median/max + a low-percentile
            train_data["eig_min"] = float(lam.min().item())
            train_data["eig_p01"] = float(torch.quantile(lam, 0.01).item())  # catches tail collapse
            train_data["eig_p50"] = float(torch.quantile(lam, 0.50).item())
            train_data["eig_max"] = float(lam.max().item())

            # Near-zero count (relative tolerance so it scales with your problem)
            lam_max = lam.max().clamp_min(1.0)
            tol = 1e-10 * lam_max  # tweak to 1e-8..1e-12 if needed
            train_data["eig_n_near0"] = int((lam < tol).sum().item())

            # Smallest nonzero-ish (more informative than raw min if lambda0 ~ 0 always)
            lam_sorted, _ = torch.sort(lam)
            # use 2nd smallest as "lambda1" proxy; safe if you have >=2 modes
            if lam_sorted.numel() >= 2:
                train_data["eig_lambda1"] = float(lam_sorted[1].item())

        solve_mf_start_time = time.time()

        # ---------- forward pass (single-call when blended) ----------
        if loss_blend_anneal and (0.0 < alpha_blend < 1.0):
            if use_reduced:
                mf = solve_mf_reduced(
                    femesh_split_opt, setup, lap_used,
                    target_seq=[dt_old, dt_new],
                )
            else:
                mf = solve_mf(
                    femesh_split_opt, setup, lap_used,
                    target_seq=[dt_old, dt_new],
                    faces_prob=face_perm,
                )
            sig = torch.abs(mf["signal_allcmpts"])  # [n_amp, 2, n_dir]
            S0_pair = S0_ref[[dt_old, dt_new], :].clamp_min(1e-6)
            sig_norm = sig / S0_pair.unsqueeze(0)
            sig_old = sig_norm[:, 0:1, :]
            sig_new = sig_norm[:, 1:2, :]
            signal_opt_curr_norm = (1.0 - alpha_blend) * sig_old + alpha_blend * sig_new
            ref_old = signal_ref_normed[:, dt_old, :]
            ref_new = signal_ref_normed[:, dt_new, :]
            L_old = ((sig_old[:, 0, :] - ref_old) ** 2).mean()
            L_new = ((sig_new[:, 0, :] - ref_new) ** 2).mean()
            sim_loss = (1.0 - alpha_blend) * L_old + alpha_blend * L_new
        else:
            dt_id = get_current_dt_id(it, dt_schedule)
            if use_reduced:
                mf = solve_mf_reduced(
                    femesh_split_opt, setup, lap_used,
                    target_seq=dt_id,
                )
            else:
                mf = solve_mf(
                    femesh_split_opt, setup, lap_used,
                    target_seq=dt_id,
                    faces_prob=face_perm,
                )
            sig = torch.abs(mf["signal_allcmpts"])
            sig = sig / S0_ref[dt_id, :].clamp_min(1e-6).unsqueeze(0)
            signal_opt_curr_norm = sig
            ref = signal_ref_normed[:, dt_id, :]
            sim_loss = ((sig[:, 0, :] - ref) ** 2).mean()

        solve_mf_time = time.time() - solve_mf_start_time

        # ---------- per-b-value loss diagnostics (NO grad) ----------
        with torch.no_grad():
            eps = 1e-12
            # Per-b-value loss across currently active diffusion time(s) and all directions
            n_amp_total = signal_opt_curr_norm.shape[0]
            if loss_blend_anneal and (0.0 < alpha_blend < 1.0):
                # Blending case: compare against blended reference
                ref_curr = (1.0 - alpha_blend) * signal_ref_normed[:, dt_old:dt_old+1, :] + alpha_blend * signal_ref_normed[:, dt_new:dt_new+1, :]
            else:
                # Single dt case
                ref_curr = signal_ref_normed[:, dt_id:dt_id+1, :]
            
            # Compute per-b losses
            diff_all = signal_opt_curr_norm - ref_curr  # [n_amp, n_dt_active, n_dir]
            per_amp = diff_all.pow(2).mean(dim=2).mean(dim=1)  # [n_amp] avg over dir then dt
            
            for a_idx in range(n_amp_total):
                train_data[f"loss_per_b/b{a_idx}"] = float(per_amp[a_idx].item())
            
            # Print diagnostic: which b-value contributes most
            top_idx = int(torch.argmax(per_amp).item())
            if loss_blend_anneal and (0.0 < alpha_blend < 1.0):
                dt_info = f"dt_blend=[{dt_old},{dt_new}] alpha={alpha_blend:.3f}"
            else:
                dt_info = f"dt_id={dt_id}"
            print(
                f"[diag] top b={top_idx} contrib={per_amp[top_idx].item():.3e} "
                f"min={per_amp.min().item():.3e} max={per_amp.max().item():.3e} {dt_info}"
            )
            print("[diag] per_b:", per_amp.cpu().numpy())
            
            # General signal statistics
            train_data["sig/opt_min"] = float(signal_opt_curr_norm.min().item())
            train_data["sig/opt_p50"] = float(signal_opt_curr_norm.median().item())
            train_data["sig/opt_p99"] = float(torch.quantile(signal_opt_curr_norm, 0.99).item())
            train_data["sig/opt_max"] = float(signal_opt_curr_norm.max().item())
            train_data["sig/opt_allfinite"] = float(torch.isfinite(signal_opt_curr_norm).all().item())

        # ---------- regularisation (merged) ----------
        lambda_continuity = update_regularizer(cfg["lambda_continuity"], it)
        lambda_manifold_reg = update_regularizer(cfg.get("lambda_manifold_reg", {"start": 0.0, "end": 0.0}), it)
        lambda_edge_reg = update_regularizer(cfg.get("lambda_edge_reg", {"start": 0.0, "end": 0.0}), it)

        if edge_reg_anneal_rate > 0.0:
            lambda_manifold_reg_eff = float(lambda_manifold_reg) * (1.0 - edge_reg_anneal_rate * float(alpha_blend))
            lambda_edge_reg_eff = float(lambda_edge_reg) * (1.0 - edge_reg_anneal_rate * float(alpha_blend))
        else:
            lambda_manifold_reg_eff = float(lambda_manifold_reg)
            lambda_edge_reg_eff = float(lambda_edge_reg)

        # Determine non-perm probability for current log values
        log_threshold = math.log10(1e-3)
        tau_decades = float(cfg.get("edge_tau_decades", 0.5))
        p_non = torch.sigmoid((log_threshold - face_perm_log) / tau_decades)
        commitment = (2.0 * p_non - 1.0).abs()

        # Continuity loss term
        continuity_loss = torch.zeros((), device=DEVICE)
        if edges_ij.numel() > 0:
            x = face_perm_log
            d = x[edges_ij[:, 0]] - x[edges_ij[:, 1]]
            delta = cfg.get("continuity_huber_delta", 5.0)
            absd = d.abs()
            huber = torch.where(absd <= delta, 0.5 * d * d, delta * (absd - 0.5 * delta))

            # Optional boundary gating (no extra knobs)
            if bool(cfg.get("continuity_weighting", False)):
                # requires p_non already computed (same as manifold/edge/binarize)
                pi = p_non[edges_ij[:, 0]]
                pj = p_non[edges_ij[:, 1]]
                w_gate = (1.0 - (pi - pj).abs()).clamp(0.0, 1.0).detach()
                huber = w_gate * huber

            continuity_loss = float(lambda_continuity) * huber.mean()

        # Calculate manifold regularization
        manifold_reg_loss = torch.zeros((), device=DEVICE)
        if lambda_manifold_reg_eff > 0.0:
            # Get soft manifold config
            soft_manifold_cfg = cfg.get("soft_manifold", {"enabled": False})
            use_soft_manifold = soft_manifold_cfg.get("enabled", False)
            
            if use_soft_manifold:
                # Use soft-min manifold regularizer
                softmin_tau = float(soft_manifold_cfg.get("softmin_tau", 0.1))
                degree_norm = soft_manifold_cfg.get("degree_norm", True)
                manifold_reg_loss = float(lambda_manifold_reg_eff) * compute_manifold_reg_softmin(
                    edge_face_map, face_perm_log, log_threshold, tau_decades,
                    softmin_tau=softmin_tau,
                    degree_norm=degree_norm,
                )
            else:
                # Use original hard manifold regularizer
                manifold_reg_loss = float(lambda_manifold_reg_eff) * compute_manifold_reg(
                    edge_face_map, face_perm_log, log_threshold, tau_decades
                )

        # Calculate edge regularization loss
        edge_reg_loss = torch.zeros((), device=DEVICE)
        if lambda_edge_reg_eff > 0.0:
            edge_reg_loss = float(lambda_edge_reg_eff) * compute_pairwise_edge_reg(
                edges_ij, p_non,
                adaptive_edge_coherence=use_adaptive_edge,
                adaptive_edge_floor=adaptive_edge_floor,
                commitment=commitment,
            )

        # Calculate binarity loss term
        binarize_loss = torch.zeros((), device=DEVICE)
        if float(lambda_binarize_val) > 0.0:
            binarize_loss = float(lambda_binarize_val) * compute_binarize_loss(
                binarize_on=binarize_on,
                p_non=p_non,
                face_perm_v=face_perm_v,
                k_tau=float(k_tau),
            )

        sim_loss_scaled = sim_loss * sim_lambda
        loss = sim_loss_scaled + continuity_loss + manifold_reg_loss + edge_reg_loss + binarize_loss

        # ---------- gradient diagnostics ----------
        g_data = torch.autograd.grad(sim_loss_scaled, face_perm_v, retain_graph=True)[0]
        g_cont_reg = torch.autograd.grad(continuity_loss, face_perm_v, retain_graph=True)[0]

        # Get edge and bin grads if exist
        if lambda_manifold_reg_eff > 0.0:
            g_man_reg = torch.autograd.grad(manifold_reg_loss, face_perm_v, retain_graph=True)[0]
        else:
            g_man_reg = torch.zeros_like(face_perm_v)

        if lambda_edge_reg_eff > 0.0:
            g_edge_reg = torch.autograd.grad(edge_reg_loss, face_perm_v, retain_graph=True)[0]
        else:
            g_edge_reg = torch.zeros_like(face_perm_v)

        if lambda_binarize_val > 0.0:
            g_binarize = torch.autograd.grad(binarize_loss, face_perm_v, retain_graph=True)[0]
        else:
            g_binarize = torch.zeros_like(face_perm_v)

        print("||dL_data/dv|| =", g_data.norm().item(),
              " ||dl_man_reg/dv|| =", g_man_reg.norm().item(),
              " ||dL_edge_reg/dv|| =", g_edge_reg.norm().item(),
              " ||dL_cont_reg/dv|| =", g_cont_reg.norm().item(),
              " ||dL_binarize/dv|| =", g_binarize.norm().item())

        train_data["sim_grad_norm"] = float(g_data.norm().item())
        train_data["man_reg_grad_norm"] = float(g_man_reg.norm().item())
        train_data["edge_grad_norm"] = float(g_edge_reg.norm().item())
        train_data["cont_grad_norm"] = float(g_cont_reg.norm().item())
        train_data["binarize_grad_norm"] = float(g_binarize.norm().item())
        train_data["cos_sim_edge"] = float(cos_sim(g_data.flatten(), g_edge_reg.flatten()))
        train_data["cos_sim_cont"] = float(cos_sim(g_data.flatten(), g_cont_reg.flatten()))
        train_data["cos_edge_cont"] = float(cos_sim(g_edge_reg.flatten(), g_cont_reg.flatten()))
        train_data["cos_sim_binarize"] = float(cos_sim(g_data.flatten(), g_binarize.flatten()))

        with torch.no_grad():
            v = face_perm_v.detach()
            alpha_sig = torch.sigmoid(v / k_tau)
            sprime = alpha_sig * (1 - alpha_sig)
            commitment = (2.0 * p_non - 1.0).abs()
            pct_committed = ((commitment > 0.8).float().mean().item()) * 100.0
            pct_uncommitted = ((commitment < 0.3).float().mean().item()) * 100.0
            print("----------------------Grad Diagnostics----------------------")
            print("lr:", float(lr_now), "alpha_blend:", float(alpha_blend),
                  f"cycle: {cyc}")
            print(f"sim_lambda: {sim_lambda:.2f}, binarize_lambda: {lambda_binarize_val:.4f}")
            print("v min/max:", v.min().item(), v.max().item())
            print("alpha min/max:", alpha_sig.min().item(), alpha_sig.max().item())
            print("sigmoid' min/max:", sprime.min().item(), sprime.max().item())
            print("pct saturated (alpha<1e-6 or >1-1e-6):",
                  ((alpha_sig < 1e-6) | (alpha_sig > 1 - 1e-6)).float().mean().item())
            print(f"p_non stats: mean={p_non.mean().item():.4f} "
                  f"std={p_non.std().item():.4f}")
            print(f"commitment: {pct_committed:.1f}% committed (>0.8), "
                  f"{pct_uncommitted:.1f}% uncommitted (<0.3)")
            print("------------------------------------------------------------")

        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start

        # ---- v3: Gradient fill-in boost for uncommitted faces ----
        # After warmup, scale up gradients for undecided faces so the
        # boundary carving propagates inward from the outer shell.
        if gradient_fill_boost > 0.0 and it >= gradient_fill_start and face_perm_v.grad is not None:
            with torch.no_grad():
                fill_scale = 1.0 + gradient_fill_boost * (1.0 - commitment)
                face_perm_v.grad.mul_(fill_scale)

        grad_snapshot = face_perm_v.grad.detach().clone()
        qs = torch.tensor([0, 0.5, 0.9, 0.99, 0.999], device=grad_snapshot.device)
        print("grad quantiles:", torch.quantile(grad_snapshot, qs).tolist())
        print("pct < 1e-20:", (grad_snapshot < 1e-20).float().mean().item(),
              "pct < 1e-30:", (grad_snapshot < 1e-30).float().mean().item())

        if grad_clip_anneal:
            clip_norm = float(current_clip(it))
        else:
            clip_norm = float(base_clip)
        if clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_([face_perm_v], clip_norm)
        grad_snapshot_clipped = face_perm_v.grad.detach().clone()

        v_before = face_perm_v.detach().clone()
        alpha_before = torch.sigmoid(v_before / k_tau)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            v_after = face_perm_v.detach()
            alpha_after = torch.sigmoid(v_after / k_tau)
            dv = v_after - v_before
            da = alpha_after - alpha_before
            train_data["dv_norm"] = float(dv.norm().item())
            train_data["dv_maxabs"] = float(dv.abs().max().item())
            train_data["dv_rel"] = float((dv.norm() / (v_before.norm().clamp_min(1e-12))).item())
            train_data["dalpha_norm"] = float(da.norm().item())
            train_data["dalpha_maxabs"] = float(da.abs().max().item())
            train_data["lr"] = float(lr_now)
            train_data["clip_norm"] = float(clip_norm)

        nonperm_indices = face_perm_snapshot < 1e-3
        
        # Calculate edge structure metrics for non-manifold analysis
        with torch.no_grad():
            edge_metrics = compute_edge_structure_metrics(faces_opt, nonperm_indices)
            train_data["edge/open_pct"] = edge_metrics['open_edge_pct']
            train_data["edge/branch_pct"] = edge_metrics['branch_edge_pct']
            train_data["edge/bad_pct"] = edge_metrics['bad_edge_pct']
            train_data["edge/num_edges"] = edge_metrics['num_edges']
            train_data["edge/num_open"] = edge_metrics['num_open_edges']
            train_data["edge/num_branch"] = edge_metrics['num_branch_edges']
            train_data["edge/num_bad"] = edge_metrics['num_bad_edges']
        
        with torch.no_grad():
            n_chamfer_samples = int(cfg.get("chamfer_n_samples", 10))
            chamfer_loss = chamfer_centroid_loss_centered(verts_opt, faces_opt[:, nonperm_indices], C_ref, n_samples=n_chamfer_samples)

        log_start_time = time.time()

        train_data["total_loss"] = float(loss.item())
        train_data["sim_loss"] = float(sim_loss_scaled.item())
        train_data["continuity_loss"] = float(continuity_loss.item())
        train_data["manifold_reg_loss"] = float(manifold_reg_loss.item())
        train_data["edge_reg_loss"] = float(edge_reg_loss.item())
        train_data["binarize_loss"] = float(binarize_loss.item())
        train_data["chamfer_loss"] = float(chamfer_loss.item())
        train_data["face_perm"] = face_perm_snapshot
        train_data["face_perm_log"] = face_perm_log_snapshot
        train_data["face_perm_grad"] = grad_snapshot
        train_data["face_perm_grad_clipped"] = grad_snapshot_clipped
        train_data["clip_ratio"] = float((grad_snapshot_clipped.norm() / grad_snapshot.norm().clamp_min(1e-12)).item())
        train_data["signal_normed"] = signal_opt_curr_norm.detach().cpu()
        train_data["eigenvalues"] = lap_stats["values"].detach()
        train_data["length_scales"] = lap_stats["length_scales"].detach()
        train_data["rom/use_reduced"] = float(use_reduced)
        train_data["alpha_blend"] = float(alpha_blend)
        train_data["dt_old"] = int(dt_old)
        train_data["dt_new"] = int(dt_new)
        train_data["cycle_idx"] = int(cyc)
        train_data["lambda_edge_reg_eff"] = float(lambda_edge_reg_eff)
        train_data["sim_lambda"] = float(sim_lambda)
        train_data["lambda_binarize_val"] = float(lambda_binarize_val)
        train_data["pct_committed"] = float(pct_committed)
        train_data["pct_uncommitted"] = float(pct_uncommitted)

        correct_perm_faces = face_perm[inter_face_indices] > 1e-3
        pct_correct_perm_faces = 100.0 * correct_perm_faces.sum().item() / max(1, inter_face_indices.numel())
        correct_nonperm_faces = face_perm[bound_face_indices] < 1e-3
        pct_correct_nonperm_faces = 100.0 * correct_nonperm_faces.sum().item() / max(1, bound_face_indices.numel())
        train_data["pct_correct_perm_faces"] = float(pct_correct_perm_faces)
        train_data["pct_correct_nonperm_faces"] = float(pct_correct_nonperm_faces)

        if hasattr(mesh_logger, "log_iteration_diagnostics"):
            mesh_logger.log_iteration_diagnostics(
                train_data, optimizer=optimizer, group_names=["perm"],
            )

        if it % LOG_ITER == 0 or it == (NUM_ITERS - 1):
            with torch.no_grad():
                mesh_logger.log_data(train_data)

        log_time = time.time() - log_start_time
        iter_end_time = time.time()
        iter_duration = iter_end_time - iter_start_time
        iteration_times.append(iter_duration)
        train_data["iteration_time"] = float(iter_duration)
        train_data["avg_iteration_time"] = float(sum(iteration_times) / len(iteration_times))

        log_str = (
            f"+++++++++++++++++++++++++++++++++++++++ Iteration {it} +++++++++++++++++++++++++++++++++++++++++++\n"
            f"Time: {iter_duration:.2f}s (avg: {train_data['avg_iteration_time']:.2f}s)\n"
            f"  - Laplace solve: {laplace_time:.2f}s ({100*laplace_time/iter_duration:.1f}%)\n"
            f"  - Solve MF: {solve_mf_time:.2f}s ({100*solve_mf_time/iter_duration:.1f}%)\n"
            f"  - Backward pass: {backward_time:.2f}s ({100*backward_time/iter_duration:.1f}%)\n"
            f"  - Logging: {log_time:.2f}s ({100*log_time/iter_duration:.1f}%)\n"
            f"ROM reduced step: {use_reduced}\n"
            f"Cosine-anneal alpha: {alpha_blend:.4f} (cycle {cyc}, old={dt_old}, new={dt_new})\n"
            f"LR: {lr_now:.6f} | GradClip: {clip_norm:.6f}\n"
            f"sim_lambda: {sim_lambda:.2f} | binarize: {lambda_binarize_val:.4f}\n"
            f"Commitment: {pct_committed:.1f}% done, {pct_uncommitted:.1f}% undecided\n"
            f"--------------------------------------- Permeabilities -------------------------------------------\n"
            f"I/E Cellular Log K Mean: {face_perm_log[inter_face_indices].mean().item():.6f}\n"
            f"Boundary Log K Mean: {face_perm_log[bound_face_indices].mean().item():.6f}\n"
            f"---------------------------------------- Loss Values ---------------------------------------------\n"
            f"Simulation Loss: {sim_loss_scaled.item():.6e} (lambda={sim_lambda:.2f})\n"
            f"Continuity Loss: {continuity_loss.item():.6e} (lambda={float(lambda_continuity):.4f})\n"
            f"Manfiold Loss: {manifold_reg_loss.item():.6e} (lambda={float(lambda_manifold_reg):.4f})\n"
            f"Edge Reg Loss: {edge_reg_loss.item():.6e} (lambda_base={float(lambda_edge_reg):.4f}, "
            f"lambda_eff={float(lambda_edge_reg_eff):.4f})\n"
            f"Binarize Loss: {binarize_loss.item():.6e} (lambda={float(lambda_binarize_val):.4f})\n"
            f"Chamfer Loss: {chamfer_loss.item():.6e}\n"
            f"Total Loss: {loss.item():.6e}\n"
            f"---------------------------------------- Eigenvalues ---------------------------------------------\n"
            f"Num Eigenvalues: {lap_stats['values'].shape}\n"
            f"Eigenvalue Min/Max: {lap_stats['values'].min().item():.6f}, {lap_stats['values'].max().item():.6f}\n"
            f"Length Scale Min/Max: {lap_stats['length_scales'].min().item():.6f}, {lap_stats['length_scales'].max().item():.6f}\n"
        )
        mesh_logger.log_text(log_str)
        print(log_str)

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------
    if hasattr(mesh_logger, "write_log"):
        mesh_logger.write_log()
    if hasattr(mesh_logger, "create_loss_plot"):
        mesh_logger.create_loss_plot()
    if hasattr(mesh_logger, "create_mesh_threshold_video_fast"):
        mesh_logger.create_mesh_threshold_video_fast()
    elif hasattr(mesh_logger, "create_mesh_threshold_video"):
        mesh_logger.create_mesh_threshold_video()
    if hasattr(mesh_logger, "create_pct_correct_plot"):
        mesh_logger.create_pct_correct_plot()

    cfg_path = os.path.join(outdir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)

    print("\nOptimisation complete!")
    print("Final faces_prob:", current_perm().detach().cpu().numpy())
    print("Wrote config:", cfg_path)

if __name__ == "__main__":
    main()
