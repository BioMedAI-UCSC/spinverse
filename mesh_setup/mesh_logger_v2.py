# import os, io, json
# from datetime import datetime

# import numpy as np
# import imageio
# import plotly.graph_objects as go
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from PIL import Image

# # ---------- Fixed rendering sizes (tweak if you want) ----------
# PLOTLY_W, PLOTLY_H = 1000, 800      # each mesh panel
# H_W, H_H = 2000, 1200               # histogram panel
# FRAME_W, FRAME_H = 2000, 2000       # final video frame target
# H_DPI = 200                         # histogram dpi

# def _pad_to_multiple(img: np.ndarray, mult: int = 16) -> np.ndarray:
#     h, w = img.shape[:2]
#     nw = ((w + mult - 1) // mult) * mult
#     nh = ((h + mult - 1) // mult) * mult
#     if (nw, nh) == (w, h): return img
#     pw, ph = nw - w, nh - h
#     if img.ndim == 2:
#         return np.pad(img, ((0, ph), (0, pw)), mode="constant")
#     return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="constant")

# def _fig_to_rgba(fig, width, height, scale=1):
#     # Plotly -> PNG bytes -> np array RGBA
#     png = fig.to_image(format="png", width=width, height=height, scale=scale)
#     im = Image.open(io.BytesIO(png)).convert("RGBA")
#     return np.array(im)

# def _mpl_hist_to_rgba(bins, b_counts, i_counts, iteration, width_px=H_W, height_px=H_H, dpi=H_DPI):
#     fig_w_in, fig_h_in = width_px / dpi, height_px / dpi
#     fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
#     bin_w = np.diff(bins)
#     ax.bar(bins[:-1], b_counts, width=bin_w, align="edge", edgecolor='black', alpha=0.6, label="Boundary")
#     ax.bar(bins[:-1], i_counts, width=bin_w, align="edge", edgecolor='black', alpha=0.6, label="Interior")
#     ax.set_xscale("log")
#     xticks = bins
#     ax.set_xticks(xticks)
#     ax.set_xticklabels([f"{x:.1e}" for x in xticks], rotation=45)
#     ax.set_xlabel("Face Permeability")
#     ax.set_ylabel("Count")
#     ax.set_title(f"Face κ Distribution (iter {iteration})")
#     ax.legend()
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches=None)
#     plt.close(fig)
#     buf.seek(0)
#     im = Image.open(buf).convert("RGBA")
#     return np.array(im)

# class MeshLogger:
#     """
#     Fast, in-memory mesh logger:
#       - In-loop: cache fp + histogram counts only
#       - End: render selected frames directly to video (no temp images)
#     """
#     def __init__(self, mesh, log_dir=None, stride=10, keep_last_html=False):
#         from mesh_setup.mesh_utils import get_split_indices

#         self.log_dir = log_dir or os.path.join("logs", datetime.now().strftime("mesh_data_%Y%m%d_%H%M%S"))
#         os.makedirs(self.log_dir, exist_ok=True)

#         # Cache static geometry (CPU numpy)
#         self.vertices = mesh["points"].T.detach().cpu().numpy()  # (V,3)
#         self.faces    = mesh["facets"].T.detach().cpu().numpy()  # (F,3)

#         b_idx, i_idx = get_split_indices(mesh)
#         self.bound_idx = b_idx.detach().cpu().numpy()
#         self.inter_idx = i_idx.detach().cpu().numpy()

#         # Runtime store
#         self.entries = []   # dicts of {iter, loss, fp(np), b_counts, i_counts}

#         # Config
#         self.stride = int(max(1, stride))
#         self.keep_last_html = bool(keep_last_html)

#         # Histogram binning (logspace)
#         self.hist_bins = np.logspace(-10, 0, 21)

#     def log_mesh(self, faces_prob, loss, iteration: int):
#         # detach → CPU numpy
#         fp = faces_prob.detach().cpu().numpy()
#         b_vals = fp[self.bound_idx]
#         i_vals = fp[self.inter_idx]
#         b_counts, _ = np.histogram(b_vals, bins=self.hist_bins)
#         i_counts, _ = np.histogram(i_vals, bins=self.hist_bins)

#         self.entries.append({
#             "iter": int(iteration),
#             "loss": float(getattr(loss, "item", lambda: loss)()),
#             "fp": fp.astype(np.float32, copy=False),
#             "b_counts": b_counts.astype(np.int32, copy=False),
#             "i_counts": i_counts.astype(np.int32, copy=False),
#         })

#     # ---------- rendering helpers ----------
#     def _plotly_mesh(self, fp_np: np.ndarray, title: str, mask=None):
#         log_fp = np.log10(np.clip(fp_np, 1e-30, None))
#         mn, mx = np.floor(log_fp.min()), np.ceil(log_fp.max())
#         if mx == mn: mx += 1
#         tickvals = np.linspace(mn, mx, 10)
#         ticktext = [f"{10**tv:.1e}" for tv in tickvals]

#         faces = self.faces if mask is None else self.faces[mask]
#         intens = log_fp if mask is None else log_fp[mask]

#         mesh3d = go.Mesh3d(
#             x=self.vertices[:,0], y=self.vertices[:,1], z=self.vertices[:,2],
#             i=faces[:,0], j=faces[:,1], k=faces[:,2],
#             intensity=intens, intensitymode="cell",
#             colorscale="Viridis", cmin=mn, cmax=mx,
#             colorbar=dict(title="Face κ", tickmode="array", tickvals=tickvals, ticktext=ticktext),
#             opacity=0.5, showscale=True
#         )
#         fig = go.Figure(mesh3d)
#         fig.update_layout(
#             title=title,
#             scene=dict(aspectmode="cube"),
#             width=PLOTLY_W, height=PLOTLY_H,
#             margin=dict(l=0, r=0, t=50, b=0),
#         )
#         return fig

#     def write_video_direct(self, out_name="mesh_vis.mp4", fps=10, render_boundary_and_interior=False):
#         """
#         Render every `stride`-th entry + final to MP4, in memory.
#         If render_boundary_and_interior=False, uses one full-mesh panel (duplicated to fill top row).
#         """
#         path = os.path.join(self.log_dir, out_name)

#         # choose frames to render
#         idxs = list(range(0, len(self.entries), self.stride))
#         if (len(self.entries)-1) not in idxs:
#             idxs.append(len(self.entries)-1)

#         with imageio.get_writer(path, fps=fps, codec="libx264") as writer:
#             for k in idxs:
#                 e = self.entries[k]
#                 it, fp = e["iter"], e["fp"]
#                 b_counts, i_counts = e["b_counts"], e["i_counts"]

#                 # top row: either (boundary + interior) OR (full + full)
#                 if render_boundary_and_interior:
#                     b_mask = self.bound_idx
#                     i_mask = self.inter_idx
#                     b_fig = self._plotly_mesh(fp, f"Boundary κ (iter {it})", mask=b_mask)
#                     i_fig = self._plotly_mesh(fp, f"Interior κ (iter {it})", mask=i_mask)
#                     top_left  = _fig_to_rgba(b_fig, PLOTLY_W, PLOTLY_H)
#                     top_right = _fig_to_rgba(i_fig, PLOTLY_W, PLOTLY_H)
#                 else:
#                     full_fig = self._plotly_mesh(fp, f"Mesh Face κ (iter {it})", mask=None)
#                     top_left  = _fig_to_rgba(full_fig, PLOTLY_W, PLOTLY_H)
#                     top_right = top_left  # duplicate to fill width cheaply

#                 # bottom: histogram
#                 bottom = _mpl_hist_to_rgba(self.hist_bins, b_counts, i_counts, it, H_W, H_H, H_DPI)

#                 # ensure consistent sizes (RGBA)
#                 def _ensure_rgba(img, w, h):
#                     if img.shape[1] == w and img.shape[0] == h: return img
#                     return np.array(Image.fromarray(img).resize((w, h), Image.LANCZOS))

#                 top_left  = _ensure_rgba(top_left,  PLOTLY_W, PLOTLY_H)
#                 top_right = _ensure_rgba(top_right, PLOTLY_W, PLOTLY_H)
#                 bottom    = _ensure_rgba(bottom,    H_W,       H_H)

#                 top_row = np.hstack((top_left, top_right))   # (PLOTLY_H, 2*PLOTLY_W, 4)
#                 frame   = np.vstack((top_row, bottom))       # (PLOTLY_H+H_H, 2*PLOTLY_W, 4)

#                 # convert RGBA -> RGB (white background) for libx264
#                 if frame.shape[2] == 4:
#                     alpha = frame[..., 3:4].astype(np.float32) / 255.0
#                     bg = np.full_like(frame[..., :3], 255, dtype=np.uint8)
#                     frame_rgb = (alpha * frame[..., :3] + (1 - alpha) * bg).astype(np.uint8)
#                 else:
#                     frame_rgb = frame

#                 frame_rgb = _pad_to_multiple(frame_rgb, 16)
#                 writer.append_data(frame_rgb)

#         # optional: final HTML snapshot (costly; off by default)
#         if self.keep_last_html:
#             last = self.entries[-1]
#             final_fig = self._plotly_mesh(last["fp"], "Final Mesh Face κ", mask=None)
#             final_fig.write_html(os.path.join(self.log_dir, "final_mesh.html"))

#         print(f"[MeshLogger] wrote video: {path}")

import os, io, json
from datetime import datetime
from typing import Optional

import numpy as np
import imageio
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ---------- Fixed rendering sizes (tweak if you want) ----------
PLOTLY_W, PLOTLY_H = 1000, 800      # each mesh panel
H_W, H_H = 2000, 1200               # histogram panel
FRAME_W, FRAME_H = 2000, 2000       # final video frame target
H_DPI = 200                         # histogram dpi

def _pad_to_multiple(img: np.ndarray, mult: int = 16) -> np.ndarray:
    h, w = img.shape[:2]
    nw = ((w + mult - 1) // mult) * mult
    nh = ((h + mult - 1) // mult) * mult
    if (nw, nh) == (w, h): return img
    pw, ph = nw - w, nh - h
    if img.ndim == 2:
        return np.pad(img, ((0, ph), (0, pw)), mode="constant")
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="constant")

def _fig_to_rgba(fig, width, height, scale=1):
    # Plotly -> PNG bytes -> np array RGBA
    png = fig.to_image(format="png", width=width, height=height, scale=scale)
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    return np.array(im)

def _mpl_hist_to_rgba(
    bins,
    b_counts,
    i_counts,
    iteration,
    width_px=H_W,
    height_px=H_H,
    dpi=H_DPI,
    conv_text: Optional[str] = None
):
    fig_w_in, fig_h_in = width_px / dpi, height_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    # widths for each bin
    bin_w = np.diff(bins)

    # --- stacked bars ---
    # first category on the base
    ax.bar(
        bins[:-1], b_counts,
        width=bin_w, align="edge",
        edgecolor="black", alpha=0.7, label="Boundary"
    )
    # second category stacked on top of the first
    ax.bar(
        bins[:-1], i_counts,
        width=bin_w, align="edge",
        edgecolor="black", alpha=0.7, label="Interior",
        bottom=b_counts
    )

    ax.set_xscale("log")
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{x:.1e}" for x in bins], rotation=45)
    ax.set_xlabel("Face Permeability")
    ax.set_ylabel("Count")

    title = f"Face κ Distribution (iter {iteration})"
    if conv_text:
        title += f" — {conv_text}"
    ax.set_title(title)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches=None)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    return np.array(im)

def _rotate_z(verts_v3: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate Nx3 vertices around Z by +degrees (counterclockwise)."""
    if abs(degrees) < 1e-12:
        return verts_v3
    th = np.deg2rad(degrees)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return (verts_v3 @ R.T)

class MeshLogger:
    """
    Fast, in-memory mesh logger with optional reference-comparison and rotation.
    Now includes:
      - Convergence stats vs. reference (log10 tolerance).
      - Per-frame convergence overlay in the histogram panel.
      - CSV export of convergence over iterations.
      - Optional final HTML snapshot(s).
    """
    def __init__(self, mesh,
                 log_dir=None,
                 stride=10,
                 keep_last_html=False,
                 ref_faces_prob=None,
                 ref_mesh=None,
                 converge_log_tol: float = 0.25):
        from mesh_setup.mesh_utils import get_split_indices

        self.log_dir = log_dir or os.path.join("logs", datetime.now().strftime("mesh_data_%Y%m%d_%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)

        # Base (current) geometry (CPU numpy)
        self.vertices = mesh["points"].T.detach().cpu().numpy().astype(np.float64)  # (V,3)
        self.faces    = mesh["facets"].T.detach().cpu().numpy().astype(np.int32)    # (F,3)

        b_idx, i_idx = get_split_indices(mesh)
        self.bound_idx = b_idx.detach().cpu().numpy()
        self.inter_idx = i_idx.detach().cpu().numpy()

        # Optional reference mesh (defaults to same geom as current)
        if ref_mesh is None:
            self.ref_vertices = self.vertices
            self.ref_faces    = self.faces
            self._has_ref_geom = False
        else:
            self.ref_vertices = ref_mesh["points"].T.detach().cpu().numpy().astype(np.float64)
            self.ref_faces    = ref_mesh["facets"].T.detach().cpu().numpy().astype(np.int32)
            self._has_ref_geom = True

        # Optional reference face probabilities (numpy float32)
        if ref_faces_prob is not None:
            self.ref_fp = ref_faces_prob.detach().cpu().numpy().astype(np.float32)
            if self.ref_fp.shape[0] != self.ref_faces.shape[0]:
                raise ValueError(f"ref_faces_prob has length {self.ref_fp.shape[0]} "
                                 f"but ref_mesh has {self.ref_faces.shape[0]} faces.")
        else:
            self.ref_fp = None

        # Runtime store
        self.entries = []   # dicts of {iter, loss, fp, b_counts, i_counts, conv_*}

        # Config
        self.stride = int(max(1, stride))
        self.keep_last_html = bool(keep_last_html)
        self.converge_log_tol = float(converge_log_tol)

        # Histogram binning (logspace)
        self.hist_bins = np.logspace(-10, 0, 21)

    def _compute_convergence(self, fp_np: np.ndarray):
        """
        Returns (conv_mask_all, conv_mask_boundary, conv_mask_interior) if ref present, else (None, None, None).
        Converged if |log10(fp) - log10(ref)| <= self.converge_log_tol
        """
        if self.ref_fp is None:
            return None, None, None
        safe = np.clip(fp_np, 1e-30, None)
        ref_safe = np.clip(self.ref_fp, 1e-30, None)
        diff = np.abs(np.log10(safe) - np.log10(ref_safe))
        mask_all = diff <= self.converge_log_tol
        return mask_all, mask_all[self.bound_idx], mask_all[self.inter_idx]

    def log_mesh(self, faces_prob, loss, iteration: int):
        # detach → CPU numpy
        fp = faces_prob.detach().cpu().numpy()
        b_vals = fp[self.bound_idx]
        i_vals = fp[self.inter_idx]
        b_counts, _ = np.histogram(b_vals, bins=self.hist_bins)
        i_counts, _ = np.histogram(i_vals, bins=self.hist_bins)

        # convergence (if reference present)
        conv_all, conv_b, conv_i = self._compute_convergence(fp)
        conv_entry = {}
        if conv_all is not None:
            conv_entry = {
                "conv_total_frac": float(conv_all.mean()),
                "conv_total_count": int(conv_all.sum()),
                "conv_total_N": int(conv_all.size),
                "conv_b_frac": float(conv_b.mean()),
                "conv_b_count": int(conv_b.sum()),
                "conv_b_N": int(conv_b.size),
                "conv_i_frac": float(conv_i.mean()),
                "conv_i_count": int(conv_i.sum()),
                "conv_i_N": int(conv_i.size),
            }

        self.entries.append({
            "iter": int(iteration),
            "loss": float(getattr(loss, "item", lambda: loss)()),
            "fp": fp.astype(np.float32, copy=False),
            "b_counts": b_counts.astype(np.int32, copy=False),
            "i_counts": i_counts.astype(np.int32, copy=False),
            **conv_entry
        })

    # ---------- rendering helpers ----------
    def _plotly_mesh(self, verts_v3: np.ndarray, faces_fx3: np.ndarray,
                     fp_np: np.ndarray, title: str, mask=None):
        log_fp = np.log10(np.clip(fp_np, 1e-30, None))
        mn, mx = np.floor(log_fp.min()), np.ceil(log_fp.max())
        if mx == mn: mx += 1
        tickvals = np.linspace(mn, mx, 10)
        ticktext = [f"{10**tv:.1e}" for tv in tickvals]

        faces = faces_fx3 if mask is None else faces_fx3[mask]
        intens = log_fp if mask is None else log_fp[mask]

        mesh3d = go.Mesh3d(
            x=verts_v3[:,0], y=verts_v3[:,1], z=verts_v3[:,2],
            i=faces[:,0], j=faces[:,1], k=faces[:,2],
            intensity=intens, intensitymode="cell",
            colorscale="Viridis", cmin=mn, cmax=mx,
            colorbar=dict(title="Face κ", tickmode="array", tickvals=tickvals, ticktext=ticktext),
            opacity=0.5, showscale=True
        )
        fig = go.Figure(mesh3d)
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="cube"),
            width=PLOTLY_W, height=PLOTLY_H,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    def _write_convergence_csv(self):
        """Dump convergence over all logged iterations if reference is present."""
        if self.ref_fp is None:
            return None
        path = os.path.join(self.log_dir, "convergence.csv")
        with open(path, "w") as f:
            f.write("iter,loss,conv_total_frac,conv_total_count,conv_total_N,"
                    "conv_b_frac,conv_b_count,conv_b_N,conv_i_frac,conv_i_count,conv_i_N\n")
            for e in self.entries:
                if "conv_total_frac" not in e:
                    continue
                f.write(
                    f'{e["iter"]},{e["loss"]},'
                    f'{e["conv_total_frac"]:.6f},{e["conv_total_count"]},{e["conv_total_N"]},'
                    f'{e["conv_b_frac"]:.6f},{e["conv_b_count"]},{e["conv_b_N"]},'
                    f'{e["conv_i_frac"]:.6f},{e["conv_i_count"]},{e["conv_i_N"]}\n'
                )
        return path

    def write_video_direct(self,
                           out_name="mesh_vis.mp4",
                           fps=10,
                           render_boundary_and_interior=False,
                           show_reference=False,
                           rotate_z_degrees_per_frame=0.0,
                           start_rotation_degrees=0.0,
                           save_last_html: Optional[bool] = None):
        """
        Render every `stride`-th entry + final to MP4, in memory.

        Args:
          render_boundary_and_interior: if True, top row shows boundary vs interior for CURRENT mesh.
          show_reference: if True and ref_faces_prob is provided, top row shows CURRENT (left) vs REFERENCE (right).
                          (If both this and render_boundary_and_interior are True, `show_reference` takes priority.)
          rotate_z_degrees_per_frame: rotate the vertices by this many degrees *per rendered frame* (around Z).
          start_rotation_degrees: initial rotation offset.
          save_last_html: override for saving final HTML (None → use self.keep_last_html).
        """
        path = os.path.join(self.log_dir, out_name)

        # choose frames to render
        idxs = list(range(0, len(self.entries), self.stride))
        if (len(self.entries)-1) not in idxs:
            idxs.append(len(self.entries)-1)

        with imageio.get_writer(path, fps=fps, codec="libx264") as writer:
            for frame_idx, k in enumerate(idxs):
                e = self.entries[k]
                it, fp = e["iter"], e["fp"]
                b_counts, i_counts = e["b_counts"], e["i_counts"]

                # ---- rotation for this frame ----
                angle = start_rotation_degrees + rotate_z_degrees_per_frame * frame_idx
                V_rot  = _rotate_z(self.vertices, angle)
                Vref_rot = _rotate_z(self.ref_vertices, angle) if self.ref_fp is not None else None

                # ---- top row panels ----
                if show_reference and (self.ref_fp is not None):
                    # current vs reference (full mesh both sides)
                    # Add conv % to CURRENT panel title if available
                    conv_text = ""
                    if "conv_total_frac" in e:
                        conv_text = f" — Conv {100*e['conv_total_frac']:.1f}%"
                    full_fig_cur = self._plotly_mesh(V_rot,  self.faces, fp,        f"Current κ (iter {it}){conv_text}", mask=None)
                    full_fig_ref = self._plotly_mesh(Vref_rot if Vref_rot is not None else V_rot,
                                                     self.ref_faces, self.ref_fp, "Reference κ", mask=None)
                    top_left  = _fig_to_rgba(full_fig_cur, PLOTLY_W, PLOTLY_H)
                    top_right = _fig_to_rgba(full_fig_ref, PLOTLY_W, PLOTLY_H)
                elif render_boundary_and_interior:
                    b_mask = self.bound_idx
                    i_mask = self.inter_idx
                    b_fig = self._plotly_mesh(V_rot, self.faces, fp, f"Boundary κ (iter {it})", mask=b_mask)
                    i_fig = self._plotly_mesh(V_rot, self.faces, fp, f"Interior  κ (iter {it})", mask=i_mask)
                    top_left  = _fig_to_rgba(b_fig, PLOTLY_W, PLOTLY_H)
                    top_right = _fig_to_rgba(i_fig, PLOTLY_W, PLOTLY_H)
                else:
                    conv_text = ""
                    if "conv_total_frac" in e:
                        conv_text = f" — Conv {100*e['conv_total_frac']:.1f}%"
                    full_fig = self._plotly_mesh(V_rot, self.faces, fp, f"Mesh Face κ (iter {it}){conv_text}", mask=None)
                    top_left  = _fig_to_rgba(full_fig, PLOTLY_W, PLOTLY_H)
                    top_right = top_left  # duplicate to fill width cheaply

                # bottom: histogram with convergence overlay text
                conv_line = None
                if "conv_total_frac" in e:
                    conv_line = (f"Converged: {100*e['conv_total_frac']:.1f}% "
                                 f"(B {100*e['conv_b_frac']:.1f}%, I {100*e['conv_i_frac']:.1f}%)")
                bottom = _mpl_hist_to_rgba(self.hist_bins, b_counts, i_counts, it, H_W, H_H, H_DPI, conv_text=conv_line)

                # ensure consistent sizes (RGBA)
                def _ensure_rgba(img, w, h):
                    if img.shape[1] == w and img.shape[0] == h: return img
                    return np.array(Image.fromarray(img).resize((w, h), Image.LANCZOS))

                top_left  = _ensure_rgba(top_left,  PLOTLY_W, PLOTLY_H)
                top_right = _ensure_rgba(top_right, PLOTLY_W, PLOTLY_H)
                bottom    = _ensure_rgba(bottom,    H_W,       H_H)

                top_row = np.hstack((top_left, top_right))   # (PLOTLY_H, 2*PLOTLY_W, 4)
                frame   = np.vstack((top_row, bottom))       # (PLOTLY_H+H_H, 2*PLOTLY_W, 4)

                # convert RGBA -> RGB (white background) for libx264
                if frame.shape[2] == 4:
                    alpha = frame[..., 3:4].astype(np.float32) / 255.0
                    bg = np.full_like(frame[..., :3], 255, dtype=np.uint8)
                    frame_rgb = (alpha * frame[..., :3] + (1 - alpha) * bg).astype(np.uint8)
                else:
                    frame_rgb = frame

                frame_rgb = _pad_to_multiple(frame_rgb, 16)
                writer.append_data(frame_rgb)

        # write convergence CSV (if any)
        csv_path = self._write_convergence_csv()

        # final HTML snapshot(s)
        do_html = self.keep_last_html if (save_last_html is None) else bool(save_last_html)
        if do_html and len(self.entries) > 0:
            last = self.entries[-1]
            final_fig_cur = self._plotly_mesh(self.vertices, self.faces, last["fp"], "Final Mesh Face κ", mask=None)
            final_cur_path = os.path.join(self.log_dir, "final_current_mesh.html")
            final_fig_cur.write_html(final_cur_path)

            if self.ref_fp is not None:
                final_fig_ref = self._plotly_mesh(self.ref_vertices, self.ref_faces, self.ref_fp, "Final Reference κ", mask=None)
                final_ref_path = os.path.join(self.log_dir, "final_reference_mesh.html")
                final_fig_ref.write_html(final_ref_path)

        print(f"[MeshLogger] wrote video: {path}")
        if csv_path:
            print(f"[MeshLogger] wrote convergence CSV: {csv_path}")