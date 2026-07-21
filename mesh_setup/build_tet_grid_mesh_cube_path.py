#!/usr/bin/env python3
import os
import numpy as np
import torch
import plotly.graph_objects as go
from mindiffdt.tgrid import TetGrid

DEVICE = 'cuda:0' if torch.cuda.is_available() else "cpu"

# ---------------------- I/O / VIS ----------------------

def save_mesh_as_pth(points, faces_all_grid, boundary_faces, tets_inner, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh = {
        "points": torch.from_numpy(points.T.astype(np.float32)),
        "elements": torch.from_numpy(tets_inner.T.astype(np.int64)),
        "facets": torch.from_numpy(faces_all_grid.T.astype(np.int64)),   # ALL grid faces
        "boundary_faces": torch.from_numpy(boundary_faces.T.astype(np.int64))
    }
    torch.save(mesh, path)
    return mesh

def save_mesh_html(verts_np, faces_all_grid_np, boundary_faces_np, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid_faces = go.Mesh3d(
        x=verts_np[:, 0], y=verts_np[:, 1], z=verts_np[:, 2],
        i=faces_all_grid_np[:, 0], j=faces_all_grid_np[:, 1], k=faces_all_grid_np[:, 2],
        opacity=0.15, color='lightgray', showscale=False, name="grid"
    )
    boundary = go.Mesh3d(
        x=verts_np[:, 0], y=verts_np[:, 1], z=verts_np[:, 2],
        i=boundary_faces_np[:, 0], j=boundary_faces_np[:, 1], k=boundary_faces_np[:, 2],
        opacity=0.6, color='blue', showscale=False, name="inner-cube-boundary"
    )
    pts = go.Scatter3d(
        x=verts_np[:, 0], y=verts_np[:, 1], z=verts_np[:, 2],
        mode='markers', marker=dict(size=2, color='red'), name="points"
    )
    fig = go.Figure([grid_faces, boundary, pts])
    fig.update_layout(scene=dict(aspectmode="data"))
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)

# ---------------------- GRID / MESH ----------------------

def boundary_faces_from_tets(elements: torch.Tensor) -> torch.Tensor:
    f0 = elements[:, [0, 1, 2]]
    f1 = elements[:, [0, 1, 3]]
    f2 = elements[:, [0, 2, 3]]
    f3 = elements[:, [1, 2, 3]]
    all_faces = torch.vstack([f0, f1, f2, f3])
    faces_sorted, _ = torch.sort(all_faces, dim=1)
    P = int(elements.max().item()) + 1
    keys = (faces_sorted[:, 0] * P + faces_sorted[:, 1]) * P + faces_sorted[:, 2]
    unique_keys, counts = torch.unique(keys, return_counts=True)
    boundary_keys = set(unique_keys[counts == 1].tolist())
    mask = torch.tensor([k.item() in boundary_keys for k in keys],
                        device=elements.device, dtype=torch.bool)
    return faces_sorted[mask]

def _build_grid(domain_min, domain_max, num_grids):
    grid_size = (domain_max - domain_min) / float(num_grids)
    tg = TetGrid(DEVICE)
    tg.init((domain_min, domain_min, domain_min),
            (domain_max, domain_max, domain_max),
            grid_size)
    verts = tg.verts - tg.verts.mean(dim=0)  # keep your centering
    faces = tg.tri_idx
    elts  = tg.tet_idx
    return verts, faces, elts, grid_size

# ---------------------- INNER CUBE (CLOSED) ----------------------

def build_inner_cube_closed_by_cells(
    domain_min=-10.0,
    domain_max=10.0,
    num_grids=10,
    cube_cells=(5, 5, 5),
    center_world=(0.0, 0.0, 0.0),
    save_base='mesh_setup/mesh_files/inner_cube_closed'
):
    verts, faces_all_grid, elts, grid_size = _build_grid(domain_min, domain_max, num_grids)

    # Ensure elts is [T,4] long on same device
    if elts.dtype != torch.long:
        elts = elts.long()
    if elts.ndim != 2:
        raise RuntimeError(f"Unexpected tet_idx shape: {elts.shape}")
    if elts.shape[1] != 4 and elts.shape[0] == 4:
        elts = elts.t().contiguous()  # make [T,4]

    # convert cells -> world half-extent along each axis
    cx, cy, cz = cube_cells
    hx = 0.5 * float(cx) * grid_size
    hy = 0.5 * float(cy) * grid_size
    hz = 0.5 * float(cz) * grid_size
    cx0, cy0, cz0 = map(float, center_world)

    # Gather tet vertex positions: [T,4,3]
    tet_pts = verts[elts]  # advanced indexing

    # Inside test per-vertex -> [T,4] boolean
    in_x = (tet_pts[..., 0] >= cx0 - hx) & (tet_pts[..., 0] <= cx0 + hx)
    in_y = (tet_pts[..., 1] >= cy0 - hy) & (tet_pts[..., 1] <= cy0 + hy)
    in_z = (tet_pts[..., 2] >= cz0 - hz) & (tet_pts[..., 2] <= cz0 + hz)

    # ALL FOUR vertices must be inside for a closed inner surface
    inside_all_verts = (in_x & in_y & in_z).all(dim=1)  # <-- was dim=2 (bug)

    contained_elts = elts[inside_all_verts]
    boundary_faces  = boundary_faces_from_tets(contained_elts)

    # Save
    mesh_path = f'{save_base}_cells_{cx}x{cy}x{cz}.pth'
    html_path = f'{save_base}_cells_{cx}x{cy}x{cz}.html'
    _ = save_mesh_as_pth(
        points=verts.detach().cpu().numpy(),
        faces_all_grid=faces_all_grid.detach().cpu().numpy(),   # ALL grid faces
        boundary_faces=boundary_faces.detach().cpu().numpy(),   # closed inner-cube boundary
        tets_inner=contained_elts.detach().cpu().numpy(),
        path=mesh_path
    )
    save_mesh_html(
        verts_np=verts.detach().cpu().numpy(),
        faces_all_grid_np=faces_all_grid.detach().cpu().numpy(),
        boundary_faces_np=boundary_faces.detach().cpu().numpy(),
        out_path=html_path
    )
    print(f"PTH saved: {mesh_path}")
    print(f"HTML saved: {html_path}")

    return {
        "points": verts,
        "facets_all_grid": faces_all_grid,
        "elements_inner": contained_elts,
        "boundary_faces": boundary_faces
    }

# ---------------------- EXAMPLES ----------------------
if __name__ == '__main__':
    # Grid 10x10x10, inner cube 5x5x5 cells (closed)
    build_inner_cube_closed_by_cells(
        domain_min=-10.0, domain_max=10.0, num_grids=10,
        cube_cells=(5, 5, 5), center_world=(0.0, 0.0, 0.0),
        save_base='mesh_setup/mesh_files/inner_cube_5'
    )

    # Same grid, inner cube 7x7x7 cells (closed)
    build_inner_cube_closed_by_cells(
        domain_min=-10.0, domain_max=10.0, num_grids=10,
        cube_cells=(7, 7, 7), center_world=(0.0, 0.0, 0.0),
        save_base='mesh_setup/mesh_files/inner_cube_7'
    )

    # Same grid, inner cube 8x8x8 cells (closed)
    build_inner_cube_closed_by_cells(
        domain_min=-10.0, domain_max=10.0, num_grids=10,
        cube_cells=(8, 8, 8), center_world=(0.0, 0.0, 0.0),
        save_base='mesh_setup/mesh_files/inner_cube_8'
    )
