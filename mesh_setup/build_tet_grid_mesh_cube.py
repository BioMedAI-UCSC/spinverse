#!/usr/bin/env python3
import os
import numpy as np
import torch
import plotly.graph_objects as go
import trimesh
from mindiffdt.tgrid import TetGrid

DEVICE = 'cuda:0' if torch.cuda.is_available() else "cpu"

def save_mesh_html(verts_np: np.ndarray,
                   all_grid_faces_np: np.ndarray,
                   boundary_faces_np: np.ndarray,
                   out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # light gray: all grid faces
    grid_faces = go.Mesh3d(
        x=verts_np[:, 0], y=verts_np[:, 1], z=verts_np[:, 2],
        i=all_grid_faces_np[:, 0], j=all_grid_faces_np[:, 1], k=all_grid_faces_np[:, 2],
        opacity=0.15, color='lightgray', showscale=False, name="grid"
    )

    # blue: boundary of the selected path/shape
    boundary = go.Mesh3d(
        x=verts_np[:, 0], y=verts_np[:, 1], z=verts_np[:, 2],
        i=boundary_faces_np[:, 0], j=boundary_faces_np[:, 1], k=boundary_faces_np[:, 2],
        opacity=0.6, color='blue', showscale=False, name="boundary"
    )

    # red points: all vertices
    pts = go.Scatter3d(
        x=verts_np[:, 0], y=verts_np[:, 1], z=verts_np[:, 2],
        mode='markers', marker=dict(size=2, color='red'), name="points"
    )

    fig = go.Figure([grid_faces, boundary, pts])
    fig.update_layout(scene=dict(aspectmode="data"))
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)

def plot_scatter(verts):
    scatter = go.Scatter3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        mode='markers',
        marker=dict(size=4)
    )
    fig = go.Figure(scatter)
    fig.show()

def plot_scatter_mesh(verts, tets):
    scatter = go.Scatter3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        mode='markers',
        marker=dict(size=4, color='red')
    )
    mesh3d = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=tets[:, 0],
        j=tets[:, 1],
        k=tets[:, 2],
        opacity=0.7,
        color='lightblue',
    )
    fig = go.Figure([scatter, mesh3d])
    fig.show()

def plot_mesh_from_faces(vertices, faces, title="3D Mesh", show_edges=False, opacity=0.7, color='lightblue'):
    mesh3d = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=opacity,
        color=color,
        showscale=False
    )
    fig = go.Figure(data=[mesh3d])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )
    )
    fig.show()
    return fig

def get_centered_grid_size(domain_min, domain_max, desired_elements):
    x_min, y_min, z_min = domain_min
    x_max, y_max, z_max = domain_max
    x_length = x_max - x_min
    y_length = y_max - y_min
    z_length = z_max - z_min
    grid_size_x = x_length / desired_elements
    grid_size_y = y_length / desired_elements
    grid_size_z = z_length / desired_elements
    return max(grid_size_x, grid_size_y, grid_size_z)

def save_mesh_as_pth(points, faces, boundary_faces, tets, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh = {
        "points": torch.from_numpy(points.T.astype(np.float32)),
        "elements": torch.from_numpy(tets.T.astype(np.int64)),
        "facets": torch.from_numpy(faces.T.astype(np.int64)),
        "boundary_faces": torch.from_numpy(boundary_faces.T.astype(np.int64))
    }
    torch.save(mesh, path)
    return mesh

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
    boundary_keys = unique_keys[counts == 1]
    boundary_key_set = set(boundary_keys.tolist())
    mask = torch.tensor([k.item() in boundary_key_set for k in keys],
                        device=elements.device, dtype=torch.bool)
    boundary_faces = faces_sorted[mask]
    return boundary_faces

def _grid_verts_faces_elems(MIN_DOMAIN, MAX_DOMAIN, num_grids):
    grid_size = (MAX_DOMAIN - MIN_DOMAIN) / num_grids
    tg = TetGrid(DEVICE)
    tg.init((MIN_DOMAIN, MIN_DOMAIN, MIN_DOMAIN),
            (MAX_DOMAIN, MAX_DOMAIN, MAX_DOMAIN),
            grid_size)
    verts = tg.verts - tg.verts.mean(dim=0)
    faces = tg.tri_idx
    elts = tg.tet_idx
    return verts, faces, elts

def _contained_indices(shape_trimesh: trimesh.Trimesh, verts_torch: torch.Tensor):
    mask_np = shape_trimesh.contains(verts_torch.detach().cpu().numpy())
    mask = torch.from_numpy(mask_np).to(DEVICE)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    return idx

def _filter_by_indices(faces_or_elts: torch.Tensor, idx: torch.Tensor):
    return torch.isin(faces_or_elts, idx).all(dim=1)

def build_cube_in_grid(
    domain_min=-10.0,
    domain_max=10.0,
    num_grids=3,
    cube_extents=(8.0, 8.0, 8.0),
    cube_center=(0.0, 0.0, 0.0),
    save_path='mesh_setup/mesh_files/cube_in_grid.pth',
    plot_boundary=True
):
    verts, faces, elts = _grid_verts_faces_elems(domain_min, domain_max, num_grids)
    cube = trimesh.creation.box(extents=cube_extents)
    T = np.eye(4)
    T[:3, 3] = np.asarray(cube_center, dtype=float)
    cube.apply_transform(T)

    idx = _contained_indices(cube, verts)
    f_mask = _filter_by_indices(faces, idx)
    e_mask = _filter_by_indices(elts, idx)
    contained_faces = faces[f_mask]
    contained_elts = elts[e_mask]

    boundary_faces = boundary_faces_from_tets(contained_elts)

    if plot_boundary and boundary_faces.numel() > 0:
        plot_mesh_from_faces(verts.detach().cpu().numpy(),
                             boundary_faces.detach().cpu().numpy(),
                             title="Cube-in-Grid Boundary Faces", opacity=0.6, color='lightblue')

    mesh = save_mesh_as_pth(
        points=verts.detach().cpu().numpy(),
        faces=contained_faces.detach().cpu().numpy(),
        boundary_faces=boundary_faces.detach().cpu().numpy(),
        tets=contained_elts.detach().cpu().numpy(),
        path=save_path
    )
    html_out = 'mesh_setup/mesh_files/cube_8x8x8_in_grid.html'
    save_mesh_html(
        verts_np=verts.detach().cpu().numpy(),
        all_grid_faces_np=faces.detach().cpu().numpy(),           # the full grid faces (tg.tri_idx)
        boundary_faces_np=boundary_faces.detach().cpu().numpy(),  # the blue boundary
        out_path=html_out
    )
    print(f"HTML saved: {html_out}")

    return mesh

def build_two_cubes_in_grid(
    domain_min=-10.0,
    domain_max=10.0,
    num_grids=6,
    cube1_extents=(6.0, 6.0, 6.0),
    cube1_center=(-5.0, 0.0, 0.0),
    cube2_extents=(6.0, 6.0, 6.0),
    cube2_center=(5.0, 0.0, 0.0),
    save_path='mesh_setup/mesh_files/two_cubes_6_in_grid.pth',
    plot_boundary=True
):
    verts, faces, elts = _grid_verts_faces_elems(domain_min, domain_max, num_grids)

    def make_cube(ext, cen):
        m = trimesh.creation.box(extents=ext)
        T = np.eye(4); T[:3, 3] = np.asarray(cen, dtype=float)
        m.apply_transform(T)
        return m

    cube1 = make_cube(cube1_extents, cube1_center)
    cube2 = make_cube(cube2_extents, cube2_center)

    # Exact union for point classification: inside either cube
    V = verts.detach().cpu().numpy()
    mask1 = cube1.contains(V)
    mask2 = cube2.contains(V)
    mask_union = np.logical_or(mask1, mask2)
    idx = torch.nonzero(torch.from_numpy(mask_union).to(DEVICE), as_tuple=False).squeeze(1)

    f_mask = torch.isin(faces, idx).all(dim=1)
    e_mask = torch.isin(elts, idx).all(dim=1)
    contained_faces = faces[f_mask]
    contained_elts = elts[e_mask]

    boundary_faces = boundary_faces_from_tets(contained_elts)

    if plot_boundary and boundary_faces.numel() > 0:
        plot_mesh_from_faces(
            verts.detach().cpu().numpy(),
            boundary_faces.detach().cpu().numpy(),
            title="Two-Cubes-in-Grid Boundary Faces",
            opacity=0.6, color='lightblue'
        )

    mesh = save_mesh_as_pth(
        points=verts.detach().cpu().numpy(),
        faces=contained_faces.detach().cpu().numpy(),
        boundary_faces=boundary_faces.detach().cpu().numpy(),
        tets=contained_elts.detach().cpu().numpy(),
        path=save_path
    )

    html_out = 'mesh_setup/mesh_files/two_cubes_6_in_grid.html'
    save_mesh_html(
        verts_np=verts.detach().cpu().numpy(),
        all_grid_faces_np=faces.detach().cpu().numpy(),
        boundary_faces_np=boundary_faces.detach().cpu().numpy(),
        out_path=html_out
    )
    print(f"HTML saved: {html_out}")

    return mesh

def build_cyl_in_grid():
    MIN_DOMAIN = -10.0
    MAX_DOMAIN = 10.0
    num_grids = 3
    grid_size = (MAX_DOMAIN - MIN_DOMAIN) / num_grids

    test_grid = TetGrid(DEVICE)
    test_grid.init((MIN_DOMAIN, MIN_DOMAIN, MIN_DOMAIN), (MAX_DOMAIN, MAX_DOMAIN, MAX_DOMAIN), grid_size)

    verts = test_grid.verts
    verts = verts - verts.mean(dim=0)

    cyl = trimesh.creation.cylinder(radius=10, height=40.0)
    mask_np = cyl.contains(verts.cpu().numpy())
    idx = torch.nonzero(torch.from_numpy(mask_np).to(DEVICE), as_tuple=False).squeeze(1)

    faces = test_grid.tri_idx
    f_mask = torch.isin(faces, idx).all(dim=1)
    contained_faces = faces[f_mask]

    elts = test_grid.tet_idx
    e_mask = torch.isin(elts, idx).all(dim=1)
    contained_elts = elts[e_mask]

    boundary_faces = boundary_faces_from_tets(contained_elts)
    plot_mesh_from_faces(verts.cpu().numpy(), boundary_faces.cpu().numpy())

    mesh = {
        "points": verts.T,
        "facets": faces.T,
        "elements": elts.T,
        "boundary_faces": boundary_faces.T
    }
    path = 'mesh_setup/mesh_files/cyl_10r_40h_in_grid_20x20.pth'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(mesh, path)
    return mesh

if __name__ == '__main__':
    # Single cube example
    _ = build_cube_in_grid(
        domain_min=-10.0,
        domain_max=10.0,
        num_grids=6,
        cube_extents=(8.0, 8.0, 8.0),
        cube_center=(0.0, 0.0, 0.0),
        save_path='mesh_setup/mesh_files/cube_8x8x8_in_grid.pth',
        plot_boundary=True
    )

    # Two cubes example
    _ = build_two_cubes_in_grid(
        domain_min=-10.0,
        domain_max=10.0,
        num_grids=6,
        cube1_extents=(6.0, 6.0, 6.0),
        cube1_center=(-5.0, 0.0, 0.0),
        cube2_extents=(6.0, 6.0, 6.0),
        cube2_center=(5.0, 0.0, 0.0),
        save_path='mesh_setup/mesh_files/two_cubes_6_in_grid.pth',
        plot_boundary=True
    )
