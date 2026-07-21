import numpy as np
from mindiffdt.tgrid import TetGrid, EffTetGrid
import torch
from plotly import graph_objects as go
import trimesh
from scipy.spatial import Delaunay, ConvexHull
from src.ellipsoidal_scale_mesh import ellipsoidal_scale_mesh
from plot.plot_femesh_plotly import plot_femesh_plotly
from plot.plot_femesh import plot_femesh
from mesh_utils import boundary_faces_from_tets

DEVICE = 'cuda:0' if torch.cuda.is_available() else "cpu"

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

    # Create scatter
    scatter = go.Scatter3d(
        x=verts[:, 0],
        y=verts[:, 1], 
        z=verts[:, 2],
        mode='markers',
        marker=dict(size=4, color='red')
    )

    # Create 3D mesh plot
    mesh3d = go.Mesh3d(
            x=verts[:, 0],  # X coordinates
            y=verts[:, 1],  # Y coordinates  
            z=verts[:, 2],  # Z coordinates
            i=tets[:, 0],  # Indices for triangle vertices
            j=tets[:, 1],
            k=tets[:, 2],
            opacity=0.7,
            color='lightblue',
    )

    fig = go.Figure([scatter, mesh3d])
    fig.show()

def plot_mesh_from_faces(vertices, faces, title="3D Mesh", show_edges=False, opacity=0.7, color='lightblue'):
    """
    Plot a 3D mesh using Plotly from vertices and faces.
    
    Args:
        vertices (np.ndarray): Array of shape (N, 3) containing vertex coordinates
        faces (np.ndarray): Array of shape (M, 3) containing face indices  
        title (str): Title for the plot
        show_edges (bool): Whether to show mesh edges
        opacity (float): Mesh opacity (0-1)
        color (str): Mesh color
    """
    import plotly.graph_objects as go
    
    # Create the mesh
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
    
    traces = [mesh3d]
    
    # Create figure
    fig = go.Figure(data=traces)
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

# Adjust grid_size for perfect centering
def get_centered_grid_size(domain_min, domain_max, desired_elements):
    """Calculate grid size that results in perfectly centered mesh"""
    x_min, y_min, z_min = domain_min
    x_max, y_max, z_max = domain_max
    
    x_length = x_max - x_min
    y_length = y_max - y_min  
    z_length = z_max - z_min
    
    # For perfect centering, use exact division
    grid_size_x = x_length / desired_elements
    grid_size_y = y_length / desired_elements
    grid_size_z = z_length / desired_elements
    
    # Use the maximum to ensure we don't exceed bounds
    return max(grid_size_x, grid_size_y, grid_size_z)

def save_mesh_as_pth(points, 
                     faces,
                     boundary_faces,
                     tets,
                     path):

    mesh = {
        "points": torch.from_numpy(points.T.astype(np.float32)),    # 3 x V
        "elements": torch.from_numpy(tets.T.astype(np.int64)),      # 4 x T
        "facets": torch.from_numpy(faces.T.astype(np.int64)),       # 3 x F
        "boundary_faces": torch.from_numpy(boundary_faces.T.astype(np.int64))
    }

    torch.save(mesh, path)

def boundary_faces_from_tets(elements: torch.Tensor) -> torch.Tensor:
    """
    elements: (T, 4) int/long tensor of tet vertex indices
    return: (B, 3) long tensor of boundary faces (vertex indices sorted within each face)
    """

    # All 4 faces of each tet
    f0 = elements[:, [0, 1, 2]]
    f1 = elements[:, [0, 1, 3]]
    f2 = elements[:, [0, 2, 3]]
    f3 = elements[:, [1, 2, 3]]
    all_faces = torch.vstack([f0, f1, f2, f3])        # (4T, 3)

    # Canonicalize: sort verts within each face so shared faces match
    faces_sorted, _ = torch.sort(all_faces, dim=1)     # (4T, 3)

    # Hash each face (a,b,c) -> single integer key
    # Pick base P > max vertex index so (a,b,c) maps uniquely
    P = int(elements.max().item()) + 1
    keys = (faces_sorted[:, 0] * P + faces_sorted[:, 1]) * P + faces_sorted[:, 2]  # (4T,)

    # Count occurrences
    unique_keys, counts = torch.unique(keys, return_counts=True)

    # Keep keys that appear exactly once
    boundary_keys = unique_keys[counts == 1]           # (B,)

    # Build membership mask: which rows of `faces_sorted` are boundary?
    boundary_key_set = set(boundary_keys.tolist())
    mask = torch.tensor(
        [k.item() in boundary_key_set for k in keys],
        device=elements.device,
        dtype=torch.bool
    )

    boundary_faces = faces_sorted[mask]                # (B, 3), sorted rows
    return boundary_faces


def build_sphere_in_grid():
    MIN_DOMAIN = -10.0
    MAX_DOMAIN = 10.0
    num_grids = 3
    grid_size = (MAX_DOMAIN - MIN_DOMAIN) / num_grids

    test_grid = TetGrid(DEVICE)
    test_grid.init((MIN_DOMAIN, MIN_DOMAIN, MIN_DOMAIN), (MAX_DOMAIN, MAX_DOMAIN, MAX_DOMAIN), grid_size)

    # Extract x, y, z coordinates from vertices
    verts = test_grid.verts
    verts = verts - verts.mean(dim=0)
    print(f"Number of vertices: {verts.shape[0]}")
    print(f"Domain max: test_grid.domain_max")
    print(f"Domain min: test_grid.domain_min")

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10)
    # cube   = trimesh.creation.box(extents=[2,2,2])
    # cyl    = trimesh.creation.cylinder(radius=10, height=50.0)

    # filter sphere
    sphere_filter = sphere.contains(verts.cpu().numpy())
    contained_verts = verts[sphere_filter, :]

    # find contained vertex indices within original verts tensor
    indices = []
    for v in contained_verts:
        idx = torch.all(torch.isclose(verts, v, atol=1e-8), dim=1).nonzero(as_tuple=True)[0]
        indices.append(idx.item())
    indices = torch.tensor(indices)

    # find contained faces
    faces = test_grid.tri_idx
    f_mask = torch.isin(faces, indices).all(dim=1)
    contained_faces = faces[f_mask]

    # find contained elements
    elts = test_grid.tet_idx
    e_mask = torch.isin(elts, indices).all(dim=1)
    contained_elts = elts[e_mask]

    # find boundary faces
    boundary_faces = boundary_faces_from_tets(contained_elts)
    boundary_faces.shape
    plot_mesh_from_faces(verts.cpu().numpy(),
                        boundary_faces.cpu().numpy())

    mesh = {
        "points": verts.T,    # 3 x V
        "facets": faces.T,      # 4 x Tbbbb
        "elements": elts.T,       # 3 x Fb
        "boundary_faces": boundary_faces.T
        }

    path = 'mesh_setup/mesh_files/sphere_10r_in_grid_20x20.pth'
    torch.save(mesh, path)

def build_cyl_in_grid():
    MIN_DOMAIN = -10.0
    MAX_DOMAIN = 10.0
    num_grids = 5
    RANGE = MAX_DOMAIN - MIN_DOMAIN
    grid_size = (RANGE) / num_grids

    test_grid = TetGrid(DEVICE)
    test_grid.init((MIN_DOMAIN, MIN_DOMAIN, MIN_DOMAIN), (MAX_DOMAIN, MAX_DOMAIN, MAX_DOMAIN), grid_size)

    # Extract x, y, z coordinates from vertices
    verts = test_grid.verts
    verts = verts - verts.mean(dim=0)
    print(f"Number of vertices: {verts.shape[0]}")
    print(f"Domain max: test_grid.domain_max")
    print(f"Domain min: test_grid.domain_min")

    # sphere = trimesh.creation.icosphere(subdivisions=3, radius=10)
    # cube   = trimesh.creation.box(extents=[2,2,2])
    radius = 8
    height = 22
    rotation = 45
    cyl = trimesh.creation.cylinder(radius=radius, height=height)

    if rotation != 0:
        # Manual rotation instead of using trimesh's apply_transform
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('x', rotation, degrees=True)
        
        # Apply rotation manually to cylinder vertices
        vertices = cyl.vertices
        rotated_vertices = rot.apply(vertices)
        cyl.vertices = rotated_vertices
    # if rotation != 0:
    #     # Rotate cylinder 45 degrees around X-axis
    #     rotation_matrix = trimesh.transformations.rotation_matrix(
    #         angle=np.radians(rotation),  # Use the rotation variable
    #         direction=np.array([1.0, 0.0, 0.0], dtype=np.float64),  # Explicit numpy array
    #         point=np.array([0.0, 0.0, 0.0], dtype=np.float64)       # Explicit numpy array
    #     )

        # cyl.apply_transform(rotation_matrix)
    
    # filter sphere
    sphere_filter = cyl.contains(verts.cpu().numpy())
    contained_verts = verts[sphere_filter, :]

    # find contained vertex indices within original verts tensor
    indices = []
    for v in contained_verts:
        idx = torch.all(torch.isclose(verts, v, atol=1e-8), dim=1).nonzero(as_tuple=True)[0]
        indices.append(idx.item())
    indices = torch.tensor(indices, device=DEVICE)

    # find contained faces
    faces = test_grid.tri_idx
    f_mask = torch.isin(faces, indices).all(dim=1)
    contained_faces = faces[f_mask]

    # find contained elements
    elts = test_grid.tet_idx
    e_mask = torch.isin(elts, indices).all(dim=1)
    contained_elts = elts[e_mask]

    # find boundary faces
    boundary_faces = boundary_faces_from_tets(contained_elts)
    boundary_faces.shape
    plot_mesh_from_faces(verts.cpu().numpy(),
                        boundary_faces.cpu().numpy())

    mesh = {
        "points": verts.T,    # 3 x V
        "facets": faces.T,      # 4 x Tbbbb
        "elements": elts.T,       # 3 x Fb
        "boundary_faces": boundary_faces.T
    }

    path = f'mesh_setup/mesh_files/cyl_{int(radius)}r_{int(height)}h_{int(rotation)}d_in_{int(num_grids)}g_{int(RANGE)}r_mesh.pth'
    torch.save(mesh, path)
    
    print(f"Saved file: {path}")


def build_torus():
    MIN_DOMAIN = -10.0
    MAX_DOMAIN = 10.0
    num_grids = 3
    RANGE = MAX_DOMAIN - MIN_DOMAIN
    grid_size = (RANGE) / num_grids

    test_grid = TetGrid(DEVICE)
    test_grid.init((MIN_DOMAIN, MIN_DOMAIN, MIN_DOMAIN), (MAX_DOMAIN, MAX_DOMAIN, MAX_DOMAIN), grid_size)

    # Extract x, y, z coordinates from vertices
    verts = test_grid.verts
    verts = verts - verts.mean(dim=0)
    print(f"Number of vertices: {verts.shape[0]}")
    print(f"Domain max: test_grid.domain_max")
    print(f"Domain min: test_grid.domain_min")

    # sphere = trimesh.creation.icosphere(subdivisions=3, radius=10)
    # cube   = trimesh.creation.box(extents=[2,2,2])
    maj_radius = 10
    min_radius = 6
    height = 22
    rotation = 90
    tor = trimesh.creation.torus(maj_radius, min_radius)

    # if rotation != 0:
    #     # Manual rotation instead of using trimesh's apply_transform
    #     from scipy.spatial.transform import Rotation as R
    #     rot = R.from_euler('x', rotation, degrees=True)
        
    #     # Apply rotation manually to cylinder vertices
    #     vertices = cyl2.vertices
    #     rotated_vertices = rot.apply(vertices)
    #     cyl2.vertices = rotated_vertices
    
    # filter sphere
    tor_filter = tor.contains(verts.cpu().numpy())
    # cyl2_filter = cyl2.contains(verts.cpu().numpy())
    # cyl_filter = cyl1_filter # | cyl2_filter
    contained_verts = verts[tor_filter, :]

    # find contained vertex indices within original verts tensor
    indices = []
    for v in contained_verts:
        idx = torch.all(torch.isclose(verts, v, atol=1e-8), dim=1).nonzero(as_tuple=True)[0]
        indices.append(idx.item())
    indices = torch.tensor(indices, device=DEVICE)

    # find contained faces
    faces = test_grid.tri_idx
    f_mask = torch.isin(faces, indices).all(dim=1)
    contained_faces = faces[f_mask]

    # find contained elements
    elts = test_grid.tet_idx
    e_mask = torch.isin(elts, indices).all(dim=1)
    contained_elts = elts[e_mask]

    # find boundary faces
    boundary_faces = boundary_faces_from_tets(contained_elts)
    boundary_faces.shape
    plot_mesh_from_faces(verts.cpu().numpy(),
                        boundary_faces.cpu().numpy())

    mesh = {
        "points": verts.T,    # 3 x V
        "facets": faces.T,      # 4 x Tbbbb
        "elements": elts.T,       # 3 x Fb
        "boundary_faces": boundary_faces.T
    }

    path = f'mesh_setup/mesh_files/torus_majr{int(maj_radius)}_minr{int(min_radius)}.pth'
    torch.save(mesh, path)
    
    print(f"Saved file: {path}")

def build_cross_cyl_in_grid():
    MIN_DOMAIN = -10.0
    MAX_DOMAIN = 10.0
    num_grids = 4
    RANGE = MAX_DOMAIN - MIN_DOMAIN
    grid_size = (RANGE) / num_grids

    test_grid = TetGrid(DEVICE)
    test_grid.init((MIN_DOMAIN, MIN_DOMAIN, MIN_DOMAIN), (MAX_DOMAIN, MAX_DOMAIN, MAX_DOMAIN), grid_size)

    # Extract x, y, z coordinates from vertices
    verts = test_grid.verts
    verts = verts - verts.mean(dim=0)
    print(f"Number of vertices: {verts.shape[0]}")
    print(f"Domain max: test_grid.domain_max")
    print(f"Domain min: test_grid.domain_min")

    # sphere = trimesh.creation.icosphere(subdivisions=3, radius=10)
    # cube   = trimesh.creation.box(extents=[2,2,2])
    radius = 5
    height = 22
    rotation = 90
    cyl1 = trimesh.creation.cylinder(radius=radius, height=height)

    cyl2 = trimesh.creation.cylinder(radius=radius, height=height)
    # if rotation != 0:
    #     # Manual rotation instead of using trimesh's apply_transform
    #     from scipy.spatial.transform import Rotation as R
    #     rot = R.from_euler('x', rotation, degrees=True)
        
    #     # Apply rotation manually to cylinder vertices
    #     vertices = cyl2.vertices
    #     rotated_vertices = rot.apply(vertices)
    #     cyl2.vertices = rotated_vertices
    
    # filter sphere
    cyl1_filter = cyl1.contains(verts.cpu().numpy())
    # cyl2_filter = cyl2.contains(verts.cpu().numpy())
    cyl_filter = cyl1_filter # | cyl2_filter
    contained_verts = verts[cyl_filter, :]

    # find contained vertex indices within original verts tensor
    indices = []
    for v in contained_verts:
        idx = torch.all(torch.isclose(verts, v, atol=1e-8), dim=1).nonzero(as_tuple=True)[0]
        indices.append(idx.item())
    indices = torch.tensor(indices, device=DEVICE)

    # find contained faces
    faces = test_grid.tri_idx
    f_mask = torch.isin(faces, indices).all(dim=1)
    contained_faces = faces[f_mask]

    # find contained elements
    elts = test_grid.tet_idx
    e_mask = torch.isin(elts, indices).all(dim=1)
    contained_elts = elts[e_mask]

    # find boundary faces
    boundary_faces = boundary_faces_from_tets(contained_elts)
    boundary_faces.shape
    plot_mesh_from_faces(verts.cpu().numpy(),
                        boundary_faces.cpu().numpy())

    mesh = {
        "points": verts.T,    # 3 x V
        "facets": faces.T,      # 4 x Tbbbb
        "elements": elts.T,       # 3 x Fb
        "boundary_faces": boundary_faces.T
    }

    # path = f'mesh_setup/mesh_files/cyl_{int(radius)}r_{int(height)}h_{int(rotation)}d_in_{int(num_grids)}g_{int(RANGE)}r_mesh.pth'
    # torch.save(mesh, path)
    
    # print(f"Saved file: {path}")

def poly_sweep_mesh():
    pass


def build_cross_cyl_in_grid():
    MIN_DOMAIN = -10.0
    MAX_DOMAIN = 10.0
    num_grids = 4
    RANGE = MAX_DOMAIN - MIN_DOMAIN
    grid_size = (RANGE) / num_grids

    test_grid = TetGrid(DEVICE)
    test_grid.init((MIN_DOMAIN, MIN_DOMAIN, MIN_DOMAIN), (MAX_DOMAIN, MAX_DOMAIN, MAX_DOMAIN), grid_size)

    # Extract x, y, z coordinates from vertices
    verts = test_grid.verts
    verts = verts - verts.mean(dim=0)
    print(f"Number of vertices: {verts.shape[0]}")
    print(f"Domain max: test_grid.domain_max")
    print(f"Domain min: test_grid.domain_min")

    # sphere = trimesh.creation.icosphere(subdivisions=3, radius=10)
    # cube   = trimesh.creation.box(extents=[2,2,2])
    radius = 5
    height = 22
    rotation = 90
    cyl1 = trimesh.creation.cylinder(radius=radius, height=height)

    cyl2 = trimesh.creation.cylinder(radius=radius, height=height)
    # if rotation != 0:
    #     # Manual rotation instead of using trimesh's apply_transform
    #     from scipy.spatial.transform import Rotation as R
    #     rot = R.from_euler('x', rotation, degrees=True)
        
    #     # Apply rotation manually to cylinder vertices
    #     vertices = cyl2.vertices
    #     rotated_vertices = rot.apply(vertices)
    #     cyl2.vertices = rotated_vertices
    
    # filter sphere
    cyl1_filter = cyl1.contains(verts.cpu().numpy())
    # cyl2_filter = cyl2.contains(verts.cpu().numpy())
    cyl_filter = cyl1_filter # | cyl2_filter
    contained_verts = verts[cyl_filter, :]

    # find contained vertex indices within original verts tensor
    indices = []
    for v in contained_verts:
        idx = torch.all(torch.isclose(verts, v, atol=1e-8), dim=1).nonzero(as_tuple=True)[0]
        indices.append(idx.item())
    indices = torch.tensor(indices, device=DEVICE)

    # find contained faces
    faces = test_grid.tri_idx
    f_mask = torch.isin(faces, indices).all(dim=1)
    contained_faces = faces[f_mask]

    # find contained elements
    elts = test_grid.tet_idx
    e_mask = torch.isin(elts, indices).all(dim=1)
    contained_elts = elts[e_mask]

    # find boundary faces
    boundary_faces = boundary_faces_from_tets(contained_elts)
    boundary_faces.shape
    plot_mesh_from_faces(verts.cpu().numpy(),
                        boundary_faces.cpu().numpy())

    mesh = {
        "points": verts.T,    # 3 x V
        "facets": faces.T,      # 4 x Tbbbb
        "elements": elts.T,       # 3 x Fb
        "boundary_faces": boundary_faces.T
    }

    # path = f'mesh_setup/mesh_files/cyl_{int(radius)}r_{int(height)}h_{int(rotation)}d_in_{int(num_grids)}g_{int(RANGE)}r_mesh.pth'
    # torch.save(mesh, path)
    
    # print(f"Saved file: {path}")

if __name__ == '__main__':

    build_cyl_in_grid()
    # mesh_data = torch.load("brain_segment_mesh.pth", map_location=DEVICE, weights_only=False)

    # mesh = {
    #     "points": mesh_data["points"],
    #     "facets": mesh_data["facets"],
    #     "boundary_faces": mesh_data['facets'],
    #     "elements": mesh_data["elements"]
    # }
    # print(
    #     f"Mesh: {mesh['points'].shape[1]} points, "
    #     f"{mesh['facets'].shape[1]} facets, "
    #     f"{mesh['elements'].shape[1]} elements"
    # )


    # F = mesh["facets"].shape[1]
    # E = mesh["elements"].shape[1]

    # elementmarkers = torch.arange(E, dtype=torch.long, device=DEVICE)
    # facetmarkers   = torch.arange(F, dtype=torch.long, device=DEVICE)
    
    # femesh = {
    #     "points": mesh["points"],
    #     "facets": mesh["facets"],
    #     "boundary_faces": mesh["boundary_faces"],
    #     "elements": mesh["elements"],
    #     "facetmarkers": facetmarkers,
    #     "elementmarkers": elementmarkers,
    # }
    
    # print("FeMesh initialized")

    # femesh_scaled = ellipsoidal_scale_mesh(
    #     femesh, torch.tensor([10, 10, 10], device=DEVICE)
    # )

    # verts = femesh_scaled['points'].T.cpu().numpy()
    # faces = femesh_scaled['facets'].T.cpu().numpy()

    # print(femesh_scaled['boundary_faces'].shape)
    # plot_mesh_from_faces(verts, femesh_scaled['boundary_faces'].T.cpu().numpy())