import os
import numpy as np
from mindiffdt.tgrid import TetGrid, EffTetGrid
import torch
from plotly import graph_objects as go
import trimesh as tm
from scipy.spatial import Delaunay, ConvexHull
from src.ellipsoidal_scale_mesh import ellipsoidal_scale_mesh
from plot.plot_femesh_plotly import plot_femesh_plotly
from plot.plot_femesh import plot_femesh
from mesh_setup.mesh_utils import boundary_faces_from_tets
from shapely.geometry import Polygon

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


def compute_tet_centroids(verts, elts):
    """
    Compute centroid of each tetrahedron.
    
    Args:
        verts: (N_verts, 3) tensor
        elts: (N_tets, 4) tensor
    
    Returns:
        (N_tets, 3) tensor of centroids
    """
    # Centroid is average of 4 vertices
    v0 = verts[elts[:, 0]]
    v1 = verts[elts[:, 1]]
    v2 = verts[elts[:, 2]]
    v3 = verts[elts[:, 3]]
    return (v0 + v1 + v2 + v3) / 4.0


def find_tet_containing_point(point, verts, elts, tet_centroids):
    """
    Find tet that contains point (or closest tet if point is outside).
    
    Returns:
        int: tet index
    """
    # Simple approach: find tet with closest centroid
    distances = torch.norm(tet_centroids - point, dim=1)
    return int(torch.argmin(distances).item())


def build_tet_adjacency_with_faces(elts):
    """
    Build adjacency with face information.
    
    Returns:
        tet_neighbors: dict {tet_idx: [neighbor_tet_idx, ...]}
        tet_face_map: dict {(tet_i, tet_j): face_key} - which face connects two tets
    """
    N_tets = elts.shape[0]
    
    # Each tet has 4 faces
    tet_faces = []
    for i in range(N_tets):
        v = elts[i]
        f0 = tuple(sorted([v[0].item(), v[1].item(), v[2].item()]))
        f1 = tuple(sorted([v[0].item(), v[1].item(), v[3].item()]))
        f2 = tuple(sorted([v[0].item(), v[2].item(), v[3].item()]))
        f3 = tuple(sorted([v[1].item(), v[2].item(), v[3].item()]))
        tet_faces.append([f0, f1, f2, f3])
    
    # Build face -> tets mapping
    face_to_tets = {}
    for tet_idx in range(N_tets):
        for face in tet_faces[tet_idx]:
            if face not in face_to_tets:
                face_to_tets[face] = []
            face_to_tets[face].append(tet_idx)
    
    # Build adjacency and face map
    tet_neighbors = {i: [] for i in range(N_tets)}
    tet_face_map = {}
    
    for face, tets_sharing in face_to_tets.items():
        if len(tets_sharing) == 2:
            t0, t1 = tets_sharing
            tet_neighbors[t0].append(t1)
            tet_neighbors[t1].append(t0)
            
            # Store which face connects these tets (use sorted tuple as key)
            tet_pair_key = tuple(sorted([t0, t1]))
            tet_face_map[tet_pair_key] = face
    
    return tet_neighbors, tet_face_map


def get_path_vertices(path_tets, elts):
    """
    Get all unique vertices used by tets in the path.
    
    Args:
        path_tets: tensor of tet indices forming the path
        elts: (N_tets, 4) tensor of all tet vertex indices
    
    Returns:
        set of vertex indices (ints)
    """
    path_verts = set()
    for tet_idx in path_tets:
        tet_vertices = elts[tet_idx].tolist()
        path_verts.update(tet_vertices)
    return path_verts


def astar_single_chain_path_no_vertex_sharing(start_pt, end_pt, verts, elts, tet_centroids,
                                               tet_neighbors, occupied_vertices):
    """
    Find single-chain path of tetrahedra from start to end using A*.
    Path cannot use any tetrahedra that contain occupied vertices.
    
    Returns:
        torch.Tensor of tet indices forming the path, or None if no path found
    """
    device = start_pt.device
    
    # Find start and end tets
    start_tet = find_tet_containing_point(start_pt, verts, elts, tet_centroids)
    end_tet = find_tet_containing_point(end_pt, verts, elts, tet_centroids)
    
    # Check if start or end tets contain occupied vertices
    start_verts = set(elts[start_tet].tolist())
    end_verts = set(elts[end_tet].tolist())
    
    if start_verts.intersection(occupied_vertices) or end_verts.intersection(occupied_vertices):
        # Start or end tet already contains occupied vertices
        return None
    
    if start_tet == end_tet:
        return torch.tensor([start_tet], device=device, dtype=torch.long)
    
    # A* search
    import heapq
    
    # Priority queue: (f_score, tet_idx)
    open_set = []
    heapq.heappush(open_set, (0.0, start_tet))
    
    # Track path
    came_from = {}
    
    # Scores
    g_score = {start_tet: 0.0}
    f_score = {start_tet: torch.norm(tet_centroids[start_tet] - tet_centroids[end_tet]).item()}
    
    visited = set()
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)
        
        # Goal reached
        if current == end_tet:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return torch.tensor(path, device=device, dtype=torch.long)
        
        # Explore neighbors
        for neighbor in tet_neighbors[current]:
            if neighbor in visited:
                continue
            
            # Check if neighbor tet contains any occupied vertices
            neighbor_verts = set(elts[neighbor].tolist())
            if neighbor_verts.intersection(occupied_vertices):
                # Cannot use this tet - it contains occupied vertices
                continue
            
            # Calculate scores
            tentative_g = g_score[current] + torch.norm(
                tet_centroids[current] - tet_centroids[neighbor]
            ).item()
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = torch.norm(tet_centroids[neighbor] - tet_centroids[end_tet]).item()
                f_score[neighbor] = tentative_g + h
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return None


def build_poly_sweep_v4_single_chain(num_paths=1, force_opp=False, debug=False):
    """
    Build mesh with single-chain tet paths. Each path is a sequence of 
    connected tetrahedra (tet -> neighbor tet -> ...). Paths cannot share 
    any vertices on their boundaries.
    
    Uses A* to find shortest path through tet graph, with vertex-sharing constraint.
    """
    MIN_DOMAIN = -20.0
    MAX_DOMAIN = 20.0
    num_grids = 3
    RANGE = MAX_DOMAIN - MIN_DOMAIN
    grid_size = (RANGE) / num_grids
    
    test_grid = TetGrid(DEVICE)
    test_grid.init((MIN_DOMAIN, MIN_DOMAIN, MIN_DOMAIN), (MAX_DOMAIN, MAX_DOMAIN, MAX_DOMAIN), grid_size)
    
    verts = test_grid.verts
    verts = verts - verts.mean(dim=0)
    faces = test_grid.tri_idx
    elts = test_grid.tet_idx
    
    print(f"Grid: {verts.shape[0]} verts, {elts.shape[0]} tets")
    
    # Build tet adjacency with face information
    tet_neighbors, tet_face_map = build_tet_adjacency_with_faces(elts)
    tet_centroids = compute_tet_centroids(verts, elts)
    
    # Get boundary points
    xyz_mins, _ = verts.min(dim=0)
    xyz_maxs, _ = verts.max(dim=0)
    eps = 1e-3
    
    is_x_min = verts[:, 0] <= xyz_mins[0] + eps
    is_x_max = verts[:, 0] >= xyz_maxs[0] - eps
    is_y_min = verts[:, 1] <= xyz_mins[1] + eps
    is_y_max = verts[:, 1] >= xyz_maxs[1] - eps
    is_z_min = verts[:, 2] <= xyz_mins[2] + eps
    is_z_max = verts[:, 2] >= xyz_maxs[2] - eps
    
    num_faces_per_vert = (is_x_min.int() + is_x_max.int() + 
                          is_y_min.int() + is_y_max.int() + 
                          is_z_min.int() + is_z_max.int())
    on_one_face_only = num_faces_per_vert == 1
    
    boundary_points_combined = [
        verts[is_x_min & on_one_face_only],
        verts[is_x_max & on_one_face_only],
        verts[is_y_min & on_one_face_only],
        verts[is_y_max & on_one_face_only],
        verts[is_z_min & on_one_face_only],
        verts[is_z_max & on_one_face_only]
    ]
    
    # Track occupied vertices globally (vertices used by any path)
    occupied_vertices = set()
    all_path_tets = []
    
    max_attempts = 100
    
    for path_idx in range(num_paths):
        print(f"\n=== Creating Path {path_idx} ===")
        
        attempts = 0
        path_found = False
        
        while attempts < max_attempts and not path_found:
            # Generate start/end points
            if force_opp:
                random_axis = np.random.randint(0, 3)
                random_pair = (2*random_axis, 2*random_axis+1)
                face_points_1 = boundary_points_combined[random_pair[0]]
                start_pt = face_points_1[torch.randint(0, len(face_points_1), (1,)).item()]
                face_points_2 = boundary_points_combined[random_pair[1]]
                end_pt = face_points_2[torch.randint(0, len(face_points_2), (1,)).item()]
            else:
                random_pair = np.random.choice(6, size=2, replace=False)
                face_points_1 = boundary_points_combined[random_pair[0]]
                start_pt = face_points_1[torch.randint(0, len(face_points_1), (1,)).item()]
                face_points_2 = boundary_points_combined[random_pair[1]]
                end_pt = face_points_2[torch.randint(0, len(face_points_2), (1,)).item()]
            
            # Find path using A* that avoids occupied vertices
            path_tets = astar_single_chain_path_no_vertex_sharing(
                start_pt, end_pt, verts, elts, tet_centroids,
                tet_neighbors, occupied_vertices
            )
            
            if path_tets is not None and len(path_tets) > 0:
                # Get vertices used by this path
                path_verts = get_path_vertices(path_tets, elts)
                
                # Check if any path vertices are already occupied
                if not path_verts.intersection(occupied_vertices):
                    # Valid path found - mark its vertices as occupied
                    occupied_vertices.update(path_verts)
                    all_path_tets.append(path_tets)
                    
                    print(f"Path {path_idx}: {len(path_tets)} tets, {len(path_verts)} unique vertices, after {attempts + 1} attempts")
                    path_found = True
                else:
                    attempts += 1
            else:
                attempts += 1
        
        if not path_found:
            print(f"Warning: Could not find valid path {path_idx} after {max_attempts} attempts")
            break
    
    if len(all_path_tets) == 0:
        print("Error: No valid paths created!")
        return
    
    # Combine all tets from all paths
    combined_tet_indices = torch.cat(all_path_tets).unique()
    selected_elts = elts[combined_tet_indices]
    
    # Get boundary faces of the combined mesh
    boundary_faces = boundary_faces_from_tets(selected_elts)
    
    if debug:
        plot_mesh_from_faces(verts.cpu().numpy(),
                           boundary_faces.cpu().numpy(),
                           title="Single-Chain Paths (No Shared Vertices)")
    
    # Build final mesh
    mesh = {
        "points": verts.T,
        "facets": faces.T,
        "elements": elts.T,
        "boundary_faces": boundary_faces.T
    }
    
    # Save
    base_path = "mesh_setup/mesh_files/random_mesh_chain"
    ext = ".pth"
    path = base_path + ext
    id = 0
    while os.path.exists(path):
        id += 1
        path = f"{base_path}{id}{ext}"
    
    torch.save(mesh, path)
    print(f"\nSaved mesh to: {path}")
    print(f"Total paths: {len(all_path_tets)}")
    print(f"Total unique tets: {len(combined_tet_indices)}")
    print(f"Total occupied vertices: {len(occupied_vertices)}")


def build_poly_sweep_v4_single_chain_alt(num_paths=1, 
                                         force_opp=False, 
                                         debug=False, 
                                         max_attempts=100,
                                         min_max_domain=(-10.0, 10.0),
                                         num_grids=3,
                                         exclude_boundary_tets=False):
    """
    Build mesh with single-chain tet paths. Each path is a sequence of 
    connected tetrahedra (tet -> neighbor tet -> ...). Paths cannot share 
    any vertices on their boundaries.
    
    Uses A* to find shortest path through tet graph, with vertex-sharing constraint.
    
    Args:
        exclude_boundary_tets: If True, removes boundary tetrahedra from final mesh.
                              Default is False (boundary tets allowed).
    """
    MIN_DOMAIN = min_max_domain[0]
    MAX_DOMAIN = min_max_domain[1]
    RANGE = MAX_DOMAIN - MIN_DOMAIN
    grid_size = (RANGE) / num_grids
    
    test_grid = TetGrid(DEVICE)
    test_grid.init((MIN_DOMAIN, MIN_DOMAIN, MIN_DOMAIN), (MAX_DOMAIN, MAX_DOMAIN, MAX_DOMAIN), grid_size)
    
    verts = test_grid.verts
    verts = verts - verts.mean(dim=0)
    faces = test_grid.tri_idx
    elts = test_grid.tet_idx
    
    print(f"Grid: {verts.shape[0]} verts, {elts.shape[0]} tets")
    
    # Build tet adjacency with face information
    tet_neighbors, tet_face_map = build_tet_adjacency_with_faces(elts)
    tet_centroids = compute_tet_centroids(verts, elts)
    
    # Get boundary points
    xyz_mins, _ = verts.min(dim=0)
    xyz_maxs, _ = verts.max(dim=0)
    eps = 1e-3
    
    is_x_min = verts[:, 0] <= xyz_mins[0] + eps
    is_x_max = verts[:, 0] >= xyz_maxs[0] - eps
    is_y_min = verts[:, 1] <= xyz_mins[1] + eps
    is_y_max = verts[:, 1] >= xyz_maxs[1] - eps
    is_z_min = verts[:, 2] <= xyz_mins[2] + eps
    is_z_max = verts[:, 2] >= xyz_maxs[2] - eps
    
    num_faces_per_vert = (is_x_min.int() + is_x_max.int() + 
                          is_y_min.int() + is_y_max.int() + 
                          is_z_min.int() + is_z_max.int())
    on_one_face_only = num_faces_per_vert == 1
    
    boundary_points_combined = [
        verts[is_x_min & on_one_face_only],
        verts[is_x_max & on_one_face_only],
        verts[is_y_min & on_one_face_only],
        verts[is_y_max & on_one_face_only],
        verts[is_z_min & on_one_face_only],
        verts[is_z_max & on_one_face_only]
    ]
    
    # Track occupied vertices globally (vertices used by any path)
    occupied_vertices = set()
    all_path_tets = []
    
    max_attempts_per_mode = max_attempts  # attempts per sampling mode
    
    for path_idx in range(num_paths):
        print(f"\n=== Creating Path {path_idx} ===")
        
        path_found = False
        
        # Try force_opp first if requested
        if force_opp:
            print("Trying opposite faces...")
            for attempts in range(max_attempts_per_mode):
                # Generate start/end points on opposite faces
                random_axis = np.random.randint(0, 3)
                random_pair = (2*random_axis, 2*random_axis+1)
                face_points_1 = boundary_points_combined[random_pair[0]]
                start_pt = face_points_1[torch.randint(0, len(face_points_1), (1,)).item()]
                face_points_2 = boundary_points_combined[random_pair[1]]
                end_pt = face_points_2[torch.randint(0, len(face_points_2), (1,)).item()]
                
                # Find path using A* that avoids occupied vertices
                path_tets = astar_single_chain_path_no_vertex_sharing(
                    start_pt, end_pt, verts, elts, tet_centroids,
                    tet_neighbors, occupied_vertices
                )
                
                if path_tets is not None and len(path_tets) > 0:
                    # Get vertices used by this path
                    path_verts = get_path_vertices(path_tets, elts)
                    
                    # Check if any path vertices are already occupied
                    if not path_verts.intersection(occupied_vertices):
                        # Valid path found - mark its vertices as occupied
                        occupied_vertices.update(path_verts)
                        all_path_tets.append(path_tets)
                        
                        print(f"Path {path_idx}: {len(path_tets)} tets, {len(path_verts)} unique vertices (opposite faces, attempt {attempts + 1})")
                        path_found = True
                        break
        
        # If not found with opposite faces, try any face pair
        if not path_found:
            print("Trying adjacent/any face pairs...")
            for attempts in range(max_attempts_per_mode):
                # Generate start/end points on any two different faces
                random_pair = np.random.choice(6, size=2, replace=False)
                face_points_1 = boundary_points_combined[random_pair[0]]
                start_pt = face_points_1[torch.randint(0, len(face_points_1), (1,)).item()]
                face_points_2 = boundary_points_combined[random_pair[1]]
                end_pt = face_points_2[torch.randint(0, len(face_points_2), (1,)).item()]
                
                # Find path using A* that avoids occupied vertices
                path_tets = astar_single_chain_path_no_vertex_sharing(
                    start_pt, end_pt, verts, elts, tet_centroids,
                    tet_neighbors, occupied_vertices
                )
                
                if path_tets is not None and len(path_tets) > 0:
                    # Get vertices used by this path
                    path_verts = get_path_vertices(path_tets, elts)
                    
                    # Check if any path vertices are already occupied
                    if not path_verts.intersection(occupied_vertices):
                        # Valid path found - mark its vertices as occupied
                        occupied_vertices.update(path_verts)
                        all_path_tets.append(path_tets)
                        
                        print(f"Path {path_idx}: {len(path_tets)} tets, {len(path_verts)} unique vertices (any faces, attempt {attempts + 1})")
                        path_found = True
                        break
        
        if not path_found:
            print(f"Warning: Could not find valid path {path_idx} after {max_attempts_per_mode * (2 if force_opp else 1)} attempts")
            break
    
    if len(all_path_tets) == 0:
        print("Error: No valid paths created!")
        return
    
    # Combine all tets from all paths
    combined_tet_indices = torch.cat(all_path_tets).unique()
    
    # Optionally filter out boundary tets
    if exclude_boundary_tets:
        boundary_tets = get_boundary_tets(verts, elts)
        print(f"Boundary tets before filtering: {len(boundary_tets)}")
        
        # Filter out boundary tets from combined indices
        combined_tet_indices_list = combined_tet_indices.tolist()
        filtered_tet_indices = [idx for idx in combined_tet_indices_list if idx not in boundary_tets]
        
        if len(filtered_tet_indices) == 0:
            print("Error: No tets remaining after filtering boundary tets!")
            return
        
        combined_tet_indices = torch.tensor(filtered_tet_indices, device=combined_tet_indices.device, dtype=combined_tet_indices.dtype)
        print(f"Tets after filtering boundary: {len(combined_tet_indices)} (removed {len(combined_tet_indices_list) - len(filtered_tet_indices)})")
    
    selected_elts = elts[combined_tet_indices]
    
    # Get boundary faces of the combined mesh
    boundary_faces = boundary_faces_from_tets(selected_elts)
    
    if debug:
        plot_mesh_from_faces(verts.cpu().numpy(),
                            boundary_faces.cpu().numpy(),
                            title="Single-Chain Paths (No Shared Vertices)",
                            opacity=1.0,
                            color='cyan')
    
    # Build final mesh
    mesh = {
        "points": verts.T,
        "facets": faces.T,
        "elements": elts.T,
        "boundary_faces": boundary_faces.T
    }
    
    # Save
    base_path = "mesh_setup/mesh_files/random_mesh_chain"
    if exclude_boundary_tets:
        base_path += "_no_bound"
    ext = ".pth"
    path = base_path + ext
    id = 0
    while os.path.exists(path):
        id += 1
        path = f"{base_path}{id}{ext}"
    
    torch.save(mesh, path)
    print(f"\nSaved mesh to: {path}")
    print(f"Total paths: {len(all_path_tets)}")
    print(f"Total unique tets: {len(combined_tet_indices)}")
    print(f"Total occupied vertices: {len(occupied_vertices)}")

def get_boundary_tets(verts, elts, eps=1e-3):
    """
    Identify tetrahedra that have at least one vertex on the boundary of the tet grid.
    
    Args:
        verts: (N_verts, 3) tensor of vertices
        elts: (N_tets, 4) tensor of tetrahedron vertex indices
        eps: tolerance for boundary detection
        
    Returns:
        set of tet indices that are on the boundary
    """
    xyz_mins, _ = verts.min(dim=0)
    xyz_maxs, _ = verts.max(dim=0)
    
    # Check which vertices are on any boundary face
    is_x_min = verts[:, 0] <= xyz_mins[0] + eps
    is_x_max = verts[:, 0] >= xyz_maxs[0] - eps
    is_y_min = verts[:, 1] <= xyz_mins[1] + eps
    is_y_max = verts[:, 1] >= xyz_maxs[1] - eps
    is_z_min = verts[:, 2] <= xyz_mins[2] + eps
    is_z_max = verts[:, 2] >= xyz_maxs[2] - eps
    
    is_boundary = (is_x_min | is_x_max | is_y_min | is_y_max | is_z_min | is_z_max)
    boundary_vert_indices = set(torch.where(is_boundary)[0].tolist())
    
    # Find tets that have at least one boundary vertex
    boundary_tets = set()
    for tet_idx in range(elts.shape[0]):
        tet_verts = set(elts[tet_idx].tolist())
        if tet_verts.intersection(boundary_vert_indices):
            boundary_tets.add(tet_idx)
    
    return boundary_tets


if __name__ == '__main__':
    # Original version
    # build_poly_sweep_v4_single_chain_alt(num_paths=5, force_opp=True, debug=True)
    
    # New version that excludes boundary tets
    build_poly_sweep_v4_single_chain_alt(num_paths=1, force_opp=True, debug=True)