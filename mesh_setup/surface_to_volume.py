import numpy as np
import trimesh
import numpy as np
import torch
import trimesh
from typing import Tuple
from mindiffdt.cgaldt import CGALDTStruct
from src.ellipsoidal_scale_mesh import ellipsoidal_scale_mesh

# -------------------------------
# I/O
# -------------------------------
def load_surface(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a surface mesh and return (V,F) where:
      V: (Ns,3) float64, F: (Ms,3) int32 (0-based triangle indices)
    """
    m = trimesh.load(path, process=True)
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError("Expected a single mesh object.")
    if not m.is_watertight:
        raise ValueError("Surface is not watertight. Repair before volumetric meshing.")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int32)
    return V, F

# -------------------------------
# Interior sampling
# -------------------------------
def uniform_grid_with_jitter(V: np.ndarray, 
                             v_in_count: int,
                             mesh_volume: float,
                             jitter_frac: float = 0.1) -> np.ndarray:
    """
    Uniform 3D grid with small random jitter (each axis in [-j,+j] with j=jitter_frac*h).
    Returns interior points (Ni,3).
    """
    xmin, ymin, zmin = V.min(axis=0)
    xmax, ymax, zmax = V.max(axis=0)
    
    # find grid size h from inside to surface ratio
    # grid_volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
    # Rough approximation of h for in count
    h = (mesh_volume / (v_in_count)) ** (1/3)

    # small padding to ensure coverage
    pad = 0.25 * h
    xmin -= pad; ymin -= pad; zmin -= pad
    xmax += pad; ymax += pad; zmax += pad

    nx = max(2, int(np.ceil((xmax - xmin) / h)) + 1)
    ny = max(2, int(np.ceil((ymax - ymin) / h)) + 1)
    nz = max(2, int(np.ceil((zmax - zmin) / h)) + 1)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # jitter
    j = jitter_frac * h
    if j > 0:
        P += np.random.uniform(-j, +j, size=P.shape)

    return P, h

# -------------------------------
# Tetrahedralization
# -------------------------------
def tetrahedralize(points: torch.Tensor):
    """
    Run CGAL delaunay tetrahedralization on volume mesh
    """
    # Run tetrahedralization on combined points
    dt_result = CGALDTStruct.forward(points)
    tets = dt_result.dsimp_point_id.to(dtype=torch.long)
    tet_faces = torch.vstack([tets[:, [0,1,2]],
                              tets[:, [0,1,3]],
                              tets[:, [0,2,3]],
                              tets[:, [1,2,3]]])

    # Find tet faces and interior/boundary split
    faces_sorted = torch.sort(tet_faces, dim=1)
    faces_unique, inverse, counts = torch.unique(
        faces_sorted.values, dim=0, return_inverse=True, return_counts=True
    )

    # boundary faces are those that appear exactly once
    boundary_mask = counts[inverse] == 1
    boundary_faces = tet_faces[boundary_mask]    # keep original (unsorted) orientation

    # interior faces appear >= 2 times (usually exactly 2)
    interior_faces = tet_faces[~boundary_mask]

    return {
        "points": points,
        "tets": tets,
        "facets": faces_unique,
        "boundary_faces": boundary_faces,
        "interior_faces": interior_faces,
    }

# -------------------------------
# End-to-end: surface -> interior points -> tets
# -------------------------------
def surface_to_volume_convex(
    surface_path: str,
    in_pt_ratio: float) -> dict:

    # load your surface (triangles must form a closed watertight mesh)
    V, F = load_surface(surface_path)

    # normalize mesh surface (can be rescaled later)
    center = V.mean(axis=0)  # Compute the center
    V -= center              # Center the mesh at the origin
    norm_scale = np.linalg.norm(V, axis=1).max()  # Compute the maximum distance from the origin
    V /= norm_scale              

    # Use trimesh to determine volume and interior points
    tm = trimesh.Trimesh(V, F)

    # Add interior points to mesh
    in_pt_count = int(V.shape[0] * in_pt_ratio)
    P_noisy, h = uniform_grid_with_jitter(V, in_pt_count, tm.volume)
    P_in = P_noisy[tm.contains(P_noisy)]
    points = np.vstack([V, P_in])
    points = torch.from_numpy(points.astype(np.float32)).contiguous()

    # Run tetrahedralization on combined points
    dt_result = CGALDTStruct.forward(points)
    tets = dt_result.dsimp_point_id.to(dtype=torch.long)
    tet_faces = torch.vstack([tets[:, [0,1,2]],
                              tets[:, [0,1,3]],
                              tets[:, [0,2,3]],
                              tets[:, [1,2,3]]])

    # Find tet faces and interior/boundary split
    faces_sorted = torch.sort(tet_faces, dim=1)
    faces_unique, inverse, counts = torch.unique(
        faces_sorted.values, dim=0, return_inverse=True, return_counts=True
    )

    # boundary faces are those that appear exactly once
    boundary_mask = counts[inverse] == 1
    boundary_faces = tet_faces[boundary_mask]    # keep original (unsorted) orientation

    # interior faces appear >= 2 times (usually exactly 2)
    interior_faces = tet_faces[~boundary_mask]

    mesh_data = {
        "points": points.T,
        "elements": tets.T,
        "facets": faces_unique.T,
        "boundary_faces": boundary_faces.T,
        "n_vertices": torch.Tensor(points.shape[0]),
    }

    return mesh_data

if __name__ == '__main__':
    ply_path = "rh.pial_lowres.ply"
    mesh_data = surface_to_volume_convex(
        surface_path=ply_path,
        in_pt_ratio=0.1)

    torch.save(mesh_data, "brain_segment_mesh.pth")