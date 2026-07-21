import os
import importlib
import imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch

from mindiffdt.tetra import TetraSet

def get_split_indices(mesh):
    '''
    Get indices for boundary and interior faces in mesh
    '''

    verts = mesh["points"].T
    elems = mesh["elements"].T
    tetra_set = TetraSet(verts, elems)
    all_faces = tetra_set.faces
    bidx = tetra_set.boundary_faces()
    is_bndry = torch.zeros(all_faces.shape[0], dtype=torch.bool, device=mesh["points"].device)
    is_bndry[bidx] = True
    interior = torch.where(~is_bndry)[0]
    mesh_fs = mesh["facets"].T

    interior_in_mesh = []
    for f in all_faces[interior]:
        for i, mf in enumerate(mesh_fs):
            if torch.equal(torch.sort(f)[0], torch.sort(mf)[0]):
                interior_in_mesh.append(i)
                break

    is_interior = torch.zeros(all_faces.shape[0], dtype=torch.bool, device=mesh['points'].device)
    is_interior[interior_in_mesh] = True

    boundary_in_mesh = torch.where(~is_interior)[0]
    interior_in_mesh = torch.tensor(interior_in_mesh, device=mesh['points'].device)

    return boundary_in_mesh, interior_in_mesh

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