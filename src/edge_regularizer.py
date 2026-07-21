import torch
import numpy as np
import plotly.graph_objects as go

def edge_to_faces(facets: torch.Tensor):
    """
    facets: (nF,3) or (3,nF) LongTensor (CPU or CUDA)
    returns:
      edges_idx   : (nE,2) LongTensor
      face_edges  : (nF,3) LongTensor (edge id per face-edge)
      edge_indptr : (nE+1,) LongTensor
      edge_faces  : (3*nF,) LongTensor (concatenated face indices per edge)
    """
    if facets.ndim != 2 or 3 not in facets.shape:
        raise ValueError("facets must be (nF,3) or (3,nF)")
    faces = facets if facets.shape[1] == 3 else facets.t()
    dev = faces.device
    nF  = faces.shape[0]

    v0, v1, v2 = faces[:,0], faces[:,1], faces[:,2]
    e01 = torch.stack([torch.minimum(v0,v1), torch.maximum(v0,v1)], dim=1)
    e02 = torch.stack([torch.minimum(v0,v2), torch.maximum(v0,v2)], dim=1)
    e12 = torch.stack([torch.minimum(v1,v2), torch.maximum(v1,v2)], dim=1)
    all_edges = torch.cat([e01, e02, e12], dim=0).long()           # (3*nF,2)

    nV = int(all_edges.max().item()) + 1
    hash_vals = (all_edges[:,0].long() * nV + all_edges[:,1].long())
    uniq_hash, inv = torch.unique(hash_vals, sorted=True, return_inverse=True)
    nE = uniq_hash.shape[0]
    edges_idx = torch.stack([uniq_hash // nV, uniq_hash % nV], dim=1).long()
    face_edges = inv.view(3, nF).t().contiguous().long()

    # Build a CSR-like edge→faces structure (indptr, indices) without Python loops
    face_ids = torch.cat([
        torch.arange(nF, device=faces.device),  # for e01 block
        torch.arange(nF, device=faces.device),  # for e02 block
        torch.arange(nF, device=faces.device),  # for e12 block
    ], dim=0)  # (3*nF,)
    
    # Sort by edge id so faces for the same edge are contiguous
    order = torch.argsort(inv)
    edge_faces_sorted = face_ids[order]
    inv_sorted = inv[order]
    # counts per edge
    counts = torch.bincount(inv, minlength=nE)
    edge_indptr = torch.empty(nE+1, device=dev, dtype=torch.long)
    edge_indptr[0] = 0
    edge_indptr[1:] = torch.cumsum(counts, dim=0)

    edge_to_faces = [
        edge_faces_sorted[edge_indptr[e]:edge_indptr[e+1]].tolist()
        for e in range(edge_indptr.numel() - 1)
    ]
    return edges_idx, edge_to_faces

def _to_np(x):
    """Accept torch.Tensor or np.ndarray / list, return numpy array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def plot_edge_with_faces(points, facets, edge, faces,
                               show_base=True, base_opacity=0.15,
                               face_color="orange", edge_color="red",
                               edge_width=8, title=None):
    """
    points : (N,3)  torch.Tensor | np.ndarray  float
    facets : (F,3)  torch.Tensor | np.ndarray  int
    edge   : (2,)   torch.Tensor | np.ndarray | tuple/list of ints (v0, v1)
    faces  : (k,)   torch.Tensor | np.ndarray | list of face indices attached to the edge
    """
    P = _to_np(points)                      # (N,3)
    F = _to_np(facets).astype(np.int64)     # (F,3)
    e = _to_np(edge).astype(np.int64).ravel()
    faces = _to_np(faces).astype(np.int64).ravel()

    if P.ndim != 2 or P.shape[1] != 3: raise ValueError("points must be (N,3)")
    if F.ndim != 2 or F.shape[1] != 3: raise ValueError("facets must be (F,3)")
    if e.size != 2: raise ValueError("edge must have 2 vertex indices")

    v0, v1 = int(e[0]), int(e[1])
    p0, p1 = P[v0], P[v1]

    fig = go.Figure()

    if show_base:
        fig.add_trace(go.Mesh3d(
            x=P[:,0], y=P[:,1], z=P[:,2],
            i=F[:,0], j=F[:,1], k=F[:,2],
            color="lightgray", opacity=base_opacity, name="mesh", hoverinfo="skip"
        ))

    if faces.size > 0:
        Fsel = F[faces]
        fig.add_trace(go.Mesh3d(
            x=P[:,0], y=P[:,1], z=P[:,2],
            i=Fsel[:,0], j=Fsel[:,1], k=Fsel[:,2],
            color=face_color, opacity=0.75, name="attached faces"
        ))

    fig.add_trace(go.Scatter3d(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
        mode="lines+markers",
        line=dict(width=edge_width, color=edge_color),
        marker=dict(size=6, color=edge_color),
        name=f"edge ({v0},{v1})",
        hovertemplate=f"edge ({v0},{v1})<extra></extra>"
    ))

    # auto bounds
    xyz_min, xyz_max = P.min(0), P.max(0)
    pad = 0.05 * float(np.linalg.norm(xyz_max - xyz_min) or 1.0)
    xr = [xyz_min[0]-pad, xyz_max[0]+pad]
    yr = [xyz_min[1]-pad, xyz_max[1]+pad]
    zr = [xyz_min[2]-pad, xyz_max[2]+pad]

    fig.update_layout(
        title=title or f"Edge ({v0},{v1}) with attached faces",
        scene=dict(xaxis=dict(title="X", range=xr),
                   yaxis=dict(title="Y", range=yr),
                   zaxis=dict(title="Z", range=zr),
                   aspectmode="data"),
        legend=dict(itemsizing="constant")
    )
    return fig

if __name__ == "__main__":
    mesh_data = torch.load("small_mesh.pth", weights_only=False)
    edges, edge_faces_sorted = edge_to_faces(mesh_data['facets'])

    # points = mesh_data['points']
    # facets = mesh_data['facets']

    # for i in range(10):
    #     rand_edge = np.random.randint(0, edges.shape[0])
    #     fig = plot_edge_with_faces_torch(points.T, facets.T, edges[rand_edge], edge_faces_sorted[rand_edge])
    #     fig.show()