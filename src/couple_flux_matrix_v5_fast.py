import torch
import logging

logger = logging.getLogger("mvrecon_3d")

def compute_areas(tris, pts):
    """Vectorized area computation."""
    verts = pts[tris]  # [nfaces, 3, 3]
    vec1 = verts[:, 1, :] - verts[:, 0, :]
    vec2 = verts[:, 2, :] - verts[:, 0, :]
    cross = torch.cross(vec1, vec2, dim=1)
    areas = 0.5 * torch.norm(cross, p=2, dim=1)
    return areas.clamp(min=1e-12)

def precompute_flux_geometry(femesh):
    """
    Precompute geometry-dependent parts of flux matrix that don't depend on faces_prob.
    Call this once before optimization loop.
    
    Returns:
        dict with precomputed data for fast flux matrix assembly
    """
    facets = femesh["facets"]
    ncomp = len(facets)
    nboundary = len(facets[0])
    npoint_cmpts = [pts.shape[1] for pts in femesh["points"]]
    N = sum(npoint_cmpts)
    device = femesh["points"][0].device
    
    c_inds = torch.cumsum(torch.tensor([0] + npoint_cmpts, device=device), dim=0)
    local_mat = torch.tensor([[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]], device=device)
    
    precomputed = {
        "N": N,
        "c_inds": c_inds,
        "local_mat": local_mat,
        "device": device,
        "boundary_data": []
    }
    
    face_counter = 0
    
    for ibound in range(nboundary):
        touches = [c for c in range(ncomp) if facets[c][ibound] is not None]
        if not touches:
            continue
        
        nfaces = facets[touches[0]][ibound].shape[1]
        
        if len(touches) == 1:
            # Single-touch (Robin boundary)
            c = touches[0]
            pts_c = femesh["points"][c].t().contiguous()
            tris = facets[c][ibound].t().long()
            slc_start = c_inds[c].item()
            
            areas = compute_areas(tris, pts_c)
            local_Q = (areas[:, None, None] / 12.) * local_mat[None, :, :]
            
            g_rows = (tris[:, :, None].expand(-1, -1, 3) + slc_start).flatten()
            g_cols = (tris[:, None, :].expand(-1, 3, -1) + slc_start).flatten()
            
            precomputed["boundary_data"].append({
                "type": "robin",
                "face_start": face_counter,
                "face_end": face_counter + nfaces,
                "local_Q": local_Q,
                "g_rows": g_rows,
                "g_cols": g_cols,
                "N": N
            })
            
        elif len(touches) == 2:
            # Two-touch (exchange coupling)
            c1, c2 = touches
            pts1 = femesh["points"][c1].t().contiguous()
            pts2 = femesh["points"][c2].t().contiguous()
            tris1 = facets[c1][ibound].t().long()
            tris2 = facets[c2][ibound].t().long()
            slc1_start = c_inds[c1].item()
            slc2_start = c_inds[c2].item()
            
            areas1 = compute_areas(tris1, pts1)
            local_Q1 = (areas1[:, None, None] / 12.) * local_mat[None, :, :]
            areas2 = compute_areas(tris2, pts2)
            local_Q2 = (areas2[:, None, None] / 12.) * local_mat[None, :, :]
            
            # Precompute indices for self blocks
            g_rows1 = (tris1[:, :, None].expand(-1, -1, 3) + slc1_start).flatten()
            g_cols1 = (tris1[:, None, :].expand(-1, 3, -1) + slc1_start).flatten()
            g_rows2 = (tris2[:, :, None].expand(-1, -1, 3) + slc2_start).flatten()
            g_cols2 = (tris2[:, None, :].expand(-1, 3, -1) + slc2_start).flatten()
            
            # Precompute matching for exchange terms
            pm1 = femesh["point_map"][c1]
            pm2 = femesh["point_map"][c2]
            labels1 = pm1[tris1]
            labels2 = pm2[tris2]
            
            equal = labels1[:, :, None] == labels2[:, None, :]
            
            exchange_data = None
            if equal.any():
                face_idx, local_a, local_b = torch.nonzero(equal, as_tuple=True)
                num_matches = face_idx.size(0)
                if num_matches > 0:
                    face_idx_rep = face_idx[:, None].expand(-1, 3).reshape(-1)
                    local_a_rep = local_a[:, None].expand(-1, 3).reshape(-1)
                    local_b_rep = local_b[:, None].expand(-1, 3).reshape(-1)
                    local_row = torch.arange(3, device=device).repeat(num_matches)
                    
                    exchange_data = {
                        "face_idx_rep": face_idx_rep,
                        "local_a_rep": local_a_rep,
                        "local_b_rep": local_b_rep,
                        "local_row": local_row,
                        "g_rows12": tris1[face_idx_rep, local_row] + slc1_start,
                        "g_cols12": tris2[face_idx_rep, local_b_rep] + slc2_start,
                    }
            
            precomputed["boundary_data"].append({
                "type": "exchange",
                "face_start": face_counter,
                "face_end": face_counter + nfaces,
                "local_Q1": local_Q1,
                "local_Q2": local_Q2,
                "g_rows1": g_rows1,
                "g_cols1": g_cols1,
                "g_rows2": g_rows2,
                "g_cols2": g_cols2,
                "exchange_data": exchange_data,
                "N": N
            })
        
        face_counter += nfaces
    
    return precomputed


def couple_flux_matrix_fast(precomputed, faces_prob=None):
    """
    Fast flux matrix assembly using precomputed geometry.
    Only the permeability (faces_prob) changes between calls.
    
    Args:
        precomputed: dict from precompute_flux_geometry()
        faces_prob: 1D tensor of per-face permeabilities
    
    Returns:
        Q: [N x N] flux matrix
    """
    N = precomputed["N"]
    device = precomputed["device"]
    
    Q = torch.zeros((N, N), device=device)
    flat_Q = Q.view(-1)
    
    for bdata in precomputed["boundary_data"]:
        face_start = bdata["face_start"]
        face_end = bdata["face_end"]
        
        # Extract permeability slice
        if faces_prob is not None:
            p_slice = faces_prob[face_start:face_end]
        else:
            p_slice = torch.full((face_end - face_start,), 1e-6, device=device)
        
        p = p_slice.clamp(min=0.0)
        
        if bdata["type"] == "robin":
            # Single-touch assembly
            values = (p[:, None, None] * bdata["local_Q"]).flatten()
            flat_Q.index_add_(0, bdata["g_rows"] * N + bdata["g_cols"], values)
            
        elif bdata["type"] == "exchange":
            # Two-touch self blocks
            k1 = p[:, None, None]
            k2 = p[:, None, None]
            
            values1 = (k1 * bdata["local_Q1"]).flatten()
            flat_Q.index_add_(0, bdata["g_rows1"] * N + bdata["g_cols1"], values1)
            
            values2 = (k2 * bdata["local_Q2"]).flatten()
            flat_Q.index_add_(0, bdata["g_rows2"] * N + bdata["g_cols2"], values2)
            
            # Exchange terms
            if bdata["exchange_data"] is not None:
                ex = bdata["exchange_data"]
                loc_values = bdata["local_Q1"][ex["face_idx_rep"], ex["local_row"], ex["local_a_rep"]]
                p_rep = p[ex["face_idx_rep"]]
                
                # Q12 block
                values12 = -p_rep * loc_values
                flat_Q.index_add_(0, ex["g_rows12"] * N + ex["g_cols12"], values12)
                
                # Q21 block (transpose)
                values21 = -p_rep * loc_values
                flat_Q.index_add_(0, ex["g_cols12"] * N + ex["g_rows12"], values21)
    
    # Symmetrize
    Q = 0.5 * (Q + Q.t())
    return Q


def couple_flux_matrix(femesh, pde, faces_prob=None):
    """
    Original interface for backward compatibility.
    For repeated calls with same geometry, use precompute_flux_geometry() + couple_flux_matrix_fast().
    """
    precomputed = precompute_flux_geometry(femesh)
    return couple_flux_matrix_fast(precomputed, faces_prob)