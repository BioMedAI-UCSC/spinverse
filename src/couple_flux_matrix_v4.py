import torch
import logging

# ensure logger is defined
logger = logging.getLogger("mvrecon_3d")

def compute_areas(tris, pts):
    verts = pts[tris]  # [nfaces, 3, 3]
    vec1 = verts[:, 1, :] - verts[:, 0, :]
    vec2 = verts[:, 2, :] - verts[:, 0, :]
    cross = torch.cross(vec1, vec2, dim=1)
    areas = 0.5 * torch.norm(cross, p=2, dim=1)
    eps = 1e-12
    neumann_areas = areas.clamp(min=eps)
    
    return neumann_areas

def couple_flux_matrix(femesh, pde, faces_prob=None):
    """
    Assemble the global flux matrix Q via weak-form Robin/exchange coupling.

    Args:
      femesh: dict with
        - "points":  list of [3 x n_pts_c] tensors, one per compartment
        - "facets":  list of lists, facets[c][b] is [3 x nfaces] or None
        - "point_map": list of 1D LongTensors mapping each compartment's local DOF to a global label
      pde:
        - pde.initial_density: list of length ncomp for density-weighting
      faces_prob: 1D tensor of length = total #faces across all (c,b)
                   giving per-face permeability in [0,1].

    Returns:
      Q: [N x N] symmetric torch.Tensor, N = total dofs.
    """
    facets    = femesh["facets"]
    ncomp     = len(facets)
    nboundary = len(facets[0])

    # how many points in each compartment?
    npoint_cmpts = [pts.shape[1] for pts in femesh["points"]]
    N = sum(npoint_cmpts)
    device = femesh["points"][0].device

    # build cumulative indices so that compartment c maps to slice(c_inds[c]:c_inds[c+1])
    c_inds = torch.cumsum(torch.tensor([0] + npoint_cmpts, device=device), dim=0)
    def global_slice(c):
        return slice(c_inds[c].item(), c_inds[c+1].item())

    Q = torch.zeros((N, N), device=device)
    flat_Q = Q.view(-1)
    face_counter = 0

    local_mat = torch.tensor([[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]], device=device)

    for ibound in range(nboundary):
        touches = [c for c in range(ncomp) if facets[c][ibound] is not None]
        if not touches:
            continue

        # count faces on this boundary
        nfaces = facets[touches[0]][ibound].shape[1]

        # extract per-face permeabilities
        if faces_prob is not None:
            p_slice = faces_prob[face_counter:face_counter + nfaces]
        else:
            p_slice = torch.full((nfaces,), 1e-6, device=device)
        face_counter += nfaces

        # single-touch: Robin (outer boundary)
        if len(touches) == 1:
            c = touches[0]
            pts_c = femesh["points"][c].t().contiguous()
            tris = facets[c][ibound].t().long()
            slc_start = c_inds[c].item()

            areas = compute_areas(tris, pts_c)
            local_Q = (areas[:, None, None] / 12.) * local_mat[None, :, :]

            p = p_slice.clamp(min=0.0)[:, None, None]
            values = p * local_Q
            g_rows = (tris[:, :, None].expand(-1, -1, 3) + slc_start).flatten()
            g_cols = (tris[:, None, :].expand(-1, 3, -1) + slc_start).flatten()
            flat_Q.index_add_(0, g_rows * N + g_cols, values.flatten())

        # two-touch: exchange coupling
        elif len(touches) == 2:
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

            # symmetrical coupling constants
            c12 = c21 = 1.0
            p = p_slice.clamp(min=0.0)
            k1 = c21 * p[:, None, None]
            k2 = c12 * p[:, None, None]

            # assemble self blocks
            values1 = k1 * local_Q1
            g_rows1 = (tris1[:, :, None].expand(-1, -1, 3) + slc1_start).flatten()
            g_cols1 = (tris1[:, None, :].expand(-1, 3, -1) + slc1_start).flatten()
            flat_Q.index_add_(0, g_rows1 * N + g_cols1, values1.flatten())

            values2 = k2 * local_Q2
            g_rows2 = (tris2[:, :, None].expand(-1, -1, 3) + slc2_start).flatten()
            g_cols2 = (tris2[:, None, :].expand(-1, 3, -1) + slc2_start).flatten()
            flat_Q.index_add_(0, g_rows2 * N + g_cols2, values2.flatten())

            # point_map for matching
            pm1 = femesh["point_map"][c1]
            pm2 = femesh["point_map"][c2]
            labels1 = pm1[tris1]
            labels2 = pm2[tris2]

            equal = labels1[:, :, None] == labels2[:, None, :]
            if equal.any():
                face_idx, local_a, local_b = torch.nonzero(equal, as_tuple=True)
                num_matches = face_idx.size(0)
                if num_matches > 0:
                    # expand for local rows
                    face_idx_rep = face_idx[:, None].expand(-1, 3).reshape(-1)
                    local_a_rep = local_a[:, None].expand(-1, 3).reshape(-1)
                    local_b_rep = local_b[:, None].expand(-1, 3).reshape(-1)
                    local_row = torch.arange(3, device=device).repeat(num_matches)
                    loc_values = local_Q1[face_idx_rep, local_row, local_a_rep]
                    p_rep = p[face_idx_rep]

                    # Q12 block
                    values12 = -c12 * p_rep * loc_values
                    g_rows12 = tris1[face_idx_rep, local_row] + slc1_start
                    g_cols12 = tris2[face_idx_rep, local_b_rep] + slc2_start
                    flat_Q.index_add_(0, g_rows12 * N + g_cols12, values12)

                    # Q21 block (transpose indices, adjusted scale)
                    values21 = -c21 * p_rep * loc_values
                    g_rows21 = g_cols12
                    g_cols21 = g_rows12
                    flat_Q.index_add_(0, g_rows21 * N + g_cols21, values21)

        else:
            logging.warning(
                f"[couple_flux_matrix] boundary {ibound} "
                f"touches {len(touches)} compartments, skipping."
            )

    # final symmetrize (optional if blocks written symmetrically)
    Q = 0.5 * (Q + Q.t())
    return Q