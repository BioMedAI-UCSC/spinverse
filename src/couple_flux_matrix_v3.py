import torch
import logging
from src.flux_matrixP1_3D import flux_matrixP1_3D

# ensure logger is defined
logger = logging.getLogger("mvrecon_3d")

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
    # breakpoint()
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
    face_counter = 0

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
            slc = global_slice(c)
            for j in range(nfaces):
                p_j = p_slice[j].clamp(min=0.0)
                tri = facets[c][ibound][:, j].long().unsqueeze(0)
                Qf, _ = flux_matrixP1_3D(tri, pts_c)
                Qf = Qf.to_dense() if Qf.is_sparse else Qf
                Q[slc, slc] += p_j * Qf

        # two-touch: exchange coupling
        elif len(touches) == 2:
            # breakpoint()
            c1, c2 = touches
            pts1 = femesh["points"][c1].t().contiguous()
            pts2 = femesh["points"][c2].t().contiguous()
            slc1 = global_slice(c1)
            slc2 = global_slice(c2)

            # point_map for matching
            pm1 = femesh["point_map"][c1]
            pm2 = femesh["point_map"][c2]

            # symmetrical coupling constants
            c12 = c21 = 1.0

            # loop over faces
            for j in range(nfaces):
                p_j = p_slice[j].clamp(min=0.0)

                # local triangle vertex indices (within each compartment)
                tri1 = facets[c1][ibound][:, j].long()
                tri2 = facets[c2][ibound][:, j].long()

                # compute local flux on each side
                Q11, _ = flux_matrixP1_3D(tri1.unsqueeze(0), pts1)
                Q22, _ = flux_matrixP1_3D(tri2.unsqueeze(0), pts2)
                Q11 = Q11.to_dense() if Q11.is_sparse else Q11
                Q22 = Q22.to_dense() if Q22.is_sparse else Q22

                # find matching facet DOFs via point_map labels
                inds1 = torch.unique(tri1)
                inds2 = torch.unique(tri2)
                labels1 = pm1[inds1]
                labels2 = pm2[inds2]
                I, J = torch.nonzero(labels1[:, None] == labels2[None, :], as_tuple=True)
                if I.numel() == 0:
                    continue  # no matching DOFs for this face
                inds1_act = inds1[I]
                inds2_act = inds2[J]

                # build cross-term blocks
                Q12 = torch.zeros((npoint_cmpts[c1], npoint_cmpts[c2]), device=device)
                Q12[:, inds2_act] = Q11[:, inds1_act]
                Q21 = Q12.t()

                # permeability‐weighted coefficients
                k1 = c21 * p_j
                k2 = c12 * p_j

                # assemble global blocks
                Q[slc1, slc1] +=  k1 * Q11
                Q[slc1, slc2] += -k2 * Q12
                Q[slc2, slc1] += -k1 * Q21
                Q[slc2, slc2] +=  k2 * Q22

        else:
            logging.warning(
                f"[couple_flux_matrix] boundary {ibound} "
                f"touches {len(touches)} compartments, skipping."
            )

    # final symmetrize (optional if blocks written symmetrically)
    Q = 0.5 * (Q + Q.t())
    return Q
