import torch
import logging
from src.flux_matrixP1_3D import flux_matrixP1_3D

# logger = logging.getLogger("mvrecon_3d")

def couple_flux_matrix(femesh, pde, faces_prob=None):
    """
    Assemble the global flux matrix Q via weak-form Robin/exchange coupling.

    Args:
      femesh: dict with
        - "points":  list of [3 x n_pts_c] tensors, one per compartment
        - "facets":  list of lists, facets[c][b] is [3 x nfaces] or None
        - "point_map": only used if you want nontrivial matching (unused here)
      pde:
        - pde.initial_density: list of length ncomp for density-weighting
      faces_prob: 1D tensor of length = total #faces across all (c,b)
                   giving per-face permeability in [0,1].

    Returns:
      Q: [N x N] dense symmetric torch.Tensor, N = total dofs.
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
    face_counter = 0

    for ibound in range(nboundary):
        # 1) figure out which compartments touch boundary index ibound
        touches = [c for c in range(ncomp) if facets[c][ibound] is not None]
        # print(f"[couple_flux_matrix] boundary {ibound}: touches={touches}")
        if not touches:
            continue

        print(f"touches: {touches}")
        print(f"number of touches: {len(touches)}")

        # how many triangles (faces) on this boundary?
        nfaces = facets[touches[0]][ibound].shape[1]

        # grab the slice of face-permeabilities (or default tiny eps)
        if faces_prob is not None:
            p_slice = faces_prob[face_counter:face_counter + nfaces]
        else:
            p_slice = torch.full((nfaces,), 1e-6, device=device)
        face_counter += nfaces

        # 2) single-touch → Robin relaxation
        if len(touches) == 1:
            print("len(touches) == 1")
            c = touches[0]
            pts_c = femesh["points"][c].t().contiguous()     # [n_pts_c x 3]
            slc = global_slice(c)
            for j in range(nfaces):
                p_j = p_slice[j].clamp(0.0, 1.0)
                tri = facets[c][ibound][:, j].long().unsqueeze(0)  # [1 x 3]
                Qf, _ = flux_matrixP1_3D(tri.contiguous(), pts_c)
                if Qf.is_sparse: Qf = Qf.to_dense()
                Q[slc, slc] += p_j * Qf

        # 3) two-touch → exchange coupling
        elif len(touches) == 2:
            print("len(touches) == 2")
            c1, c2 = touches
            pts1 = femesh["points"][c1].t().contiguous()
            pts2 = femesh["points"][c2].t().contiguous()
            slc1 = global_slice(c1)
            slc2 = global_slice(c2)

            # density-weighting (if non-symmetrical); here we do symmetrical:
            # rho1 = pde.initial_density[c1]
            # rho2 = pde.initial_density[c2]
            # correct dictionary ordering
            rho1 = pde["initial_density"][c1]
            rho2 = pde["initial_density"][c2]   
            # symmetrical coupling
            c12 = c21 = 1.0

            for j in range(nfaces):
                p_j = p_slice[j].clamp(0.0, 1.0)
                # local flux on each side
                tri1 = facets[c1][ibound][:, j].long().unsqueeze(0)
                tri2 = facets[c2][ibound][:, j].long().unsqueeze(0)
                Q11, _ = flux_matrixP1_3D(tri1.contiguous(), pts1)
                Q22, _ = flux_matrixP1_3D(tri2.contiguous(), pts2)
                if Q11.is_sparse: Q11 = Q11.to_dense()
                if Q22.is_sparse: Q22 = Q22.to_dense()

                # approximate cross-term by copying local ordering
                Q12 = Q11.clone()
                Q21 = Q22.clone()

                k1 = c21 * p_j
                k2 = c12 * p_j

                # assemble the four blocks
                Q[slc1, slc1] +=  k1 * Q11
                Q[slc1, slc2] += -k2 * Q12
                Q[slc2, slc1] += -k1 * Q21
                Q[slc2, slc2] +=  k2 * Q22

        else:
            logger.warning(
                f"[couple_flux_matrix] boundary {ibound} "
                f"touches {len(touches)} compartments, skipping."
            )

    # symmetrize and return
    Q = 0.5 * (Q + Q.t())
    print(f"[couple_flux_matrix] final Q.shape = {Q.shape}")
    return Q
