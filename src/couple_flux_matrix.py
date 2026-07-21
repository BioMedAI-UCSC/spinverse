import torch
import logging

logger = logging.getLogger("mvrecon_3d")

def couple_flux_matrix(femesh, pde, faces_prob=None):
    point_map    = femesh["point_map"]
    facets       = femesh["facets"]
    nboundary    = len(facets[0])
    npoint_cmpts = [pts.shape[1] for pts in femesh["points"]]
    npoint       = sum(npoint_cmpts)

    # build cumulative indices for slicing into the big Q
    cmpt_inds = torch.cumsum(torch.tensor([0] + npoint_cmpts,
                                          device=femesh["points"][0].device), 0)
    def get_inds(ic):
        return slice(cmpt_inds[ic].item(), cmpt_inds[ic+1].item())

    Q = torch.zeros((npoint, npoint), device=femesh["points"][0].device)
    face_counter = 0

    for ibound in range(nboundary):
        # which compartments touch this boundary index?
        touches = [i for i in range(len(facets))
                   if facets[i][ibound] is not None]
        if not touches:
            continue

        for cmpt in touches:
            boundary = facets[cmpt][ibound]    # shape [3, nfaces]
            nfaces   = boundary.shape[1]
            inds     = get_inds(cmpt)

            # compute raw flux block
            from src.flux_matrixP1_3D import flux_matrixP1_3D
            Q_block, _ = flux_matrixP1_3D(
                boundary.t().contiguous(),
                femesh["points"][cmpt].t().contiguous()
            )
            if Q_block.is_sparse:
                Q_block = Q_block.to_dense()

            logger.debug(f"[couple_flux_matrix] ibound={ibound} "
                         f"cmpt={cmpt} nfaces={nfaces} "
                         f"Q_block.shape={tuple(Q_block.shape)}")

            if faces_prob is not None:
                # grab the next slice of face-probabilities
                probs = faces_prob[face_counter:face_counter + nfaces]
                face_counter = face_counter + nfaces

                logger.debug(f"[couple_flux_matrix] faces_prob slice "
                             f"min={probs.min():.4e}, max={probs.max():.4e}")

                # flatten each triangle's 3 verts into a single vector
                vs = boundary.reshape(-1).long()            # [3 * nfaces]
                ws = probs.repeat_interleave(3)             # [3 * nfaces]

                # scatter_add into per-vertex totals & counts (non-inplace)
                n_verts = Q_block.shape[0]
                vertex_w_sum = torch.zeros(n_verts, device=ws.device, dtype=ws.dtype)
                counts = torch.zeros(n_verts, device=ws.device, dtype=ws.dtype)
                
                # Use scatter_add (non-inplace version)
                vertex_w_sum = torch.scatter_add(vertex_w_sum, 0, vs, ws)
                counts = torch.scatter_add(counts, 0, vs, torch.ones_like(ws))

                # mean weight per vertex with small epsilon to avoid div0
                epsilon = 1e-8
                vertex_w = vertex_w_sum / (counts + epsilon)

                # Apply weights without creating diagonal matrix
                # This is equivalent to W @ Q_block @ W but more efficient
                Q_block_weighted = vertex_w.unsqueeze(1) * Q_block * vertex_w.unsqueeze(0)
                Q_block = Q_block_weighted
            else:
                Q_block_scaled = 1e-6 * Q_block
                Q_block = Q_block_scaled

            # accumulate into the global Q (create new tensor)
            # Q_new = Q.clone()
            # Q_new[inds, inds] = Q[inds, inds] + Q_block
            # Q = Q_new

            # instead of Q_new = Q.clone() + Q_block in a block
            mask = torch.zeros_like(Q)
            mask[inds, inds] = 1.0
            Q = Q + mask * Q_block_weighted

    # final symmetrization (create new tensor)
    Q_symmetric = 0.5 * (Q + Q.t())
    logger.debug(f"[couple_flux_matrix] final Q.shape={tuple(Q_symmetric.shape)}")
    return Q_symmetric
