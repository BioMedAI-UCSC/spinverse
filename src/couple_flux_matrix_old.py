import torch
import logging

logger = logging.getLogger("mvrecon_3d")

# def couple_flux_matrix(femesh, pde, faces_prob=None):
#     point_map = femesh["point_map"]
#     facets = femesh["facets"]
#     nboundary = femesh["nboundary"]
#     npoint_cmpts = [points.shape[1] for points in femesh["points"]]
#     npoint = sum(npoint_cmpts)
#     cmpt_inds = torch.cumsum(torch.tensor([0] + npoint_cmpts, device=femesh["points"][0].device), 0)

#     def get_inds(icmpt):
#         return slice(cmpt_inds[icmpt].item(), cmpt_inds[icmpt + 1].item())

#     Q = torch.zeros((npoint, npoint), device=femesh["points"][0].device, dtype=torch.float32)
#     face_counter = 0
#     default_perm = 1e-6

#     for ibound in range(nboundary):
#         cmpts_touch = [i for i, facet in enumerate(facets) if facet[ibound] is not None]
#         ntouch = len(cmpts_touch)
#         nfaces = facets[cmpts_touch[0]][ibound].shape[1] if ntouch > 0 else 0

#         logger.debug(f"[couple_flux_matrix] ibound={ibound}, nfaces={nfaces}, ntouch={ntouch}, face_counter={face_counter}")

#         if ntouch == 1:
#             cmpt = cmpts_touch[0]
#             npoints = npoint_cmpts[cmpt]
#             inds = get_inds(cmpt)
#             from src.flux_matrixP1_3D import flux_matrixP1_3D
#             Q_block = flux_matrixP1_3D(facets[cmpt][ibound].t().contiguous(), femesh["points"][cmpt].t().contiguous())[0]
#             if Q_block.is_sparse:
#                 Q_block = Q_block.to_dense()

#             if faces_prob is not None:
#                 if face_counter + nfaces > len(faces_prob):
#                     logger.error(f"[couple_flux_matrix] faces_prob too short: face_counter={face_counter}, nfaces={nfaces}, faces_prob.shape={faces_prob.shape}")
#                     raise ValueError("faces_prob does not have enough entries")

#                 logger.debug(f"[couple_flux_matrix] raw faces_prob min={faces_prob[face_counter:face_counter + nfaces].min().item()}, max={faces_prob[face_counter:face_counter + nfaces].max().item()}")
#                 face_weights = faces_prob[face_counter:face_counter + nfaces] * default_perm
#                 face_counter += nfaces

#                 threshold = 1e-8
#                 face_weights = torch.where(face_weights < threshold, torch.zeros_like(face_weights), face_weights)
#                 logger.debug(f"[couple_flux_matrix] face_weights min={face_weights.min().item()}, max={face_weights.max().item()}")

#                 vertex_weights = torch.zeros(npoints, device=Q.device, dtype=faces_prob.dtype)
#                 vertex_counts = torch.zeros(npoints, device=Q.device, dtype=torch.long)
#                 for face_idx in range(nfaces):
#                     vertices = facets[cmpt][ibound][:, face_idx].long()
#                     vertex_weights[vertices] += face_weights[face_idx]
#                     vertex_counts[vertices] += 1

#                 vertex_weights = vertex_weights / torch.clamp(vertex_counts, min=1)
#                 vertex_weights = torch.clamp(vertex_weights, min=0)
#                 logger.debug(f"[couple_flux_matrix] vertex_weights min={vertex_weights.min().item()}, max={vertex_weights.max().item()}")

#                 Q_block = vertex_weights.unsqueeze(1) * Q_block
#             else:
#                 Q_block = default_perm * Q_block

#             Q[inds, inds] += Q_block

#         elif ntouch == 2:
#             cmpt1, cmpt2 = cmpts_touch
#             npoints1 = npoint_cmpts[cmpt1]
#             npoints2 = npoint_cmpts[cmpt2]
#             from src.flux_matrixP1_3D import flux_matrixP1_3D
#             Q11 = flux_matrixP1_3D(facets[cmpt1][ibound].t().contiguous(), femesh["points"][cmpt1].t().contiguous())[0]
#             Q22 = flux_matrixP1_3D(facets[cmpt2][ibound].t().contiguous(), femesh["points"][cmpt2].t().contiguous())[0]
#             if Q11.is_sparse:
#                 Q11 = Q11.to_dense()
#             if Q22.is_sparse:
#                 Q22 = Q22.to_dense()

#             Q12 = torch.zeros((npoints1, npoints2), device=Q.device)
#             inds1 = torch.unique(facets[cmpt1][ibound])
#             inds2 = torch.unique(facets[cmpt2][ibound])

#             if torch.all(point_map[cmpt1][inds1] == point_map[cmpt2][inds2]):
#                 indinds1 = torch.arange(len(inds1))
#                 indinds2 = torch.arange(len(inds2))
#             else:
#                 indinds1, indinds2 = torch.where(
#                     point_map[cmpt1][inds1, None] == point_map[cmpt2][None, inds2]
#                 )

#             Q12[:, inds2[indinds2].long()] = Q11[:, inds1[indinds1].long()]
#             Q21 = Q12.transpose(0, 1)

#             if pde.get("symmetrical", True):
#                 c12 = c21 = 1
#             else:
#                 rho1 = pde["initial_density"][cmpt1]
#                 rho2 = pde["initial_density"][cmpt2]
#                 c21 = 2 * rho2 / (rho1 + rho2)
#                 c12 = 2 * rho1 / (rho1 + rho2)

#             if faces_prob is not None:
#                 if face_counter + nfaces > len(faces_prob):
#                     logger.error(f"[couple_flux_matrix] faces_prob too short: face_counter={face_counter}, nfaces={nfaces}, faces_prob.shape={faces_prob.shape}")
#                     raise ValueError("faces_prob does not have enough entries")

#                 logger.debug(f"[couple_flux_matrix] raw faces_prob min={faces_prob[face_counter:face_counter + nfaces].min().item()}, max={faces_prob[face_counter:face_counter + nfaces].max().item()}")
#                 face_weights = faces_prob[face_counter:face_counter + nfaces] * default_perm
#                 face_counter += nfaces

#                 threshold = 1e-8
#                 face_weights = torch.where(face_weights < threshold, torch.zeros_like(face_weights), face_weights)
#                 logger.debug(f"[couple_flux_matrix] face_weights min={face_weights.min().item()}, max={face_weights.max().item()}")

#                 vertex_weights1 = torch.zeros(npoints1, device=Q.device, dtype=faces_prob.dtype)
#                 vertex_counts1 = torch.zeros(npoints1, device=Q.device, dtype=torch.long)
#                 for face_idx in range(nfaces):
#                     vertices = facets[cmpt1][ibound][:, face_idx].long()
#                     vertex_weights1[vertices] += face_weights[face_idx]
#                     vertex_counts1[vertices] += 1
#                 vertex_weights1 = vertex_weights1 / torch.clamp(vertex_counts1, min=1)
#                 vertex_weights1 = torch.clamp(vertex_weights1, min=0)

#                 vertex_weights2 = torch.zeros(npoints2, device=Q.device, dtype=faces_prob.dtype)
#                 vertex_counts2 = torch.zeros(npoints2, device=Q.device, dtype=torch.long)
#                 for face_idx in range(nfaces):
#                     vertices = facets[cmpt2][ibound][:, face_idx].long()
#                     vertex_weights2[vertices] += face_weights[face_idx]
#                     vertex_counts2[vertices] += 1
#                 vertex_weights2 = vertex_weights2 / torch.clamp(vertex_counts2, min=1)
#                 vertex_weights2 = torch.clamp(vertex_weights2, min=0)

#                 logger.debug(f"[couple_flux_matrix] vertex_weights1 min={vertex_weights1.min().item()}, max={vertex_weights1.max().item()}")
#                 logger.debug(f"[couple_flux_matrix] vertex_weights2 min={vertex_weights2.min().item()}, max={vertex_weights2.max().item()}")

#                 k1 = c21 * vertex_weights1
#                 k2 = c12 * vertex_weights2
#             else:
#                 k1 = c21 * default_perm
#                 k2 = c12 * default_perm

#             inds1 = get_inds(cmpt1)
#             inds2 = get_inds(cmpt2)
#             Q[inds1, inds1] += k1.unsqueeze(1) * Q11
#             Q[inds1, inds2] -= k2.unsqueeze(1) * Q12
#             Q[inds2, inds1] -= k1.unsqueeze(1) * Q21
#             Q[inds2, inds2] += k2.unsqueeze(1) * Q22

#         elif ntouch > 2:
#             raise ValueError("Each interface touches only 1 or 2 compartments")

#     Q_sym = (Q + Q.transpose(0, 1)) / 2
#     Q_sparse = Q_sym.to_sparse()
#     logger.debug(f"[couple_flux_matrix] Q_sparse.shape={Q_sparse.shape}, type={type(Q_sparse)}, grad_fn={Q_sparse.grad_fn}")
#     return Q_sparse

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
                face_counter += nfaces

                logger.debug(f"[couple_flux_matrix] faces_prob slice "
                             f"min={probs.min():.4e}, max={probs.max():.4e}")

                # flatten each triangle's 3 verts into a single vector
                vs = boundary.reshape(-1).long()            # [3 * nfaces]
                ws = probs.repeat_interleave(3)             # [3 * nfaces]

                # scatter_add into per-vertex totals & counts
                n_verts = Q_block.shape[0]
                vertex_w = torch.zeros(n_verts, device=ws.device, dtype=ws.dtype)
                counts   = torch.zeros_like(vertex_w)
                vertex_w.scatter_add_(0, vs, ws)
                counts.scatter_add_(0, vs, torch.ones_like(ws))

                # mean weight per vertex (avoid div0)
                vertex_w = vertex_w / counts.clamp(min=1.0)

                # build diagonal weighting matrix and apply
                W = torch.diag(vertex_w)                    # [n_verts, n_verts]
                Q_block = W @ Q_block @ W

            # accumulate into the global Q
            Q[inds, inds] += Q_block

    # final symmetrization
    Q = 0.5 * (Q + Q.t())
    logger.debug(f"[couple_flux_matrix] final Q.shape={tuple(Q.shape)}")
    return Q
