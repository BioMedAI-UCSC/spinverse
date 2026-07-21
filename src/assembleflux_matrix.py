# from src.flux_matrixP1_3D import flux_matrixP1_3D


# def assembleflux_matrix(points, facets, faces_prob=None):
#     ncompartment = len(facets)
#     nboundary = len(facets[0]) if ncompartment > 0 else 0

#     flux_matrices = []

#     for icmpt in range(ncompartment):
#         compartment_matrices = []
#         for iboundary in range(nboundary):
#             boundary = facets[icmpt][iboundary]

#             if boundary is not None and boundary.nelement() > 0:
#                 # Assuming boundary and points are already tensors and transposed as needed
#                 matrix, _ = flux_matrixP1_3D(
#                     boundary.t().contiguous(), points[icmpt].t().contiguous()
#                 )
#                 compartment_matrices.append(matrix)

#             else:
#                 compartment_matrices.append(None)  # Or an appropriate placeholder
#         flux_matrices.append(compartment_matrices)

#     # Returning as a list of lists
#     return flux_matrices

# from src.flux_matrixP1_3D import flux_matrixP1_3D

# def assembleflux_matrix(points, facets, faces_prob=None):
#     ncompartment = len(facets)
#     nboundary = len(facets[0]) if ncompartment > 0 else 0

#     flux_matrices = []
#     face_counter = 0  # Global counter to track face ordering

#     for icmpt in range(ncompartment):
#         compartment_matrices = []
#         for iboundary in range(nboundary):
#             boundary = facets[icmpt][iboundary]

#             if boundary is not None and boundary.nelement() > 0:
#                 matrix, _ = flux_matrixP1_3D(
#                     boundary.t().contiguous(), points[icmpt].t().contiguous()
#                 )
#                 if faces_prob is not None:
#                     nfaces = boundary.shape[1]
#                     weight = faces_prob[face_counter:face_counter + nfaces]
#                     matrix = weight.unsqueeze(1) * matrix  # Scale each face's flux contribution
#                     face_counter += nfaces
#                 else:
#                     nfaces = boundary.shape[1]
#                     face_counter += nfaces

#                 compartment_matrices.append(matrix)
#             else:
#                 compartment_matrices.append(None)
#         flux_matrices.append(compartment_matrices)

#     return flux_matrices

# from src.flux_matrixP1_3D import flux_matrixP1_3D

# def assembleflux_matrix(points, facets, faces_prob=None):
#     ncompartment = len(facets)
#     nboundary = len(facets[0]) if ncompartment > 0 else 0

#     flux_matrices = []
#     face_counter = 0  # Global counter to track face ordering

#     for icmpt in range(ncompartment):
#         compartment_matrices = []
#         for iboundary in range(nboundary):
#             boundary = facets[icmpt][iboundary]

#             if boundary is not None and boundary.nelement() > 0:
#                 # Compute the flux matrix for the current boundary.
#                 matrix, _ = flux_matrixP1_3D(
#                     boundary.t().contiguous(), points[icmpt].t().contiguous()
#                 )

#                 if faces_prob is not None:
#                     # Instead of using boundary.shape[1],
#                     # use the number of rows in matrix as the number of flux contributions.
#                     nflux = matrix.shape[0]
#                     weight = faces_prob[face_counter:face_counter + nflux]
#                     # Multiply each flux contribution (row in matrix) by its corresponding weight.
#                     matrix = weight.unsqueeze(1) * matrix
#                     # Increment the counter by the number of flux rows.
#                     face_counter += nflux
#                 else:
#                     # If no faces_prob provided, update counter using the original number of faces.
#                     nfaces = boundary.shape[1]
#                     face_counter += nfaces

#                 compartment_matrices.append(matrix)
#             else:
#                 compartment_matrices.append(None)
#         flux_matrices.append(compartment_matrices)

#     return flux_matrices

# from src.flux_matrixP1_3D import flux_matrixP1_3D
# import torch

# def assembleflux_matrix(points, facets, faces_prob=None):
#     ncompartment = len(facets)
#     nboundary = len(facets[0]) if ncompartment > 0 else 0

#     flux_matrices = []
#     face_counter = 0  # Global counter to track face ordering

#     for icmpt in range(ncompartment):
#         compartment_matrices = []
#         for iboundary in range(nboundary):
#             boundary = facets[icmpt][iboundary]

#             if boundary is not None and boundary.nelement() > 0:
#                 matrix, _ = flux_matrixP1_3D(
#                     boundary.t().contiguous(), points[icmpt].t().contiguous()
#                 )
#                 # Convert the sparse matrix to dense if needed
#                 if matrix.is_sparse:
#                     matrix = matrix.to_dense()

#                 # Track the number of faces for this boundary
#                 nfaces = boundary.shape[1]

#                 if faces_prob is not None:
#                     # Use faces_prob to weight the matrix
#                     weight = faces_prob[face_counter:face_counter + nfaces]
#                     w_matrix = weight.unsqueeze(1) * matrix  # Scale each row by its weight
#                     face_counter += nfaces
#                 else:
#                     w_matrix = matrix
#                     face_counter += nfaces

#                 compartment_matrices.append(w_matrix)
#             else:
#                 compartment_matrices.append(None)
#                 # No faces to count
#         flux_matrices.append(compartment_matrices)

#     return flux_matrices

from src.flux_matrixP1_3D import flux_matrixP1_3D
import torch
import logging

logger = logging.getLogger("mvrecon_3d")

def assembleflux_matrix(points, facets, faces_prob=None):
    ncompartment = len(facets)
    nboundary = len(facets[0]) if ncompartment > 0 else 0

    flux_matrices = []
    face_counter = 0

    for icmpt in range(ncompartment):
        compartment_matrices = []
        npoints = points[icmpt].shape[1]
        for ibound in range(nboundary):
            boundary = facets[icmpt][ibound]

            if boundary is not None and boundary.nelement() > 0:
                nfaces = boundary.shape[1]
                matrix, _ = flux_matrixP1_3D(
                    boundary.t().contiguous(), points[icmpt].t().contiguous()
                )
                if matrix.is_sparse:
                    matrix = matrix.to_dense()

                logger.debug(f"[assembleflux_matrix] icmpt={icmpt}, ibound={ibound}, nfaces={nfaces}, npoints={npoints}, matrix.shape={matrix.shape}, face_counter={face_counter}")

                if faces_prob is not None:
                    if face_counter + nfaces > len(faces_prob):
                        logger.error(f"[assembleflux_matrix] faces_prob too short: face_counter={face_counter}, nfaces={nfaces}, faces_prob.shape={faces_prob.shape}")
                        raise ValueError("faces_prob does not have enough entries")

                    face_weights = faces_prob[face_counter:face_counter + nfaces]
                    face_counter += nfaces

                    vertex_weights = torch.zeros(npoints, device=points[icmpt].device, dtype=faces_prob.dtype)
                    vertex_counts = torch.zeros(npoints, device=points[icmpt].device, dtype=torch.long)
                    for face_idx in range(nfaces):
                        vertices = boundary[:, face_idx].long()
                        vertex_weights[vertices] += face_weights[face_idx]
                        vertex_counts[vertices] += 1

                    vertex_weights = vertex_weights / torch.clamp(vertex_counts, min=1)
                    logger.debug(f"[assembleflux_matrix] vertex_weights.shape={vertex_weights.shape}, requires_grad={vertex_weights.requires_grad}")

                    w_matrix = vertex_weights.unsqueeze(1) * matrix
                else:
                    w_matrix = matrix
                    face_counter += nfaces

                compartment_matrices.append(w_matrix)
            else:
                compartment_matrices.append(None)
                logger.debug(f"[assembleflux_matrix] icmpt={icmpt}, ibound={ibound}, no faces")
        flux_matrices.append(compartment_matrices)

    logger.info(f"[assembleflux_matrix] Total faces processed: {face_counter}")
    return flux_matrices