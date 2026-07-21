import xitorch
import xitorch.linalg
import time
import torch
from src.get_volume_mesh import get_volume_mesh
# from src.mass_matrixP1_3D import mass_matrixP1_3D
# from src.mass_matrixP1_3D_e_v2 import mass_matrixP1_3D
from src.mass_matrixP1_3D_e_v3 import mass_matrixP1_3D
# from src.stiffness_matrixP1_3D import stiffness_matrixP1_3D
# from src.mass_matrixP1_3D_e import mass_matrixP1_3D
from src.stiffness_matrixP1_3D_e import stiffness_matrixP1_3D
from src.sparse_block_diagonal import sparse_block_diagonal
# from src.assembleflux_matrix import assembleflux_matrix
# from src.couple_flux_matrix import couple_flux_matrix
# from src.couple_flux_matrix_v2 import couple_flux_matrix
# from src.couple_flux_matrix_v3 import couple_flux_matrix
from src.couple_flux_matrix_v4 import couple_flux_matrix
from xitorch.linalg import symeig
from xitorch import LinearOperator
import logging

logger = logging.getLogger("mvrecon_3d")

def compute_laplace_eig_diff(femesh, setup, pde, eiglim, neig_max=None, faces_prob=None):
    start_time = time.time()

    diffusivity = pde["diffusivity"].float()
    relaxation = pde["relaxation"].float()
    ncompartment = femesh["ncompartment"]
    initial_density = pde["initial_density"]

    points_list = [femesh["points"][ic].float() for ic in range(ncompartment)]
    elements_list = [femesh["elements"][ic].long() for ic in range(ncompartment)]
    npoint_cmpts = [pts.size(1) for pts in points_list]
    volumes_list = [get_volume_mesh(pts, els)[1].float() for pts, els in zip(points_list, elements_list)]

    M_cmpts = [mass_matrixP1_3D(els, vols).float() for els, vols in zip(elements_list, volumes_list)]
    K_cmpts = [stiffness_matrixP1_3D(els, pts, diffusivity[ic, :, :]).float() for ic, (els, pts) in enumerate(zip(elements_list, points_list))]
    R_cmpts = [(1 / relaxation[ic] * M_cmpt).float() for ic, M_cmpt in enumerate(M_cmpts)]

    Jx_cmpts = [[mass_matrixP1_3D(els, vols, pts[idim, :]).float() for els, vols, pts in zip(elements_list, volumes_list, points_list)] for idim in range(3)]

    rho_cmpts = [initial_density[ic].to(dtype=torch.complex128) * torch.ones((npt, 1), dtype=torch.complex128) for ic, npt in enumerate(npoint_cmpts)]

    M = sparse_block_diagonal(M_cmpts).float()
    # print("M dense:\n", M.to_dense())
    K = sparse_block_diagonal(K_cmpts).float()
    # print("K dense:\n", K.to_dense())
    R = sparse_block_diagonal(R_cmpts).float()

    Jx = [sparse_block_diagonal(dim_cmpts).float() for dim_cmpts in Jx_cmpts]

    # Q_blocks = assembleflux_matrix(femesh["points"], femesh["facets"], faces_prob)
    # flux_matrix = couple_flux_matrix(femesh, pde, faces_prob=faces_prob).float()  # Renamed Q to flux_matrix
    flux_matrix = couple_flux_matrix(femesh, setup.pde, faces_prob=faces_prob).float()

    logger.debug(f"[compute_laplace_eig_diff] flux_matrix.type={type(flux_matrix)}, flux_matrix.shape={flux_matrix.shape}, flux_matrix.grad_fn={flux_matrix.grad_fn}")
    if not isinstance(flux_matrix, torch.Tensor):
        logger.error(f"[compute_laplace_eig_diff] flux_matrix is not a tensor: {type(flux_matrix)}")
        raise TypeError("flux_matrix must be a torch.Tensor")

    rho = torch.vstack(rho_cmpts)

    original_neig_max = neig_max
    neig_max = min(original_neig_max, M.shape[0])
    
    #Debug print
    # breakpoint()
    # print("flux_matrix dense:\n", flux_matrix.to_dense())

    K_dense = K.to_dense()
    flux_matrix_dense = flux_matrix.to_dense()  # Use renamed variable
    A = K_dense + flux_matrix_dense

    B = M.to_dense()

    A = LinearOperator.m(A)
    B = LinearOperator.m(B)

    eigenvalues, eigenvectors = symeig(A, neig=neig_max, M=B, mode="lowest")
    
    # print('eigenvalues:\n', eigenvalues)
    # print('eigenfuncs:\n', eigenvectors)

    values, indices = torch.sort(eigenvalues)
    funcs = eigenvectors[:, indices]

    values = torch.clamp(values, min=0.0)

    keep = torch.zeros_like(values, dtype=torch.bool)
    mask_limit = values <= eiglim
    mask_size = mask_limit.sum().item()

    if mask_size <= original_neig_max:
        keep[mask_limit] = True
    else:
        keep[:original_neig_max] = True

    values = values[keep]
    funcs = funcs[:, keep]
    neig = values.shape[0]

    # if orig_inf and neig == neig_all and not torch.isinf(eiglim):
    #     print("Warning: No eigenvalues were outside [0, eiglim]. Consider increasing neig_max.")

    funcs = funcs / torch.sqrt(torch.sum(funcs * (M.to_dense() @ funcs), dim=0))[None, :]

    Jx_dense = torch.stack([jx.to_dense() for jx in Jx])  # [3, N, N]
    tmp = torch.matmul(Jx_dense, funcs)  # [3, N, neig]
    moments = torch.matmul(funcs.T.unsqueeze(0), tmp).permute(1, 2, 0)  # [1, neig, N] @ [3, N, neig] -> [3, neig, neig] -> [neig, neig, 3]

    massrelax = funcs.T @ R.to_dense() @ funcs

    lap_eig = {
        "values": values,
        "funcs": funcs,
        "moments": moments,
        "massrelax": massrelax,
        "totaltime": time.time() - start_time,
    }

    return lap_eig