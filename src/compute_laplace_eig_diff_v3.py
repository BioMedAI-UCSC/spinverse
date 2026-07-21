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
from src.assembleflux_matrix import assembleflux_matrix
# from src.couple_flux_matrix import couple_flux_matrix
# from src.couple_flux_matrix_v2 import couple_flux_matrix
from src.couple_flux_matrix_v3 import couple_flux_matrix
# from src.couple_flux_matrix_v4 import couple_flux_matrix
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

    M_cmpts, K_cmpts, R_cmpts = [], [], []
    Jx_cmpts = [[] for _ in range(3)]
    rho_cmpts = []

    for icmpt in range(ncompartment):
        points = femesh["points"][icmpt].float()
        elements = femesh["elements"][icmpt].long()
        npoint_cmpt = points.size(1)

        _, volumes, _ = get_volume_mesh(points, elements)
        volumes = volumes.float()

        M_cmpt = mass_matrixP1_3D(elements, volumes).float()
        K_cmpt = stiffness_matrixP1_3D(elements, points, diffusivity[icmpt, :, :]).float()
        R_cmpt = (1 / relaxation[icmpt] * M_cmpt).float()

        for idim in range(3):
            Jx_cmpts[idim].append(mass_matrixP1_3D(elements, volumes, points[idim, :]).float())

        initial_density_icmpt = initial_density[icmpt].to(dtype=torch.complex128)
        rho_cmpt = initial_density_icmpt * torch.ones((npoint_cmpt, 1), dtype=torch.complex128)

        rho_cmpts.append(rho_cmpt)
        M_cmpts.append(M_cmpt)
        K_cmpts.append(K_cmpt)
        R_cmpts.append(R_cmpt)

    M = sparse_block_diagonal(M_cmpts).float()
    print("M dense:\n", M.to_dense())
    K = sparse_block_diagonal(K_cmpts).float()
    print("K dense:\n", K.to_dense())
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
    breakpoint()
    print("flux_matrix dense:\n", flux_matrix.to_dense())

    K_dense = K.to_dense()
    flux_matrix_dense = flux_matrix.to_dense()  # Use renamed variable
    A = K_dense + flux_matrix_dense

    B = M.to_dense()

    A = LinearOperator.m(A)
    B = LinearOperator.m(B)

    eigenvalues, eigenvectors = symeig(A, neig=neig_max, M=B, mode="lowest")
    
    print('eigenvalues:\n', eigenvalues)
    print('eigenfuncs:\n', eigenvectors)

    values, indices = torch.sort(eigenvalues)
    funcs = eigenvectors[:, indices]

    values = torch.clamp(values, min=0.0)

    orig_inf = torch.isinf(torch.tensor(original_neig_max, device=values.device))
    mask_all = torch.ones_like(values, dtype=torch.bool)
    mask_limit = values <= eiglim
    keep = torch.where(orig_inf, mask_limit, mask_all)

    neig_all = values.numel()
    values = values[keep]
    funcs = funcs[:, keep]
    neig = values.shape[0]

    if orig_inf and neig == neig_all and not torch.isinf(eiglim):
        print("Warning: No eigenvalues were outside [0, eiglim]. Consider increasing neig_max.")

    funcs = funcs / torch.sqrt(torch.sum(funcs * (M.to_dense() @ funcs), dim=0))[None, :]

    moments = torch.zeros(neig, neig, 3, dtype=torch.float, device=funcs.device)
    for idim in range(3):
        moments[:, :, idim] = funcs.T @ Jx[idim].to_dense() @ funcs

    massrelax = funcs.T @ R.to_dense() @ funcs

    lap_eig = {
        "values": values,
        "funcs": funcs,
        "moments": moments,
        "massrelax": massrelax,
        "totaltime": time.time() - start_time,
    }

    return lap_eig