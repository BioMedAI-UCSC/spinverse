import xitorch
import xitorch.linalg
import time
import torch
from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D_e_v3 import mass_matrixP1_3D
from src.stiffness_matrixP1_3D_e import stiffness_matrixP1_3D
from src.sparse_block_diagonal import sparse_block_diagonal
from src.couple_flux_matrix_v5_fast import precompute_flux_geometry, couple_flux_matrix_fast
from xitorch.linalg import symeig
from xitorch import LinearOperator
import logging

logger = logging.getLogger("mvrecon_3d")

def precompute_reduced_constants(precomputed: dict, basis_funcs: torch.Tensor) -> dict:
    """
    Precompute reduced matrices that do NOT depend on faces_prob, given a frozen basis.

    Args:
        precomputed: output of precompute_laplace_matrices()
        basis_funcs: Phi (N, K) basis (e.g., eigenvectors). Usually detached if frozen.

    Returns:
        dict with Mr, Kr, Rr, Jx_r and cached Phi.
    """
    M_dense = precomputed["M_dense"]
    K_dense = precomputed["K_dense"]
    R_dense = precomputed["R_dense"]
    Jx_dense = precomputed["Jx_dense"]  # (3, N, N) dense

    Phi = basis_funcs
    if Phi.ndim != 2:
        raise ValueError(f"basis_funcs must be 2D (N,K). Got {Phi.shape}")

    # Reduced mass/stiffness/relaxation (K,K)
    Mr = Phi.T @ (M_dense @ Phi)
    Kr = Phi.T @ (K_dense @ Phi)
    Rr = Phi.T @ (R_dense @ Phi)

    # Reduced Jx per dimension (3, K, K)
    # Jx_dense is stacked as (3,N,N)
    Jx_r = torch.stack([Phi.T @ (Jx_dense[idim] @ Phi) for idim in range(3)], dim=0)

    return {
        "Phi": Phi,
        "Mr": Mr,
        "Kr": Kr,
        "Rr": Rr,
        "Jx_r": Jx_r,  # (3,K,K)
    }


def compute_laplace_reduced_projections(precomputed: dict,
                                    reduced_const: dict,
                                    faces_prob: torch.Tensor) -> dict:
    """
    Compute permeability-dependent reduced operators using a frozen basis Phi.

    This is the "non-refresh" path:
    - build sparse flux_matrix(faces_prob)
    - project it: Fr = Phi^T * F * Phi
    - assemble Ar = Kr + Fr
    - provide moments (= Phi^T Jx Phi) and massrelax (= Phi^T R Phi)

    Returns a dict compatible with a reduced solve (KxK dynamics).
    """
    flux_precomputed = precomputed["flux_precomputed"]

    Phi = reduced_const["Phi"]
    Kr  = reduced_const["Kr"]
    Mr  = reduced_const["Mr"]
    Rr  = reduced_const["Rr"]
    Jx_r = reduced_const["Jx_r"]  # (3,K,K)

    # Build flux (likely sparse COO/CSR)
    t0 = time.time()
    flux_matrix = couple_flux_matrix_fast(flux_precomputed, faces_prob=faces_prob).float()
    flux_time = time.time() - t0
    print(f"[compute_laplace_reduced_projections] Flux assembly time: {flux_time:.2f}s")

    if not isinstance(flux_matrix, torch.Tensor):
        raise TypeError(f"flux_matrix must be a torch.Tensor, got {type(flux_matrix)}")

    # Project flux to reduced space without densifying if sparse
    # Fr = Phi^T (F Phi)
    if flux_matrix.is_sparse:
        F_Phi = torch.sparse.mm(flux_matrix, Phi)          # (N,K)
    else:
        F_Phi = flux_matrix @ Phi                          # (N,K)
    Fr = Phi.T @ F_Phi                                     # (K,K)

    Ar = Kr + Fr                                           # (K,K)

    # Match your existing "moments" shape: [K, K, 3]
    moments = Jx_r.permute(1, 2, 0).contiguous()           # (K,K,3)

    # massrelax already projected
    massrelax = Rr                                         # (K,K)

    return {
        "Mr": Mr,
        "Ar": Ar,
        "Fr": Fr,
        "moments": moments,
        "massrelax": massrelax,
        "flux_time": flux_time,
    }

def precompute_laplace_matrices(femesh, setup, pde, eiglim, neig_max=None):
    """
    Precompute all matrices that are independent of faces_prob.
    This should be called once before the optimization loop.
    
    Includes both Laplace matrix assembly AND flux geometry precomputation.
    
    Returns:
        dict containing precomputed matrices and metadata
    """
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

    M = sparse_block_diagonal(M_cmpts).float()
    K = sparse_block_diagonal(K_cmpts).float()
    R = sparse_block_diagonal(R_cmpts).float()
    Jx = [sparse_block_diagonal(dim_cmpts).float() for dim_cmpts in Jx_cmpts]
    
    # Convert to dense once
    M_dense = M.to_dense()
    K_dense = K.to_dense()
    R_dense = R.to_dense()
    Jx_dense = torch.stack([jx.to_dense() for jx in Jx])
    
    original_neig_max = neig_max
    neig_max = min(original_neig_max, M.shape[0])
    
    # NEW: Precompute flux geometry as well
    print("[precompute_laplace_matrices] Precomputing flux geometry...")
    flux_precomputed = precompute_flux_geometry(femesh)
    
    precomputed = {
        "M_dense": M_dense,
        "K_dense": K_dense,
        "R_dense": R_dense,
        "Jx_dense": Jx_dense,
        "eiglim": eiglim if eiglim else None,
        "neig_max": neig_max,
        "original_neig_max": original_neig_max,
        "flux_precomputed": flux_precomputed,  # NEW: Include flux geometry
        "precompute_time": time.time() - start_time,
    }
    
    print(f"[precompute_laplace_matrices] Total precomputation time: {precomputed['precompute_time']:.2f}s")
    return precomputed


def compute_laplace_eig_diff(femesh, setup, precomputed, faces_prob=None):
    """
    Fast version that uses precomputed matrices.
    Only recomputes flux_matrix and eigendecomposition.
    
    Args:
        femesh: mesh data (only needed for logging, not computation)
        setup: experiment setup (only needed for logging, not computation)
        precomputed: dict from precompute_laplace_matrices()
        faces_prob: permeability values (changes each iteration)
    
    Returns:
        lap_eig dict with eigenvalues, eigenvectors, etc.
    """
    start_time = time.time()
    
    # Extract precomputed data
    M_dense = precomputed["M_dense"]
    K_dense = precomputed["K_dense"]
    R_dense = precomputed["R_dense"]
    Jx_dense = precomputed["Jx_dense"]
    eiglim = precomputed["eiglim"]
    neig_max = precomputed["neig_max"]
    original_neig_max = precomputed["original_neig_max"]
    flux_precomputed = precomputed["flux_precomputed"]
    
    # Compute flux matrix (depends on faces_prob) - using fast precomputed version
    t0 = time.time()
    flux_matrix = couple_flux_matrix_fast(flux_precomputed, faces_prob=faces_prob).float()
    flux_time = time.time() - t0
    print(f"[compute_laplace_eig_diff] Flux matrix assembly time: {flux_time:.2f}s")
    
    logger.debug(f"[compute_laplace_eig_diff] flux_matrix.type={type(flux_matrix)}, flux_matrix.shape={flux_matrix.shape}, flux_matrix.grad_fn={flux_matrix.grad_fn}")
    if not isinstance(flux_matrix, torch.Tensor):
        logger.error(f"[compute_laplace_eig_diff] flux_matrix is not a tensor: {type(flux_matrix)}")
        raise TypeError("flux_matrix must be a torch.Tensor")
    
    # Assemble A matrix
    flux_matrix_dense = flux_matrix.to_dense()
    A = K_dense + flux_matrix_dense
    
    # Wrap in LinearOperator and solve eigenvalue problem
    t0 = time.time()
    A_op = LinearOperator.m(A)
    B_op = LinearOperator.m(M_dense)
    
    eigenvalues, eigenvectors = symeig(A_op, neig=neig_max, M=B_op, mode="lowest")
    eig_time = time.time() - t0
    print(f"[compute_laplace_eig_diff] Eigenvalue computation time: {eig_time:.2f}s")
    
    # Sort and filter eigenvalues
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
    
    # Normalize eigenfunctions
    funcs = funcs / torch.sqrt(torch.sum(funcs * (M_dense @ funcs), dim=0))[None, :]
    
    # Compute moments
    tmp = torch.matmul(Jx_dense, funcs)  # [3, N, neig]
    moments = torch.matmul(funcs.T.unsqueeze(0), tmp).permute(1, 2, 0)  # [neig, neig, 3]
    
    # Compute massrelax
    massrelax = funcs.T @ R_dense @ funcs
    
    total_time = time.time() - start_time
    
    lap_eig = {
        "values": values,
        "funcs": funcs,
        "moments": moments,
        "massrelax": massrelax,
        "totaltime": total_time,
        "flux_time": flux_time,
        "eig_time": eig_time,
    }
    
    return lap_eig