import time
import torch
from mesh_setup.PGSE import PGSE
from src.sparse_block_diagonal import sparse_block_diagonal
from src.get_volume_mesh import get_volume_mesh
# from src.mass_matrixP1_3D import mass_matrixP1_3D
from src.mass_matrixP1_3D_e_v3 import mass_matrixP1_3D
from src.krylov_tools import batched_krylov_matrix_exp

def solve_mf(femesh, setup, lap_eig, direction=None, debug_checks=False, n_eig = 1000):
    """
    Matrix-formalism Bloch-Torrey solver (differentiable).
    Computes signal loss for all amplitudes, sequences, and directions.
    """
    t_start = time.time()
    COMPLEX = torch.complex64
    
    # Get device from first available tensor
    device = lap_eig["funcs"].device

    # breakpoint()
    
    # Unpack PDE & gradient settings - ENSURE ON DEVICE
    init_density = setup.pde["initial_density"]       # list length n_comp
    q_values     = setup.gradient["qvalues"].to(device)         # [n_amp, n_seq]
    b_values     = setup.gradient["bvalues"].to(device)         # [n_amp, n_seq]
    sequences    = setup.gradient["sequences"]       # list length n_seq

    if len(direction.shape) < 2:
        directions = direction.to(dtype=COMPLEX, device=device).unsqueeze(1)  # [3, 1]
    # n_dir = 1
    n_dir = directions.shape[1]

    # Promote everything that goes into @ or expm to complex64 ON DEVICE
    eig_funcs   = lap_eig["funcs"].to(dtype=COMPLEX, device=device)       # [n_point, n_eig]
    moments     = lap_eig["moments"].to(dtype=COMPLEX, device=device)     # [n_eig, n_eig, 3]
    relax_mat   = lap_eig["massrelax"].to(dtype=COMPLEX, device=device)   # [n_eig, n_eig]
    lambda_mat  = torch.diag(lap_eig["values"].to(device)).to(dtype=COMPLEX, device=device)  # [n_eig, n_eig]

    # Sizes
    n_comp = femesh["ncompartment"]
    n_amp  = len(setup.gradient["values"])
    n_seq  = len(sequences)
    n_int  = setup.mf["ninterval"]

    # Assemble FEM mass blocks & initial ν₀
    n_pts = femesh["points"][0].shape[1]
    pts_per_comp = [n_pts] * n_comp
    mass_blocks = [mass_matrixP1_3D(femesh["elements"][c].to(device), get_volume_mesh(femesh["points"][c].to(device), femesh["elements"][c].to(device))[1]).to(device) for c in range(n_comp)]
    rho0 = torch.cat([torch.full((n_pts, 1), init_density[c], dtype=COMPLEX, device=device) for c in range(n_comp)], dim=0)

    # Big block-diagonal mass & initial coeffs
    Mbig = sparse_block_diagonal(mass_blocks).to(dtype=COMPLEX, device=device) 
    H    = torch.conj(eig_funcs).T                    # Hermitian transpose
    nu0  = H @ (Mbig.to_dense() @ rho0)               # [n_eig × 1]

    # Keep dense mass blocks for final signal integration
    dense_mass = [blk.to_dense().to(device) for blk in mass_blocks]

    # Initialize output containers ON DEVICE
    n_pts = pts_per_comp[0]  # Assuming uniform size per compartment
    magnetization = torch.zeros((n_comp, n_amp, n_seq, n_dir, n_pts), dtype=COMPLEX, device=device)
    signal = torch.zeros(n_comp, n_amp, n_seq, n_dir, dtype=COMPLEX, device=device)
    itertimes = torch.zeros(n_amp, n_seq, n_dir, dtype=torch.float32, device=device)

    # Precompute A for all directions
    A_all = torch.einsum('ijk,kl->ijl', moments, directions)  # [n_eig, n_eig, n_dir]

    # Stack dense_mass for batched computation
    dense_mass_stack = torch.stack(dense_mass).to(dtype=COMPLEX)  # [n_comp, n_pts, n_pts]

    # Loop over sequences (kept due to potential varying types; batched over amp/dir/comps)
    for s in range(n_seq):
        t0 = time.time()

        seq = sequences[s]
        q_as = q_values[:, s][:, None, None, None]  # [n_amp, 1, 1, 1]
        A_batch = A_all.permute(2, 0, 1).unsqueeze(0)  # [1, n_dir, n_eig, n_eig]
        lambda_batch = lambda_mat.unsqueeze(0).unsqueeze(0)  # [1, 1, n_eig, n_eig]
        relax_batch = relax_mat.unsqueeze(0).unsqueeze(0)    # [1, 1, n_eig, n_eig]

        # Batch nu0 over amp and dir
        nu = nu0.unsqueeze(0).unsqueeze(0)  # [1, 1, n_eig, 1]
        nu = nu.expand(n_amp, n_dir, -1, -1).clone()  # [n_amp, n_dir, n_eig, 1]

        # breakpoint()
        # if isinstance(seq, PGSE):
        #     # Batched K1 = lambda + relax + 1j * q * A
        #     K1 = lambda_batch + relax_batch + 1j * q_as * A_batch  # [n_amp, n_dir, n_eig, n_eig]

        #     # Batched E1 = exp(-delta * K1)
        #     # E1 = torch.linalg.matrix_exp(-seq.delta * K1)
            
        #     # print(f"All 1000x1000 matrices are symmetric in K1 = {all(torch.allclose(K1[i, 0], K1[i, 0].T, atol=1e-6) for i in range(21))}") "True"
        #     # print(f"All 1000x1000 matrices are hermetian in K1 = {all(torch.allclose(K1[i, 0], K1[i, 0].conj().T, atol=1e-6) for i in range(21))}") "False"
            
        #     # E1 = torch.linalg.matrix_exp((-seq.delta * K1).contiguous())
        #     E1 = batched_krylov_matrix_exp(-seq.delta * K1, m=krylov_dim)

        #     # E2 is independent of amp/dir
        #     # E2 = torch.linalg.matrix_exp(-(seq.Delta - seq.delta) * (lambda_mat + relax_mat))
        #     E2 = batched_krylov_matrix_exp(-(seq.Delta - seq.delta) * (lambda_mat + relax_mat), m=krylov_dim)

        #     # nu = E1^H @ (E2 @ (E1 @ nu))
        #     tmp1 = torch.matmul(E1, nu)  # [n_amp, n_dir, n_eig, 1]
        #     tmp2 = torch.matmul(E2.unsqueeze(0).unsqueeze(0), tmp1)  # broadcast E2
        #     E1_H = torch.conj(E1).transpose(-1, -2)  # [n_amp, n_dir, n_eig, n_eig]
        #     nu_final = torch.matmul(E1_H, tmp2)  # [n_amp, n_dir, n_eig, 1]

        if isinstance(seq, PGSE):
            # Batched K1 = lambda + relax + 1j * q * A
            K1 = lambda_batch + relax_batch + 1j * q_as * A_batch
            
            # ===== E1 with Krylov (15 seconds faster as you noted) =====
            krylov_dim = min(50, n_eig // 20)  # Adaptive dimension
            E1 = batched_krylov_matrix_exp(-seq.delta * K1, m=krylov_dim)
            
            # ===== E2 WITHOUT matrix_exp (much faster) =====
            L_plus_R = lambda_mat.real + relax_mat.real
            
            # Check if diagonal (very common case)
            is_diag = torch.allclose(L_plus_R, torch.diag(torch.diagonal(L_plus_R)))
            
            if is_diag:
                # Super fast: just exponential of diagonal elements
                exp_diag = torch.exp(-(seq.Delta - seq.delta) * torch.diagonal(L_plus_R))
                
                # Apply operations
                tmp1 = torch.matmul(E1, nu)
                tmp2 = tmp1 * exp_diag.view(1, 1, -1, 1)  # Element-wise multiply
            else:
                print("Not diagonal E2")
                # Still fast: eigendecomposition (done once, not per batch)
                eigenvalues, eigenvectors = torch.linalg.eigh(L_plus_R)
                exp_eigenvalues = torch.exp(-(seq.Delta - seq.delta) * eigenvalues)
                
                # Apply operations via eigenspace
                tmp1 = torch.matmul(E1, nu)
                tmp1_eigen = torch.matmul(eigenvectors.T.unsqueeze(0).unsqueeze(0), tmp1)
                tmp2_eigen = tmp1_eigen * exp_eigenvalues.view(1, 1, -1, 1)
                tmp2 = torch.matmul(eigenvectors.unsqueeze(0).unsqueeze(0), tmp2_eigen)
            
            # Final step
            E1_H = torch.conj(E1).transpose(-1, -2)
            nu_final = torch.matmul(E1_H, tmp2)

        else:
            tgrid = torch.linspace(0, seq.echotime, n_int+1, device=device)
            for k in range(n_int):
                dt = tgrid[k+1] - tgrid[k]
                ft = 0.5 * (seq.call(tgrid[k+1]) + seq.call(tgrid[k]))
                ft = torch.tensor(ft, dtype=torch.float32, device=device)  # Keep as float; complex mul later

                # Batched Kt = lambda + relax + 1j * q * ft * A
                Kt = lambda_batch + relax_batch + 1j * q_as * ft * A_batch

                # exp(-dt * Kt)
                exp_dt = torch.linalg.matrix_exp(-dt * Kt)

                # nu = exp @ nu
                nu = torch.matmul(exp_dt, nu)

            nu_final = nu

        # Back to spatial basis: mag_full = eig_funcs @ nu_final
        mag_full = torch.matmul(eig_funcs.unsqueeze(0).unsqueeze(0), nu_final).squeeze(-1)  # [n_amp, n_dir, n_point]

        # Split by compartment and stack
        mag_parts = torch.split(mag_full, n_pts, dim=-1)  # list of [n_amp, n_dir, n_pts]
        mag_stack = torch.stack(mag_parts, dim=0)  # [n_comp, n_amp, n_dir, n_pts]

        # Store magnetization
        magnetization[:, :, s, :, :] = mag_stack

        # Compute signal batched
        mag_c = mag_stack.unsqueeze(-1)  # [n_comp, n_amp, n_dir, n_pts, 1]
        integ = torch.matmul(dense_mass_stack.unsqueeze(1).unsqueeze(2), mag_c)  # [n_comp, n_amp, n_dir, n_pts, 1]
        signal_c = integ.sum(dim=-2).squeeze(-1)  # [n_comp, n_amp, n_dir]
        signal[:, :, s, :] = signal_c

        # Approximate iteration times
        dt = time.time() - t0
        itertimes[:, s, :] = dt / (n_amp * n_dir)

    signal_allcmpts = signal.sum(dim=0)

    return {
        "magnetization": magnetization.squeeze(3),
        "signal": signal.squeeze(3),
        "signal_allcmpts": signal_allcmpts.squeeze(2),
        "itertimes": itertimes.squeeze(2),
        "time_taken": time.time() - t_start,
    }