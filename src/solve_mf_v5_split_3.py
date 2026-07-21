import time
import torch
from mesh_setup.PGSE import PGSE
from src.sparse_block_diagonal import sparse_block_diagonal
from src.get_volume_mesh import get_volume_mesh
# from src.mass_matrixP1_3D import mass_matrixP1_3D
from src.mass_matrixP1_3D_e_v3 import mass_matrixP1_3D

def solve_mf(femesh, setup, lap_eig, direction=None, debug_checks=False):
    """
    Matrix-formalism Bloch-Torrey solver (differentiable).
    Computes signal loss for all amplitudes, sequences, and directions.
    """
    t_start = time.time()
    COMPLEX = torch.complex64
    
    # Get device from first available tensor
    device = lap_eig["funcs"].device
    
    # Unpack PDE & gradient settings - ENSURE ON DEVICE
    init_density = setup.pde["initial_density"]        # list length n_comp
    q_values     = setup.gradient["qvalues"].to(device)  # [n_amp, n_seq]
    b_values     = setup.gradient["bvalues"].to(device)  # [n_amp, n_seq] (unused here)
    sequences    = setup.gradient["sequences"]          # list length n_seq

    # Directions: expect [3] or [3, n_dir]; promote to complex
    directions = direction.to(dtype=COMPLEX, device=device)
    if directions.dim() == 1:
        directions = directions.unsqueeze(1)  # [3, 1]
    n_dir = directions.shape[1]

    # Promote Laplacian-eig objects to complex64 ON DEVICE
    eig_funcs  = lap_eig["funcs"].to(dtype=COMPLEX, device=device)       # [n_point, n_eig]
    moments    = lap_eig["moments"].to(dtype=COMPLEX, device=device)     # [n_eig, n_eig, 3]
    relax_mat  = lap_eig["massrelax"].to(dtype=COMPLEX, device=device)   # [n_eig, n_eig]
    lambda_mat = torch.diag(lap_eig["values"].to(device)).to(dtype=COMPLEX, device=device)  # [n_eig, n_eig]
    n_eig = eig_funcs.shape[1]

    # Sizes
    n_comp = femesh["ncompartment"]
    n_amp  = len(setup.gradient["values"])
    n_seq  = len(sequences)
    n_int  = setup.mf["ninterval"]

    # Assemble FEM mass blocks & initial ν₀  (unchanged)
    n_pts = femesh["points"][0].shape[1]
    pts_per_comp = [n_pts] * n_comp
    mass_blocks = [
        mass_matrixP1_3D(
            femesh["elements"][c].to(device),
            get_volume_mesh(femesh["points"][c].to(device), femesh["elements"][c].to(device)
        )[1]).to(device)
        for c in range(n_comp)
    ]
    rho0 = torch.cat(
        [torch.full((n_pts, 1), init_density[c], dtype=COMPLEX, device=device) for c in range(n_comp)],
        dim=0
    )

    # Big block-diagonal mass & initial coeffs (unchanged)
    Mbig = sparse_block_diagonal(mass_blocks).to(dtype=COMPLEX, device=device)
    H    = torch.conj(eig_funcs).T
    nu0  = H @ (Mbig.to_dense() @ rho0)  # [n_eig, 1]

    # Keep dense mass blocks for final signal integration (unchanged)
    dense_mass = [blk.to_dense().to(device) for blk in mass_blocks]

    # Initialize output containers ON DEVICE (unchanged)
    n_pts = pts_per_comp[0]
    magnetization = torch.zeros((n_comp, n_amp, n_seq, n_dir, n_pts), dtype=COMPLEX, device=device)
    signal        = torch.zeros(n_comp, n_amp, n_seq, n_dir, dtype=COMPLEX, device=device)
    itertimes     = torch.zeros(n_amp, n_seq, n_dir, dtype=torch.float32, device=device)

    # Precompute A for all directions: A_all[i,j,d] = sum_k moments[i,j,k] * directions[k,d]
    A_all = torch.einsum('ijk,kd->ijd', moments, directions)  # [n_eig, n_eig, n_dir]

    # Stack dense_mass for batched computation (unchanged)
    dense_mass_stack = torch.stack(dense_mass).to(dtype=COMPLEX)  # [n_comp, n_pts, n_pts]

    # Amplitude block size for tall-skinny GEMMs (tune 8–16 typically)
    AMP_BLOCK = max(1, min(16, n_amp))

    # Loop over sequences
    for s in range(n_seq):
        t0 = time.time()
        seq = sequences[s]

        if isinstance(seq, PGSE):
            # ===== PGSE EXACT, CHUNKED (no approximations) =====
            # Common part (amp/dir independent)
            # L = (lambda_mat + relax_mat).contiguous()                             # [n,n] complex
            # E2 = torch.linalg.matrix_exp(-(seq.Delta - seq.delta) * L)           # [n,n], once/seq
            # n = L.shape[0]
            L = (lambda_mat + relax_mat).contiguous()
            n = L.shape[0]
            tau = seq.Delta - seq.delta
            
            diag_L = torch.diag(lambda_mat) + torch.diag(relax_mat)
            E2 = torch.diag(torch.exp(-tau * diag_L))

            # How many amplitudes per chunk (tune: 4–16; bigger = faster, too big = OOM)
            # AMP_BLOCK = min(max(8, 1), n_amp)  # start with 8; adjust to your GPU
            AMP_BLOCK = 10

            # Prepare container for nu_final: [n_amp, n_dir, n, 1]
            nu_final = torch.empty((n_amp, n_dir, n, 1), dtype=COMPLEX, device=device)

            # Scalars per amplitude
            alpha_all = q_values[:, s].to(eig_funcs.real.dtype)  # [n_amp]

            # Loop directions
            for d in range(n_dir):
                A_d = A_all[..., d].contiguous()                 # [n,n] complex
                iA  = 1j * A_d                                   # precompute once

                # Chunk amplitudes to cap peak memory; still vectorized *within* each chunk
                a0 = 0
                while a0 < n_amp:
                    a1 = min(a0 + AMP_BLOCK, n_amp)
                    alphas = alpha_all[a0:a1].view(-1, 1, 1)     # [B,1,1]
                    # Build K1 chunk: L + i*alpha*A_d  (broadcast over B)
                    K1_B = L.unsqueeze(0) + alphas * iA.unsqueeze(0)   # [B,n,n]
                    # Exact exp for the chunk
                    E1_B = torch.linalg.matrix_exp((-seq.delta * K1_B).contiguous())  # [B,n,n]

                    # Forward multiply by nu0 (broadcast to [B,n,1])
                    tmp1 = E1_B @ nu0.unsqueeze(0)                # [B,n,1]
                    # Middle E2 (same for all B)
                    tmp2 = E2.unsqueeze(0) @ tmp1                 # [B,n,1]
                    # Adjoint multiply
                    E1H_B = torch.conj(E1_B).transpose(-1, -2)    # [B,n,n]
                    u3_B  = E1H_B @ tmp2                          # [B,n,1]

                    # Store
                    nu_final[a0:a1, d, :, 0] = u3_B.squeeze(-1)   # [B,n]

                    # free chunk temps (helps peak memory)
                    del K1_B, E1_B, tmp1, tmp2, E1H_B, u3_B
                    a0 = a1
            # ===== end PGSE EXACT, CHUNKED =====
        else:
            # Original non-PGSE path (unchanged)
            q_as = q_values[:, s][:, None, None, None]                 # [n_amp, 1, 1, 1]
            A_batch = A_all.permute(2, 0, 1).unsqueeze(0)              # [1, n_dir, n_eig, n_eig]
            lambda_batch = lambda_mat.unsqueeze(0).unsqueeze(0)        # [1, 1, n_eig, n_eig]
            relax_batch  = relax_mat.unsqueeze(0).unsqueeze(0)         # [1, 1, n_eig, n_eig]

            # Expand nu0 over amp/dir for stepping
            nu = nu0.unsqueeze(0).unsqueeze(0).expand(n_amp, n_dir, -1, -1)  # [n_amp,n_dir,n_eig,1]

            tgrid = torch.linspace(0, seq.echotime, n_int + 1, device=device)
            for k in range(n_int):
                dt = tgrid[k + 1] - tgrid[k]
                ft = 0.5 * (seq.call(tgrid[k + 1]) + seq.call(tgrid[k]))
                ft = torch.tensor(ft, dtype=torch.float32, device=device)
                Kt = lambda_batch + relax_batch + 1j * q_as * ft * A_batch
                exp_dt = torch.linalg.matrix_exp(-dt * Kt)
                nu = torch.matmul(exp_dt, nu)
            nu_final = nu  # [n_amp,n_dir,n_eig,1]

        # Back to spatial basis: mag_full = eig_funcs @ nu_final  (unchanged)
        mag_full = torch.matmul(eig_funcs.unsqueeze(0).unsqueeze(0), nu_final).squeeze(-1)  # [n_amp, n_dir, n_point]

        # Split by compartment and stack (unchanged)
        mag_parts = torch.split(mag_full, n_pts, dim=-1)     # list of [n_amp, n_dir, n_pts]
        mag_stack = torch.stack(mag_parts, dim=0)            # [n_comp, n_amp, n_dir, n_pts]

        # Store magnetization (unchanged)
        magnetization[:, :, s, :, :] = mag_stack

        # Compute signal batched (unchanged)
        mag_c = mag_stack.unsqueeze(-1)  # [n_comp, n_amp, n_dir, n_pts, 1]
        integ = torch.matmul(dense_mass_stack.unsqueeze(1).unsqueeze(2), mag_c)  # [n_comp, n_amp, n_dir, n_pts, 1]
        signal_c = integ.sum(dim=-2).squeeze(-1)  # [n_comp, n_amp, n_dir]
        signal[:, :, s, :] = signal_c

        # Iteration timing
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
