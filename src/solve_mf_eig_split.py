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
    Uses a single input direction like: directions = direction[...,None]  # [3,1]
    """
    t_start = time.time()
    COMPLEX = torch.complex64

    # Device
    device = lap_eig["funcs"].device

    # Unpack PDE & gradient settings
    init_density = setup.pde["initial_density"]             # list length n_comp
    q_values     = setup.gradient["qvalues"].to(device)     # [n_amp, n_seq]
    b_values     = setup.gradient["bvalues"].to(device)     # [n_amp, n_seq]
    sequences    = setup.gradient["sequences"]              # list length n_seq

    # Single direction like your reference code
    directions = direction.to(dtype=COMPLEX, device=device).unsqueeze(1)  # [3, 1]
    n_dir = 1

    # Promote inputs to complex64 on device
    eig_funcs  = lap_eig["funcs"].to(dtype=COMPLEX, device=device)      # [n_point, n_eig]
    moments    = lap_eig["moments"].to(dtype=COMPLEX, device=device)    # [n_eig, n_eig, 3]
    relax_mat  = lap_eig["massrelax"].to(dtype=COMPLEX, device=device)  # [n_eig, n_eig]
    lambda_mat = torch.diag(lap_eig["values"].to(device)).to(dtype=COMPLEX, device=device)  # [n_eig, n_eig]

    # Sizes
    n_comp = femesh["ncompartment"]
    n_amp  = len(setup.gradient["values"])
    n_seq  = len(sequences)
    n_int  = setup.mf["ninterval"]

    # Assemble FEM mass blocks & initial ν0
    n_pts = femesh["points"][0].shape[1]
    pts_per_comp = [n_pts] * n_comp
    mass_blocks = [
        mass_matrixP1_3D(
            femesh["elements"][c].to(device),
            get_volume_mesh(femesh["points"][c].to(device), femesh["elements"][c].to(device)
        )[1]).to(device)
        for c in range(n_comp)
    ]
    rho0 = torch.cat([
        torch.full((n_pts, 1), init_density[c], dtype=COMPLEX, device=device)
        for c in range(n_comp)
    ], dim=0)

    # Big block-diagonal mass & initial coeffs
    Mbig = sparse_block_diagonal(mass_blocks).to(dtype=COMPLEX, device=device)
    H    = torch.conj(eig_funcs).T
    nu0  = H @ (Mbig.to_dense() @ rho0)   # [n_eig, 1]

    # Keep dense mass blocks for final signal integration
    dense_mass = [blk.to_dense().to(device) for blk in mass_blocks]

    # Output containers
    magnetization = torch.zeros((n_comp, n_amp, n_seq, n_dir, n_pts), dtype=COMPLEX, device=device)
    signal        = torch.zeros(n_comp, n_amp, n_seq, n_dir, dtype=COMPLEX, device=device)
    itertimes     = torch.zeros(n_amp, n_seq, n_dir, dtype=torch.float32, device=device)

    # Precompute A for the (single) direction
    # moments: [n_eig,n_eig,3], directions: [3,1]  -> A_all: [n_eig,n_eig,1]
    A_all = torch.einsum('ijk,kl->ijl', moments, directions)  # [n_eig, n_eig, n_dir]

    # Stack dense_mass for batched computation
    dense_mass_stack = torch.stack(dense_mass).to(dtype=COMPLEX)  # [n_comp, n_pts, n_pts]

    # Loop over sequences
    for s in range(n_seq):
        t0 = time.time()
        seq = sequences[s]

        # Prepare broadcasting shells (dir=1 here)
        q_as         = q_values[:, s][:, None, None, None]        # [n_amp, 1, 1, 1]
        A_batch      = A_all.permute(2, 0, 1).unsqueeze(0)         # [1, n_dir, n_eig, n_eig]
        lambda_batch = lambda_mat.unsqueeze(0).unsqueeze(0)        # [1, 1, n_eig, n_eig]
        relax_batch  = relax_mat.unsqueeze(0).unsqueeze(0)         # [1, 1, n_eig, n_eig]

        # Expand ν0 across amp,dir
        nu = nu0.unsqueeze(0).unsqueeze(0).expand(n_amp, n_dir, -1, -1).clone()  # [n_amp, n_dir, n_eig, 1]

        if isinstance(seq, PGSE):
            # ---- Precompute S and E2 action without forming E2 ----
            S_raw = (lambda_mat + relax_mat)                                        # [n_eig,n_eig]
            # Make S explicitly Hermitian for eigh (small projection only on S)
            S_h = 0.5 * (S_raw + S_raw.conj().transpose(-1, -2))                    # [n_eig,n_eig]
            sS, US = torch.linalg.eigh(S_h)                                         # sS: [n_eig] (real), US: [n_eig,n_eig]
            E2_scale = torch.exp(-(seq.Delta - seq.delta) * sS).to(COMPLEX)         # [n_eig]

            # Direction matrix (n_dir=1 in your current call)
            A_dir = A_all[..., 0]                                                   # [n_eig,n_eig]
            n_eig = S_h.shape[0]

            # Output buffer (no in-place writes on grads)
            nu_final = torch.empty(n_amp, 1, n_eig, 1, dtype=COMPLEX, device=device)

            # ---- amplitude chunking to cap memory (set chunk size to taste) ----
            amp_chunk = max(1, min(n_amp, 4))   # try 2–8 for a speed/mem trade-off

            for a0 in range(0, n_amp, amp_chunk):
                a1 = min(a0 + amp_chunk, n_amp)
                q_chunk = q_values[a0:a1, s]                                        # [m], m = a1-a0

                # Build K only for this chunk; don't stack all amps
                K = S_h.unsqueeze(0) + 1j * q_chunk[:, None, None] * A_dir.unsqueeze(0)  # [m,n_eig,n_eig]

                # Hermiticity check; use eigh if close, else fallback to matrix_exp for that q
                Kh = K.conj().transpose(-1, -2)
                num = torch.linalg.norm(K - Kh, dim=(-2, -1))
                den = torch.linalg.norm(K,     dim=(-2, -1)) + 1e-20
                herm_mask = (num / den) < 1e-6                                       # tune if needed

                idx_h  = torch.nonzero(herm_mask, as_tuple=False).flatten()
                idx_nh = torch.nonzero(~herm_mask, as_tuple=False).flatten()

                # Containers for this chunk
                nu_chunk_blocks = []
                idx_blocks      = []

                # ---------- Hermitian fast path (exact) ----------
                if idx_h.numel() > 0:
                    print("Hermetian case")
                    K_h = 0.5 * (K[idx_h] + Kh[idx_h])                               # [mh,n,n] (make exactly Hermitian)
                    w, U = torch.linalg.eigh(K_h)                                     # w: [mh,n], U: [mh,n,n]

                    # First hit: y = U * exp(-δ w) * (U^H ν0)
                    expw = torch.exp(-seq.delta * w).to(COMPLEX).unsqueeze(-1)        # [mh,n,1]
                    nu0_b = nu0.expand(w.shape[0], n_eig, 1)                          # [mh,n,1]
                    Uh_nu0 = U.conj().transpose(-1, -2) @ nu0_b                       # [mh,n,1]
                    y = U @ (expw * Uh_nu0)                                           # [mh,n,1]

                    # Apply E2 in S_h basis: y <- US * diag(E2_scale) * (US^H y)
                    yS = US.conj().transpose(-1, -2).unsqueeze(0) @ y                 # [mh,n,1]
                    yS = E2_scale.view(1, -1, 1) * yS
                    y  = US.unsqueeze(0) @ yS                                         # [mh,n,1]

                    # Second hit: E1^H == E1 (since K_h Hermitian)
                    Uh_y = U.conj().transpose(-1, -2) @ y                              # [mh,n,1]
                    Uh_y = expw * Uh_y                                                 # [mh,n,1]
                    nu_h = U @ Uh_y                                                    # [mh,n,1]

                    nu_chunk_blocks.append(nu_h)
                    idx_blocks.append(idx_h)

                # ---------- Non-Hermitian fallback (exact matrix_exp) ----------
                if idx_nh.numel() > 0:
                    print("Non hermetian case")
                    K_nh = K[idx_nh]                                                   # [mk,n,n]
                    E1   = torch.linalg.matrix_exp((-seq.delta * K_nh).contiguous())   # [mk,n,n]

                    # y = E1 @ ν0
                    nu0_b = nu0.expand(K_nh.shape[0], n_eig, 1)                        # [mk,n,1]
                    y = E1 @ nu0_b                                                     # [mk,n,1]

                    # Apply E2 in S_h basis (still cheap)
                    yS = US.conj().transpose(-1, -2).unsqueeze(0) @ y                  # [mk,n,1]
                    yS = E2_scale.view(1, -1, 1) * yS
                    y  = US.unsqueeze(0) @ yS                                          # [mk,n,1]

                    # nu = E1^H @ y
                    nu_nh = E1.conj().transpose(-1, -2) @ y                            # [mk,n,1]

                    nu_chunk_blocks.append(nu_nh)
                    idx_blocks.append(idx_nh)

                # Merge back in original order (no in-place scatter)
                all_idx   = torch.cat(idx_blocks, dim=0)
                all_vals  = torch.cat(nu_chunk_blocks, dim=0)                          # [m,n,1]
                perm      = torch.argsort(all_idx)                                     # order by original amplitude index
                nu_chunk  = all_vals[perm]                                             # [m,n,1]

                # write into result slice (this assignment is fine; it's a fresh buffer)
                nu_final[a0:a1, 0, :, :] = nu_chunk

            # done: nu_final shape [n_amp,1,n_eig,1]
        else:
            # Generic time-discretized case (unchanged; works with n_dir=1)
            tgrid = torch.linspace(0, seq.echotime, n_int+1, device=device)
            for k in range(n_int):
                dt = tgrid[k+1] - tgrid[k]
                ft = 0.5 * (seq.call(tgrid[k+1]) + seq.call(tgrid[k]))
                ft = torch.tensor(ft, dtype=torch.float32, device=device)

                Kt    = lambda_batch + relax_batch + 1j * q_as * ft * A_batch   # [n_amp,1,n_eig,n_eig]
                exp_dt = torch.linalg.matrix_exp(-dt * Kt)
                nu     = torch.matmul(exp_dt, nu)

            nu_final = nu  # [n_amp,1,n_eig,1]

        # Back to spatial basis
        mag_full = (eig_funcs.unsqueeze(0).unsqueeze(0) @ nu_final).squeeze(-1)  # [n_amp,1,n_point]

        # Split by compartment and stack
        mag_parts = torch.split(mag_full, n_pts, dim=-1)   # list of [n_amp,1,n_pts]
        mag_stack = torch.stack(mag_parts, dim=0)          # [n_comp,n_amp,1,n_pts]

        # Store magnetization
        magnetization[:, :, s, :, :] = mag_stack

        # Compute signal
        mag_c  = mag_stack.unsqueeze(-1)  # [n_comp,n_amp,1,n_pts,1]
        integ  = dense_mass_stack.unsqueeze(1).unsqueeze(2) @ mag_c
        signal_c = integ.sum(dim=-2).squeeze(-1)          # [n_comp,n_amp,1]
        signal[:, :, s, :] = signal_c

        # Iter time
        itertimes[:, s, :] = (time.time() - t0) / (n_amp * n_dir)

    # Match your reference return (squeeze dir dim)
    signal_allcmpts = signal.sum(dim=0)  # [n_amp,n_seq,1]
    return {
        "magnetization": magnetization.squeeze(3),      # [n_comp,n_amp,n_seq,n_pts]
        "signal": signal.squeeze(3),                    # [n_comp,n_amp,n_seq]
        "signal_allcmpts": signal_allcmpts.squeeze(2),  # [n_amp,n_seq]
        "itertimes": itertimes.squeeze(2),              # [n_amp,n_seq]
        "time_taken": time.time() - t_start,
    }
