import time
import torch
from mesh_setup.PGSE import PGSE
from src.sparse_block_diagonal import sparse_block_diagonal
from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D_e_v3 import mass_matrixP1_3D

def solve_mf(femesh, setup, lap_eig, direction=None):
    t_start = time.time()
    COMPLEX = torch.complex64
    device = lap_eig["funcs"].device

    # --------------------------
    # Unpack PDE & gradient info
    # --------------------------
    init_density = setup.pde["initial_density"]               # list len n_comp
    q_values     = setup.gradient["qvalues"].to(device)       # [n_amp, n_seq]
    sequences    = setup.gradient["sequences"]                # list len n_seq

    # --------------------------
    # Canonicalize directions -> [3, n_dir] and KEEP COMPLEX
    # --------------------------
    assert direction is not None, "direction tensor required"
    dirs = direction.to(device)
    if dirs.ndim == 1 and dirs.shape[0] == 3:
        dirs = dirs.view(3, 1)
    elif dirs.ndim == 2 and dirs.shape[0] == 3:
        pass  # already [3, n_dir]
    elif dirs.ndim == 2 and dirs.shape[1] == 3:
        dirs = dirs.transpose(0, 1)  # -> [3, n_dir]
    else:
        raise ValueError(f"direction must be [3], [3,n_dir], or [n_dir,3], got {dirs.shape}")
    directions = dirs.to(dtype=COMPLEX, device=device)        # [3, n_dir] complex
    n_dir = directions.shape[1]

    # ---------------------------------------------
    # Promote everything used in @ or expm to c64
    # ---------------------------------------------
    eig_funcs  = lap_eig["funcs"].to(dtype=COMPLEX, device=device)      # [n_point, n_eig]
    moments    = lap_eig["moments"].to(dtype=COMPLEX, device=device)    # [n_eig, n_eig, 3] complex
    relax_mat  = lap_eig["massrelax"].to(dtype=COMPLEX, device=device)  # [n_eig, n_eig]
    lambda_mat = torch.diag(lap_eig["values"].to(device)).to(dtype=COMPLEX, device=device)  # [n_eig,n_eig]

    # --------------------------
    # Sizes and initial rho/nu0
    # --------------------------
    n_comp = femesh["ncompartment"]
    n_amp  = len(setup.gradient["values"])
    n_seq  = len(sequences)
    n_int  = setup.mf["ninterval"]

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
        [torch.full((n_pts, 1), init_density[c], dtype=COMPLEX, device=device)
         for c in range(n_comp)],
        dim=0
    )

    Mbig = sparse_block_diagonal(mass_blocks).to(dtype=COMPLEX, device=device)
    H    = torch.conj(eig_funcs).T
    nu0  = H @ (Mbig.to_dense() @ rho0)                              # [n_eig, 1]

    dense_mass = [blk.to_dense().to(device) for blk in mass_blocks]
    dense_mass_stack = torch.stack(dense_mass).to(dtype=COMPLEX)     # [n_comp, n_pts, n_pts]

    # --------------------------------------
    # Build A for all directions (KEEP COMPLEX) and batched EVD of Hermitian part
    # --------------------------------------
    # moments: [n,n,3] complex, directions: [3,d] complex -> A_all: [n,n,d] complex
    # Use 'ijc,cd->ijd' to make it explicit that the 3-axis is 'c'
    A_all = torch.einsum('ijc,cd->ijd', moments, directions)         # [n_eig, n_eig, n_dir] complex
    A_per_dir = A_all.permute(2, 0, 1)                               # [n_dir, n, n] complex

    # Hermitianize in the complex sense (keeps everything complex & differentiable)
    A_herm = 0.5 * (A_per_dir + A_per_dir.conj().transpose(-1, -2))  # [d, n, n] Hermitian complex

    # Batched Hermitian EVD (eigh returns real eigenvalues + complex eigenvectors)
    S_dirs, U_dirs = torch.linalg.eigh(A_herm)                       # S:[d,n] float32, U:[d,n,n] complex64
    U_H = U_dirs.conj().transpose(-1, -2)                            # [d,n,n] complex
    # (S_dirs is real-valued; keep as float32)
    # If you want, ensure dtype explicitly:
    S_dirs = S_dirs.to(torch.float32)
    U_dirs = U_dirs.to(COMPLEX)
    U_H    = U_H.to(COMPLEX)

    # --------------------------------------
    # Allocate outputs that include n_dir
    # --------------------------------------
    magnetization = torch.zeros((n_comp, n_amp, n_seq, n_dir, n_pts), dtype=COMPLEX, device=device)
    signal        = torch.zeros(n_comp, n_amp, n_seq, n_dir, dtype=COMPLEX, device=device)
    itertimes     = torch.zeros(n_amp, n_seq, n_dir, dtype=torch.float32, device=device)

    # ---------------
    # Main loop on seq
    # ---------------
    for s in range(n_seq):
        t0 = time.time()
        seq = sequences[s]

        # Batch nu0 over amp and dir: [n_amp, n_dir, n, 1]
        nu = nu0.view(1, 1, -1, 1).expand(n_amp, n_dir, -1, -1).clone()

        if isinstance(seq, PGSE):
            # D = lambda + relax
            D = (lambda_mat + relax_mat).to(dtype=COMPLEX, device=device)          # [n,n]

            # Shared exponentials
            Ehalf   = torch.linalg.matrix_exp(-0.5 * seq.delta * D)                # [n,n]
            Ehalf_H = Ehalf.conj().transpose(-1, -2)                               # [n,n]
            E2      = torch.linalg.matrix_exp(-(seq.Delta - seq.delta) * D)        # [n,n]

            # Broadcast helpers
            Ehalf_b   = Ehalf.unsqueeze(0).unsqueeze(0)                             # [1,1,n,n]
            Ehalf_H_b = Ehalf_H.unsqueeze(0).unsqueeze(0)                           # [1,1,n,n]
            E2_b      = E2.unsqueeze(0).unsqueeze(0)                                # [1,1,n,n]

            # q for this sequence across amplitudes
            q_s = q_values[:, s].to(device).to(torch.float32)                       # [n_amp]

            # Phase in eigenbasis for all amps×dirs×eigs: [a,d,n] complex
            phase      = torch.exp(-1j * seq.delta * q_s.view(-1, 1, 1) * S_dirs.unsqueeze(0)).to(COMPLEX)
            phase_conj = phase.conj()

            # ---- E1 @ nu (vectorized over [amp, dir]) ----
            x = Ehalf_b @ nu                                                        # [a,d,n,1]
            # into eigenbasis: Uᴴ per dir (complex)
            x_eig = torch.einsum('dij,adjl->adil', U_H, x)                          # [a,d,n,1]
            # apply diagonal phase in eigenbasis
            x_eig = (x_eig.squeeze(-1) * phase).unsqueeze(-1)                       # [a,d,n,1]
            # back to spatial: U per dir
            x = torch.einsum('dij,adjl->adil', U_dirs, x_eig)                       # [a,d,n,1]
            # right half-step by D
            tmp1 = Ehalf_b @ x                                                      # [a,d,n,1]

            # ---- E2 @ (...) ----
            tmp2 = E2_b @ tmp1                                                      # [a,d,n,1]

            # ---- E1^H @ (...) ----
            z = Ehalf_H_b @ tmp2                                                    # [a,d,n,1]
            z_eig = torch.einsum('dij,adjl->adil', U_H, z)                          # [a,d,n,1]
            z_eig = (z_eig.squeeze(-1) * phase_conj).unsqueeze(-1)                  # [a,d,n,1]
            z = torch.einsum('dij,adjl->adil', U_dirs, z_eig)                       # [a,d,n,1]
            nu_final = Ehalf_H_b @ z                                                # [a,d,n,1]

        else:
            # (unchanged) generic time-stepping; still supports batched dirs
            tgrid = torch.linspace(0, seq.echotime, n_int + 1, device=device)
            A_batch       = A_all.permute(2, 0, 1).unsqueeze(0)                     # [1,d,n,n]
            lambda_batch  = lambda_mat.unsqueeze(0).unsqueeze(0)                    # [1,1,n,n]
            relax_batch   = relax_mat.unsqueeze(0).unsqueeze(0)                     # [1,1,n,n]
            q_as          = q_values[:, s].view(n_amp, 1, 1, 1)                     # [a,1,1,1]

            for k in range(n_int):
                dt = tgrid[k+1] - tgrid[k]
                ft = 0.5 * (seq.call(tgrid[k+1]) + seq.call(tgrid[k]))
                ft = torch.tensor(ft, dtype=torch.float32, device=device)

                Kt = lambda_batch + relax_batch + 1j * q_as * ft * A_batch          # [a,d,n,n]
                exp_dt = torch.linalg.matrix_exp(-dt * Kt)                           # [a,d,n,n]
                nu = exp_dt @ nu                                                     # [a,d,n,1]

            nu_final = nu

        # -------------------------
        # Back to spatial basis
        # -------------------------
        mag_full = (eig_funcs.unsqueeze(0).unsqueeze(0) @ nu_final).squeeze(-1)     # [a,d,n_point]
        mag_parts = torch.split(mag_full, n_pts, dim=-1)
        mag_stack = torch.stack(mag_parts, dim=0)                                    # [n_comp,a,d,n_pts]
        magnetization[:, :, s, :, :] = mag_stack

        # -------------------------
        # Signal integration
        # -------------------------
        mag_c = mag_stack.unsqueeze(-1)                                              # [n_comp,a,d,n_pts,1]
        integ = dense_mass_stack.unsqueeze(1).unsqueeze(2) @ mag_c                   # [n_comp,a,d,n_pts,1]
        signal_c = integ.sum(dim=-2).squeeze(-1)                                     # [n_comp,a,d]
        signal[:, :, s, :] = signal_c

        # timing per amp×dir (same scalar replicated)
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
