import time
import torch
from mesh_setup.PGSE import PGSE
from src.sparse_block_diagonal import sparse_block_diagonal
from src.get_volume_mesh import get_volume_mesh
# from src.mass_matrixP1_3D import mass_matrixP1_3D
from src.mass_matrixP1_3D_e_v3 import mass_matrixP1_3D
# from src.pade_expm import pade_matrix_exp
from src.pade_expm_one import pade_matrix_exp

def solve_mf(femesh, setup, lap_eig, directions=None):
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
    directions   = setup.gradient["directions"].to(dtype=COMPLEX, device=device)      # [3, n_dir]

    # Promote everything that goes into @ or expm to complex64 ON DEVICE
    eig_funcs   = lap_eig["funcs"].to(dtype=COMPLEX, device=device)       # [n_point, n_eig]
    moments     = lap_eig["moments"].to(dtype=COMPLEX, device=device)     # [n_eig, n_eig, 3]
    relax_mat   = lap_eig["massrelax"].to(dtype=COMPLEX, device=device)   # [n_eig, n_eig]
    lambda_mat  = torch.diag(lap_eig["values"].to(device)).to(dtype=COMPLEX, device=device)  # [n_eig, n_eig]

    # Sizes
    n_comp = femesh["ncompartment"]
    n_amp  = len(setup.gradient["values"])
    n_seq  = len(sequences)
    n_dir  = directions.shape[1]
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
        if isinstance(seq, PGSE):
            # Batched K1 = lambda + relax + 1j * q * A
            K1 = lambda_batch + relax_batch + 1j * q_as * A_batch  # [n_amp, n_dir, n_eig, n_eig]

            # Batched E1 = exp(-delta * K1)
            # E1 = torch.linalg.matrix_exp(-seq.delta * K1)
            # 21 x 30 x 1000 x 1000
            # E1 = torch.linalg.matrix_exp((-seq.delta * K1).contiguous())
            E1 = pade_matrix_exp((-seq.delta * K1), q=12)

            # E2 is independent of amp/dir
            L = (lambda_batch + relax_batch).contiguous()
            n = L.shape[0]
            tau = seq.Delta - seq.delta
            
            diag_L = torch.diag(lambda_mat) + torch.diag(relax_mat)
            E2 = torch.diag(torch.exp(-tau * diag_L))
            # E2 = torch.linalg.matrix_exp(-(seq.Delta - seq.delta) * (lambda_mat + relax_mat))

            # nu = E1^H @ (E2 @ (E1 @ nu))
            tmp1 = torch.matmul(E1, nu)  # [n_amp, n_dir, n_eig, 1]
            tmp2 = torch.matmul(E2.unsqueeze(0).unsqueeze(0), tmp1)  # broadcast E2
            E1_H = torch.conj(E1).transpose(-1, -2)  # [n_amp, n_dir, n_eig, n_eig]
            nu_final = torch.matmul(E1_H, tmp2)  # [n_amp, n_dir, n_eig, 1]

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

    return {
        "magnetization": magnetization,
        "signal": signal,
        "signal_allcmpts": signal.sum(dim=0),
        "itertimes": itertimes,
        "time_taken": time.time() - t_start,
    }