import time
import torch
from mesh_setup.PGSE import PGSE
from src.sparse_block_diagonal import sparse_block_diagonal
from src.get_volume_mesh import get_volume_mesh
# from src.mass_matrixP1_3D import mass_matrixP1_3D
from src.mass_matrixP1_3D_e_v3 import mass_matrixP1_3D

def solve_mf(femesh, setup, lap_eig, target_seq=None, faces_prob=None):
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
    directions   = setup.gradient["directions"].to(dtype=COMPLEX, device=device)      # [3, n_dir]

    # Handle sequence filtering
    all_sequences = setup.gradient["sequences"]
    if target_seq is not None:
        # Convert single int to list
        if isinstance(target_seq, int):
            seq_indices = [target_seq]
        else:
            seq_indices = list(target_seq)
        
        sequences = [all_sequences[i] for i in seq_indices]
        q_values = q_values[:, seq_indices]  # Filter q_values to match
        b_values = b_values[:, seq_indices]  # Filter b_values to match
    else:
        sequences = all_sequences
        seq_indices = list(range(len(all_sequences)))

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

    # Power spectrum for logging (not used in main computation)
    mode_power_mean = torch.zeros((n_amp, n_seq, lap_eig["values"].numel()), device=device, dtype=torch.float32)
    mode_power_var  = torch.zeros_like(mode_power_mean)

    # Loop over sequences (kept due to potential varying types; batched over amp/dir/comps)
    for i_seq, seq in enumerate(sequences):
        t0 = time.time()

        q_as = q_values[:, i_seq][:, None, None, None]  # [n_amp, 1, 1, 1]
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
            E1 = torch.linalg.matrix_exp((-seq.delta * K1).contiguous())

            # E2 is independent of amp/dir
            E2 = torch.linalg.matrix_exp(-(seq.Delta - seq.delta) * (lambda_mat + relax_mat))

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

        # Direction-aggregated spectrum per (amp, eig)
        # For logging only:
        with torch.no_grad():
            nu_vec = nu_final.squeeze(-1)     # [n_amp, n_dir, n_eig]
            power = (nu_vec.abs() ** 2)       # nonnegative
            power_dir_mean = power.mean(dim=1)  # [n_amp, n_eig]
            power_dir_var  = power.var(dim=1, unbiased=False)# [n_amp, n_eig]

            # store into arrays for logging
            mode_power_mean[:, i_seq, :] = power_dir_mean
            mode_power_var[:, i_seq, :] = power_dir_var

        # Back to spatial basis: mag_full = eig_funcs @ nu_final
        mag_full = torch.matmul(eig_funcs.unsqueeze(0).unsqueeze(0), nu_final).squeeze(-1)  # [n_amp, n_dir, n_point]

        # Split by compartment and stack
        mag_parts = torch.split(mag_full, n_pts, dim=-1)  # list of [n_amp, n_dir, n_pts]
        mag_stack = torch.stack(mag_parts, dim=0)  # [n_comp, n_amp, n_dir, n_pts]

        # Store magnetization
        magnetization[:, :, i_seq, :, :] = mag_stack

        # Compute signal batched
        mag_c = mag_stack.unsqueeze(-1)  # [n_comp, n_amp, n_dir, n_pts, 1]
        integ = torch.matmul(dense_mass_stack.unsqueeze(1).unsqueeze(2), mag_c)  # [n_comp, n_amp, n_dir, n_pts, 1]
        signal_c = integ.sum(dim=-2).squeeze(-1)  # [n_comp, n_amp, n_dir]
        signal[:, :, i_seq, :] = signal_c

        # Approximate iteration times
        dt = time.time() - t0
        itertimes[:, i_seq, :] = dt / (n_amp * n_dir)

    return {
        "magnetization": magnetization,
        "signal": signal,
        "signal_allcmpts": signal.sum(dim=0),
        "itertimes": itertimes,
        "time_taken": time.time() - t_start,
        "mode_power_mean": mode_power_mean,   # [n_amp, n_seq, n_eig]
        "mode_power_var":  mode_power_var,    # [n_amp, n_seq, n_eig]
        "seq_indices": seq_indices,           # optional, for mapping back
    }

def solve_mf_reduced(femesh, setup, lap_red, target_seq=None):
    """
    Reduced-order MF Bloch–Torrey solver using projected operators.

    Expects lap_red to contain:
      - "funcs":      Phi, (N, K) basis functions in full space
      - "Mr":        (K, K) reduced mass matrix
      - "Ar":        (K, K) reduced diffusion+flux operator (i.e., Kr + Fr(p))
      - "moments":   (K, K, 3) reduced moment matrices (Phi^H Jx Phi)
      - "massrelax": (K, K) reduced relaxation matrix (Phi^H R Phi)

    Returns dict with the SAME keys/shapes as solve_mf().
    """
    t_start = time.time()
    COMPLEX = torch.complex64

    # Device
    device = lap_red["Mr"].device

    # Unpack experiment
    init_density = setup.pde["initial_density"]
    q_values     = setup.gradient["qvalues"].to(device)          # [n_amp, n_seq_total]
    b_values     = setup.gradient["bvalues"].to(device)          # [n_amp, n_seq_total]
    directions   = setup.gradient["directions"].to(dtype=COMPLEX, device=device)  # [3, n_dir]
    all_sequences = setup.gradient["sequences"]

    # Handle sequence filtering
    if target_seq is not None:
        if isinstance(target_seq, int):
            seq_indices = [target_seq]
        else:
            seq_indices = list(target_seq)
        sequences = [all_sequences[i] for i in seq_indices]
        q_values = q_values[:, seq_indices]
        b_values = b_values[:, seq_indices]
    else:
        sequences = all_sequences
        seq_indices = list(range(len(all_sequences)))

    # Reduced operators / basis
    Phi        = lap_red["funcs"].to(dtype=COMPLEX, device=device)       # (N, K)
    Mr         = lap_red["Mr"].to(dtype=COMPLEX, device=device)          # (K, K)
    Ar         = lap_red["Ar"].to(dtype=COMPLEX, device=device)          # (K, K)
    moments    = lap_red["moments"].to(dtype=COMPLEX, device=device)     # (K, K, 3)
    relax_mat  = lap_red["massrelax"].to(dtype=COMPLEX, device=device)   # (K, K)

    # Sizes
    n_comp = femesh["ncompartment"]
    n_amp  = len(setup.gradient["values"])
    n_seq  = len(sequences)
    n_dir  = directions.shape[1]
    n_int  = setup.mf["ninterval"]

    # FEM mass blocks & initial density rho0 in full space
    n_pts = femesh["points"][0].shape[1]
    mass_blocks = [
        mass_matrixP1_3D(
            femesh["elements"][c].to(device),
            get_volume_mesh(femesh["points"][c].to(device), femesh["elements"][c].to(device))[1],
        ).to(device)
        for c in range(n_comp)
    ]

    rho0 = torch.cat(
        [torch.full((n_pts, 1), init_density[c], dtype=COMPLEX, device=device) for c in range(n_comp)],
        dim=0,
    )  # (N, 1), N = n_comp*n_pts

    Mbig = sparse_block_diagonal(mass_blocks).to(dtype=COMPLEX, device=device).coalesce()

    # Project initial condition into reduced coords:
    # rhs = Phi^H (M rho0),  nu0 = Mr^{-1} rhs
    Mrho0 = torch.sparse.mm(Mbig, rho0)                     # (N, 1)
    rhs   = torch.conj(Phi).T @ Mrho0                       # (K, 1)
    nu0   = torch.linalg.solve(Mr, rhs)                     # (K, 1)

    # Dense mass blocks for signal integration
    dense_mass_stack = torch.stack([blk.to_dense().to(device) for blk in mass_blocks]).to(dtype=COMPLEX)

    # Output buffers
    magnetization = torch.zeros((n_comp, n_amp, n_seq, n_dir, n_pts), dtype=COMPLEX, device=device)
    signal        = torch.zeros((n_comp, n_amp, n_seq, n_dir), dtype=COMPLEX, device=device)
    itertimes     = torch.zeros((n_amp, n_seq, n_dir), dtype=torch.float32, device=device)

    # Logging spectra (reduced coefficients)
    Kdim = Mr.shape[0]
    mode_power_mean = torch.zeros((n_amp, n_seq, Kdim), device=device, dtype=torch.float32)
    mode_power_var  = torch.zeros_like(mode_power_mean)

    # Precompute A_r for all directions: A_dir = sum_j moments[:,:,j] * dir[j]
    # moments: (K,K,3), directions: (3,n_dir) => A_all: (K,K,n_dir)
    A_all = torch.einsum("ijk,kl->ijl", moments, directions)  # (K, K, n_dir)

    # Helper: build batched Mr^{-1} X via solve (broadcasting)
    def solve_M(X: torch.Tensor) -> torch.Tensor:
        # X: (..., K, K) or (..., K, 1), returns (..., K, K) or (..., K, 1)
        # torch.linalg.solve supports broadcasting if Mr has same trailing dims
        Mr_b = Mr
        # Add leading singleton dims so it broadcasts to X
        while Mr_b.ndim < X.ndim:
            Mr_b = Mr_b.unsqueeze(0)
        return torch.linalg.solve(Mr_b, X)

    # Base operator (no gradient): Mr^{-1}(Ar + Rr)
    K0 = solve_M(Ar + relax_mat)  # (K, K)

    for i_seq, seq in enumerate(sequences):
        t0 = time.time()

        q_as = q_values[:, i_seq][:, None, None, None].to(device)   # (n_amp,1,1,1)

        # Batched A over dirs: (1,n_dir,K,K)
        A_batch = A_all.permute(2, 0, 1).unsqueeze(0)               # (1,n_dir,K,K)

        # Expand reduced ops to batched shapes
        Ar_b = Ar.unsqueeze(0).unsqueeze(0)                         # (1,1,K,K)
        Rr_b = relax_mat.unsqueeze(0).unsqueeze(0)                  # (1,1,K,K)

        # Batch nu0 over amp,dir
        nu = nu0.unsqueeze(0).unsqueeze(0).expand(n_amp, n_dir, -1, -1).clone()  # (n_amp,n_dir,K,1)

        if isinstance(seq, PGSE):
            # K1 = Mr^{-1}(Ar + Rr + i q A)
            X1 = Ar_b + Rr_b + 1j * q_as * A_batch                  # (n_amp,n_dir,K,K) via broadcast
            K1 = solve_M(X1)                                        # (n_amp,n_dir,K,K)

            E1 = torch.linalg.matrix_exp((-seq.delta * K1).contiguous())

            # E2 uses gradient-off evolution: exp(-(Delta-delta) * K0)
            E2 = torch.linalg.matrix_exp((-(seq.Delta - seq.delta) * K0).contiguous())  # (K,K)

            tmp1 = torch.matmul(E1, nu)  # (n_amp,n_dir,K,1)
            tmp2 = torch.matmul(E2.unsqueeze(0).unsqueeze(0), tmp1)
            E1_H = torch.conj(E1).transpose(-1, -2)
            nu_final = torch.matmul(E1_H, tmp2)

        else:
            tgrid = torch.linspace(0, seq.echotime, n_int + 1, device=device)
            for k in range(n_int):
                dt = tgrid[k + 1] - tgrid[k]
                ft = 0.5 * (seq.call(tgrid[k + 1]) + seq.call(tgrid[k]))
                ft = torch.tensor(ft, dtype=torch.float32, device=device)

                Xt = Ar_b + Rr_b + 1j * q_as * ft * A_batch          # (n_amp,n_dir,K,K)
                Kt = solve_M(Xt)                                     # (n_amp,n_dir,K,K)

                exp_dt = torch.linalg.matrix_exp((-dt * Kt).contiguous())
                nu = torch.matmul(exp_dt, nu)

            nu_final = nu

        # Logging only: coefficient power across dirs
        with torch.no_grad():
            nu_vec = nu_final.squeeze(-1)            # (n_amp,n_dir,K)
            power = (nu_vec.abs() ** 2)
            mode_power_mean[:, i_seq, :] = power.mean(dim=1)
            mode_power_var[:,  i_seq, :] = power.var(dim=1, unbiased=False)

        # Reconstruct full-space magnetization: m = Phi nu
        mag_full = torch.matmul(Phi.unsqueeze(0).unsqueeze(0), nu_final).squeeze(-1)  # (n_amp,n_dir,N)

        # Split by compartment, store
        mag_parts = torch.split(mag_full, n_pts, dim=-1)   # list of (n_amp,n_dir,n_pts)
        mag_stack = torch.stack(mag_parts, dim=0)          # (n_comp,n_amp,n_dir,n_pts)
        magnetization[:, :, i_seq, :, :] = mag_stack

        # Signal: integrate per compartment using mass matrices
        mag_c = mag_stack.unsqueeze(-1)  # (n_comp,n_amp,n_dir,n_pts,1)
        integ = torch.matmul(dense_mass_stack.unsqueeze(1).unsqueeze(2), mag_c)  # (n_comp,n_amp,n_dir,n_pts,1)
        signal_c = integ.sum(dim=-2).squeeze(-1)  # (n_comp,n_amp,n_dir)
        signal[:, :, i_seq, :] = signal_c

        dt = time.time() - t0
        itertimes[:, i_seq, :] = dt / (n_amp * n_dir)

    return {
        "magnetization": magnetization,
        "signal": signal,
        "signal_allcmpts": signal.sum(dim=0),
        "itertimes": itertimes,
        "time_taken": time.time() - t_start,
        "mode_power_mean": mode_power_mean,   # (n_amp,n_seq,K)
        "mode_power_var":  mode_power_var,    # (n_amp,n_seq,K)
        "seq_indices": seq_indices,
    }