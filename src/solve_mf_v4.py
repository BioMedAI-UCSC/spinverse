import time
import torch
from mesh_setup.PGSE import PGSE
from src.sparse_block_diagonal import sparse_block_diagonal
from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D import mass_matrixP1_3D

def solve_mf(femesh, setup, lap_eig, faces_prob=None):
    """
    Matrix-formalism Bloch-Torrey solver (differentiable).
    Computes signal loss for all amplitudes, sequences, and directions.
    """
    t_start = time.time()
    COMPLEX = torch.complex64
    
    # Get device from first available tensor
    device = None
    if "points" in femesh and len(femesh["points"]) > 0:
        device = femesh["points"][0].device
    elif "funcs" in lap_eig:
        device = lap_eig["funcs"].device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # print(f"solve_mf using device: {device}")

    # Unpack PDE & gradient settings - ENSURE ON DEVICE
    init_density = setup.pde["initial_density"]       # list length n_comp
    q_values     = setup.gradient["qvalues"].to(device)         # [n_amp, n_seq]
    b_values     = setup.gradient["bvalues"].to(device)         # [n_amp, n_seq]
    sequences    = setup.gradient["sequences"]       # list length n_seq
    directions   = setup.gradient["directions"].to(device)      # [3, n_dir]

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

    # print(f"Tensor devices check:")
    # print(f"  eig_funcs: {eig_funcs.device}, shape: {eig_funcs.shape}")
    # print(f"  moments: {moments.device}, shape: {moments.shape}")
    # print(f"  directions: {directions.device}, shape: {directions.shape}")
    # print(f"  lambda_mat: {lambda_mat.device}, shape: {lambda_mat.shape}")

    # Assemble FEM mass blocks & initial ν₀
    mass_blocks, rho_blocks, pts_per_comp = [], [], []
    for c in range(n_comp):
        pts   = femesh["points"][c].to(device)  # Ensure on device
        elems = femesh["elements"][c].to(device)  # Ensure on device
        _, vols, _ = get_volume_mesh(pts, elems)
        Mblk = mass_matrixP1_3D(elems, vols).to(dtype=COMPLEX, device=device)
        mass_blocks.append(Mblk)
        rho_blocks.append(torch.full((pts.shape[1],), init_density[c], dtype=COMPLEX, device=device))
        pts_per_comp.append(pts.shape[1])

    # Big block-diagonal mass & initial coeffs
    Mbig = sparse_block_diagonal(mass_blocks)
    rho0 = torch.cat(rho_blocks).unsqueeze(1).to(device)          # [sum_pts × 1]
    H    = torch.conj(eig_funcs).T                    # Hermitian transpose
    nu0  = H @ (Mbig.to_dense().to(device) @ rho0)               # [n_eig × 1]

    # Keep dense mass blocks for final signal integration
    dense_mass = [blk.to_dense().to(device) for blk in mass_blocks]

    # Initialize output containers ON DEVICE
    magnetization = [
        [[[None]*n_dir for _ in range(n_seq)] for _ in range(n_amp)]
        for _ in range(n_comp)
    ]
    signal = torch.zeros(n_comp, n_amp, n_seq, n_dir, dtype=COMPLEX, device=device)
    itertimes = torch.zeros(n_amp, n_seq, n_dir, dtype=torch.float32, device=device)

    # Triple loop over all amplitudes, sequences, and directions
    for a in range(n_amp):
        for s in range(n_seq):
            for d in range(n_dir):
                t0 = time.time()

                # Build the BT operator component - ENSURE ALL ON SAME DEVICE
                q   = q_values[a, s].to(device)  # Ensure scalar is on device
                seq = sequences[s]
                g   = directions[:, d].to(device)  # Ensure direction vector is on device
                
                # Debug print for A computation
                # print(f"Computing A for a={a}, s={s}, d={d}")
                # print(f"  moments device: {moments.device}, shape: {moments.shape}")
                # print(f"  g device: {g.device}, shape: {g.shape}")
                
                # Compute A carefully - ensure all operations on same device
                g_unsqueezed = g.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 3]
                moments_g = moments * g_unsqueezed  # [n_eig, n_eig, 3]
                A = moments_g.sum(dim=2).to(device)  # [n_eig, n_eig]
                
                # print(f"  A device: {A.device}, shape: {A.shape}")

                # Evolve ν in Laplace basis
                if isinstance(seq, PGSE):
                    # Ensure all components are on same device
                    K1 = lambda_mat.to(device) + relax_mat.to(device) + 1j*q.to(device)*A.to(device)
                    E1 = torch.matrix_exp(-seq.delta * K1)
                    E2 = torch.matrix_exp(-(seq.Delta - seq.delta)*(lambda_mat.to(device) + relax_mat.to(device)))
                    nu = E1.conj().T @ (E2 @ (E1 @ nu0.to(device)))
                else:
                    nu = nu0.clone().to(device)
                    tgrid = torch.linspace(0, seq.echotime, n_int+1, device=device)
                    for k in range(n_int):
                        dt = tgrid[k+1] - tgrid[k]
                        ft = 0.5*(seq.call(tgrid[k+1]) + seq.call(tgrid[k]))
                        ft = torch.tensor(ft, dtype=COMPLEX, device=device)  # Ensure ft is on device
                        Kt = lambda_mat.to(device) + relax_mat.to(device) + 1j*q.to(device)*ft*A.to(device)
                        nu = torch.matrix_exp(-dt * Kt) @ nu

                # Back to spatial basis & split by compartment
                mag_full  = eig_funcs.to(device) @ nu.to(device)
                mag_parts = torch.split(mag_full, pts_per_comp)

                # Store magnetization and compute signal
                for c in range(n_comp):
                    magnetization[c][a][s][d] = mag_parts[c].squeeze()
                    signal[c, a, s, d] = (dense_mass[c].to(device) @ mag_parts[c].to(device)).sum()

                itertimes[a, s, d] = time.time() - t0

    return {
        "magnetization": magnetization,
        "signal": signal,
        "signal_allcmpts": signal.sum(dim=0),
        "itertimes": itertimes,
        "time_taken": time.time() - t_start,
    }