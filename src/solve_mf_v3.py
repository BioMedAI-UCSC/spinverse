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

    # Unpack PDE & gradient settings
    init_density = setup.pde["initial_density"]       # list length n_comp
    q_values     = setup.gradient["qvalues"]         # [n_amp, n_seq]
    b_values     = setup.gradient["bvalues"]         # [n_amp, n_seq]
    sequences    = setup.gradient["sequences"]       # list length n_seq
    directions   = setup.gradient["directions"]      # [3, n_dir]

    # Promote everything that goes into @ or expm to complex64
    eig_funcs   = lap_eig["funcs"].to(COMPLEX)       # [n_point, n_eig]
    moments     = lap_eig["moments"].to(COMPLEX)     # [n_eig, n_eig, 3]
    relax_mat   = lap_eig["massrelax"].to(COMPLEX)   # [n_eig, n_eig]
    lambda_mat  = torch.diag(lap_eig["values"]).to(COMPLEX)  # [n_eig, n_eig]

    # Sizes
    n_comp = femesh["ncompartment"]
    n_amp  = len(setup.gradient["values"])
    n_seq  = len(sequences)
    n_dir  = directions.shape[1]
    n_int  = setup.mf["ninterval"]

    # Assemble FEM mass blocks & initial ν₀
    mass_blocks, rho_blocks, pts_per_comp = [], [], []
    for c in range(n_comp):
        pts   = femesh["points"][c]
        elems = femesh["elements"][c]
        _, vols, _ = get_volume_mesh(pts, elems)
        Mblk = mass_matrixP1_3D(elems, vols).to(COMPLEX)
        mass_blocks.append(Mblk)
        rho_blocks.append(torch.full((pts.shape[1],), init_density[c], dtype=COMPLEX))
        pts_per_comp.append(pts.shape[1])

    # Big block-diagonal mass & initial coeffs
    Mbig = sparse_block_diagonal(mass_blocks)
    rho0 = torch.cat(rho_blocks).unsqueeze(1)          # [sum_pts × 1]
    H    = torch.conj(eig_funcs).T                    # Hermitian transpose
    nu0  = H @ (Mbig.to_dense() @ rho0)               # [n_eig × 1]

    # Keep dense mass blocks for final signal integration
    dense_mass = [blk.to_dense() for blk in mass_blocks]

    # Initialize output containers
    magnetization = [
        [[[None]*n_dir for _ in range(n_seq)] for _ in range(n_amp)]
        for _ in range(n_comp)
    ]
    signal = torch.zeros(n_comp, n_amp, n_seq, n_dir, dtype=COMPLEX)
    itertimes = torch.zeros(n_amp, n_seq, n_dir, dtype=torch.float32)

    # Triple loop over all amplitudes, sequences, and directions
    for a in range(n_amp):
        for s in range(n_seq):
            for d in range(n_dir):
                t0 = time.time()

                # Build the BT operator component
                q   = q_values[a, s]
                seq = sequences[s]
                g   = directions[:, d]
                A   = (moments * g.unsqueeze(0).unsqueeze(0)).sum(dim=2)

                # Evolve ν in Laplace basis
                if isinstance(seq, PGSE):
                    K1 = lambda_mat + relax_mat + 1j*q*A
                    E1 = torch.matrix_exp(-seq.delta * K1)
                    E2 = torch.matrix_exp(-(seq.Delta - seq.delta)*(lambda_mat + relax_mat))
                    nu = E1.conj().T @ (E2 @ (E1 @ nu0))
                else:
                    nu = nu0.clone()
                    tgrid = torch.linspace(0, seq.echotime, n_int+1)
                    for k in range(n_int):
                        dt = tgrid[k+1] - tgrid[k]
                        ft = 0.5*(seq.call(tgrid[k+1]) + seq.call(tgrid[k]))
                        Kt = lambda_mat + relax_mat + 1j*q*ft*A
                        nu = torch.matrix_exp(-dt * Kt) @ nu

                # Back to spatial basis & split by compartment
                mag_full  = eig_funcs @ nu
                mag_parts = torch.split(mag_full, pts_per_comp)

                # Store magnetization and compute signal
                for c in range(n_comp):
                    magnetization[c][a][s][d] = mag_parts[c].squeeze()
                    signal[c, a, s, d] = (dense_mass[c] @ mag_parts[c]).sum()

                itertimes[a, s, d] = time.time() - t0

    return {
        "magnetization": magnetization,
        "signal": signal,
        "signal_allcmpts": signal.sum(dim=0),
        "itertimes": itertimes,
        "time_taken": time.time() - t_start,
    }