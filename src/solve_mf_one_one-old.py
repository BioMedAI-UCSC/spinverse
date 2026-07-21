import time
import torch
from mesh_setup.PGSE import PGSE
from src.sparse_block_diagonal import sparse_block_diagonal
from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D import mass_matrixP1_3D

def solve_mf(femesh, setup, lap_eig, faces_prob,
             seq_idx, dir_idx):
    """
    Compute the solution to the BTPDE using Matrix Formalism.
    If seq_idx is specified, only that sequence is run.
    If dir_idx is specified, only that direction is run.
    All amplitudes are always computed.
    """
    start_time_all = time.time()

    # Extract parameters
    initial_density = setup.pde["initial_density"]
    qvalues = setup.gradient["qvalues"]
    bvalues = setup.gradient["bvalues"]
    sequences = setup.gradient["sequences"]
    directions = setup.gradient["directions"]
    eigfuncs = lap_eig["funcs"]
    moments = lap_eig["moments"]
    T2 = lap_eig["massrelax"]
    L = torch.diag(lap_eig["values"])

    # Sizes
    ncompartment = femesh["ncompartment"]
    namplitude = len(setup.gradient["values"])
    nsequence = len(sequences)
    ndirection = directions.shape[1]
    ninterval = setup.mf["ninterval"]

    # Points per compartment
    npoint_cmpts = [x.shape[1] for x in femesh["points"]]

    # Allocate outputs
    magnetization = [
        [
            [[[] for _ in range(ndirection)] for _ in range(nsequence)]
            for _ in range(namplitude)
        ]
        for _ in range(ncompartment)
    ]
    signal = torch.zeros(ncompartment, namplitude, nsequence, ndirection, dtype=torch.complex64)
    itertimes = torch.zeros(namplitude, nsequence, ndirection)

    # Assemble mass matrices and initial rho
    M_cmpts, rho_cmpts = [], []
    for icmpt in range(ncompartment):
        pts = femesh["points"][icmpt]
        elems = femesh["elements"][icmpt]
        _, volumes, _ = get_volume_mesh(pts, elems)
        M_cmpts.append(mass_matrixP1_3D(elems, volumes))
        rho_cmpts.append(torch.full((pts.shape[1],),
                                    fill_value=initial_density[icmpt],
                                    dtype=torch.complex64))
    M = sparse_block_diagonal(M_cmpts)
    rho = torch.cat(rho_cmpts, dim=0).unsqueeze(1)
    nu0 = eigfuncs.t().to(torch.complex64) @ (
        M.to_dense().to(torch.complex64) @ rho.to_dense()
    )
    
    # return nu0
    breakpoint()

    # Build index lists: all amplitudes, but optionally single seq & dir
    amps = range(namplitude)
    seqs = [seq_idx] if seq_idx is not None else range(nsequence)
    dirs = [dir_idx] if dir_idx is not None else range(ndirection)

    # Main loop: A × (1 or S) × (1 or D)
    for iamp in amps:
        for iseq in seqs:
            for idir in dirs:
                itertime = time.time()

                # experiment parameters
                q = qvalues[iamp, iseq]
                seq = sequences[iseq]
                g = directions[:, idir]

                # time intervals
                time_intervals = torch.linspace(0, seq.echotime, ninterval + 1)

                # BT‐operator component
                A = (moments * g.unsqueeze(0).unsqueeze(0)).sum(dim=2)

                # Compute nu
                if isinstance(seq, PGSE):
                    K = L + T2 + 1j * q * A
                    edK = torch.matrix_exp(-seq.delta * K)
                    edL = torch.matrix_exp(-(seq.Delta - seq.delta) * (L + T2))
                    nu = edK.conj().t() @ (edL.to(torch.complex64) @ (edK @ nu0))
                else:
                    def K_ft(ft):
                        return L + T2 + 1j * q * ft * A
                    nu = nu0.clone()
                    for i in range(ninterval):
                        dt = time_intervals[i + 1] - time_intervals[i]
                        ft = (seq.call(time_intervals[i + 1]) +
                              seq.call(time_intervals[i])) / 2
                        nu = torch.mm(torch.matrix_exp(-dt * K_ft(ft)), nu)

                # Back to nodal basis
                mag = eigfuncs.to(torch.complex64) @ nu
                mag_cmpts = torch.split(mag, npoint_cmpts)
                for icmpt, mag_c in enumerate(mag_cmpts):
                    magnetization[icmpt][iamp][iseq][idir] = mag_c.squeeze()

                # Compute compartment signals
                M_cmpts_cmplx = [Mc.to_dense().to(torch.complex64) for Mc in M_cmpts]
                for icmpt in range(ncompartment):
                    signal[icmpt, iamp, iseq, idir] = torch.sum(
                        torch.mm(M_cmpts_cmplx[icmpt], mag_cmpts[icmpt])
                    )

                itertimes[iamp, iseq, idir] = time.time() - itertime

    # Sum over compartments
    signal_allcmpts = signal.sum(dim=0)

    # For flattening
    # signal_allcmpts_flattened = signal_allcmpts.flatten()
    
    # return signal_allcmpts

    results_mf = {
        "magnetization": magnetization,
        "signal": signal,
        "signal_allcmpts": signal_allcmpts,
        "itertimes": itertimes,
        "time_taken": time.time() - start_time_all,
    }

    return results_mf
