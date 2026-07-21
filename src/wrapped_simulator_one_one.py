from src.get_vol_sa import get_vol_sa
from src.compute_adc_sta import compute_adc_sta
from src.calculate_generalized_mean_diffusivity import calculate_generalized_mean_diffusivity
from src.length2eig import length2eig
from src.compute_laplace_eig_diff_v2 import compute_laplace_eig_diff
from src.eig2length import eig2length
from src.solve_mf_v4 import solve_mf
# from src.solve_mf_one_one import solve_mf
from src.split_mesh import split_mesh
import torch as tch
import logging

# Assuming a logger is set up elsewhere; if not, configure it here
logger = logging.getLogger("mvrecon_3d")

def wrapped_simulator(femesh_all, setup, faces_prob, seq_idx=None, dir_idx=None):
    # Initial input check
    logger.info(f"[wrapped_simulator] faces_prob grad: {faces_prob.requires_grad}")
    # logger.info(f"[wrapped_simulator] faces_prob grad_fn: {faces_prob.grad_fn}")
    # logger.info(f"[wrapped_simulator] femesh_all['points'] requires_grad: {femesh_all['points'].requires_grad}")
    breakpoint() 
    # Step 1: split_mesh
    femesh_all_2_split = split_mesh(femesh_all)
    # logger.info(f"[wrapped_simulator] After split_mesh: femesh_all_2_split['points'] requires_grad: {femesh_all_2_split['points'][0].requires_grad}")
    # logger.info(f"[wrapped_simulator] After split_mesh: femesh_all_2_split['points'].grad_fn: {femesh_all_2_split['points'][0].grad_fn is not None}")

    # breakpoint()
    
    # Step 2: get_vol_sa
    volumes, surface_areas = get_vol_sa(femesh_all_2_split, faces_prob)
    # logger.info(f"[wrapped_simulator] After get_vol_sa: volumes.grad_fn: {volumes.grad_fn}")
    # logger.info(f"[wrapped_simulator] After surface_areas: surface_areas.grad_fn: {surface_areas.grad_fn}")
    # logger.info(f"[wrapped_simulator] After surface_areas: surface_areas.requires_grad: {surface_areas.requires_grad}")

    # return surface_areas

    # Step 3: calculate_generalized_mean_diffusivity
    mean_diffusivity = calculate_generalized_mean_diffusivity(setup.pde['diffusivity'], volumes)
    # logger.info(f"[wrapped_simulator] After calculate_generalized_mean_diffusivity: mean_diffusivity.grad_fn: {mean_diffusivity.grad_fn}")

    # Step 4: length2eig
    eiglim = length2eig(setup.mf['length_scale'], mean_diffusivity)
    # logger.info(f"[wrapped_simulator] After length2eig: eiglim.grad_fn: {eiglim.grad_fn}")

    # Step 5: compute_laplace_eig_diff
    lap_eig = compute_laplace_eig_diff(femesh_all_2_split, setup, setup.pde, eiglim, setup.mf['neig_max'], faces_prob)
    # logger.info(f"[wrapped_simulator] After compute_laplace_eig_diff: lap_eig['funcs'].grad_fn: {lap_eig['funcs'].grad_fn}")
    # logger.info(f"[wrapped_simulator] After compute_laplace_eig_diff: lap_eig['funcs'].grad: {lap_eig['funcs'].grad}")
    # logger.info(f"[wrapped_simulator] After compute_laplace_eig_diff: lap_eig['funcs'].requires_grad: {lap_eig['funcs'].requires_grad}")
    
    # return lap_eig

    # Step 6: eig2length
    lap_eig['length_scales'] = eig2length(lap_eig['values'], mean_diffusivity)
    
    # return lap_eig['length_scales']

    # logger.info(f"[wrapped_simulator] After eig2length: lap_eig['length_scales'].grad_fn: {lap_eig['length_scales'].grad_fn}")

    # Step 7: solve_mf
    # mf_signal = solve_mf(
    #     femesh_all_2_split, setup, lap_eig, faces_prob,
    #     seq_idx=seq_idx, dir_idx=dir_idx
    # )
    mf_signal = solve_mf(
        femesh_all_2_split, setup, lap_eig, faces_prob=faces_prob)
    
    # return mf_signal

    # logger.info(f"[wrapped_simulator] After solve_mf: mf_signal['signal_allcmpts'].grad_fn: {mf_signal['signal_allcmpts'].grad_fn}")
    # logger.info(f"[wrapped_simulator] After solve_mf: mf_signal component .grad_fn: {mf_signal['signal_allcmpts'].grad_fn}")

    # Final computation
    signal = tch.abs(tch.abs(tch.abs(mf_signal["signal_allcmpts"]) / tch.abs(mf_signal["signal_allcmpts"][0, :, 0]).view(1, -1, 1).clamp(min=1e-6))) * 100
    logger.info(f"[wrapped_simulator] Final signal.grad_fn: {signal.grad_fn}")

    with tch.no_grad():
        logger.info(f"[wrapped_simulator] Max signal value: {tch.max(signal)}")

    return signal
