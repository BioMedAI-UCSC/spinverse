import torch
from src.get_volume_mesh import get_volume_mesh
from src.get_surface_mesh import get_surface_mesh
import logging

# Assuming a logger is set up elsewhere; if not, configure it here
logger = logging.getLogger("mvrecon_3d")

def get_vol_sa(femesh, faces_prob = None):
    ncompartment = len(femesh["points"])
    volumes = torch.zeros(ncompartment, dtype=torch.float32)
    surface_areas = torch.zeros(ncompartment, dtype=torch.float32)
    
    # logger.info(f"[get_vol_sa] After get_vol_sa: faces_prob.grad_fn: {faces_prob.grad_fn}")
    # logger.info(f"[get_vol_sa] After get_vol_sa: faces_prob.requires_grad: {faces_prob.requires_grad}")

    # breakpoint()
    for i in range(ncompartment):
        points = femesh["points"][i]
        elements = femesh["elements"][i]
        facets = femesh["facets"][i]

        if facets is not None and len(facets) > 0:
            total_volume, _, _ = get_volume_mesh(points, elements)
            volumes[i] = total_volume

            # breakpoint()
            
            total_area, _, _, _ = get_surface_mesh(points, facets, faces_prob)
            surface_areas[i] = total_area
            
            # logger.info(f"[get_vol_sa] After get_vol_sa: total_area.grad_fn: {total_area.grad_fn}")
            # logger.info(f"[get_vol_sa] After get_vol_sa: total_area.requires_grad: {total_area.requires_grad}")
        else:
            # Handle the case where no facets are defined for a compartment
            volumes[i], _, _ = get_volume_mesh(points, elements)
            surface_areas[i] = 0

    return volumes, surface_areas
