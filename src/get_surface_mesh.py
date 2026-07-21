# import torch


# def get_surface_mesh(points, facets, faces_prob = None):
#     if facets is None or len(facets) == 0:
#         return 0, torch.tensor([]), torch.tensor([]), torch.tensor([])

#     total_area = 0
#     all_areas = []
#     all_centers = []
#     all_normals = []

#     for facet_sublist in facets:
#         if facet_sublist is None or len(facet_sublist) == 0:
#             continue

#         facet_sublist = facet_sublist.to(dtype=torch.long)

#         nfacet = facet_sublist.shape[1]
#         areas = torch.zeros(nfacet, dtype=points.dtype, device=points.device)
#         centers = torch.zeros((3, nfacet), dtype=points.dtype, device=points.device)
#         normals = torch.zeros((3, nfacet), dtype=points.dtype, device=points.device)

#         for i in range(nfacet):
#             tri = points[:, facet_sublist[:, i]]
#             normal = torch.linalg.cross(tri[:, 0] - tri[:, 1], tri[:, 2] - tri[:, 1])
#             area = torch.norm(normal) / 2
#             center = torch.mean(tri, dim=1)

#             norm = torch.norm(normal)
#             if norm != 0:
#                 normal = normal / norm

#             normals[:, i] = normal
#             areas[i] = area
#             centers[:, i] = center

#         total_area += torch.sum(areas)
#         all_areas.append(areas)
#         all_centers.append(centers)
#         all_normals.append(normals)

#     all_areas = torch.cat(all_areas, dim=0)
#     all_centers = torch.cat(all_centers, dim=1)
#     all_normals = torch.cat(all_normals, dim=1)

#     return total_area, all_areas, all_centers, all_normals

import torch
import logging

# Assuming a logger is set up elsewhere; if not, configure it here
logger = logging.getLogger("mvrecon_3d")

def get_surface_mesh(points, facets, faces_prob=None):
    
    # logger.info(f"[get_surface_mesh] After get_surface_mesh: faces_prob.grad_fn: {faces_prob.grad_fn}")
    # logger.info(f"[get_surface_mesh] After get_surface_mesh: faces_prob.requires_grad: {faces_prob.requires_grad}")
    
    if facets is None or len(facets) == 0:
        return 0, torch.tensor([]), torch.tensor([]), torch.tensor([])

    total_area = 0
    all_areas = []
    all_centers = []
    all_normals = []
    face_counter = 0  # counter to index faces_prob if provided

    for facet_sublist in facets:
        if facet_sublist is None or len(facet_sublist) == 0:
            continue

        facet_sublist = facet_sublist.to(dtype=torch.long)
        nfacet = facet_sublist.shape[1]
        areas = torch.zeros(nfacet, dtype=points.dtype, device=points.device)
        centers = torch.zeros((3, nfacet), dtype=points.dtype, device=points.device)
        normals = torch.zeros((3, nfacet), dtype=points.dtype, device=points.device)

        for i in range(nfacet):
            tri = points[:, facet_sublist[:, i]]
            normal = torch.linalg.cross(tri[:, 0] - tri[:, 1], tri[:, 2] - tri[:, 1])
            area = torch.norm(normal) / 2.0

            # If faces_prob is provided, weight the area accordingly.
            # Make sure faces_prob is differentiable and matches the face ordering.
            area = area * faces_prob[face_counter]
            face_counter += 1

            center = torch.mean(tri, dim=1)

            norm = torch.norm(normal)
            if norm != 0:
                normal = normal / norm

            normals[:, i] = normal
            areas[i] = area
            centers[:, i] = center

        total_area += torch.sum(areas)
        all_areas.append(areas)
        all_centers.append(centers)
        all_normals.append(normals)

    all_areas = torch.cat(all_areas, dim=0)
    all_centers = torch.cat(all_centers, dim=1)
    all_normals = torch.cat(all_normals, dim=1)

    return total_area, all_areas, all_centers, all_normals
