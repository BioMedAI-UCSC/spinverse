import os
import torch

import itertools
import torch
from collections import defaultdict

def split_mesh_per_element(femesh_all):
    """
    Per-element split WITH diffusion coupling via shared global face IDs.
    Compatible with existing solver expectations.
    """

    points_all   = torch.as_tensor(femesh_all["points"])
    elements_all = torch.as_tensor(femesh_all["elements"], dtype=torch.long)

    nv_elem, Ne = elements_all.shape
    assert nv_elem == 4, "Only tetrahedra supported"

    # Fixed local element connectivity
    local_element = torch.arange(4, dtype=torch.long).view(4, 1)

    # Local faces: columns = faces
    local_faces = torch.tensor(
        [[0, 0, 0, 1],
         [1, 1, 2, 2],
         [2, 3, 3, 3]],
        dtype=torch.long
    )

    # ------------------------------------------------------------
    # 1) Build global face ID map
    #    key = sorted(global vertex ids)
    # ------------------------------------------------------------
    face_id = {}
    face_map = defaultdict(list)
    next_face_id = 0

    for e in range(Ne):
        gnodes = elements_all[:, e]
        for lf in range(4):
            verts = gnodes[local_faces[:, lf]].tolist()
            key = tuple(sorted(verts))

            if key not in face_id:
                face_id[key] = next_face_id
                next_face_id += 1

            face_map[key].append((e, lf))

    # ------------------------------------------------------------
    # 2) Allocate outputs (KEEP SHAPE COMPATIBLE)
    # ------------------------------------------------------------
    points    = [None] * Ne
    elements  = [None] * Ne
    facets    = [[None] for _ in range(Ne)]   # one interface group
    face_ids  = [None] * Ne                   # NEW: per-face global IDs
    point_map = [None] * Ne

    # ------------------------------------------------------------
    # 3) Populate per-element data
    # ------------------------------------------------------------
    for e in range(Ne):
        gnodes = elements_all[:, e]

        points[e]    = points_all[:, gnodes]
        elements[e]  = local_element
        facets[e][0] = local_faces
        point_map[e] = gnodes

        # one global face ID per local face
        ids = []
        for lf in range(4):
            verts = gnodes[local_faces[:, lf]].tolist()
            key = tuple(sorted(verts))
            ids.append(face_id[key])

        face_ids[e] = torch.tensor(ids, dtype=torch.long)

    return {
        "ncompartment": Ne,
        "nboundary": 1,          # single interface group
        "points": points,
        "elements": elements,
        "facets": facets,
        "face_ids": face_ids,    # <-- THIS restores coupling
        "point_map": point_map,
    }

def split_mesh_fast_per_element(femesh_all, tet_grid=False):
    """
    Fast per-element split:
      - Treats each element as its own compartment.
      - Attaches only those boundary facets that belong to the element via a facet hash.
      - Renumbers connectivity to local indices [0..nv_elem-1].

    Input dict keys:
      points:         (dim, Np) float
      elements:       (nv_elem, Ne) long
      facets:         (nv_facet, Nf) long
      facetmarkers:   (Nf,) long

    Output dict:
      ncompartment:   Ne
      nboundary:      number of unique facet markers
      points:         list[Ne] of (dim, nv_elem)
      elements:       list[Ne] of (nv_elem, 1)   (always [[0..nv_elem-1]]^T)
      facets:         list[Ne][nboundary] of (nv_facet, k) or None
      point_map:      list[Ne] of length nv_elem (local->global node ids)
    """

    # Check if cached mesh exists
    num_elements = femesh_all['elements'].shape[1]
    path = f"mesh_setup/fem_tet_grids/tet_grid_{num_elements}_elements.pth"
    
    if tet_grid and os.path.exists(path):
        femesh_split = torch.load(path)
        points_all = femesh_all["points"].clone()
        # Update points for each compartment
        for e in range(len(femesh_split['points'])):
            gnodes = femesh_split['point_map'][e]
            femesh_split['points'][e] = points_all[:, gnodes]
        return femesh_split

    # Pull + normalize dtypes
    points_all   = femesh_all["points"].clone()
    elements_all = femesh_all["elements"]
    facets_all   = femesh_all["facets"]
    facetmarkers = femesh_all["facetmarkers"]

    if not isinstance(elements_all, torch.Tensor):
        elements_all = torch.tensor(elements_all, dtype=torch.long)
    else:
        elements_all = elements_all.clone().detach().long()

    if not isinstance(facets_all, torch.Tensor):
        facets_all = torch.tensor(facets_all, dtype=torch.long)
    else:
        facets_all = facets_all.clone().detach().long()

    if not isinstance(facetmarkers, torch.Tensor):
        facetmarkers = torch.tensor(facetmarkers, dtype=torch.long)
    else:
        facetmarkers = facetmarkers.clone().detach().long()

    nv_elem, Ne = elements_all.shape
    nv_facet, Nf = facets_all.shape

    # Compute unique boundary labels and index them
    boundaries = torch.unique(facetmarkers)
    nboundary = int(boundaries.numel())
    b2i = {int(b.item()): i for i, b in enumerate(boundaries)}

    # ----------------------------------------------------------------------
    # 1) Build a facet hash: unordered vertex set -> (facet_index, boundary_idx)
    #    We canonicalize a facet by sorting its vertex ids and using a Python tuple key.
    #    This is a one-time O(Nf * nv_facet) pass and extremely fast in CPython.
    # ----------------------------------------------------------------------
    facet_hash = {}
    # Use .tolist() per column = tiny (nv_facet), acceptable and fast
    for j in range(Nf):
        verts = facets_all[:, j].tolist()
        verts.sort()
        key = tuple(verts)
        # In the (rare) case of duplicates, keep first; change to list append if needed
        if key not in facet_hash:
            boundary_idx = b2i[int(facetmarkers[j].item())]
            facet_hash[key] = (j, boundary_idx)

    # Precompute the combinations of local element vertex indices that form a facet
    # (e.g., for tet (nv_elem=4, nv_facet=3): [(0,1,2),(0,1,3),(0,2,3),(1,2,3)])
    local_face_combos = list(itertools.combinations(range(nv_elem), nv_facet))

    # ----------------------------------------------------------------------
    # 2) Prepare outputs
    # ----------------------------------------------------------------------
    ncompartment = Ne
    points   = [None] * Ne
    elements = [None] * Ne
    point_map = [None] * Ne
    facets = [[None for _ in range(nboundary)] for _ in range(ncompartment)]

    # For local connectivity, each "compartment" has a single element with nodes [0..nv_elem-1]
    local_element_template = torch.arange(nv_elem, dtype=torch.long).view(nv_elem, 1)

    # ----------------------------------------------------------------------
    # 3) Main loop: O(Ne * faces_per_element)
    #    - Gather per-element points and mapping (global->local is size nv_elem)
    #    - Enumerate faces, look them up in facet_hash, bucket by boundary
    #    - Remap found facets to local indices
    # ----------------------------------------------------------------------
    dim = points_all.shape[0]

    for e in range(Ne):
        # Global nodes of this element
        gnodes = elements_all[:, e]  # (nv_elem,)
        point_map[e] = gnodes.clone()

        # Per-element points (dim, nv_elem)
        # NOTE: using advanced indexing on columns is fast and avoids copies beyond the slice
        points[e] = points_all[:, gnodes]

        # Local elements connectivity: trivial [0..nv_elem-1]
        elements[e] = local_element_template

        # Build tiny global->local map (size nv_elem). Dict is fastest here given ~3-8 entries.
        gl2loc = {int(g): i for i, g in enumerate(gnodes.tolist())}

        # Collect facet columns per boundary (as lists of local int tuples)
        # We'll convert to tensors once per boundary to minimize small allocations
        per_boundary_loc_facets = [[] for _ in range(nboundary)]

        # Enumerate each face of this element, canonicalize via sorted global ids, and look up
        # This is constant-time per face since nv_facet is small.
        for comb in local_face_combos:
            face_globals = [int(gnodes[i]) for i in comb]
            key = tuple(sorted(face_globals))
            hit = facet_hash.get(key, None)
            if hit is not None:
                _, bidx = hit
                # Remap globals to locals with gl2loc; keep local order consistent with comb
                per_boundary_loc_facets[bidx].append(tuple(gl2loc[g] for g in face_globals))

        # Materialize tensors for any boundary that has facets on this element
        # Shape becomes (nv_facet, k) with columns as local vertex ids
        for bidx, flist in enumerate(per_boundary_loc_facets):
            if flist:
                # Transpose list-of-tuples into (nv_facet, k)
                # This is efficient due to small k; convert directly to torch.long
                facets[e][bidx] = torch.tensor(flist, dtype=torch.long).T.contiguous()

    femesh_split = {
        "ncompartment": ncompartment,
        "nboundary": nboundary,
        "points": points,            # list[Ne] of (dim, nv_elem)
        "facets": facets,            # list[Ne][nboundary] of (nv_facet, k) or None
        "elements": elements,        # list[Ne] of (nv_elem, 1) with [0..nv_elem-1]
        "point_map": point_map,      # list[Ne] of (nv_elem,) local->global ids
        "boundaries": boundaries,    # optional: expose mapping used
    }

    if tet_grid and not os.path.exists(path):
        torch.save(femesh_split, path)

    return femesh_split

def split_mesh(femesh_all, grad=False):
    # Extract global mesh
    # points_all = femesh_all['points'].clone().detach().requires_grad_(grad) if isinstance(femesh_all['points'], torch.Tensor) else torch.tensor(femesh_all['points'], dtype=torch.float, requires_grad=grad)
    points_all = femesh_all["points"].clone()
    facets_all = (
        femesh_all["facets"].clone().detach()
        if isinstance(femesh_all["facets"], torch.Tensor)
        else torch.tensor(femesh_all["facets"], dtype=torch.long)
    )
    elements_all = (
        femesh_all["elements"].clone().detach()
        if isinstance(femesh_all["elements"], torch.Tensor)
        else torch.tensor(femesh_all["elements"], dtype=torch.long)
    )
    facetmarkers = (
        femesh_all["facetmarkers"].clone().detach()
        if isinstance(femesh_all["facetmarkers"], torch.Tensor)
        else torch.tensor(femesh_all["facetmarkers"], dtype=torch.long)
    )
    elementmarkers = (
        femesh_all["elementmarkers"].clone().detach()
        if isinstance(femesh_all["elementmarkers"], torch.Tensor)
        else torch.tensor(femesh_all["elementmarkers"], dtype=torch.long)
    )

    # Identify compartments and boundaries
    compartments, _ = torch.unique(elementmarkers, return_inverse=True)
    boundaries, _ = torch.unique(facetmarkers, return_inverse=True)
    ncompartment = compartments.size(0)
    nboundary = boundaries.size(0)

    # Split points and elements into compartments
    elements = [None] * ncompartment
    point_map = [None] * ncompartment
    points = [None] * ncompartment
    for i, compartment in enumerate(compartments):
        mask = elementmarkers == compartment
        elements[i] = elements_all[:, mask]
        _, inverse_indices = torch.unique(elements[i], return_inverse=True)
        point_map[i] = torch.unique(elements[i]).long()
        points[i] = points_all[:, point_map[i]]

    # Split facets into boundaries
    boundary_facets = [None] * nboundary
    for i, boundary in enumerate(boundaries):
        mask = facetmarkers == boundary
        boundary_facets[i] = facets_all[:, mask]

    # Renumber nodes in elements and facets
    facets = [[None for _ in range(nboundary)] for _ in range(ncompartment)]
    for icmpt in range(ncompartment):
        old_to_new_map = {
            int(old): int(new) for new, old in enumerate(point_map[icmpt])
        }
        elements[icmpt] = torch.tensor(
            [old_to_new_map[int(old)] for old in elements[icmpt].view(-1)],
            dtype=torch.long,
        ).view(*elements[icmpt].shape)

        for iboundary in range(nboundary):
            boundary_on_compartment = torch.all(
                torch.isin(boundary_facets[iboundary], point_map[icmpt]), dim=0
            )
            if boundary_on_compartment.any():
                facets[icmpt][iboundary] = torch.tensor(
                    [
                        old_to_new_map[int(old)]
                        for old in boundary_facets[iboundary].view(-1)
                    ],
                    dtype=torch.long,
                ).view(*boundary_facets[iboundary].shape)

    femesh = {
        "ncompartment": ncompartment,
        "nboundary": nboundary,
        "points": points,
        "facets": facets,
        "elements": elements,
        "point_map": point_map,
    }

    return femesh
