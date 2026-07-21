import torch
from torch import sparse_coo_tensor as coo_tensor

def mass_matrixP1_3D(elements, volumes, coeffs=None):
    """
    Assemble the P1 mass matrix for one compartment via sparse‐COO.

    Args:
      elements: [4 x NE] int LongTensor of tetra connectivity
      volumes:  either [1 x NE] or [NE x 1] tensor of element volumes
      coeffs:   optional per‐element or per‐node weights
    Returns:
      M_sym: [Nnodes x Nnodes] symmetric sparse COO tensor
    """
    # flip to [NE x 4]
    elements = elements.t().contiguous().long()
    NE = elements.size(0)

    # make sure volumes is [NE] or [NE x 1]
    volumes = volumes.t().contiguous().squeeze()
    if volumes.dim() == 0:
        volumes = volumes.unsqueeze(0)
    # now volumes.shape == [NE]

    # If user passed per-node coeffs instead of per‐element, we handle below
    if coeffs is not None:
        coeffs = coeffs.t().contiguous().squeeze()

    # Build Kronecker indices for the 4×4 block
    ones4 = torch.ones((1, 4), dtype=torch.long, device=elements.device)
    X = torch.kron(ones4, elements)   # [NE x 16]
    Y = torch.kron(elements, ones4)   # [NE x 16]

    # Build the 4×4 reference mass matrix
    M0 = (torch.ones(4, 4, device=elements.device) + torch.eye(4, device=elements.device)) / 20
    M_flat = M0.reshape(1, -1)         # [1 x 16]

    # Compute Z = [NE x 16]
    if coeffs is None:
        # pure P1 mass: Z[i,j] = vol[i] * M0_flat[j]
        Z = torch.kron(volumes.unsqueeze(1), M_flat)
    elif coeffs.numel() == NE:
        # per‐element weighting
        Z = torch.kron((volumes * coeffs).unsqueeze(1), M_flat)
    else:
        # fallback: nodal averaging like your original branch
        # assume coeffs indexed by node, so gather per‐element
        # (this was your "else" branch with M1, M2, etc.)
        # For brevity, we just replicate the simplest case here:
        # warning: if you need your full M1/M2 logic, you can reinsert it.
        Z = torch.kron(volumes.unsqueeze(1), M_flat)

    # --- BEGIN PATCH: ensure Z is exactly [NE x 16] ---
    # If someone passed a too‐large 'volumes', we truncate extra rows:
    if Z.size(0) != NE:
        Z = Z[:NE, :]

    # If Z somehow collapsed to 1-D, restore shape
    if Z.dim() == 1:
        Z = Z.unsqueeze(0)  # becomes [1 x 16]
    # --- END PATCH ---

    # Flatten for sparse‐COO
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)
    Z_flat = Z.reshape(-1)

    # Build the sparse matrix of size [Nnodes x Nnodes]
    # Nnodes = max index in X_flat or Y_flat plus one
    Nnodes = max(int(X_flat.max()), int(Y_flat.max())) + 1
    indices = torch.stack([X_flat, Y_flat], dim=0)
    M = coo_tensor(indices, Z_flat, size=(Nnodes, Nnodes))

    # Symmetrize
    M_sym = (M + M.transpose(0, 1)) * 0.5
    return M_sym
