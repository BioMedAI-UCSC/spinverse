import torch
from torch import lgamma
from src.phider import phider
from src.torch_functions import torch_amtam, torch_astam, torch_smamt

def stiffness_matrixP1_3D(elements, coordinates, coeffs=None):
    # Adjusting shapes for PyTorch
    elements = elements.t().contiguous()
    coordinates = coordinates.t().contiguous()
    NE = elements.shape[0]    # Number of elements
    DIM = coordinates.shape[1] # Problem dimension (should be 3)
    NLB = 4                    # Number of local basis functions for P1

    # Build per-element coordinate array: [DIM x NLB x NE]
    coord = torch.zeros((DIM, NLB, NE),
                        dtype=coordinates.dtype,
                        device=coordinates.device)
    for d in range(DIM):
        for i in range(NLB):
            coord[d, i, :] = coordinates[elements[:, i], d]

    # Integration points for reference tetra
    IP = torch.tensor([1/4, 1/4, 1/4],
                      dtype=coordinates.dtype,
                      device=coordinates.device).reshape(-1, 1)

    # Compute shape‐function gradients dphi: [NLB x DIM x NE]
    dphi, detj, _ = phider(coord, IP, "P1")

    # Element volumes
    volumes = torch.abs(detj).squeeze() / lgamma(torch.tensor([DIM+1],
                                dtype=coordinates.dtype,
                                device=coordinates.device)).exp()
    volumes = volumes.reshape(1, NE)

    # Remove any extra singleton dims
    dphi = dphi.squeeze()

    # Handle coefficients
    if coeffs is None:
        coeffs = torch.tensor(1.0, device=coordinates.device)
    elif coeffs.dim() == 2:
        coeffs = coeffs.squeeze()

    # --- BEGIN PATCH for single‐element case ---
    # Ensure dphi is dense
    if dphi.is_sparse:
        dphi = dphi.to_dense()
    # If only one element the third dim may be collapsed; restore it
    if dphi.dim() == 2:
        # dphi shape is now [NLB x DIM], so add element axis
        dphi = dphi.unsqueeze(2)  # [NLB x DIM x 1]
    # --- END PATCH ---

    # Transpose to [DIM x NLB x NE]
    dphi_transposed = dphi.permute(1, 0, 2)

    # Assemble local stiffness contributions Z
    if coeffs is None or (torch.is_tensor(coeffs) and coeffs.numel() == 1):
        amtam_result = torch_amtam(dphi_transposed, dphi_transposed)
        if coeffs is None:
            Z = torch_astam(volumes.t(), amtam_result)
        else:
            Z = torch_astam((volumes * coeffs).t(), amtam_result)
    else:
        if coeffs.dim() == 2:
            Z = torch_astam(
                volumes.t(),
                torch.einsum(
                    "jik,ljk->ilk",
                    torch_smamt(coeffs, dphi_transposed),
                    dphi_transposed,
                ),
            )
        elif coeffs.dim() == 3 and coeffs.size(2) == 1:
            coeffs_rep = coeffs.repeat(1, 1, NE)
            multiplied = torch_amtam(dphi_transposed, coeffs_rep)
            Z = torch_astam(volumes.t(), multiplied)
        else:
            raise NotImplementedError(
                "P1->P0 averaging not implemented for coeffs dimension",
            )

    # Build global sparse indices
    Y = torch.reshape(
        torch.tile(elements, (1, NLB)).t().contiguous(),
        (NLB, NLB, NE)
    )
    X = Y.permute(1, 0, 2)
    indices = torch.stack([X.flatten(), Y.flatten()], dim=0)
    Z_flat = Z.flatten()

    # Matrix size = number of global nodes
    nnodes = coordinates.size(0)
    M = torch.sparse_coo_tensor(indices, Z_flat, size=(nnodes, nnodes))

    # Symmetrize and return
    M_sym = (M + M.transpose(0, 1)) * 0.5
    return M_sym
