import torch
from torch import sparse_coo_tensor as coo_tensor

def mass_matrixP1_3D(elements, volumes, coeffs=None):
    # breakpoint()
    elements = elements.t().contiguous().long()
    volumes = volumes.t().contiguous()
    num_elements = elements.size(1)

    if coeffs is not None:
        coeffs = coeffs.t()

    ones_4 = torch.ones((1, 4), dtype=torch.long, device=elements.device)
    X = torch.kron(ones_4, elements)
    Y = torch.kron(elements, ones_4)

    if coeffs is None:
        matrix = (torch.ones(4, 4, device=elements.device) + torch.eye(4, device=elements.device)) / 20
        matrix_flat = matrix.reshape(1, -1)
        Z = torch.kron(volumes.unsqueeze(1), matrix_flat)
    elif coeffs.numel() == num_elements:
        # per-element scaling
        matrix = (torch.ones(4, 4, device=elements.device) + torch.eye(4, device=elements.device)) / 20
        matrix_flat = matrix.reshape(1, -1)
        Z = torch.kron((volumes * coeffs).unsqueeze(1), matrix_flat)
    else:
        # Nodal averaging logic
        M1 = (torch.tensor(
            [[6, 2, 2, 2], [2, 2, 1, 1], [2, 1, 2, 1], [2, 1, 1, 2]],
            dtype=torch.float32, device=elements.device
        ) / 120)
        sequence = [3, 0, 1, 2]
        M2 = M1[sequence, :][:, sequence]
        M3 = M2[sequence, :][:, sequence]
        M4 = M3[sequence, :][:, sequence]

        M1_flat = M1.reshape(1, -1)
        M2_flat = M2.reshape(1, -1)
        M3_flat = M3.reshape(1, -1)
        M4_flat = M4.reshape(1, -1)

        Z1 = torch.kron(volumes * coeffs[elements[:, 0]], M1_flat)
        Z2 = torch.kron(volumes * coeffs[elements[:, 1]], M2_flat)
        Z3 = torch.kron(volumes * coeffs[elements[:, 2]], M3_flat)
        Z4 = torch.kron(volumes * coeffs[elements[:, 3]], M4_flat)
        Z = Z1 + Z2 + Z3 + Z4

    Z = Z.view(-1)
    X_flat = X.view(-1)
    Y_flat = Y.view(-1)
    Z_flat = Z.view(-1)

    indices = torch.stack([X_flat, Y_flat], dim=0)
    size = (torch.max(X_flat).item() + 1, torch.max(Y_flat).item() + 1)
    M = torch.sparse_coo_tensor(indices, Z_flat, size=size)

    # Symmetrize
    M_sym = (M + M.transpose(0, 1)) / 2
    return M_sym
