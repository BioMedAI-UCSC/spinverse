import torch
from torch import sparse_coo_tensor as coo_tensor

def mass_matrixP1_3D(elements, volumes, coeffs=None):
    elements = elements.t().contiguous().long()
    volumes = volumes.t().contiguous().view(-1)
    num_elements = elements.size(0)

    if coeffs is not None:
        coeffs = coeffs.t().contiguous().view(-1)

    ones_4 = torch.ones((1, 4), dtype=torch.long, device=elements.device)
    X = torch.kron(ones_4, elements)
    Y = torch.kron(elements, ones_4)

    if (coeffs is None) or (coeffs.numel() == 0):
        # Pure P1 mass
        matrix = (torch.ones(4, 4, device=elements.device) + torch.eye(4, device=elements.device)) / 20
        matrix_flat = matrix.reshape(1, -1)
        Z = torch.kron(volumes.unsqueeze(1), matrix_flat)
    elif coeffs.numel() == num_elements:
        # Per-element scaling
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
        Mblocks = [M1, M2, M3, M4]
        # This part is now completely safe for NE=1 or NE>1:
        Z = torch.zeros((num_elements, 16), dtype=volumes.dtype, device=elements.device)
        for i in range(4):
            local_coeff = coeffs[elements[:, i]]    # shape: (num_elements,)
            M_flat = Mblocks[i].reshape(1, -1)      # shape: (1, 16)
            # (num_elements, 1) * (1, 16) => (num_elements, 16)
            Z += (volumes * local_coeff).unsqueeze(1) * M_flat
    # flatten
    Z = Z.reshape(-1)
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)
    Z_flat = Z.reshape(-1)

    indices = torch.stack([X_flat, Y_flat], dim=0)
    size = (torch.max(X_flat).item() + 1, torch.max(Y_flat).item() + 1)
    M = torch.sparse_coo_tensor(indices, Z_flat, size=size)

    # Symmetrize
    M_sym = (M + M.transpose(0, 1)) / 2
    return M_sym
