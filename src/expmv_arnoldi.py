import torch

def expmv_arnoldi(M_fn, v, t=1.0, m=30, tol=1e-7):
    """Approximate exp(t * M) @ v using Arnoldi Krylov (batched, complex).
    M_fn: callable for matvec M @ x (sparse-friendly, batched).
    v: [batch_dims, n, 1] vector.
    t: scalar multiplier (e.g., -seq.delta).
    m: Krylov dim (20-50; increase for accuracy).
    Returns: approx exp(t M) @ v.
    """
    batch_shape = v.shape[:-2]
    n = v.shape[-2]
    dtype = v.dtype
    device = v.device
    Q = torch.zeros(*batch_shape, n, m+1, dtype=dtype, device=device)  # Krylov basis
    H = torch.zeros(*batch_shape, m+1, m, dtype=dtype, device=device)  # Hessenberg
    beta = torch.linalg.norm(v, dim=-2, keepdim=True)  # [batch..., 1, 1]
    denom_beta = beta[..., 0, :].clamp(min=tol)
    is_small_beta = beta[..., 0, 0] < tol
    Q_temp = v[..., :, 0] / denom_beta
    Q[..., :, 0] = Q_temp * (~is_small_beta).unsqueeze(-1)

    for j in range(m):
        w = M_fn(Q[..., :, j].unsqueeze(-1))  # Batched matvec: [batch..., n, 1]
        # First orthogonalization
        for i in range(j+1):
            proj = torch.sum(torch.conj(Q[..., :, i]) * w[..., 0], dim=-1)  # Project
            H_temp = H.clone()
            H_temp[..., i, j] = proj
            H = H_temp
            w_temp = w - proj.unsqueeze(-1).unsqueeze(-1) * Q[..., :, i].unsqueeze(-1)
            w = w_temp
        # Second orthogonalization for stability
        delta = torch.zeros(*batch_shape, j+1, dtype=dtype, device=device)
        for i in range(j+1):
            h_add = torch.sum(torch.conj(Q[..., :, i]) * w[..., 0], dim=-1)
            delta_temp = delta.clone()
            delta_temp[..., i] = h_add
            delta = delta_temp
            w_temp = w - h_add.unsqueeze(-1).unsqueeze(-1) * Q[..., :, i].unsqueeze(-1)
            w = w_temp
        # Update H with delta after loop
        col_j = H[..., :j+1, j] + delta
        H_temp = H.clone()
        H_temp[..., :j+1, j] = col_j
        H = H_temp

        h_next = torch.linalg.norm(w, dim=-2, keepdim=True)
        is_small = h_next[..., 0, 0] < tol
        if torch.all(is_small):
            break
        denom = h_next[..., 0, :].clamp(min=tol)
        Q_temp = w[..., 0] / denom
        Q_temp2 = Q_temp * (~is_small).unsqueeze(-1)
        Q[..., :, j+1] = Q_temp2
        h_next_val = h_next[..., 0, 0] * (~is_small)
        H_temp = H.clone()
        H_temp[..., j+1, j] = h_next_val
        H = H_temp

    j = min(j + 1, m)  # Ensure proper truncation index
    H_trunc = H[..., :j, :j].clone()  # Create new tensor
    e1 = torch.zeros(*batch_shape, j, 1, dtype=dtype, device=device)
    e1_temp = e1.clone()
    e1_temp[..., 0, 0] = 1.0
    e1 = e1_temp
    exp_H = torch.linalg.matrix_exp(t * H_trunc)  # Small exp
    result = beta * (Q[..., :, :j] @ (exp_H @ (beta * e1)))  # Backproject
    return result  # [batch..., n, 1]