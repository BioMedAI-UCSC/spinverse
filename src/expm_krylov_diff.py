# expm_krylov.py
# Fully differentiable Arnoldi–Krylov expm-multiply for batched blocks.
# Usage:
#   from expm_krylov import expmv_krylov_bd
#   tmp1 = expmv_krylov_bd(K1, nu, float(seq.delta), m=32, steps=1)
#   tmp2 = E2.unsqueeze(0).unsqueeze(0) @ tmp1
#   K1_H = torch.conj(K1).transpose(-1, -2)
#   nu_final = expmv_krylov_bd(K1_H, tmp2, float(seq.delta), m=32, steps=1)

import torch
from typing import Optional

def expmv_krylov_bd(
    K: torch.Tensor,
    v: torch.Tensor,
    t: float,
    m: int = 32,
    steps: int = 1,
    reorth: bool = False,
    checkpoint: bool = False,
    eps: float = 1e-38,
) -> torch.Tensor:
    """
    Compute exp(-t*K) @ v without forming exp(-t*K), using Arnoldi–Krylov.

    Shapes (match your code):
      K: [n_amp, n_dir, n, n]  (real or complex; autograd-friendly)
      v: [n_amp, n_dir, n, r]  (r=1 in your code; r>1 ok, treated column-wise)
      t: scalar float time (e.g., seq.delta)

    Returns:
      same shape as v: [n_amp, n_dir, n, r], complex dtype if K or v is complex.

    Args:
      m         Krylov subspace size (24–48 typical). Larger = more accurate, more compute/memory.
      steps     Optional time-splitting count; if t*||K|| is big, 2–4 helps stability/accuracy.
      reorth    If True, perform one pass of reorthogonalization (more stable, slightly slower).
      checkpoint If True, wrap the small m-step in torch.utils.checkpoint to save backward memory
                 at the cost of recomputation in the backward pass.
      eps       Tiny constant to avoid division by zero.

    Notes:
      * Everything is implemented with standard PyTorch ops -> fully differentiable.
      * We only do tiny matrix_exp on [B, m, m] Hessenberg blocks (B = n_amp*n_dir), never on [n, n].
      * Works fine with complex dtypes; inputs are promoted to complex64 if needed.
    """
    a, d, n, _ = K.shape
    _, _, nv, r = v.shape
    assert nv == n, "K and v mismatch on the state dimension"

    B = a * d
    # Flatten amp×dir into batch
    Kb = K.reshape(B, n, n).contiguous()
    vb = v.reshape(B, n, r).contiguous()

    # Promote to complex for 1j terms safety
    ctype = Kb.dtype if Kb.is_complex() else (torch.complex64 if Kb.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.complex128)
    Kb = Kb.to(ctype)
    vb = vb.to(ctype)

    # Optional time splitting
    if steps <= 0:
        steps = 1
    dt = float(t) / steps

    # Core Arnoldi–Krylov on the flattened batch B
    out = _expmv_krylov_batched_flat(
        Kb, vb, dt, m=m, steps=steps, reorth=reorth, checkpoint=checkpoint, eps=eps
    )
    return out.view(a, d, n, r)


def _expmv_krylov_batched_flat(
    Kb: torch.Tensor,
    vb: torch.Tensor,
    dt: float,
    m: int,
    steps: int,
    reorth: bool,
    checkpoint: bool,
    eps: float,
) -> torch.Tensor:
    """
    Kb: [B, n, n], vb: [B, n, r]  -> returns [B, n, r]
    Performs `steps` substeps of Arnoldi–Krylov with subspace size m.
    """
    # We apply the one-step map repeatedly if steps>1.
    u = vb
    for _ in range(steps):
        if checkpoint:
            u = torch.utils.checkpoint.checkpoint(
                _krylov_one_step,
                Kb,
                u,
                torch.tensor(dt, dtype=Kb.real.dtype, device=Kb.device),
                torch.tensor(m, dtype=torch.int64, device=Kb.device),
                torch.tensor(int(reorth), dtype=torch.int64, device=Kb.device),
                torch.tensor(eps, dtype=Kb.real.dtype, device=Kb.device)
            )
        else:
            u = _krylov_one_step(Kb, u, dt, m, reorth, eps)
    return u


def _krylov_one_step(
    Kb: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    m: int,
    reorth: bool,
    eps: float,
) -> torch.Tensor:
    """
    One expm step: u <- exp(-dt * Kb) @ u
    Vectorized across batch B; loops only over Krylov dim m and columns r (usually 1).
    """
    # Handle possible tensor-wrapped scalars from checkpoint
    dt = dt.item() if isinstance(dt, torch.Tensor) else dt
    m = int(m.item()) if isinstance(m, torch.Tensor) else m
    reorth = bool(reorth.item()) if isinstance(reorth, torch.Tensor) else reorth
    eps = eps.item() if isinstance(eps, torch.Tensor) else eps

    B, n, _ = Kb.shape
    _, nv, r = u.shape
    assert nv == n

    ctype = Kb.dtype
    zeros_bn = torch.zeros(B, n, dtype=ctype, device=Kb.device)

    U_cols = []

    for j in range(r):
        uj = u[:, :, j]  # [B, n]

        beta = torch.linalg.vector_norm(uj, dim=1)  # [B]
        zero_mask = beta == 0

        denom_beta = (beta + eps).unsqueeze(1)
        v1 = torch.where(zero_mask.unsqueeze(1), zeros_bn, uj / denom_beta)

        V_list = [v1]  # list of [B, n]
        H_cols = []    # list of [B, m] (columns of H)

        for k in range(m):
            vk = V_list[k]  # [B, n]
            w = torch.bmm(Kb, vk.unsqueeze(-1)).squeeze(-1)  # [B, n]

            # Initialize H column for this k
            H_col_k = torch.zeros(B, m, dtype=ctype, device=Kb.device)  # [B, m]

            # Modified Gram-Schmidt orthogonalization
            for i in range(k + 1):
                hij = torch.einsum('bn,bn->b', torch.conj(V_list[i]), w)  # [B]
                w = w - hij.unsqueeze(1) * V_list[i]
                H_col_k[:, i] = hij

            if reorth:
                # Second pass for reorthogonalization
                for i in range(k + 1):
                    corr = torch.einsum('bn,bn->b', torch.conj(V_list[i]), w)  # [B]
                    w = w - corr.unsqueeze(1) * V_list[i]
                    H_col_k[:, i] = H_col_k[:, i] + corr

            if k + 1 < m:
                h_next = torch.linalg.vector_norm(w, dim=1)  # [B]
                H_col_k[:, k + 1] = h_next.to(ctype)
                small_h = h_next == 0

                denom_h = (h_next + eps).unsqueeze(1)
                v_next = torch.where(small_h.unsqueeze(1), zeros_bn, w / denom_h)

                V_list.append(v_next)

            H_cols.append(H_col_k)

        # Assemble H: [B, m, m]
        H = torch.stack(H_cols, dim=2)  # [B, m, m]

        # e1 = [1, 0, ..., 0]^T -> [B, m, 1]
        ones_b11 = torch.ones(B, 1, 1, dtype=ctype, device=Kb.device)
        zeros_bm1_1 = torch.zeros(B, m - 1, 1, dtype=ctype, device=Kb.device) if m > 1 else torch.empty(B, 0, 1, dtype=ctype, device=Kb.device)
        e1 = torch.cat([ones_b11, zeros_bm1_1], dim=1)

        # y = beta * exp(-dt * H) @ e1
        E = torch.linalg.matrix_exp((-dt) * H)  # [B, m, m]
        y = torch.bmm(E, e1)  # [B, m, 1]
        y = beta.to(ctype).unsqueeze(1).unsqueeze(1) * y

        # V = [B, n, m]
        V = torch.stack(V_list, dim=2)  # [B, n, m]

        # Uj = V @ y -> [B, n]
        Uj = torch.bmm(V, y).squeeze(-1)
        Uj = torch.where(zero_mask.unsqueeze(1), zeros_bn, Uj)

        U_cols.append(Uj)

    U_out = torch.stack(U_cols, dim=2)  # [B, n, r]
    return U_out