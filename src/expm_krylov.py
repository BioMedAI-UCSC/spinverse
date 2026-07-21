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
            u = torch.utils.checkpoint.checkpoint(_krylov_one_step, Kb, u, torch.tensor(dt, dtype=Kb.real.dtype, device=Kb.device), torch.tensor(float(m)), torch.tensor(int(reorth)), torch.tensor(eps))
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
    B, n, _ = Kb.shape
    _, nv, r = u.shape
    assert nv == n

    ctype = Kb.dtype if Kb.is_complex() else (torch.complex64 if Kb.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.complex128)
    U_out = torch.empty_like(u, dtype=ctype)
    u = u.to(ctype)
    Kb = Kb.to(ctype)

    for j in range(r):
        uj = u[:, :, j:j+1]  # [B, n, 1]

        # Initial norm per batch
        beta = torch.linalg.vector_norm(uj.squeeze(-1), dim=1)  # [B], real
        zero_mask = (beta == 0)

        # V: [B, n, m], H: [B, m, m]
        V = torch.zeros((B, n, m), dtype=ctype, device=Kb.device)
        H = torch.zeros((B, m, m), dtype=ctype, device=Kb.device)

        # v1 = v / ||v||
        v1 = torch.where(
            zero_mask.view(B, 1),
            torch.zeros((B, n), dtype=ctype, device=Kb.device),
            uj.squeeze(-1) / (beta.view(B, 1) + eps)
        )
        V[:, :, 0] = v1

        # Arnoldi (Modified Gram-Schmidt)
        for k in range(m):
            w = torch.bmm(Kb, V[:, :, k].unsqueeze(-1)).squeeze(-1)  # [B, n]

            # Orthogonalize vs current basis
            for i in range(k + 1):
                hij = torch.sum(torch.conj(V[:, :, i]) * w, dim=1)  # [B]
                H[:, i, k] = hij
                w = w - hij.view(B, 1) * V[:, :, i]

            if reorth:
                # One extra MGS pass (helps stability on tough spectra)
                for i in range(k + 1):
                    corr = torch.sum(torch.conj(V[:, :, i]) * w, dim=1)  # [B]
                    H[:, i, k] += corr
                    w = w - corr.view(B, 1) * V[:, :, i]

            if k + 1 < m:
                hj1k = torch.linalg.vector_norm(w, dim=1)  # [B], real
                H[:, k + 1, k] = hj1k.to(ctype)
                V[:, :, k + 1] = torch.where(
                    (hj1k.view(B, 1) > 0),
                    w / (hj1k.view(B, 1) + eps),
                    torch.zeros_like(w)
                )

        # y = beta * exp(-dt * H) e1
        e1 = torch.zeros((B, m, 1), dtype=ctype, device=Kb.device)
        e1[:, 0, 0] = 1.0 + 0.0j
        E = torch.linalg.matrix_exp((-dt) * H)    # tiny [B, m, m]
        y = torch.bmm(E, e1)                      # [B, m, 1]
        y = (beta.to(ctype).view(B, 1, 1)) * y

        # Uj = V @ y
        Uj = torch.bmm(V, y)                      # [B, n, 1]
        Uj = torch.where(zero_mask.view(B, 1, 1), torch.zeros_like(Uj), Uj)
        U_out[:, :, j:j+1] = Uj

    return U_out
