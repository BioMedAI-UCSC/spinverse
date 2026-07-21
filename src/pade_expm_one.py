import torch
import torch.nn.functional as F

def pade_matrix_exp(A, q=6, scaling=True):
    """
    Memory-efficient matrix exponential using Padé approximation.
    Processes each matrix individually to minimize memory usage.
    
    Args:
        A: Input matrix of shape (..., n, n), can be complex
        q: Order of Padé approximation (default: 6)
        scaling: Whether to use scaling and squaring (default: True)
    
    Returns:
        e^A computed using diagonal Padé approximation
    """
    original_shape = A.shape
    *batch_dims, n, _ = A.shape
    
    # Flatten batch dimensions
    A_flat = A.reshape(-1, n, n)
    num_matrices = A_flat.shape[0]
    
    # Allocate output tensor
    result = torch.zeros_like(A_flat)
    
    # Process each matrix individually to save memory
    for i in range(num_matrices):
        result[i] = _pade_matrix_exp_single(A_flat[i:i+1], q, scaling)
    
    # Reshape back to original batch dimensions
    result = result.reshape(original_shape)
    
    return result


def _pade_matrix_exp_single(A, q, scaling):
    """
    Process a single matrix (shape [1, n, n]).
    This minimizes memory by only working with one matrix at a time.
    """
    device = A.device
    is_complex = torch.is_complex(A)
    n = A.shape[-1]
    
    # Compute 1-norm
    if is_complex:
        A_norm = torch.sum(torch.abs(A), dim=-2).max().real
    else:
        A_norm = torch.sum(torch.abs(A), dim=-2).max()
    
    # Scaling
    if scaling and A_norm > 0.5:
        m = torch.ceil(A_norm / 0.5).clamp(min=1).long()
        m_log2 = torch.ceil(torch.log2(m.float())).long().item()
        m = 2 ** m_log2
        A_scaled = A / float(m)
    else:
        A_scaled = A
        m_log2 = 0
    
    # Compute Padé approximation
    R = pade_approximant_single(A_scaled.squeeze(0), q).unsqueeze(0)
    
    # Squaring
    for _ in range(m_log2):
        R = torch.matmul(R, R)
    
    return R.squeeze(0)


def pade_approximant_single(A, q):
    """
    Compute Padé approximant for a single matrix [n, n].
    Memory-efficient: doesn't store all powers simultaneously.
    """
    n = A.shape[0]
    device = A.device
    dtype = A.dtype
    
    # Identity matrix
    I = torch.eye(n, device=device, dtype=dtype)
    
    # Initialize with identity term
    c_0 = compute_pade_coeff(q, q, 0)
    N = c_0 * I
    D = c_0 * I
    
    # Iteratively compute powers and accumulate
    A_power = A  # A^1
    
    for j in range(1, q + 1):
        c_n = compute_pade_coeff(q, q, j)
        c_d = c_n * ((-1) ** j)
        
        N = N + c_n * A_power
        D = D + c_d * A_power
        
        # Compute next power if needed
        if j < q:
            A_power = A_power @ A
    
    # Solve D * R = N
    R = torch.linalg.solve(D, N)
    
    return R


def compute_pade_coeff(p, q, j):
    """Compute Padé coefficient."""
    from math import factorial
    
    if j > min(p, q):
        return 0.0
    
    numerator = factorial(p + q - j) * factorial(p)
    denominator = factorial(p + q) * factorial(j) * factorial(p - j)
    
    return float(numerator) / float(denominator)


# If you need slightly better performance, use this vectorized version with checkpointing
def pade_matrix_exp_vectorized(A, q=6, scaling=True, checkpoint=True):
    """
    More efficient version using gradient checkpointing to reduce memory during backward pass.
    """
    if checkpoint and A.requires_grad:
        from torch.utils.checkpoint import checkpoint
        # Split computation into chunks and checkpoint them
        return checkpoint(_pade_matrix_exp_impl, A, q, scaling, use_reentrant=False)
    else:
        return _pade_matrix_exp_impl(A, q, scaling)


def _pade_matrix_exp_impl(A, q, scaling):
    """Implementation that can be checkpointed."""
    device = A.device
    is_complex = torch.is_complex(A)
    *batch_dims, n, _ = A.shape
    
    # Compute 1-norm
    if is_complex:
        A_norm = torch.max(torch.sum(torch.abs(A), dim=-2), dim=-1)[0].real
    else:
        A_norm = torch.max(torch.sum(torch.abs(A), dim=-2), dim=-1)[0]
    
    # Scaling
    if scaling:
        m = torch.ceil(A_norm / 0.5).clamp(min=1).long()
        m_log2 = torch.ceil(torch.log2(m.float())).long()
        m = 2 ** m_log2
        max_m_log2 = m_log2.max().item()
        
        m_expanded = m.view(*batch_dims, 1, 1)
        if is_complex:
            m_expanded = m_expanded.to(dtype=A.dtype)
        A_scaled = A / m_expanded
    else:
        A_scaled = A
        max_m_log2 = 0
        m_log2 = torch.zeros(*batch_dims, dtype=torch.long, device=device)
    
    # Compute Padé approximation  
    R = pade_approximant_batch(A_scaled, q)
    
    # Squaring with sequential approach
    for k in range(max_m_log2):
        needs_squaring = (m_log2 > k).view(*batch_dims, 1, 1)
        R_squared = torch.matmul(R, R)
        R = torch.where(needs_squaring, R_squared, R)
    
    return R


def pade_approximant_batch(A, q):
    """Batched Padé approximant."""
    *batch_dims, n, _ = A.shape
    device = A.device
    dtype = A.dtype
    
    I = torch.eye(n, device=device, dtype=dtype).view(*([1]*len(batch_dims)), n, n).expand(*batch_dims, n, n)
    
    c_0 = compute_pade_coeff(q, q, 0)
    N = c_0 * I
    D = c_0 * I
    
    A_power = A.clone()
    
    for j in range(1, q + 1):
        c_n = compute_pade_coeff(q, q, j)
        c_d = c_n * ((-1) ** j)
        
        N = N + c_n * A_power
        D = D + c_d * A_power
        
        if j < q:
            A_power = torch.matmul(A_power, A)
    
    R = torch.linalg.solve(D, N)
    
    return R


# Example usage
if __name__ == "__main__":
    n_amp, n_dir, n_eig = 21, 30, 1000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test with smaller size first
    print("\nTesting with small matrices first...")
    K1_small = torch.randn(2, 3, 100, 100, device=device, dtype=torch.cfloat)
    K1_small.requires_grad_(True)
    
    E1 = pade_matrix_exp(K1_small, q=6)
    loss = E1.abs().sum()
    loss.backward()
    print("Small test passed!")
    
    print(f"\nNow testing with full size: [{n_amp}, {n_dir}, {n_eig}, {n_eig}]")
    print("Processing each matrix individually...")
    
    lambda_batch = torch.randn(n_amp, n_dir, n_eig, n_eig, device=device)
    relax_batch = torch.randn(n_amp, n_dir, n_eig, n_eig, device=device)
    q_as = torch.randn(n_amp, n_dir, device=device)
    A_batch = torch.randn(n_amp, n_dir, n_eig, n_eig, device=device)
    seq_delta = torch.tensor(0.01, device=device)
    
    K1 = lambda_batch + relax_batch + 1j * q_as.unsqueeze(-1).unsqueeze(-1) * A_batch
    
    # Use the individual matrix processing version
    E1 = pade_matrix_exp(-seq_delta * K1, q=6)
    print("Computation complete!")