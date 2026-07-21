import torch
import torch.nn.functional as F

def pade_matrix_exp(A, q=6, scaling=True, batch_size=None):
    """
    Memory-efficient matrix exponential using Padé approximation.
    Supports both real and complex matrices.
    
    Args:
        A: Input matrix of shape (..., n, n), can be complex
        q: Order of Padé approximation (default: 6)
        scaling: Whether to use scaling and squaring (default: True)
        batch_size: Process this many matrices at once (default: None = all at once)
    
    Returns:
        e^A computed using diagonal Padé approximation
    """
    original_shape = A.shape
    *batch_dims, n, _ = A.shape
    
    # Flatten batch dimensions for easier processing
    A_flat = A.reshape(-1, n, n)
    num_matrices = A_flat.shape[0]
    
    # Determine batch size
    if batch_size is None:
        batch_size = num_matrices
    
    # Process in chunks to save memory
    results = []
    
    for start_idx in range(0, num_matrices, batch_size):
        end_idx = min(start_idx + batch_size, num_matrices)
        A_batch = A_flat[start_idx:end_idx]
        
        result_batch = _pade_matrix_exp_batch(A_batch, q, scaling)
        results.append(result_batch)
    
    # Concatenate results and reshape
    result = torch.cat(results, dim=0)
    result = result.reshape(original_shape)
    
    return result


def _pade_matrix_exp_batch(A, q, scaling):
    """Process a single batch of matrices."""
    device = A.device
    is_complex = torch.is_complex(A)
    batch_size, n, _ = A.shape
    
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
        
        m_expanded = m.view(batch_size, 1, 1)
        if is_complex:
            m_expanded = m_expanded.to(dtype=A.dtype)
        A_scaled = A / m_expanded
    else:
        A_scaled = A
        m_log2 = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Compute Padé approximation
    R = pade_approximant(A_scaled, q)
    
    # Squaring - avoid in-place operations for autograd compatibility
    if scaling:
        max_k = m_log2.max().item()
        for k in range(max_k):
            # Create mask for which matrices need squaring
            needs_squaring = (m_log2 > k).view(batch_size, 1, 1)
            
            # Square all matrices, then select which ones to keep
            R_squared = torch.matmul(R, R)
            
            # Use where to select without in-place modification
            R = torch.where(needs_squaring, R_squared, R)
    
    return R


def pade_approximant(A, q):
    """
    Memory-efficient Padé approximant computation.
    """
    batch_size, n, _ = A.shape
    device = A.device
    dtype = A.dtype
    
    # Identity matrix
    I = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n, n)
    
    # Initialize N and D
    N = torch.zeros_like(A)
    D = torch.zeros_like(A)
    
    # Add identity contribution
    c_0 = compute_pade_coeff(q, q, 0)
    N = N + c_0 * I
    D = D + c_0 * I
    
    # Compute and accumulate powers iteratively
    A_power = A.clone()  # A^1
    
    for j in range(1, q + 1):
        # Compute coefficients
        c_n = compute_pade_coeff(q, q, j)
        c_d = c_n * ((-1) ** j)
        
        # Accumulate (create new tensors, don't modify in-place)
        N = N + c_n * A_power
        D = D + c_d * A_power
        
        # Compute next power if needed
        if j < q:
            A_power = torch.matmul(A_power, A)
    
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


# Example usage
if __name__ == "__main__":
    n_amp, n_dir, n_eig = 21, 30, 1000
    
    print(f"Creating tensors of shape [{n_amp}, {n_dir}, {n_eig}, {n_eig}]...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    lambda_batch = torch.randn(n_amp, n_dir, n_eig, n_eig, device=device)
    relax_batch = torch.randn(n_amp, n_dir, n_eig, n_eig, device=device)
    q_as = torch.randn(n_amp, n_dir, device=device)
    A_batch = torch.randn(n_amp, n_dir, n_eig, n_eig, device=device)
    seq_delta = torch.tensor(0.01, device=device)
    
    # Your K1 computation (complex matrix)
    K1 = lambda_batch + relax_batch + 1j * q_as.unsqueeze(-1).unsqueeze(-1) * A_batch
    K1.requires_grad_(True)
    
    print("K1 shape:", K1.shape)
    print("K1 is complex:", torch.is_complex(K1))
    
    # Using Padé implementation with batching
    print("\nComputing with Padé method (batch_size=10)...")
    E1_pade = pade_matrix_exp(-seq_delta * K1, q=6, batch_size=10)
    
    print("E1 Padé shape:", E1_pade.shape)
    
    # Test gradient computation
    print("\nTesting gradients...")
    loss = E1_pade.abs().sum()
    loss.backward()
    print("Gradients computed successfully!")
    print(f"K1 grad shape: {K1.grad.shape}")