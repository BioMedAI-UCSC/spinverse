import torch
import torch.nn.functional as F

def krylov_matrix_exp(A, m=None, tol=1e-7, hermitian=False):
    """
    Compute matrix exponential exp(A) using Krylov subspace method (Arnoldi/Lanczos).
    Fully differentiable PyTorch implementation.
    
    Args:
        A: Input matrix/matrices. Shape: [..., n, n] where ... are batch dimensions
        m: Dimension of Krylov subspace (default: min(n-1, 30))
        tol: Tolerance for early stopping
        hermitian: If True, use Lanczos (more efficient for Hermitian matrices)
    
    Returns:
        exp(A) with same shape as input
    """
    *batch_shape, n, n2 = A.shape
    assert n == n2, "Matrix must be square"
    
    if m is None:
        m = min(n - 1, 30)
    m = min(m, n - 1)
    
    device = A.device
    dtype = A.dtype
    
    # Reshape for batch processing
    A_batch = A.reshape(-1, n, n)
    batch_size = A_batch.shape[0]
    
    # Initialize output
    expA = torch.zeros_like(A_batch)
    
    # Process each matrix in batch
    for b in range(batch_size):
        if hermitian:
            expA[b] = _lanczos_exp_single(A_batch[b], m, tol)
        else:
            expA[b] = _arnoldi_exp_single(A_batch[b], m, tol)
    
    return expA.reshape(*batch_shape, n, n)


def _arnoldi_exp_single(A, m, tol=1e-7):
    """
    Compute exp(A) for a single matrix using Arnoldi iteration.
    """
    n = A.shape[0]
    dtype = A.dtype
    device = A.device
    
    # Start with standard basis vector e1
    v = torch.zeros(n, 1, dtype=dtype, device=device)
    v[0, 0] = 1.0
    
    # Arnoldi process to build orthonormal basis V and upper Hessenberg matrix H
    V = torch.zeros(n, m + 1, dtype=dtype, device=device)
    H = torch.zeros(m + 1, m, dtype=dtype, device=device)
    
    beta = torch.norm(v)
    V[:, 0] = v.squeeze() / beta
    
    for j in range(m):
        # Matrix-vector product
        w = A @ V[:, j]
        
        # Orthogonalization (Modified Gram-Schmidt)
        for i in range(j + 1):
            H[i, j] = torch.dot(torch.conj(V[:, i]), w)
            w = w - H[i, j] * V[:, i]
        
        H[j + 1, j] = torch.norm(w)
        
        # Check for breakdown
        if H[j + 1, j] < tol:
            m = j + 1
            break
        
        V[:, j + 1] = w / H[j + 1, j]
    
    # Compute exp(H) in the smaller subspace
    H_small = H[:m, :m]
    exp_H = torch.linalg.matrix_exp(H_small)
    
    # Build the approximation: exp(A) ≈ V * exp(H) * V^H
    V_m = V[:, :m]
    expA = V_m @ exp_H @ torch.conj(V_m).T
    
    # For better accuracy, compute exp(A) from the relation:
    # exp(A) * v = V_m * exp(H_small) * e1 * ||v||
    # Then build full matrix
    I = torch.eye(n, dtype=dtype, device=device)
    expA = torch.zeros_like(A)
    
    for i in range(n):
        v_i = I[:, i:i+1]
        expA[:, i] = _arnoldi_matvec(A, v_i, m, tol).squeeze()
    
    return expA


def _arnoldi_matvec(A, v, m, tol=1e-7):
    """
    Compute exp(A) @ v using Arnoldi iteration.
    More efficient when you only need matrix-vector products.
    """
    n = A.shape[0]
    dtype = A.dtype
    device = A.device
    
    # Normalize initial vector
    beta = torch.norm(v)
    if beta < tol:
        return torch.zeros_like(v)
    
    # Arnoldi process
    V = torch.zeros(n, m + 1, dtype=dtype, device=device)
    H = torch.zeros(m + 1, m, dtype=dtype, device=device)
    
    V[:, 0] = v.squeeze() / beta
    
    for j in range(m):
        w = A @ V[:, j]
        
        for i in range(j + 1):
            H[i, j] = torch.dot(torch.conj(V[:, i]), w)
            w = w - H[i, j] * V[:, i]
        
        H[j + 1, j] = torch.norm(w)
        
        if H[j + 1, j] < tol:
            m = j + 1
            break
        
        V[:, j + 1] = w / H[j + 1, j]
    
    # Compute exp(H) * e1 in the smaller subspace
    H_small = H[:m, :m]
    exp_H = torch.linalg.matrix_exp(H_small)
    
    # First column of exp(H) scaled by beta
    y = exp_H[:, 0:1] * beta
    
    # Map back to original space
    result = V[:, :m] @ y
    
    return result


def _lanczos_exp_single(A, m, tol=1e-7):
    """
    Compute exp(A) for a single Hermitian matrix using Lanczos iteration.
    More efficient than Arnoldi for Hermitian matrices.
    """
    n = A.shape[0]
    dtype = A.dtype
    device = A.device
    
    # For Hermitian matrices, H is tridiagonal
    alpha = torch.zeros(m, dtype=torch.real(dtype) if dtype.is_complex else dtype, device=device)
    beta = torch.zeros(m, dtype=torch.real(dtype) if dtype.is_complex else dtype, device=device)
    
    V = torch.zeros(n, m, dtype=dtype, device=device)
    
    # Start with random vector for better numerical stability
    v = torch.randn(n, dtype=dtype, device=device)
    v = v / torch.norm(v)
    V[:, 0] = v
    
    w = A @ v
    alpha[0] = torch.real(torch.dot(torch.conj(v), w))
    w = w - alpha[0] * v
    
    for j in range(1, m):
        beta[j-1] = torch.norm(w)
        
        if beta[j-1] < tol:
            m = j
            break
        
        v_prev = V[:, j-1]
        V[:, j] = w / beta[j-1]
        v = V[:, j]
        
        w = A @ v - beta[j-1] * v_prev
        alpha[j] = torch.real(torch.dot(torch.conj(v), w))
        w = w - alpha[j] * v
    
    # Build tridiagonal matrix
    T = torch.diag(alpha[:m])
    if m > 1:
        T = T + torch.diag(beta[:m-1], 1) + torch.diag(beta[:m-1], -1)
    
    # Compute exp(T)
    exp_T = torch.linalg.matrix_exp(T)
    
    # Build approximation
    V_m = V[:, :m]
    expA = V_m @ exp_T @ V_m.T
    
    return expA


def krylov_matrix_exp_mv(A, V, m=None, tol=1e-7):
    """
    Compute exp(A) @ V for multiple vectors V simultaneously.
    More efficient than computing full exp(A) when you only need action on vectors.
    
    Args:
        A: Matrix of shape [..., n, n]
        V: Vectors of shape [..., n, k] where k is number of vectors
        m: Krylov subspace dimension
        tol: Tolerance
    
    Returns:
        exp(A) @ V of shape [..., n, k]
    """
    *batch_shape, n, n2 = A.shape
    *v_batch_shape, n3, k = V.shape
    
    assert n == n2 == n3, "Dimension mismatch"
    assert batch_shape == v_batch_shape, "Batch shapes must match"
    
    if m is None:
        m = min(n - 1, 30)
    
    A_flat = A.reshape(-1, n, n)
    V_flat = V.reshape(-1, n, k)
    batch_size = A_flat.shape[0]
    
    result = torch.zeros_like(V_flat)
    
    for b in range(batch_size):
        for i in range(k):
            result[b, :, i] = _arnoldi_matvec(A_flat[b], V_flat[b, :, i:i+1], m, tol).squeeze()
    
    return result.reshape(*batch_shape, n, k)


# Batched version optimized for your use case
def batched_krylov_matrix_exp(A, m=30, tol=1e-7):
    """
    Optimized batched version for your specific use case.
    Handles complex symmetric (but not Hermitian) matrices efficiently.
    
    Args:
        A: Batched matrices of shape [batch_size, n_dir, n, n]
        m: Krylov subspace dimension (reduce if memory is tight)
        tol: Tolerance for convergence
    
    Returns:
        exp(A) with same shape as input
    """
    batch_size, n_dir, n, _ = A.shape
    device = A.device
    dtype = A.dtype
    
    # Flatten batch dimensions
    A_flat = A.reshape(batch_size * n_dir, n, n)
    expA_flat = torch.zeros_like(A_flat)
    
    # Process each matrix
    for idx in range(batch_size * n_dir):
        # Build Krylov approximation for each matrix
        expA_flat[idx] = _arnoldi_exp_single(A_flat[idx], m, tol)
    
    return expA_flat.reshape(batch_size, n_dir, n, n)


# Your specific use case wrapper
def solve_mf_with_krylov(K1, nu, seq, lambda_mat, relax_mat, krylov_dim=30):
    """
    Drop-in replacement for your PGSE branch using Krylov method.
    
    Args:
        K1: Your K1 matrix [n_amp, n_dir, n_eig, n_eig]
        nu: Your nu vector [n_amp, n_dir, n_eig, 1]
        seq: Your sequence object with delta and Delta attributes
        lambda_mat: Lambda matrix
        relax_mat: Relaxation matrix
        krylov_dim: Dimension of Krylov subspace (tune for memory/accuracy tradeoff)
    
    Returns:
        nu_final: Result [n_amp, n_dir, n_eig, 1]
    """
    # Use Krylov for E1 computation
    E1 = batched_krylov_matrix_exp(-seq.delta * K1, m=krylov_dim)
    
    # E2 can still use standard method since it's smaller/simpler
    E2 = torch.linalg.matrix_exp(-(seq.Delta - seq.delta) * (lambda_mat + relax_mat))
    
    # Apply operations
    tmp1 = torch.matmul(E1, nu)
    tmp2 = torch.matmul(E2.unsqueeze(0).unsqueeze(0), tmp1)
    E1_H = torch.conj(E1).transpose(-1, -2)
    nu_final = torch.matmul(E1_H, tmp2)
    
    return nu_final


# Test function to verify correctness
def test_krylov_exp():
    """Test that Krylov approximation matches torch.linalg.matrix_exp"""
    torch.manual_seed(42)
    
    # Test real matrix
    A_real = torch.randn(10, 10)
    A_real = A_real + A_real.T  # Make symmetric
    
    exp_torch = torch.linalg.matrix_exp(A_real)
    exp_krylov = krylov_matrix_exp(A_real, m=9)
    
    print(f"Real matrix error: {torch.norm(exp_torch - exp_krylov) / torch.norm(exp_torch):.2e}")
    
    # Test complex matrix
    A_complex = torch.randn(10, 10, dtype=torch.complex64)
    A_complex = A_complex + A_complex.T  # Make symmetric (not Hermitian)
    
    exp_torch = torch.linalg.matrix_exp(A_complex)
    exp_krylov = krylov_matrix_exp(A_complex, m=9)
    
    print(f"Complex matrix error: {torch.norm(exp_torch - exp_krylov) / torch.norm(exp_torch):.2e}")
    
    # Test batched
    A_batch = torch.randn(3, 2, 10, 10, dtype=torch.complex64)
    exp_torch = torch.linalg.matrix_exp(A_batch)
    exp_krylov = krylov_matrix_exp(A_batch, m=9)
    
    print(f"Batched matrix error: {torch.norm(exp_torch - exp_krylov) / torch.norm(exp_torch):.2e}")
    
    # Test gradient flow
    A_grad = torch.randn(5, 5, requires_grad=True, dtype=torch.complex64)
    exp_grad = krylov_matrix_exp(A_grad, m=4)
    loss = torch.sum(torch.abs(exp_grad))
    loss.backward()
    
    print(f"Gradient computed: {A_grad.grad is not None}")


# if __name__ == "__main__":
#     test_krylov_exp()