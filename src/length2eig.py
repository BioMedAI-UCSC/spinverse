import torch


def length2eig(length_scales, diffusivity):
    """
    Convert length scales into Laplace eigenvalues
    """
    if length_scales == 0 or length_scales is None:
        return float('inf')
    
    return (diffusivity * (torch.pi**2)) / (length_scales**2)