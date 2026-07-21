import numpy as np
import torch

def unit_sphere(n_points: int) -> torch.Tensor:
    """
    Create evenly distributed points on the unit sphere using the Fibonacci lattice.
    
    Parameters
    ----------
    n_points : int
        Number of points to generate on the sphere.
    
    Returns
    -------
    points : (3, n_points) ndarray
        Array of unit vectors evenly distributed on the sphere.
    """
    # Golden angle in radians
    golden_angle = np.pi * (3. - np.sqrt(5.))  
    
    # Indices
    i = np.arange(n_points)
    
    # z values uniformly spaced between -1 and 1
    z = 1 - (2*i + 1)/n_points  
    radius = np.sqrt(1 - z*z)
    
    # azimuthal angles
    theta = golden_angle * i  
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    points = np.stack((x, y, z), axis=0)
    
    # remove negative zeros (for aesthetics only, like MATLAB code)
    points[np.isclose(points, 0)] = 0.0
    
    return torch.from_numpy(points)