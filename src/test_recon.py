import torch
import torch.optim as optim
import plotly.graph_objects as go
import numpy as np
from src.get_vol_sa import get_vol_sa
from src.calculate_generalized_mean_diffusivity import calculate_generalized_mean_diffusivity
from src.length2eig import length2eig
from src.compute_laplace_eig_diff_v2 import compute_laplace_eig_diff
from src.eig2length import eig2length
from src.solve_mf_v4 import solve_mf
from src.split_mesh import split_mesh
import importlib
from mesh_setup import microstructure
importlib.reload(microstructure)
from mesh_setup.microstructure import microstructure

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_mesh(filename):
    """Load mesh for visualization"""
    mesh_data = torch.load(filename, weights_only=True)
    return {
        'points': mesh_data['points'].to(DEVICE),
        'facets': mesh_data['facets'].to(DEVICE), 
        'elements': mesh_data['elements'].to(DEVICE)
    }

def visualize_mesh(mesh, title="Mesh", show_points=True, opacity=0.7):
    """Visualize mesh with plotly"""
    vertices = mesh['points'].T.cpu().numpy()  # [N, 3]
    faces = mesh['facets'].T.cpu().numpy()     # [M, 3]
    
    print(f"📊 Visualizing: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    
    surface_mesh = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=opacity, color='lightblue', name='Surface'
    )
    
    data = [surface_mesh]
    
    if show_points:
        face_vertices = vertices[np.unique(faces.flatten())]
        points = go.Scatter3d(
            x=face_vertices[:, 0], y=face_vertices[:, 1], z=face_vertices[:, 2],
            mode='markers', marker=dict(size=3, color='red'),
            name=f'Vertices ({len(face_vertices)})'
        )
        data.append(points)
    
    fig = go.Figure(data=data)
    fig.update_layout(
        title=title, 
        scene=dict(aspectmode='cube'),
        width=800, height=600
    )
    return fig

def main():
    print("Starting mesh optimization...")
    
    # Load meshes
    print("Loading meshes...")
    mesh1 = load_mesh("small_mesh.pth")  # reference mesh
    base_mesh = load_mesh("base_grid.pth")  # optimization mesh
    
    print(f"Reference mesh: {mesh1['points'].shape[1]} points, {mesh1['facets'].shape[1]} facets")
    print(f"Base mesh: {base_mesh['points'].shape[1]} points, {base_mesh['facets'].shape[1]} facets")
    
    # Setup microstructure
    print("Setting up microstructure...")
    setup = microstructure()
    
    # Create femesh structures
    femesh_mesh1 = {
        'points': mesh1['points'],
        'facets': mesh1['facets'],
        'elements': mesh1['elements'],
        'facetmarkers': torch.ones(mesh1['facets'].shape[1], dtype=torch.long, device=DEVICE),
        'elementmarkers': torch.ones(mesh1['elements'].shape[1], dtype=torch.long, device=DEVICE)
    }
    
    femesh_base = {
        'points': base_mesh['points'],
        'facets': base_mesh['facets'], 
        'elements': base_mesh['elements'],
        'facetmarkers': torch.ones(base_mesh['facets'].shape[1], dtype=torch.long, device=DEVICE),
        'elementmarkers': torch.ones(base_mesh['elements'].shape[1], dtype=torch.long, device=DEVICE)
    }
    
    # Split meshes
    print("Splitting meshes...")
    femesh_mesh1_split = split_mesh(femesh_mesh1)
    femesh_base_split = split_mesh(femesh_base)
    
    # Get volumes and surface areas
    print("Computing volumes and surface areas...")
    surface_area_contribution_mesh1 = torch.full((femesh_mesh1['facets'].shape[1],), 1, dtype=torch.float32, device=DEVICE)
    volumes_mesh1, surface_areas_mesh1 = get_vol_sa(femesh_mesh1_split, surface_area_contribution_mesh1)
    
    surface_area_contribution_base = torch.full((femesh_base['facets'].shape[1],), 1, dtype=torch.float32, device=DEVICE)
    volumes_base, surface_areas_base = get_vol_sa(femesh_base_split, surface_area_contribution_base)
    
    # Calculate reference signal (mesh1 with faces_prob = 1e-4)
    print("Computing reference signal...")
    faces_prob_ref = torch.full((femesh_mesh1['facets'].shape[1],), 1e-4, dtype=torch.float32, device=DEVICE)
    
    mean_diffusivity_mesh1 = calculate_generalized_mean_diffusivity(setup.pde['diffusivity'].to(DEVICE), volumes_mesh1)
    eiglim_mesh1 = length2eig(setup.mf['length_scale'], mean_diffusivity_mesh1)
    tap_eig_mesh1 = compute_laplace_eig_diff(femesh_mesh1_split, setup, setup.pde, eiglim_mesh1, setup.mf['neig_max'], faces_prob_ref)
    tap_eig_mesh1['length_scales'] = eig2length(tap_eig_mesh1['values'], mean_diffusivity_mesh1)
    mf_signal_ref = solve_mf(femesh_mesh1_split, setup, tap_eig_mesh1, faces_prob=faces_prob_ref)
    signal_ref = torch.abs(torch.abs(torch.abs(mf_signal_ref["signal_allcmpts"]) / torch.abs(mf_signal_ref["signal_allcmpts"][0, :, 0]).view(1, -1, 1)).cpu())
    
    print(f"Reference signal shape: {signal_ref.shape}")
    print(f"Reference signal range: {signal_ref.min():.6f} to {signal_ref.max():.6f}")
    
    # Initialize optimizable faces_prob for base mesh
    faces_prob = torch.full((femesh_base['facets'].shape[1],), 1e-4, dtype=torch.float32, device=DEVICE, requires_grad=True)
    
    # Setup optimizer
    optimizer = optim.Adam([faces_prob], lr=0.01)
    
    # Optimization loop
    num_iterations = 100
    losses = []
    
    print(f"Starting optimization for {num_iterations} iterations...")
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass with current faces_prob
        mean_diffusivity_base = calculate_generalized_mean_diffusivity(setup.pde['diffusivity'].to(DEVICE), volumes_base)
        eiglim_base = length2eig(setup.mf['length_scale'], mean_diffusivity_base)
        tap_eig_base = compute_laplace_eig_diff(femesh_base_split, setup, setup.pde, eiglim_base, setup.mf['neig_max'], faces_prob)
        tap_eig_base['length_scales'] = eig2length(tap_eig_base['values'], mean_diffusivity_base)
        mf_signal_base = solve_mf(femesh_base_split, setup, tap_eig_base, faces_prob=faces_prob)
        signal_base = torch.abs(torch.abs(torch.abs(mf_signal_base["signal_allcmpts"]) / torch.abs(mf_signal_base["signal_allcmpts"][0, :, 0]).view(1, -1, 1)).cpu())
        
        # Compute loss (MSE between signals)
        # Interpolate/resize if needed - for now assume same size or take subset
        min_size = min(signal_ref.shape[-1], signal_base.shape[-1])
        loss = torch.mean((signal_ref[:, :, :min_size] - signal_base[:, :, :min_size])**2)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_([faces_prob], max_norm=1.0)
        
        # Update
        optimizer.step()
        
        # Clamp faces_prob to reasonable range
        with torch.no_grad():
            faces_prob.clamp_(1e-8, 1e-1)
        
        losses.append(loss.item())
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Loss = {loss.item():.8f}, faces_prob range: {faces_prob.min():.6f} to {faces_prob.max():.6f}")
    
    print(f"Optimization complete!")
    print(f"Final loss: {losses[-1]:.8f}")
    print(f"Final faces_prob range: {faces_prob.min():.6f} to {faces_prob.max():.6f}")
    
    # Save optimized faces_prob
    torch.save({
        'faces_prob': faces_prob.detach().cpu(),
        'losses': losses,
        'mesh_file': 'base_grid.pth'
    }, 'optimized_faces_prob.pth')
    
    print("Saved optimized faces_prob to optimized_faces_prob.pth")
    
    # Optional: Visualize final result
    print("Creating final signal comparison...")
    with torch.no_grad():
        mean_diffusivity_final = calculate_generalized_mean_diffusivity(setup.pde['diffusivity'].to(DEVICE), volumes_base)
        eiglim_final = length2eig(setup.mf['length_scale'], mean_diffusivity_final)
        tap_eig_final = compute_laplace_eig_diff(femesh_base_split, setup, setup.pde, eiglim_final, setup.mf['neig_max'], faces_prob)
        tap_eig_final['length_scales'] = eig2length(tap_eig_final['values'], mean_diffusivity_final)
        mf_signal_final = solve_mf(femesh_base_split, setup, tap_eig_final, faces_prob=faces_prob)
        signal_final = torch.abs(torch.abs(torch.abs(mf_signal_final["signal_allcmpts"]) / torch.abs(mf_signal_final["signal_allcmpts"][0, :, 0]).view(1, -1, 1)).cpu())
        
        final_loss = torch.mean((signal_ref[:, :, :min_size] - signal_final[:, :, :min_size])**2)
        print(f"Final verification loss: {final_loss.item():.8f}")

if __name__ == "__main__":
    main()
