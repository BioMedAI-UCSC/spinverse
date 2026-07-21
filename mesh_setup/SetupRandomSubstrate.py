import torch
from mesh_setup.directions import directions
from mesh_setup.PGSE import PGSE


class SetupRandomSubstrate:
    def __init__(self):
        # File name to load or store cell description, surface geometry, mesh, and simulation results
        # self.name = "mesh_files/spheres/1sphere_analytical"
        # Geometry parameters
        self.geometry = {
            "cell_shape": "cylinder",
            "ncell": 1,
            "rmin": 5,
            "rmax": 5,
            "dmin": 0.1,
            "dmax": 0.2,
            "height": 20,
            "deformation": torch.tensor([0.0, 0.0]),
            "include_in": False,  # True
            "in_ratio": 0.6,
            "ecs_shape": "no_ecs",  # "ecs_shape": "tight_wrap" / "no_ecs"
            "ecs_ratio": 0.5,
            "refinement": 1,
        }

        # PDE parameters - initially set, can be modified later
        self.pde = {
            "diffusivity_in": 0.002,
            "diffusivity_out": 0.002,
            "diffusivity_ecs": 0.002,
            "relaxation_in": float("inf"),
            "relaxation_out": float("inf"),
            "relaxation_ecs": float("inf"),
            "initial_density_in": 1.0,
            "initial_density_out": 1.0,
            "initial_density_ecs": 1.0,
            "permeability_in_out": 1e-4,
            "permeability_out_ecs": 1e-4,
            "permeability_in": 0,
            "permeability_out": 0,
            "permeability_ecs": 0,
        }

        # Gradient sequences
        self.gradient = {
            # "values": torch.tensor([0.0, 800.0, 1500.0]),  # s/mm^2 (moderate b => good κ sensitivity)
            # "values": torch.tensor([0.0, 200.0, 800.0, 1500.0]),  # s/mm^2 (moderate b => good κ sensitivity)
            # "values": torch.arange(
            #     0, 1500, 750
            # ),
            # "values_type": "b",
            # "sequences": [
            #     PGSE(5000, 10000),   # δ=5 ms, Δ=10 ms  => t_eff ≈ 8.3 ms
            #     PGSE(5000, 30000),   # δ=5 ms, Δ=30 ms  => t_eff ≈ 28.3 ms
            #     PGSE(5000, 80000),   # δ=5 ms, Δ=80 ms  => t_eff ≈ 78.3 ms
            # ],
            "values": torch.tensor([0., 1000., 2000., 4000., 6000.0]),  # (+10000 if SNR allows)
            "values_type": "b",
            # "sequences": [
            #     PGSE(5000, 10000),  # Δ = 10 ms  → ℓ ≈ 6.8 µm
            #     PGSE(5000, 20000),  # Δ = 20 ms  → ℓ ≈ 9.6 µm
            #     PGSE(5000, 40000),  # Δ = 40 ms  → ℓ ≈ 13.6 µm
            #     PGSE(5000, 80000),  # Δ = 80 ms  → ℓ ≈ 19.2 µm
            # ],
            "sequences": [
                # PGSE(2000, 4000),   # Δ=4ms  - catches fast exchange (τ<10ms)
                PGSE(3000, 8000),   # Δ=8ms  - intermediate 5micron
                PGSE(4000, 16000),  # Δ=16ms - slower exchange (τ=10-30ms) 7micron
                PGSE(4000, 12000),  # Δ=12ms - long-time reference 9.6 micron
                PGSE(5000, 25000),  # Δ=25ms - long-time reference 9.6 micron
                # PGSE(5000, 40000),   # 12.4 micron
            ],
            "directions": directions("mesh_setup/PointSets/Elec030.txt"),
        }
        # --- Matrix-formalism budget (lean but safe) ---
        self.mf = {
            # "length_scale": 1,
            "length_scale": 1.0, # for micro meter scaling
            "neig_max": 760,   # 500–600 is a good fixed budget; you don’t need 1000
            "ninterval": 380,  # fewer intervals is fine with δ=5 ms
        }