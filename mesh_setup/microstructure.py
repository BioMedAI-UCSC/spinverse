from mesh_setup import Setup1AxonAnalytical_LowRes as Setup1AxonAnalytical_LowRes
from mesh_setup.update_pde import update_pde
from mesh_setup.prepare_pde import prepare_pde
from mesh_setup.prepare_experiments import prepare_experiments
import os
import torch as tch

DEVICE = 'cuda:0' if tch.cuda.is_available() else 'cpu'

def microstructure():
    # Define the file path
    file_path = 'micro_setup.pth'
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Load the existing setup
        setup = tch.load(file_path, weights_only=False, map_location=DEVICE)
        print("Loaded existing setup from micro_setup.pth")
    else:
        # Create new setup if file doesn't exist
        setup = Setup1AxonAnalytical_LowRes.Setup1AxonAnalytical_LowRes()
        update_pde(setup)
        # Call prepare_pde with this setup instance
        U_pde = prepare_pde(setup)
        # Inspect the output
        print("Updated PDE Parameters:")
        for key, value in U_pde.items():
            print(f"{key}: {value}")

        # Prepare b-values and pde
        prepare_experiments(setup)
        
        # Save the setup to a .pth file
        tch.save(setup, file_path)
        print("Created and saved new setup to micro_setup.pth")

    return setup
