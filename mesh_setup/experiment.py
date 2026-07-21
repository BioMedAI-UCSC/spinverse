import os
import torch as torch
import yaml

from mesh_setup.update_pde import update_pde
from mesh_setup.prepare_pde import prepare_pde
from mesh_setup.prepare_experiments import prepare_experiments
from mesh_setup.PGSE import PGSE
from mesh_setup.gradient_unit_sphere import unit_sphere

class ExperimentSetup():
    def __init__(self, setup_file):
        self.setup_name, _ = os.path.splitext(setup_file)
        
        with open(setup_file, 'r') as f:
            setup_dict = yaml.safe_load(f)

        # Load parameters
        self.geometry = self._load_geometry(setup_dict) 
        self.gradient = self._load_gradient(setup_dict)
        self.pde = setup_dict['pde']
        self.mf = setup_dict['mf']
        
        # Initialize pde and experiments
        self._init_pde()
        self._init_experiment()

        # TODO add torch file save if needed/wanted

    def _load_geometry(self, setup_dict):
        geometry = setup_dict['geometry']
        geometry['deformation'] = torch.tensor(geometry['deformation'])

        return geometry

    def _load_gradient(self, setup_dict):
        gradient = setup_dict['gradient']
        
        # Load b values
        if isinstance(gradient['values'], dict):
            gradient['values'] = torch.arange(gradient['values']['start'], 
                                              gradient['values']['stop'], 
                                              gradient['values']['incr'])
        else:
            gradient['values'] = torch.tensor(gradient['values'])

        sequences = []
        
        # Load diffusion sequences
        if gradient['sequence_type'] == 'PGSE':
            for seq in gradient['sequence_deltas']:
                sequences.append(PGSE(seq['delta'], seq['Delta']))
        else:
            #TODO add handling for OGSE
            pass
    
        gradient['sequences'] = sequences

        # Process gradient directions
        if 'direction_file' in gradient.keys():
            with open(gradient['direction_file'], "r") as f:
                dirs = [float(line.strip()) for line in f][1:]
                gradient['directions'] = torch.tensor(dirs).reshape(-1, 3).T
        elif 'num_directions' in gradient.keys():
            gradient['directions'] = unit_sphere(gradient['num_directions'])
        else:
            raise ValueError("Gradient config must have either 'direction_file' or 'num_directions'")

        return gradient

    def _init_pde(self):
    
        # Extract compartment information using attribute access
        ncell = self.geometry["ncell"]
        cell_shape = self.geometry["cell_shape"]
        include_in = self.geometry["include_in"]
        in_ratio = self.geometry["in_ratio"]
        include_ecs = self.geometry["ecs_shape"] != "no_ecs"
        ecs_ratio = self.geometry["ecs_ratio"]
        pde = self.pde

        # Check for correct radius ratios and that neurons do not have in-compartments
        assert not include_in or (0 < in_ratio < 1 and cell_shape != "neuron")
        assert not include_ecs or (0 < ecs_ratio)

        # Number of compartments
        ncompartment = (1 + include_in) * ncell + include_ecs

        # Find number of boundaries
        if cell_shape == "cylinder":
            nboundary = (include_in + 1) * 2 * ncell + include_ecs
        elif cell_shape in ["sphere", "neuron"]:
            nboundary = (include_in + 1) * ncell + include_ecs

        compartments = []
        boundaries = []

        if include_in:
            compartments += ["in"] * ncell
            boundaries += ["in,out"] * ncell

        compartments += ["out"] * ncell

        if include_ecs:
            compartments.append("ecs")
            boundaries += ["out,ecs"] * ncell
        else:
            boundaries += ["out"] * ncell

        if cell_shape == "cylinder":
            if include_in:
                boundaries += ["in"] * ncell
            boundaries += ["out"] * ncell
            if include_ecs:
                boundaries.append("ecs")
        elif cell_shape in ["sphere", "neuron"] and include_ecs:
            boundaries.append("ecs")

        # Diffusion coefficients (tensorize if scalars)
        diffusivity_in = pde["diffusivity_in"] * torch.eye(3)
        diffusivity_out = pde["diffusivity_out"] * torch.eye(3)
        diffusivity_ecs = pde["diffusivity_ecs"] * torch.eye(3)

        diffusivity = torch.zeros(3, 3, ncompartment)
        relaxation = torch.zeros(ncompartment)
        initial_density = torch.zeros(ncompartment)
        permeability = torch.zeros(nboundary)

        for i, comp in enumerate(compartments):
            if comp == "in":
                diffusivity[:, :, i] = diffusivity_in
                relaxation[i] = pde["relaxation_in"]
                initial_density[i] = pde["initial_density_in"]
            elif comp == "out":
                diffusivity[:, :, i] = diffusivity_out
                relaxation[i] = pde["relaxation_out"]
                initial_density[i] = pde["initial_density_out"]
            elif comp == "ecs":
                diffusivity[:, :, i] = diffusivity_ecs
                relaxation[i] = pde["relaxation_ecs"]
                initial_density[i] = pde["initial_density_ecs"]

        for i, boundary in enumerate(boundaries):
            if boundary == "in":
                permeability[i] = pde["permeability_in"]
            elif boundary == "out":
                permeability[i] = pde["permeability_out"]
            elif boundary == "ecs":
                permeability[i] = pde["permeability_ecs"]
            elif boundary == "in,out":
                permeability[i] = pde["permeability_in_out"]
            elif boundary == "out,ecs":
                permeability[i] = pde["permeability_out_ecs"]

        # Update the self.pde dictionary directly
        self.pde["diffusivity"] = diffusivity.T
        self.pde["relaxation"] = relaxation
        self.pde["initial_density"] = initial_density
        self.pde["permeability"] = permeability
        self.pde["compartments"] = compartments
        self.pde["boundaries"] = boundaries

        return self.pde
    
    def _init_experiment(self):
        # We assume we are working with water protons
        gamma = 0.2675222005

        # Assign b-values and q-values
        namplitude = len(self.gradient["values"])
        nsequence = len(self.gradient["sequences"])
        self.gradient["bvalues"] = torch.zeros(namplitude, nsequence)
        self.gradient["qvalues"] = torch.zeros(namplitude, nsequence)

        for iseq in range(nsequence):
            for i, value in enumerate(self.gradient["values"]):
                bnq = self.gradient["sequences"][iseq].bvalue_no_q()
                # print(bnq)
                if self.gradient["values_type"] == "g":
                    q = value * gamma / 1e6
                    self.gradient["qvalues"][i, iseq] = q
                    self.gradient["bvalues"][i, iseq] = bnq * (
                        q**2
                    )  # Assumedbnq = 1 for simplification earlier
                elif self.gradient["values_type"] == "q":
                    self.gradient["qvalues"][i, iseq] = value
                    self.gradient["bvalues"][i, iseq] = bnq * (
                        value**2
                    )  # Assumed bnq = 1 for simplification earlier
                elif self.gradient["values_type"] == "b":
                    self.gradient["bvalues"][i, iseq] = value
                    self.gradient["qvalues"][i, iseq] = torch.sqrt(
                        value / bnq
                    )  # Assumed bnq = 1 for simplification

        # Normalize gradient directions
        norm = torch.norm(self.gradient["directions"], p=2, dim=0)
        # print(norm.shape)
        # print(self.gradient['directions'].shape)
        self.gradient["directions"] = self.gradient["directions"] / norm