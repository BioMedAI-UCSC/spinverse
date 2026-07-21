import torch
import os


def split_pth_file(input_path, output_dir):
    # Load the tensor from the .pth file
    tensor = torch.load(input_path)  # Assuming it's a single tensor inside the file

    # Ensure it's a 3D tensor
    if not isinstance(tensor, torch.Tensor) or tensor.ndimension() != 3:
        raise ValueError("Expected a 3D tensor of shape (A, B, C) in the .pth file.")

    A, B, C = tensor.shape  # Get the dimensions

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through A and B dimensions and save each (C,) tensor
    for i in range(A):
        for j in range(B):
            file_name = f"tensor_{i}_{j}.pth"
            output_path = os.path.join(output_dir, file_name)

            # Extract and save the (C,) tensor
            torch.save(tensor[i, j], output_path)

    print(f"Saved {A * B} files in {output_dir}")


# Example usage
input_pth_file = "reference_signal.pth"  # Replace with your file path
output_directory = "output_tensors"  # Replace with desired output folder
split_pth_file(input_pth_file, output_directory)
