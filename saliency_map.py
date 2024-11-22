import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


from Dummy3DModel import Dummy3DModel
from DummyDataset import DummyDataset

base_dir = "IXI"
samples = [
    ('Guys', 'IXI158-Guys-0783_T1'),
    ('Guys', 'IXI492-Guys-1022_T1'),
    ('HH', 'IXI516-HH-2297_T1'),
    ('HH', 'IXI526-HH-2392_T1')
]

# Load your dataset
dataset = DummyDataset(base_dir=base_dir, samples=samples)
ixi_loader = DataLoader(dataset, batch_size=1)

# Instantiate the model - use here your own model with pretrained weights
dummy_model = Dummy3DModel()

# Set up the model for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_model.to(device)
dummy_model.eval()

output_dir = "ixi_example_saliency_maps"
os.makedirs(output_dir, exist_ok=True)

for idx, (patient_id, input_tensor, label) in enumerate(ixi_loader):
    input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, 1, 30, 95, 79)
    input_tensor.requires_grad = True

    # Forward pass
    output = dummy_model(input_tensor)
    output = torch.sigmoid(output)
    predicted_class = torch.argmax(output, dim=1)

    # Backward pass
    dummy_model.zero_grad()
    output[0, predicted_class].backward()

    # Compute saliency map by taking the absolute maximum gradient across the volume
    gradients = input_tensor.grad.detach().cpu().numpy()
    saliency_map = np.max(np.abs(gradients), axis=1)

    # Normalize the saliency map
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Optional - Save saliency map as a NIfTI file
    # saliency_nii = nib.Nifti1Image(saliency_map.squeeze(), affine=np.eye(4))
    # nib.save(saliency_nii, f"{output_dir}/{patient_id[0]}_saliency.nii")

    # Visualization
    plt.imshow(saliency_map[0, 5, :, :], cmap='hot')
    plt.title(f"Saliency Map - {patient_id[0]} - Slice {5}")
    plt.colorbar()
    plt.savefig(f"{output_dir}/{patient_id[0]}_slice_{5}.png")
    plt.close()
