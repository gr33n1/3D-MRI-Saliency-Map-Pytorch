import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib


class DummyDataset(Dataset):
    def __init__(self, base_dir, samples, transform=None):
        """
        Args:
            base_dir (str): Base directory where the dataset is stored.
            samples (list): List of tuples with folder names and filenames.
                            Example: [('Guys', 'IXI158-Guys-0783_T1'), ...]
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.base_dir = base_dir
        self.samples = samples
        self.transform = transform
        self.labels = {'Guys': 1, 'HH': 0}  # Mapping folder names to labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, filename = self.samples[idx]
        label = self.labels[folder]

        # Load the NIfTI file
        file_path = os.path.join(self.base_dir, folder, filename, f"w{filename}.nii")
        nifti_data = nib.load(file_path)
        image = nifti_data.get_fdata().astype(np.float32)

        # Convert image and label to tensors
        image_tensor = torch.tensor(image)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return filename, image_tensor, label_tensor
