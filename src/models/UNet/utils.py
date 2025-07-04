import random
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import torchvision.transforms.functional as F
import os
import glob

class UNetDataset(Dataset):
    def __init__(self, datadir, 
                 split='train', 
                 train_ratio=0.8,
                 val_ratio=0.1,
                 test_ratio=0.1,
                 ds_size=-1, 
                 num_c=1,
                 apply_transforms=True):
        """
        Args:
            datadir: Path to the directory containing hdf5 files
            split: One of 'train', 'val', or 'test'
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            ds_size: Number of samples to use (-1 for all)
            num_c: Number of channels
            apply_transforms: Whether to apply data augmentation
        """
        # Define path to directory
        self.data_dir = datadir
        self.num_c = num_c
        self.apply_transforms = apply_transforms
        
        # Discover HDF5 files and build (file_path, seed) mapping in one pass
        hdf5_files = sorted(glob.glob(os.path.join(self.data_dir, "*.hdf5")))
        if not hdf5_files:
            raise ValueError(f"No hdf5 files found in directory: {self.data_dir}")

        self.file_mapping = [
            (fp, os.path.splitext(os.path.basename(fp))[0].zfill(8))
            for fp in hdf5_files
        ]

        # Optionally limit dataset size
        if ds_size > 0:
            self.file_mapping = self.file_mapping[:ds_size]
        
        # Verify ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
        # Split data
        n_samples = len(self.file_mapping)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        if split == 'train':
            self.file_mapping = self.file_mapping[:train_end]
        elif split == 'val':
            self.file_mapping = self.file_mapping[train_end:val_end]
        elif split == 'test':
            self.file_mapping = self.file_mapping[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.file_mapping)

    def transform(self, x, y):
        """Apply data augmentation transforms"""
        if not self.apply_transforms:
            return x, y
            
        angle = random.choice([0, 90, 180, 270])
        x = F.rotate(x, angle)
        y = F.rotate(y, angle)

        # Random horizontal flip
        if random.random() > 0.5:
            x = F.hflip(x)
            y = F.hflip(y)

        # Random vertical flip
        if random.random() > 0.5:
            x = F.vflip(x)
            y = F.vflip(y)
        
        return x, y
        
    def __getitem__(self, idx):
        # Get file path and seed name for this index
        file_path, seed_name = self.file_mapping[idx]
        
        # Open file and read data
        with h5py.File(file_path, 'r') as hdf5_file:
            seed_group = hdf5_file[seed_name]
            # Load data once and keep in float32 torch tensor (shape preserved)
            data = torch.from_numpy(np.array(seed_group["data"], dtype=np.float32))
            
            x = data[:self.num_c, 0]      # first timestep
            y = data[:self.num_c, -1]     # last timestep

            # Vectorised normalisation (for channels beyond the first)
            if self.num_c > 1:
                # Scale factors based on timestep index
                scale_x = 1e-6                                   # t = 0
                t_last = data.shape[1] - 1
                scale_y = ((t_last * 50) + 1) * 1e-6
                x[1:self.num_c] = x[1:self.num_c] / scale_x
                y[1:self.num_c] = y[1:self.num_c] / scale_y
            
            x, y = self.transform(x, y)  
            
        return x.double(), y.double()

if __name__ == "__main__":
    data_dir = "data/tension/spect"
    dataset = UNetDataset(datadir=data_dir, split="train", ds_size=-1)
    print(len(dataset))
    print(dataset[0])