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
        
        # Find all hdf5 files in the directory
        hdf5_files = glob.glob(os.path.join(self.data_dir, "*.hdf5"))
        if not hdf5_files:
            raise ValueError(f"No hdf5 files found in directory: {self.data_dir}")
        
        # Extract list of seeds (filenames without extension)
        all_data_list = []
        self.file_mapping = []  # Maps data index to (file_path, seed_name)
        
        for file_path in sorted(hdf5_files):
            # Get filename without extension as seed name
            seed_name = os.path.splitext(os.path.basename(file_path))[0]
            # Pad seed name to 8 digits with zeros
            seed_name = seed_name.zfill(8)
            all_data_list.append(seed_name)
            self.file_mapping.append((file_path, seed_name))
        
        if ds_size > 0:
            all_data_list = all_data_list[:ds_size]
            self.file_mapping = self.file_mapping[:ds_size]
        
        # Verify ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
        # Split data
        n_samples = len(all_data_list)
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
            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)
            
            x = data[:self.num_c, 0, :, :]
            y = data[:self.num_c, -1, :, :]
            # normalize x channels separately between 0 and 1
            if self.num_c != 1:
                for ch in range(1, 3):
                    x[ch, :, :] = data[ch, 0, :, :] / (((0 * 50) + 1) * 1e-6)
                    y[ch, :, :] = data[ch, -1, :, :] / (((100 * 50) + 1) * 1e-6)
            x, y = self.transform(x, y)  
            
        return x.double(), y.double()

if __name__ == "__main__":
    data_dir = "data/tension/spect"
    dataset = UNetDataset(datadir=data_dir, split="train", ds_size=-1)
    print(len(dataset))
    print(dataset[0])