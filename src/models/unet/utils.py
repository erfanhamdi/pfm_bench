import random
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import torchvision.transforms.functional as F

class UNetDataset(Dataset):
    def __init__(self, filename, 
                 split='train', 
                 train_ratio=0.8,
                 val_ratio=0.1,
                 test_ratio=0.1,
                 ds_size=-1, 
                 num_c=1,
                 apply_transforms=True):
        """
        Args:
            filename: Path to the h5 file
            split: One of 'train', 'val', or 'test'
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            ds_size: Number of samples to use (-1 for all)
            num_c: Number of channels
            apply_transforms: Whether to apply data augmentation
        """
        # Define path to files
        self.file_path = filename
        self.num_c = num_c
        self.apply_transforms = apply_transforms
        
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())
            if ds_size > 0:
                data_list = data_list[:ds_size]
        
        # Verify ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
        # Split data
        n_samples = len(data_list)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        if split == 'train':
            self.data_list = np.array(data_list[:train_end])
        elif split == 'val':
            self.data_list = np.array(data_list[train_end:val_end])
        elif split == 'test':
            self.data_list = np.array(data_list[val_end:])
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.data_list)

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
        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
        
            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)
            
            x = data[:self.num_c, 0, :, :]
            y = data[:self.num_c, -1, :, :]
            # normalize x channels separately between 0 and 1
            for ch in range(1, 3):
                x[ch, :, :] = data[ch, 0, :, :] / (((0 * 50) + 1) * 1e-6)
                y[ch, :, :] = data[ch, -1, :, :] / (((100 * 50) + 1) * 1e-6)

                
            # y = data[:self.num_c, -1, :, :]
            # normalize y channels separately
            # for i in range(1, self.num_c):
            #     y[i, ...] /= 0.0051 
                
            x, y = self.transform(x, y)  
            
        return x.double(), y.double()


# For backward compatibility
class UNetDatasetMult(UNetDataset):
    def __init__(self, filename, initial_step=0, if_test=False, test_ratio=0.1, ds_size=-1, num_c=1):
        """Legacy interface that maps to the new UNetDataset"""
        if if_test:
            split = 'test'
            train_ratio = 1 - test_ratio
            val_ratio = 0
        else:
            split = 'train'
            train_ratio = 1 - test_ratio
            val_ratio = test_ratio
            
        super().__init__(
            filename=filename,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=0 if not if_test else test_ratio,
            ds_size=ds_size,
            num_c=num_c
        )
        
        # For backward compatibility
        if not if_test:
            with h5py.File(self.file_path, 'r') as h5_file:
                data_list = sorted(h5_file.keys())[:ds_size]
            test_idx = int(len(data_list) * (1-test_ratio))
            self.val_data_list = np.array(data_list[test_idx:])
    
    def __len__(self):
        if hasattr(self, 'val_data_list'):
            return len(self.data_list), len(self.val_data_list)
        return len(self.data_list)