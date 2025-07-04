import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import glob

class FNODataset(Dataset):
    def __init__(self, datadir,
                split='train',
                initial_step=10,
                train_ratio=0.8, 
                val_ratio=0.1,
                test_ratio=0.1,
                ds_size=-1,
                num_c=1,
                return_seed=False
                 ):
        """
        :param datadir: directory that contains the dataset
        :type datadir: STR
        :param split: which data split to use ('train', 'val', 'test')
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional
        :param train_ratio: fraction of data for training
        :param val_ratio: fraction of data for validation
        :param test_ratio: fraction of data for testing
        :param ds_size: limit dataset size (-1 for all)
        :param num_c: number of channels (currently unused)
        :param return_seed: whether to return the data seed along with data
        :type return_seed: BOOL, optional
        """
        
        # Define path to files
        self.datadir = datadir
        self.num_c = num_c
        self.initial_step = initial_step
        self.return_seed = return_seed
        
        # Extract list of seeds
        hdf5_files = sorted(glob.glob(os.path.join(self.datadir, "*.hdf5")))
        if not hdf5_files:
            raise ValueError(f"No hdf5 files found in directory: {self.datadir}")

        # Build (file_path, seed) mapping and optionally truncate for ds_size.
        self.file_mapping = [
            (fp, os.path.splitext(os.path.basename(fp))[0].zfill(8))
            for fp in hdf5_files
        ]
        if ds_size > 0:
            self.file_mapping = self.file_mapping[:ds_size]
            
        # Verify ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
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
    
    def __getitem__(self, idx):
        # Get the seed for the current index
        file_path, data_seed = self.file_mapping[idx]
        
        # Open file and read data
        with h5py.File(file_path, 'r') as h5_file:
            seed_group = h5_file[data_seed]
        
            # Load data, convert to torch tensor, and permute to [x1, x2, t, v]
            data = torch.from_numpy(np.array(seed_group["data"], dtype=np.float32)).permute(2, 3, 1, 0).contiguous()
            
            # Normalize each channel (beyond the first) across all timesteps
            if self.num_c > 1:
                t = torch.arange(data.shape[2], dtype=data.dtype, device=data.device)  # time dimension
                scale = ((t * 50) + 1) * 1e-6  # [T]
                data[..., 1:self.num_c] = data[..., 1:self.num_c] / scale.view(1, 1, -1, 1)
            
            resolution = data.shape[0]                                               
            
            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            x = torch.from_numpy(np.array(seed_group["grid"]["x"], dtype=np.float32)).view(resolution, resolution)
            y = torch.from_numpy(np.array(seed_group["grid"]["y"], dtype=np.float32)).view(resolution, resolution)
            grid = torch.stack((x,y),axis=-1)
            
        # Prepare return values
        x_data = data[:, :, :self.initial_step, :self.num_c] # Input steps
        y_data = data[..., :self.num_c]                             # Full sequence for target/comparison

        if self.return_seed:
            return x_data, y_data, grid, data_seed # Return seed if requested
        else:
            return x_data, y_data, grid # Default behavior

if __name__ == "__main__":
    data_dir = "data/tension/spect"    
    dataset = FNODataset(data_dir, split='train', initial_step=10, ds_size=-1)
    x, y, grid = dataset[0]
    print(f"Shapes x={x.shape}, y={y.shape}, grid={grid.shape}")
    print("\n--- Testing with return_seed ---")
    dataset_seed = FNODataset(data_dir, split='test', initial_step=10, ds_size=-1, return_seed=True)
    x_s, y_s, g_s, seed = dataset_seed[0]
    print(f"Returned Seed: {seed}")
    print(f"Shapes x={x_s.shape}, y={y_s.shape}, grid={g_s.shape}")