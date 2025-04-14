# from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any
import os

class NumpyDataset(Dataset):
    def __init__(self, data_dir, prefix='train'):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        # Load data
        self.data = np.load(os.path.join(data_dir, f'{prefix}_data.npz'), allow_pickle=True)
        self.simulations = self.data['simulations']
        self.params = self.data['params']

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, idx):
        x = self.params[idx]
        y = self.simulations[idx]

        x = torch.from_numpy(x).to(torch.float64)
        y = torch.from_numpy(y).to(torch.float64)

        return x, y
    

class LotkaVolterraDataset(NumpyDataset):
    def __init__(self, data_dir, prefix='train'):
        super().__init__(data_dir, prefix)

        # Calculate mean and std across all simulations and time steps
        self.s = np.mean(np.abs(self.simulations), axis=1, keepdims=True)
        # Normalize the data
        self.y_normalized = (self.simulations) / self.s

    def __getitem__(self, idx):
        x = self.params[idx]
        y = self.y_normalized[idx]

        x = torch.from_numpy(x).to(torch.float64)
        y = torch.from_numpy(y).to(torch.float64)

        return x, y

class MZBDataset(Dataset):
    def __init__(self, data_dir, prefix='train'):
        self.data_dir = data_dir
        
        # Load data
        data_path = os.path.join(data_dir, f'{prefix}_data.npz')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.data = np.load(data_path, allow_pickle=True)
        self.simulations = self.data['simulations']
        self.params = self.data['params']

        # Precompute scales for normalization
        self.scales = np.mean(np.abs(self.simulations), axis=1)[:, 0]

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, idx):
        # Retrieve data
        x = self.params[idx]
        y = self.simulations[idx]

        # Apply transformations
        mzb, donor_ki67, host_ki67, nfd = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        mzb = mzb/3e6
        donor_ki67 = np.arcsin(np.sqrt(donor_ki67))  # Transform donor_ki67
        host_ki67 = np.arcsin(np.sqrt(host_ki67))  # Transform host_ki67
        nfd = nfd  # No transformation for nfd

        # Stack the transformed data
        y = np.array([mzb, donor_ki67, host_ki67, nfd]).T
        
        # Convert to PyTorch tensors
        x = torch.from_numpy(x).float()  # Use float32 for efficiency
        y = torch.from_numpy(y).float()

        return x, y


def create_dataloaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Check data directory existence
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    train_dataset = LotkaVolterraDataset(data_dir, prefix='train')
    val_dataset = LotkaVolterraDataset(data_dir, prefix='val')

    # Ensure data type matches precision setting
    train_dataset.simulations = train_dataset.simulations.astype('float64')
    train_dataset.params = train_dataset.params.astype('float64')
    val_dataset.simulations = val_dataset.simulations.astype('float64')
    val_dataset.params = val_dataset.params.astype('float64')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader
