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
        self.scales = np.mean(np.abs(self.simulations), axis = 1, keepdims=True)

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, idx):
        x = self.params[idx]
        y = self.simulations[idx]

        x = x / 10
        y = y / self.scales[idx]

        x = torch.from_numpy(x).to(torch.float64)
        y = torch.from_numpy(y).to(torch.float64)

        return x, y


class MZBDataset(Dataset):
    def __init__(self, data_dir, prefix='train'):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        # Load data
        self.data = np.load(os.path.join(data_dir, f'{prefix}_data.npz'), allow_pickle=True)
        self.simulations = self.data['simulations']
        self.params = self.data['params']

        mzb_means = np.mean(self.simulations, axis=1)[:, 0]
        donor_ki67_means = np.ones_like(mzb_means)
        host_ki67_means = np.ones_like(mzb_means)
        nfd_means = np.ones_like(mzb_means)
        means = np.array([mzb_means, donor_ki67_means, host_ki67_means, nfd_means]).T
        self.scales = np.expand_dims(means, axis=1)  # Use np.expand_dims to add a new dimension

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, idx):
        x = self.params[idx]
        y = self.simulations[idx]

        x = x / 10 # TODO: fix this accordingly, probaly use minmax scaler from sklearn?
        y = y / self.scales[idx]

        x = torch.from_numpy(x).to(torch.float64)
        y = torch.from_numpy(y).to(torch.float64)

        return x, y

def create_dataloaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Check data directory existence
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    train_dataset = MZBDataset(data_dir, prefix='train')
    val_dataset = MZBDataset(data_dir, prefix='val')

    # Ensure data type matches precision setting
    train_dataset.simulations = train_dataset.simulations.astype('float64')
    train_dataset.params = train_dataset.params.astype('float64')
    val_dataset.simulations = val_dataset.simulations.astype('float64')
    val_dataset.params = val_dataset.params.astype('float64')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader
