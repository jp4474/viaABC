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

        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)

        return x, y
    
class LotkaVolterraDataset(NumpyDataset):
    def __init__(self, data_dir, prefix='train'):
        super().__init__(data_dir, prefix)

        # Calculate mean and std across all simulations and time steps
        self.s = np.mean(np.abs(self.simulations), axis=1, keepdims=True)
        # Normalize the data
        self.y_normalized = (self.simulations) / self.s

    def __getitem__(self, idx):
        y = self.y_normalized[idx]
        y = torch.from_numpy(y).to(torch.float32)

        return y

class SpatialSIRDataset(Dataset):
    def __init__(self, data_dir, prefix='train', transform=None):
        self.data_dir = data_dir
        # self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        # Load data
        self.data = np.load(os.path.join(data_dir, f'{prefix}_data.npz'), allow_pickle=True)
        self.simulations = self.data['simulations']
        self.params = self.data['params']

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, idx):
        x = self.params[idx]
        y = self.simulations[idx]
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32) #.permute(3, 0, 1, 2)  # Change to (C, T, H, W) format

        return y

class CARDataset(NumpyDataset):
    def __init__(self, data_dir, prefix='train'):
        super().__init__(data_dir, prefix)

        # Calculate mean and std across all simulations and time steps
        self.s = np.mean(np.abs(self.simulations), axis=1, keepdims=True)
        self.y_normalized = (self.simulations) / self.s

    def __getitem__(self, idx):
        y = self.y_normalized[idx]
        y = torch.from_numpy(y).to(torch.float32)
        return y
    
def create_dataloaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Check data directory existence
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    # train_dataset = LotkaVolterraDataset(data_dir, prefix='train')
    # val_dataset = LotkaVolterraDataset(data_dir, prefix='val')
    # train_dataset = CARDataset(data_dir, prefix='train')
    # val_dataset = CARDataset(data_dir, prefix='val')

    train_dataset = SpatialSIRDataset(data_dir, prefix='train')
    val_dataset = SpatialSIRDataset(data_dir, prefix='val')

    # # Ensure data type matches precision setting
    train_dataset.simulations = train_dataset.simulations.astype('float32')
    val_dataset.simulations = val_dataset.simulations.astype('float32')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader
