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
        self.scales = np.mean(self.simulations, axis=1, keepdims=True)

        # MZB
        # self.max_ = np.array([ 0.99998673, 19.99958512,  0.5999854 , -1.00000201,  6.49953051,
        #  0.49987116])
        # self.min_ = np.array([ 6.16673630e-06,  8.00015844e+00,  1.35937240e-05, -8.99987713e+00,
        #  1.00007737e+00, -5.99925511e+00])

        self.m = np.mean(self.simulations, axis=1)
        self.sd = np.std(self.simulations, axis=1)

        self.max_ = np.array([10, 10])
        self.min_ = np.array([0, 0])
        
        # self.scales = np.abs(self.simulations).mean(axis=1)

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, idx):
        x = self.params[idx]
        y = self.simulations[idx]

        x_scaled = (x - self.min_) / (self.max_ - self.min_)
        y_scaled = y / self.scales[idx]
        # y_scaled = (y - self.m[idx]) / (self.sd[idx])
        # Convert to torch tensors
        x = torch.from_numpy(x_scaled).to(torch.float64)
        y = torch.from_numpy(y_scaled).to(torch.float64)

        return x, y 
    

def create_dataloaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Check data directory existence
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    train_dataset = NumpyDataset(data_dir, prefix='train')
    val_dataset = NumpyDataset(data_dir, prefix='val')

    # Ensure data type matches precision setting
    train_dataset.simulations = train_dataset.simulations.astype('float64')
    train_dataset.params = train_dataset.params.astype('float64')
    val_dataset.simulations = val_dataset.simulations.astype('float64')
    val_dataset.params = val_dataset.params.astype('float64')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader