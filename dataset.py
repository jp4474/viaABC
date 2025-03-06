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

        y = y / self.scales[idx]

        x = torch.from_numpy(x).to(torch.float64)
        y = torch.from_numpy(y).to(torch.float64)

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

class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, data_dir: str, prefix='train'):
        self.data_dir = data_dir
        
        # Load data
        self.data = np.load(os.path.join(data_dir, f'{prefix}_data.npz'), allow_pickle=True)
        self.anchors = self.data['anchor_y']
        self.positives = self.data['pos_y']
        self.negatives = self.data['neg_y']

        self.max = np.array([1.23131727e+05, 5.07489562e+01])
        self.min = np.array([-6.81139813e-07,  5.17101545e-07])

    def __getitem__(self, index):
        anchor = self.anchors[index]
        positive = self.positives[index]
        negative = self.negatives[index]

        # Scale and Convert to torch tensors
        anchor = (anchor - self.min) / (self.max - self.min)
        positive = (positive - self.min) / (self.max - self.min)
        negative = (negative - self.min) / (self.max - self.min)
        
        anchor = torch.from_numpy(anchor).to(torch.float64)
        positive = torch.from_numpy(positive).to(torch.float64)
        negative = torch.from_numpy(negative).to(torch.float64)

        return anchor, positive, negative