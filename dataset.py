# construct a dataset from npy files
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    def __init__(self, data_dir, transform=None, prefix = 'train'):
        self.data_dir = data_dir
        self.transform = transform
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.simulations = [np.load(os.path.join(data_dir, f)) for f in self.files if f.startswith(prefix + '_' + 'simulations')][0]
        self.params = [np.load(os.path.join(data_dir, f)) for f in self.files if f.startswith(prefix + '_' + 'params')][0]

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, idx):
        x = self.params[idx]
        y = self.simulations[idx]

        x = torch.from_numpy(x).to(torch.float64)  # or torch.from_numpy(x).to(torch.float64)
        y = torch.from_numpy(y).to(torch.float64)  # or torch.from_numpy(y).to(torch.float64)

        return x, y
        