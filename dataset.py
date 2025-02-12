# from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class NumpyDataset(Dataset):
    def __init__(self, data_dir, prefix='train'):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        # Load data
        self.simulations = [np.load(os.path.join(data_dir, f)) 
                           for f in self.files if f.startswith(prefix + '_' + 'simulations')][0]
        
        self.params = [np.load(os.path.join(data_dir, f)) 
                      for f in self.files if f.startswith(prefix + '_' + 'params')][0]
        
        self.scales = np.abs(self.simulations).mean(axis=1)

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, idx):
        x = self.params[idx]
        y = self.simulations[idx]
        scale = self.scales[idx]
        
        # Convert to torch tensors
        x = torch.from_numpy(x).to(torch.float64)
        y = torch.from_numpy(y).to(torch.float64)

        y_scaled = y / scale

        return x, y_scaled