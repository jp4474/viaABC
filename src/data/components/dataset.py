# from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Literal
import os

class BaseNumpyDataset(Dataset):
    def __init__(self, data_dir, prefix="train", transform=None):
        self.data_dir = data_dir
        self.prefix = prefix
        self.transform = transform

        npz_file = os.path.join(data_dir, f"{prefix}_data.npz")
        self.data = np.load(npz_file, allow_pickle=True)
        self.simulations = self.data["simulations"]
        self.params = self.data["params"]

    def __len__(self):
        return len(self.simulations)

    def _apply_transform(self, x):
        if self.transform is not None:
            return self.transform(x)
        return x
    
class LotkaVolterraDataset(BaseNumpyDataset):
    def __init__(self, data_dir, prefix="train", transform=None):
        super().__init__(data_dir, prefix, transform)

    def __getitem__(self, idx):
        x = self.simulations[idx]
        x = self._apply_transform(x)
        x = torch.as_tensor(x, dtype=torch.float32)
        return x
    
class Spatial2DDataset(BaseNumpyDataset):
    def __init__(self, data_dir, prefix="train"):
        super().__init__(data_dir, prefix,)

    def __getitem__(self, idx):
        x = self.simulations[idx]
        x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        return x
    
class SpatialSIRDataset(BaseNumpyDataset):
    def __init__(self, data_dir, prefix="train", transform=None):
        super().__init__(data_dir, prefix, transform)

    def __getitem__(self, idx):
        x = self.simulations[idx]
        x = self._apply_transform(x)
        x = torch.as_tensor(x, dtype=torch.float32)
        return x