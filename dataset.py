# from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any
import os
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision
import torchvision.transforms as transforms

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
        x = self.params[idx] / 10.0
        y = self.y_normalized[idx]

        x = torch.from_numpy(x).to(torch.float64)
        y = torch.from_numpy(y).to(torch.float64)

        return x, y

class SpatialSIRDataset(Dataset):
    def __init__(self, data_dir, prefix='train', transform=None):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        # Load data
        self.data = np.load(os.path.join(data_dir, f'{prefix}_data_reshaped.npz'), allow_pickle=True)
        #self.params = self.data['params']
        self.simulations = self.data['simulations']

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, idx):
        #x = self.params[idx]
        y = self.simulations[idx]
        y = torch.from_numpy(y).to(torch.float64).permute(2, 0, 1) #.permute(0, 3, 1, 2)

        #return x, y
        return y
    
def create_dataloaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Check data directory existence
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    train_dataset = SpatialSIRDataset(data_dir, prefix='train')
    val_dataset = SpatialSIRDataset(data_dir, prefix='val')

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # trainset = torchvision.datasets.MNIST(root='./data_mnist/', train=True, download=True, transform=transform)
    # train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    # testset = torchvision.datasets.MNIST(root='./data_mnist/', train=False, download=True, transform=transform)
    # val_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # # Ensure data type matches precision setting
    train_dataset.simulations = train_dataset.simulations.astype('float64')
    #train_dataset.params = train_dataset.params.astype('float64')
    val_dataset.simulations = val_dataset.simulations.astype('float64')
    #val_dataset.params = val_dataset.params.astype('float64')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader
