# from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any
from utils import *
import os

class TripletLotkaVolterra(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, data_dir: str, prefix='train', with_params=False):
        self.data_dir = data_dir
        
        # Load data
        self.data = np.load(os.path.join(data_dir, f'{prefix}_data.npz'), allow_pickle=True)
        self.anchors = self.data['anchor_y']
        self.positives = self.data['pos_y']
        self.negatives = self.data['neg_y']

        self.with_params = with_params

        if with_params:
            self.anchors_params = self.data['anchor_x']
            self.positives_params = self.data['pos_x']
            self.negatives_params = self.data['neg_x']

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor = self.anchors[index]
        positive = self.positives[index]
        negative = self.negatives[index]

        # Scale and Convert to torch tensors
        anchor = (anchor) / np.abs(anchor).mean(0)
        positive = (positive) / np.abs(positive).mean(0)
        negative = (negative) / np.abs(negative).mean(0)

        anchor = torch.from_numpy(anchor).to(torch.float64)
        positive = torch.from_numpy(positive).to(torch.float64)
        negative = torch.from_numpy(negative).to(torch.float64)

        if self.with_params:
            anchor_params = self.anchors_params[index]
            positive_params = self.positives_params[index]
            negative_params = self.negatives_params[index]

            anchor_params = torch.from_numpy(anchor_params).to(torch.float64)
            positive_params = torch.from_numpy(positive_params).to(torch.float64)
            negative_params = torch.from_numpy(negative_params).to(torch.float64)

            return anchor_params, positive_params, negative_params, anchor, positive, negative

        return anchor, positive, negative

class TripletLotkaVolterraDynamic(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, data_dir: str, prefix='train'):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
        self.data = np.load(os.path.join(data_dir, f'{prefix}_data.npz'), allow_pickle=True)
        self.simulations = self.data['simulations']
        self.params = self.data['params']
        self.scales = np.mean(np.abs(self.simulations), axis = 1, keepdims=True)

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, index):
        anchor_x = self.params[index]
        anchor_y = self.simulations[index]/self.scales[index]

        (x_pos, y_pos), (x_neg, y_neg) = generate_triplet(anchor_x, 0, 10)

        # Scale and Convert to torch tensors
        anchor_x = torch.from_numpy(anchor_x).to(torch.float64)
        anchor_y = torch.from_numpy(anchor_y).to(torch.float64)

        pos_x = torch.from_numpy(x_pos).to(torch.float64)
        pos_y = torch.from_numpy(y_pos/y_pos.mean(0)).to(torch.float64)

        neg_x = torch.from_numpy(x_neg).to(torch.float64)
        neg_y = torch.from_numpy(y_neg/y_neg.mean(0)).to(torch.float64)

        # return anchor_x, pos_x, neg_x, anchor_y, pos_y, neg_y
        return anchor_y, pos_y, neg_y
    

def create_dataloaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Check data directory existence
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    train_dataset = TripletLotkaVolterraDynamic(data_dir, prefix='train')
    val_dataset = TripletLotkaVolterraDynamic(data_dir, prefix='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader