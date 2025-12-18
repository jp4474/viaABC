# datamodule.py

from typing import Optional, Callable, Type, Any
from torch.utils.data import DataLoader
import lightning as L
from torch.utils.data import Dataset, random_split
from hydra.utils import instantiate

class SimulationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        transform=None,
        num_workers: int = 4,
        pin_memory: bool = True,
        dataset: Any = None, 
    ):
        super().__init__()

        if dataset is None:
            raise ValueError("A dataset class must be provided to SimulationDataModule.")

        # dataset is a PARTIAL of the dataset class
        self.dataset_cls = dataset

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.dataest = self.dataset_cls
        self.train_dataset, self.val_dataset = random_split(
            self.dataest, [int(0.8 * len(self.dataest)), len(self.dataest) - int(0.8 * len(self.dataest))]
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )