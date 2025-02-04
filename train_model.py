import os
import argparse
from typing import Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger

from lightning_module import LotkaVolterraLightning

@dataclass
class Config:
    batch_size: int = 768
    num_workers: int = 8
    max_epochs: int = 300
    learning_rate: float = 1e-3
    accumulate_grad_batches: int = 10
    gradient_clip_val: float = 1.0
    seed: int = 42
    data_dir: str = "data"

class LotkaVolterraDataset(Dataset):
    def __init__(self, params: np.ndarray, simulations: np.ndarray):
        self.params = params
        self.simulations = simulations
        
    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.params[idx], dtype=torch.float32),
                torch.tensor(self.simulations[idx], dtype=torch.float32))

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load datasets and create dataloaders."""
    try:
        train_params = np.load(f'{config.data_dir}/train_params.npy')
        train_simulations = np.load(f'{config.data_dir}/train_simulations.npy')
        val_params = np.load(f'{config.data_dir}/val_params.npy')
        val_simulations = np.load(f'{config.data_dir}/val_simulations.npy')
        test_params = np.load(f'{config.data_dir}/test_params.npy')
        test_simulations = np.load(f'{config.data_dir}/test_simulations.npy')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data files not found in {config.data_dir}") from e

    datasets = {
        'train': LotkaVolterraDataset(train_params, train_simulations),
        'val': LotkaVolterraDataset(val_params, val_simulations),
        'test': LotkaVolterraDataset(test_params, test_simulations)
    }

    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders['train'], dataloaders['val'], dataloaders['test']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Lotka-Volterra model')
    parser.add_argument('--batch_size', type=int, default=768)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='data')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        data_dir=args.data_dir,
        seed=args.seed
    )
    
    # Set reproducibility
    set_seed(config.seed)
    
    # Setup hardware
    torch.set_float32_matmul_precision('high')
    
    # Verify Neptune API token
    if not os.getenv('NEPTUNE_API_TOKEN'):
        raise ValueError("NEPTUNE_API_TOKEN environment variable not set")

    # Load data
    train_dataloader, val_dataloader, test_dataloader = load_data(config)

    # Setup logging
    neptune_logger = NeptuneLogger(
        api_key=os.getenv('NEPTUNE_API_TOKEN'),
        project="RaneLab/LatentABCSMC",
        name="Lotka-Volterra",
        tags=["Lotka-Volterra"],
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            filename="best-checkpoint",
            save_top_k=1,
            mode="min",
        )
    ]

    # Initialize model and trainer
    model = LotkaVolterraLightning(learning_rate=config.learning_rate)
    
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        accumulate_grad_batches=config.accumulate_grad_batches,
        logger=neptune_logger,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks,
        deterministic=True
    )
    
    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

if __name__ == "__main__":
    main()