import os
from dotenv import load_dotenv
import argparse
from typing import Tuple, Dict, Any
import yaml
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from models import TiMAE
from lightning_module import CustomLightning
from dataset import NumpyDataset
import neptune
from lightning.pytorch.loggers import NeptuneLogger

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training the model.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train Lotka-Volterra model')
    parser.add_argument('--batch_size', type=int, default=768, help='Batch size for training and validation.')
    parser.add_argument('--max_epochs', type=int, default=300, help='Maximum number of epochs to train.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset.')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension for the model.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--depth', type=int, default=2, help='Depth of the encoder.')
    parser.add_argument('--decoder_embed_dim', type=int, default=64, help='Embedding dimension for the decoder.')
    parser.add_argument('--decoder_num_heads', type=int, default=8, help='Number of attention heads in the decoder.')
    parser.add_argument('--decoder_depth', type=int, default=1, help='Depth of the decoder.')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Masking ratio for the input data.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--beta', type=float, default=0.00025, help='Beta parameter for the VAE loss.')
    parser.add_argument('--type', type=str, default='vanilla', help='Type of model to use.')

    return parser.parse_args()

def save_model_config(args):
    config = {
        'model': {
            'name': 'TiMAE',
            'params': {
                'seq_len': 8, # change this
                'in_chans': 2, # change this
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'depth': args.depth,
                'decoder_embed_dim': args.decoder_embed_dim,
                'decoder_num_heads': args.decoder_num_heads,
                'decoder_depth': args.decoder_depth,
                'z_type': args.type,
                'lambda_': args.beta,
                'mask_ratio': args.mask_ratio,
                'bag_size': 1024, # change this if running vq-vae
                'dropout': args.dropout
            }
        }
    }
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_dataloaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation dataloaders.
    """
    train_dataset = NumpyDataset(data_dir, prefix='train')
    val_dataset = NumpyDataset(data_dir, prefix='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader


def main():
    args = parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Set seed for reproducibility
    seed_everything(args.seed)

    try:
        # Create dataloaders
        train_dataloader, val_dataloader = create_dataloaders(args.data_dir, args.batch_size)

        # TODO: get shape of the data
        # data_shape = train_dataloader.dataset[0].shape
        # print(f"Data shape: {data_shape}")

        save_model_config(args)

        # Initialize model
        model = TiMAE(
            seq_len=8, in_chans=2, embed_dim=args.embed_dim, num_heads=args.num_heads, depth=args.depth,
            decoder_embed_dim=args.decoder_embed_dim, decoder_num_heads=args.decoder_num_heads,
            decoder_depth=args.decoder_depth, z_type=args.type, lambda_=args.beta, mask_ratio=args.mask_ratio,
            bag_size=1024, dropout=args.dropout
        )

        # Initialize Lightning module
        pl_model = CustomLightning(model=model, lr=args.learning_rate)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename='lotka-volterra-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=False, mode="min")

        # Retrieve API token from environment
        api_token = os.getenv("NEPTUNE_API_TOKEN")
        if not api_token:
            raise ValueError("NEPTUNE_API_TOKEN environment variable is not set.")

        # Initialize Neptune logger
        logger = NeptuneLogger(
            project="RaneLab/LatentABCSMC",
            api_token=api_token,
            tags=["training", "lotka"],
        )

        # Log hyperparameters
        logger.log_hyperparams(vars(args))

        # Initialize trainer
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator='auto',
            devices=1,
            callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
            logger=logger,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            accumulate_grad_batches=10,
            enable_progress_bar=False,
            precision="64-true"
        )

        # Train the model
        trainer.fit(
            model=pl_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise


if __name__ == '__main__':
    main()