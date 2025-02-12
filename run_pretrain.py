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
from lightning_module import PreTrainLightning
from dataset import NumpyDataset
import neptune
from lightning.pytorch.loggers import NeptuneLogger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train MZB Cell Analysis model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and validation.')
    parser.add_argument('--max_epochs', type=int, default=400, help='Maximum number of epochs to train.')
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
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--beta', type=float, default=4, help='Beta parameter for the VAE loss.')
    parser.add_argument('--type', type=str, default='vanilla', help='Type of model to use.')
    parser.add_argument('--dirpath', type=str, default='checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--debug', action='store_true', help='Debug mode in Trainer.')    
    return parser.parse_args()

def save_model_config(args, seq_len: int, in_chans: int):
    config = {
        'model': {
            'name': 'TiMAE',
            'params': {
                'seq_len': seq_len,
                'in_chans': in_chans,
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'depth': args.depth,
                'decoder_embed_dim': args.decoder_embed_dim,
                'decoder_num_heads': args.decoder_num_heads,
                'decoder_depth': args.decoder_depth,
                'z_type': args.type,
                'lambda_': args.beta,
                'mask_ratio': args.mask_ratio,
                'bag_size': 1024,
                'dropout': args.dropout
            }
        }
    }
    
    if not os.path.exists(args.dirpath):
        os.makedirs(args.dirpath)

    config_path = os.path.join(args.dirpath, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

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

def main():
    args = parse_args()

    load_dotenv()
    seed_everything(args.seed)

    try:
        train_dataloader, val_dataloader = create_dataloaders(args.data_dir, args.batch_size)

        # Get data shape from the dataset
        sample_data = train_dataloader.dataset[0][1]  # Assumes dataset returns (seq_len, in_chans)
        seq_len, in_chans = sample_data.shape

        model = TiMAE(
            seq_len=seq_len, 
            in_chans=in_chans, 
            embed_dim=args.embed_dim, 
            num_heads=args.num_heads, 
            depth=args.depth,
            decoder_embed_dim=args.decoder_embed_dim, 
            decoder_num_heads=args.decoder_num_heads,
            decoder_depth=args.decoder_depth, 
            z_type=args.type, 
            lambda_=args.beta, 
            mask_ratio=args.mask_ratio,
            bag_size=1024, 
            dropout=args.dropout
        )

        save_model_config(args, seq_len, in_chans)

        pl_model = PreTrainLightning(model=model, lr=args.learning_rate)

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.dirpath,
            filename='TiMAE-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=30, mode="min")

        api_token = os.getenv("NEPTUNE_API_TOKEN")
        if not api_token:
            raise ValueError("NEPTUNE_API_TOKEN environment variable is not set.")

        logger = NeptuneLogger(
            project="RaneLab/LatentABCSMC",
            api_token=api_token,
            tags=["training", "MZB"],
        )

        logger.log_hyperparams({
            **vars(args),
            "seq_len": seq_len,
            "in_chans": in_chans
        })

        # torch.set_float32_matmul_precision('highest')
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator='auto',
            devices=1,
            callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
            logger=logger,
            log_every_n_steps=10,
            # accumulate_grad_batches=1,
            enable_progress_bar=False,
            precision="64-true",
            fast_dev_run=args.debug
        )

        trainer.fit(pl_model, train_dataloader, val_dataloader)

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()