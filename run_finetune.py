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
from lightning_module import FineTuneLightning
from dataset import NumpyDataset
import neptune
from lightning.pytorch.loggers import NeptuneLogger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Lotka-Volterra model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--max_epochs', type=int, default=5, help='Maximum number of epochs to train.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset.')
    parser.add_argument('--dirpath', type=str, default='checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--debug', action='store_true', help='Debug mode in Trainer.')
    return parser.parse_args()


def create_dataloaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Check data directory existence
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    train_dataset = NumpyDataset(data_dir, prefix='train')
    val_dataset = NumpyDataset(data_dir, prefix='val')
    test_dataset = NumpyDataset(data_dir, prefix='test')

    # Ensure data type matches precision setting
    train_dataset.simulations = train_dataset.simulations.astype('float64')
    train_dataset.params = train_dataset.params.astype('float64')
    val_dataset.simulations = val_dataset.simulations.astype('float64')
    val_dataset.params = val_dataset.params.astype('float64')
    test_dataset.simulations = test_dataset.simulations.astype('float64')
    test_dataset.params = test_dataset.params.astype('float64')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader

def main():
    args = parse_args()

    load_dotenv()
    seed_everything(args.seed)

    try:
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args.data_dir, args.batch_size)

        # Get data shape from the dataset
        sample_data = train_dataloader.dataset[0][1]
        seq_len, in_chans = sample_data.shape

        # get model config
        config = yaml.safe_load(open(f"{args.dirpath}/config.yaml"))
        checkpoint_files = [f for f in os.listdir(args.dirpath) if f.endswith("ckpt")]
        checkpoint_file = sorted(checkpoint_files)[-1]
        print(f"Loading model from {checkpoint_file}")

        checkpoint = torch.load(os.path.join(args.dirpath, checkpoint_file))
        model = TiMAE(**config["model"]["params"])
        model_weights = {k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(model_weights)

        pl_model = FineTuneLightning(model=model, lr=args.learning_rate)

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.dirpath,
            filename='kan_fine_tune-{epoch:02d}-{val_loss:.2f}',
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
            tags=["training", "lotka", "finetuning", f"{args.dirpath}"],
        )

        logger.log_hyperparams({
            **vars(args),
            "seq_len": seq_len,
            "in_chans": in_chans
        })

        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator='auto',
            devices=1,
            callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
            logger=logger,
            log_every_n_steps=10,
            accumulate_grad_batches=1,
            enable_progress_bar=False,
            precision="64-true",
            fast_dev_run=args.debug
        )

        trainer.fit(pl_model, train_dataloader, val_dataloader)
        # trainer.test(ckpt_path="best", test_dataloaders=test_dataloader)

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()