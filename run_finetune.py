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
from lightning_module import FineTuneLightning, PreTrainLightning
from dataset import create_dataloaders
import neptune
from lightning.pytorch.loggers import NeptuneLogger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Finetune model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs to train.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset.')
    parser.add_argument('--dirpath', type=str, default='checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--debug', action='store_true', help='Debug mode in Trainer.')
    parser.add_argument('--num_parameters', type=int, default=2, help='Number of parameters to estimate.')
    return parser.parse_args()

def main():
    args = parse_args()

    load_dotenv()
    seed_everything(args.seed)

    try:
        train_dataloader, val_dataloader = create_dataloaders(args.data_dir, args.batch_size)

        # Get data shape from the dataset
        sample_data = train_dataloader.dataset[0][1]
        seq_len, in_chans = sample_data.shape

        # get model config
        config = yaml.safe_load(open(f"{args.dirpath}/config.yaml"))
        checkpoint_files = [f for f in os.listdir(args.dirpath) if f.endswith("ckpt")]
        checkpoint_file = os.path.join(args.dirpath, sorted(checkpoint_files)[-1])
        print(f"Loading model from {checkpoint_file}")

        model = TiMAE(**config["model"]["params"])
        pretrain_module = PreTrainLightning.load_from_checkpoint(checkpoint_file, model = model)
        finetune_module = FineTuneLightning(pl_module=pretrain_module, lr=args.learning_rate, num_parameters=args.num_parameters, linear_probe=False)

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.dirpath,
            filename='fine_tune-{epoch:02d}-{val_loss:.2f}',
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

        trainer.fit(finetune_module, train_dataloader, val_dataloader)
        # trainer.test(ckpt_path="best", test_dataloaders=test_dataloader)

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()