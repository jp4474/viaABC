import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from models import MaskedAutoencoderViT3D
from lightning_module import PreTrainLightningSpatial
from dataset import create_dataloaders

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train SpatialSIR3D model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation.')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs to train.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--data_dir', type=str, default='data/SPATIAL', help='Directory containing the dataset.')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension for the model.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--depth', type=int, default=2, help='Depth of the encoder.')
    parser.add_argument('--decoder_embed_dim', type=int, default=64, help='Embedding dimension for the decoder.')
    parser.add_argument('--decoder_num_heads', type=int, default=8, help='Number of attention heads in the decoder.')
    parser.add_argument('--decoder_depth', type=int, default=1, help='Depth of the decoder.')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Masking ratio for the input data.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--beta', type=float, default=1, help='Beta parameter for the VAE loss.')
    parser.add_argument('--type', type=str, default='vanilla', help='Type of model to use.')
    parser.add_argument('--dirpath', type=str, default='checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--debug', action='store_true', help='Debug mode in Trainer.')    
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size for the input.')
    parser.add_argument('--t_patch_size', type=int, default=5, help='Temporal patch size.')
    return parser.parse_args()

def save_model_config(args, img_size: int, num_frames: int):
    config = {
        'model': {
            'name': 'SpatialSIR3D',
            'params': {
                'img_size': img_size,
                'patch_size': args.patch_size,
                'num_frames': num_frames,
                'pred_t_dim': num_frames,
                't_patch_size': args.t_patch_size,
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'depth': args.depth,
                'decoder_embed_dim': args.decoder_embed_dim,
                'decoder_num_heads': args.decoder_num_heads,
                'decoder_depth': args.decoder_depth,
                'z_type': args.type,
                'lambda_': args.beta,
            }
        }
    }

    if not os.path.exists(args.dirpath):
        os.makedirs(args.dirpath)

    config_path = os.path.join(args.dirpath, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def main():
    args = parse_args()

    seed_everything(args.seed)

    try:
        train_dataloader, val_dataloader = create_dataloaders(args.data_dir, args.batch_size)

        # Get data shape from the dataset
        sample_data = train_dataloader.dataset[0] #[1]  # Assumes dataset returns (seq_len, in_chans)
        print(sample_data.shape)
        C, T, H, W = sample_data.shape

        save_model_config(args, H, T)

        model = MaskedAutoencoderViT3D(
            img_size=H,
            patch_size=args.patch_size,
            num_frames=T,
            pred_t_dim=T,
            t_patch_size=args.t_patch_size,
            in_chans=C,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            depth=args.depth,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_num_heads=args.decoder_num_heads,
            decoder_depth=args.decoder_depth,
            z_type=args.type,
            lambda_=args.beta,
            dropout=args.dropout,
        )

        pl_model = PreTrainLightningSpatial(model=model, lr=args.learning_rate, mask_ratio=args.mask_ratio)

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.dirpath,
            filename='SpatialSIR3D-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")

        torch.set_float32_matmul_precision('medium')
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator='auto',
            devices=1,
            callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
            log_every_n_steps=10,
            enable_progress_bar=False,
            precision="32-true",
            fast_dev_run=args.debug
        )
        
        trainer.fit(pl_model, train_dataloader, val_dataloader)

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()