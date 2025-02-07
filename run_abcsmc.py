#!/usr/bin/env python3

import logging
from pathlib import Path
import numpy as np
import torch

from models import TiMAE
from lightning_module import CustomLightning
from systems import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'model': {
        'seq_len': 8,
        'in_chans': 2,
        'embed_dim': 64,
        'num_heads': 8,
        'depth': 2,
        'decoder_embed_dim': 64,
        'decoder_num_heads': 8,
        'decoder_depth': 1,
        'z_type': "vae",
        'lambda_': 0.00025,
        'mask_ratio': 0.15,
        'bag_size': 1024,
        'dropout': 0.0
    },
    'abc': {
        'num_particles': 20000
    },
    'paths': {
        'checkpoint': Path('/home/jp4474/latent-abc-smc/checkpoints/lotka-volterra-epoch=295-val_loss=0.41.ckpt'),
        'output_dir': Path('./output')
    }
}

def setup_model():
    """Initialize and load the pre-trained model."""
    try:
        model = TiMAE(**CONFIG['model'])
        pl_model = CustomLightning.load_from_checkpoint(
            CONFIG['paths']['checkpoint'],
            model=model
        )
        pl_model.eval()
        return pl_model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def run_abc_simulation(pl_model):
    """Run ABC-SMC simulation with the loaded model."""
    try:
        lotka_abc = LotkaVolterra(num_particles=CONFIG['abc']['num_particles'])
        lotka_abc.update_model(pl_model)
        
        particles, weights = lotka_abc.run()
        return particles, weights, lotka_abc
    except Exception as e:
        logger.error(f"Failed to run ABC simulation: {str(e)}")
        raise

def save_results(particles, weights):
    """Save simulation results to files."""
    output_dir = CONFIG['paths']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        np.save(output_dir / "particles.npy", particles)
        np.save(output_dir / "weights.npy", weights)
        logger.info(f"Results saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise

def main():
    """Main execution function."""
    logger.info("Starting ABC-SMC simulation")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    pl_model = setup_model()
    logger.info("Model loaded successfully")
    
    # Run simulation
    particles, weights, lotka_abc = run_abc_simulation(pl_model)
    logger.info("ABC-SMC simulation completed")
    
    # Save results
    save_results(particles, weights)
    
    # Compute statistics
    lotka_abc.compute_statistics()
    logger.info("Statistics computation completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise