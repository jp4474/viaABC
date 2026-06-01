import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate simulations')
    parser.add_argument('--train_sizes', 
                       type=int, 
                       default=50000,
                       help='Training data sizes')
    
    parser.add_argument('--seed', 
                       type=int,
                       default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of workers for parallel processing')
    
    parser.add_argument('--save_dir',
                        type=str,
                        default="data/spatial2D",
                        help='Directory to save the generated training data')
    
    parser.add_argument('--experiment',
                        type=str,
                        default="spatial2D",
                        help='Hydra experiment config to use for system shape settings')

    return parser.parse_args()

def main(train_sizes: int, seed: int, num_workers: int, save_dir: str, experiment: str) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}"])

    system = instantiate(cfg.system)
    system.generate_training_data(train_sizes, seed=seed, num_workers=num_workers, save_dir=save_dir)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.train_sizes, args.seed, args.num_workers, args.save_dir, args.experiment)
