import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
from typing import List
from src.viaABC.systems import *

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
    
    # Number of trajectories for each parameter sampling
    parser.add_argument('--num_repeats',
                        type=int,
                        default=3,
                        help='Number of trajectories for each parameter sampling')
    
    parser.add_argument('--save_dir',
                        type=str,
                        default="data/spatial2D",
                        help='Directory to save the generated training data')
    
    return parser.parse_args()

def main(train_sizes: List[int], seed: int, num_workers: int, num_repeats: int, save_dir: str) -> None:
    """
    Run the simulation with specified parameters.
    
    Args:
        train_sizes: List of three integers specifying training data sizes
        seed: Random seed for reproducibility
        num_workers: Number of workers for parallel processing
        num_repeats: Number of trajectories for each parameter sampling
        save_dir: Directory to save the generated training data
    """
    
    # This script intentionally stays thin: dataset generation policy lives in
    # the `Spatial2D` system class so training-data scripts and ABC inference
    # reuse the same simulator/prior definitions.
    model = Spatial2D()
    model.generate_training_data(train_sizes, seed=seed, num_workers=num_workers, save_dir=save_dir)
if __name__ == "__main__":
    args = parse_arguments()
    main(args.train_sizes, args.seed, args.num_workers, args.num_repeats, args.save_dir)  
