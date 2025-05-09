import argparse
from typing import List
from systems import *

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate simulations')
    parser.add_argument('--train_sizes', 
                       type=int, 
                       nargs=3,
                       default=[50000, 10000, 10000],
                       help='List of three integers for training data sizes')
    
    parser.add_argument('--seed', 
                       type=int,
                       default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='Number of workers for parallel processing')
    
    return parser.parse_args()

def main(train_sizes: List[int], seed: int, num_workers: int) -> None:
    """
    Run the simulation with specified parameters.
    
    Args:
        train_sizes: List of three integers specifying training data sizes
        seed: Random seed for reproducibility
    """
    
    # Initialize and run simulation
    lotka_abc = LotkaVolterra()
    lotka_abc.generate_training_data(train_sizes, seed=seed, num_workers=num_workers)

    # raw_data = np.load("data/mzb_data.npz")['obs_data']
    # mzb_abc = MZB(observational_data=raw_data)
    # mzb_abc.generate_training_data(train_sizes, seed=seed)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.train_sizes, args.seed, args.num_workers)