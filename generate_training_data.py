import argparse
from typing import List
from latent_abc_smc import LotkaVolterra
from lightning.pytorch import seed_everything

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Lotka-Volterra ABC-SMC simulation')
    parser.add_argument('--train_sizes', 
                       type=int, 
                       nargs=3,
                       default=[500000, 50000, 50000],
                       help='List of three integers for training data sizes')
    
    parser.add_argument('--seed', 
                       type=int,
                       default=0,
                       help='Random seed for reproducibility')
    return parser.parse_args()

def main(train_sizes: List[int], seed: int) -> None:
    """
    Run the Lotka-Volterra simulation with specified parameters.
    
    Args:
        train_sizes: List of three integers specifying training data sizes
        seed: Random seed for reproducibility
    """
    # Set seed for reproducibility
    seed_everything(seed)
    
    # Initialize and run simulation
    lotka_abc = LotkaVolterra()
    lotka_abc.generate_training_data(train_sizes, seed=seed)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.train_sizes, args.seed)