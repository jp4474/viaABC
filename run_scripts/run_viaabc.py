import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
from systems import LotkaVolterra, CARModel, SpatialSIR3D
from lightning_module import PreTrainLightning, PreTrainLightningSpatial
from models import TSMVAE, MaskedAutoencoderViT3D

from inference_utils import load_pretrained_model

def main():
    parser = argparse.ArgumentParser(description='Run viaABC inference for Lotka-Volterra system')
    parser.add_argument('--num_particles', type=int, default=1000, 
                       help='Number of particles for viaABC (default: 1000)')
    parser.add_argument('--k', type=int, default=10, 
                       help='Number of generations for viaABC (default: 10)')
    parser.add_argument('--folder_name', type=str, default='pretrained_models/lotka_volterra', help='Folder containing the pretrained model')
    parser.add_argument('--pooling_method', type=str, default='no_cls', choices=['cls', 'no_cls', 'all', 'mean'], 
                       help='Pooling method for feature aggregation')
    parser.add_argument('--metric', type=str, default='pairwise_cosine', choices=['cosine', 'l1', 'l2', 'bertscore', 'pairwise_cosine', 'bertscore_batch', 'maxSim'], 
                       help='Distance metric for viaABC')     
    parser.add_argument('--idx', type=int, default=1)

    args = parser.parse_args()
    
    model = load_pretrained_model(
        model_class=TSMVAE,
        lightning_class=PreTrainLightning,
        checkpoint_substr="TSMVAE",
        folder_name=args.folder_name,
    )

    # model = load_pretrained_model(
    #     model_class=MaskedAutoencoderViT3D,
    #     lightning_class=PreTrainLightningSpatial,
    #     checkpoint_substr="SpatialSIR",
    #     folder_name=args.folder_name,
    # )

    ####################################################################
    # lotka_abc = LotkaVolterra(
    #     model=model,
    #     pooling_method=args.pooling_method,
    #     metric=args.metric,
    # )
    
    # lotka_abc.run(num_particles=args.num_particles, k=args.k, q_threshold = 0.999)
    # generations = lotka_abc.generations
    ####################################################################

    ####################################################################
    car_abc = CARModel(model=model)
    car_abc.run(num_particles=args.num_particles, k=args.k, q_threshold = 0.999)
    generations = car_abc.generations
    ####################################################################

    ####################################################################
    # sirs_abc = SpatialSIR3D(
    #     model=model,
    #     pooling_method=args.pooling_method,
    #     metric=args.metric,
    # )
    # sirs_abc.run(num_particles=args.num_particles, k=args.k, q_threshold = 0.99)
    # generations = sirs_abc.generations
    ####################################################################

    ############################### Save Results ###############################
    # Create output filename based on parameters
    output_filename = f'car_{args.k}_{args.pooling_method}_{args.metric}_{args.num_particles}_{args.idx}.npz'
    output_path = os.path.join(args.folder_name, output_filename)
    
    np.savez(output_path, generations=generations)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()