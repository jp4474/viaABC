#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import yaml
import numpy as np
import torch
from matplotlib import pyplot as plt

from models import TSMVAE, MaskedAutoencoderViT
from lightning_module import PreTrainLightning, FineTuneLightning, PreTrainLightningSpatial2D
from systems import LotkaVolterra, SpatialSIR
from dataset import NumpyDataset, SpatialSIRDataset

class LotkaVolterraAnalyzer:
    def __init__(self, config_path, data_dir):
        self.config_path = config_path
        self.data_dir = data_dir
        self.load_config()
        self.load_model()
        self.load_data()
        self.initialize_lotka_volterra()

    def load_config(self):
        """Load configuration from YAML file"""
        self.config = yaml.safe_load(open(f"{self.config_path}/config.yaml"))
        self.folder_name = self.config_path

    def load_model(self):
        """Load the pretrained model"""
        self.model = TSMVAE(**self.config["model"]["params"], tokenize="linear")
        pretrain_model_path = next(
            f for f in os.listdir(self.folder_name) 
            if f.endswith(".ckpt") and "TSMVAE" in f
        )
        self.pl_model = PreTrainLightning.load_from_checkpoint(
            os.path.join(self.folder_name, pretrain_model_path), 
            model=self.model
        )
        #############################################################################################

        # finetune_model_path = next(
        #     f for f in os.listdir(self.folder_name) 
        #     if f.endswith(".ckpt") and "fine_tune" in f # change this line to load fine-tuned model
        # )

        # self.fine_tune = FineTuneLightning.load_from_checkpoint(
        #     os.path.join(self.folder_name, finetune_model_path), 
        #     pl_module=self.pl_model,
        #     num_parameters=2,
        # )
        
        #############################################################################################

        print("Successfully loaded model")

    def load_data(self):
        """Load and preprocess the data"""
        self.train_ds = NumpyDataset(data_dir=self.data_dir)
        self.obs_data = np.load(os.path.join(self.data_dir, "lotka_data.npz"))

    def initialize_lotka_volterra(self):
        """Initialize the Lotka-Volterra system"""
        self.lotka_abc = LotkaVolterra()
        self.lotka_abc.update_model(self.pl_model)         #############################################################################################
        self.lotka_abc.update_train_dataset(self.train_ds)
        self.raw_np_scaled = self.lotka_abc.preprocess(self.obs_data["obs_data"])

    def generate_reconstruction(self):
        """Generate model reconstruction of the observed data"""
        with torch.no_grad():
            recon_loss, _, _, _, reconstruction = self.pl_model.forward(
                torch.tensor(self.raw_np_scaled).float()
                .to(self.pl_model.device)
                .unsqueeze(0)
            )
            self.reconstruction = reconstruction.squeeze(0).cpu().numpy()

    def run_simulation(self, params):
        """Run simulation with given parameters"""
        self.simulated_np, _ = self.lotka_abc.simulate(params)
        self.simulated_np_scaled = self.lotka_abc.preprocess(self.simulated_np)

    def plot_comparisons(self):
        """Plot comparisons between prediction, ground truth and noisy data"""
        # Scaled comparison
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        for i in range(2):
            ax[i].plot(self.reconstruction[:, i], label='Prediction')
            ax[i].plot(self.simulated_np_scaled[:, i], label='Ground Truth')
            ax[i].plot(self.raw_np_scaled[:, i], label='Noisy')
            ax[i].legend()
            ax[i].set_title(f"Parameter {i+1} (Scaled)")
        #plt.show()

        fig.savefig(os.path.join(
            self.folder_name, 
            "reconstruction_plot_scaled.png"
        ))

        # Unscaled comparison
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        for i in range(2):
            ax[i].plot(
                self.reconstruction[:, i] * self.obs_data['obs_scale'][i], 
                label='Prediction'
            )
            ax[i].plot(self.simulated_np[:, i], label='Ground Truth')
            ax[i].plot(self.obs_data['obs_data'][:, i], label='Noisy')
            ax[i].legend()
            ax[i].set_title(f"Parameter {i+1} (Unscaled)")
        #plt.show()

        # save the figures
        fig.savefig(os.path.join(
            self.folder_name, 
            "reconstruction_plot_original_scale.png"
        ))

    def run_abc_analysis(self, pooling_method, metric, num_particles):
        """Run ABC analysis with specified parameters"""
        self.lotka_abc.pooling_method = pooling_method
        self.lotka_abc.metric = metric
        self.lotka_abc.encode_observational_data()
        print(f"Running ABC with {pooling_method} pooling and {metric} metric")
        self.lotka_abc.run(num_particles=num_particles)
        
        output_file = os.path.join(
            self.folder_name, 
            f"lotka_abc_{pooling_method}_{metric}.npz"
        )
        np.savez(output_file, generations=self.lotka_abc.generations)
        print(f"Results saved to {output_file}")

class SpatialSIRAnalyzer:
    def __init__(self, config_path, data_dir):
        self.config_path = config_path
        self.data_dir = data_dir
        self.load_config()
        self.load_model()
        self.load_data()
        self.initialize_spatial_sir()

    def load_config(self):
        """Load configuration from YAML file"""
        self.config = yaml.safe_load(open(f"{self.config_path}/config.yaml"))
        self.folder_name = self.config_path

    def load_model(self):
        """Load the pretrained model"""
        self.model = MaskedAutoencoderViT(**self.config["model"]["params"], in_channels=3)
        pretrain_model_path = next(
            f for f in os.listdir(self.folder_name)
            if f.endswith(".ckpt") and "SUM" in f
        )
        self.pl_model = PreTrainLightningSpatial2D.load_from_checkpoint(
            os.path.join(self.folder_name, pretrain_model_path),
            model=self.model
        )
        print("Successfully loaded model")

    def load_data(self):
        """Load and preprocess the data"""
        self.train_ds = SpatialSIRDataset(data_dir=self.data_dir)
        self.obs_data = np.load(os.path.join(self.data_dir, "sir_data.npz"))

    def initialize_spatial_sir(self):
        """Initialize the Spatial SIR system"""
        self.spatial_sir_abc = SpatialSIR()
        self.spatial_sir_abc.update_model(self.pl_model)
        self.spatial_sir_abc.update_train_dataset(self.train_ds)
        # If preprocess is defined in spatial_sir_abc, use it to scale the data
        if hasattr(self.spatial_sir_abc, 'preprocess'):
            self.raw_np_scaled = self.spatial_sir_abc.preprocess(self.obs_data["obs_data"])
        else:
            self.raw_np_scaled = self.obs_data["obs_data"]

    def run_simulation(self, params):
        """Run simulation with given parameters"""
        self.simulated_np, _ = self.spatial_sir_abc.simulate(params)
        if hasattr(self.spatial_sir_abc, 'preprocess'):
            self.simulated_np_scaled = self.spatial_sir_abc.preprocess(self.simulated_np)
        else:
            self.simulated_np_scaled = self.simulated_np

    def run_abc_analysis(self, pooling_method, metric, num_particles):
        """Run ABC analysis with specified parameters"""
        self.spatial_sir_abc.pooling_method = pooling_method
        self.spatial_sir_abc.metric = metric
        self.spatial_sir_abc.encode_observational_data()
        print(f"Running ABC with {pooling_method} pooling and {metric} metric")
        self.spatial_sir_abc.run(num_particles=num_particles)

        output_file = os.path.join(
            self.folder_name,
            f"spatial_sir_abc_{pooling_method}_{metric}.npz"
        )
        np.savez(output_file, generations=self.spatial_sir_abc.generations)
        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Lotka-Volterra ABC Analysis with TSMVAE"
    )
    parser.add_argument(
        "--config", 
        required=True,
        help="Path to config.yaml file in the model directory"
    )
    
    parser.add_argument(
        "--data-dir", 
        default="data",
        help="Directory containing the training data"
    )

    parser.add_argument(
        "--sim-params", 
        type=float, 
        nargs=2,
        default=[1, 1],
        help="Parameters for simulation"
    )

    parser.add_argument(
        "--num_particles", 
        type=int, 
        default=1000,
        help="Number of particles for ABC"
    )
 
    args = parser.parse_args()

    # analyzer = LotkaVolterraAnalyzer(args.config, args.data_dir)
    analyzer = SpatialSIRAnalyzer(args.config, args.data_dir)
    #analyzer.generate_reconstruction()
    #analyzer.run_simulation(args.sim_params)
    #analyzer.plot_comparisons()

    # Run with CLS pooling and cosine metric
    # analyzer.run_abc_analysis(
    #     pooling_method="mean",
    #     metric="cosine",
    #     num_particles=args.num_particles
    # )

    analyzer.run_abc_analysis(
        pooling_method="no_cls",
        metric="pairwise_cosine",
        num_particles=args.num_particles
    )
    
    # Run with all pooling and bertscore metric
    # analyzer.run_abc_analysis(
    #     pooling_method="all",
    #     metric="bertscore",
    #     num_particles=args.num_particles
    # )

    # analyzer.run_abc_analysis(
    #     pooling_method="cls",
    #     metric="l1",
    #     num_particles=args.num_particles
    # )

    # analyzer.run_abc_analysis(
    #     pooling_method="cls",
    #     metric="l2",
    #     num_particles=args.num_particles
    # )


if __name__ == "__main__":
    main()