import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from models import TSMVAE
from lightning_module import PreTrainLightning, FineTuneLightning
from systems import LotkaVolterra2
from dataset import NumpyDataset
from latent_abc_pmc import viaABC

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and validate the configuration file.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Dict[str, Any]: Loaded configuration.

    Raises:
        FileNotFoundError: If the config file is not found.
        ValueError: If the config file is invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    try:
        config = yaml.safe_load(config_path.read_text())
        if not isinstance(config, dict):
            raise ValueError("Config file must be a dictionary.")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")


def load_model(path: Path, finetune: bool = False, num_parameters: int = 2) -> PreTrainLightning:
    """
    Load the model from the given path.

    Args:
        path (Path): The path to the model directory.
        finetune (bool): Whether to load a fine-tuned model.
        num_parameters (int): Number of parameters for the fine-tuned model.

    Returns:
        PreTrainLightning: The pre-trained or fine-tuned model.

    Raises:
        FileNotFoundError: If the config file or model checkpoint is not found.
    """
    config_path = path / "config.yaml"
    config = load_config(config_path)

    # Load model architecture
    model = TSMVAE(**config["model"]["params"])
    # model = LSTMVAE_LINEAR_ENCODE(**config["model"]["params"])

    # Load pre-trained model
    pretrain_model_path = next((f for f in path.iterdir() if f.suffix == ".ckpt" and "TSMVAE" in f.name), None)
    if not pretrain_model_path:
        raise FileNotFoundError("No pre-trained model checkpoint found.")

    pretrain_model = PreTrainLightning.load_from_checkpoint(pretrain_model_path, model=model)
    logger.info(f"Pre-trained model loaded from {pretrain_model_path}")

    if not finetune:
        return pretrain_model

    # Load fine-tuned model
    finetune_model_path = next((f for f in path.iterdir() if f.suffix == ".ckpt" and "fine_tune" in f.name), None)
    if not finetune_model_path:
        raise FileNotFoundError("No fine-tuned model checkpoint found.")

    finetune_model = FineTuneLightning.load_from_checkpoint(
        finetune_model_path, pl_module=pretrain_model, num_parameters=num_parameters
    )
    logger.info(f"Fine-tuned model loaded from {finetune_model_path}")
    return finetune_model


def load_system(name: str = "lotka_volterra", data_dir: str = "data") -> viaABC:
    """
    Load the system from the given name.

    Args:
        name (str): The name of the system to load.

    Returns:
        LotkaVolterra: The loaded system.

    Raises:
        ValueError: If the system name is not recognized.
    """
    name = name.lower()
    if name == "lotka_volterra":
        system = LotkaVolterra2()
        train_ds = NumpyDataset(data_dir, "train")
        system.update_train_dataset(train_ds)
    elif name == "sir":
        raise NotImplementedError("SIR system is not implemented yet.")
    else:
        raise ValueError(f"System {name} not found.")

    logger.info(f"System {name} loaded.")
    return system


def load_observational_data(path: Path, system: viaABC) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load observational data for the specified system.

    Args:
        path (Path): The path to the data directory.
        system (str): The name of the system.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Raw and scaled observational data.

    Raises:
        FileNotFoundError: If the data files are not found.
    """
    system = system.lower()
    prefix = "lotka" if system == "lotka_volterra" else "sir"

    data_path = path / f"{prefix}_data.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found in {path}. Make sure it is named '{prefix}_data.npz'.")

    data = np.load(data_path, allow_pickle=True)
    obs_data = data.get('obs_data')
    scaled_obs_data = data.get('scaled_obs_data')
    # scaled_obs_data = data.get('scaled_obs_data')
    # obs_scale = data.get('obs_scale')
    # ground_truth = data.get('ground_truth')
    # scaled_ground_truth = data.get('scaled_ground_truth')
    # ground_truth_scale = data.get('ground_truth_scale')

    raw_np = obs_data
    raw_np_scaled = scaled_obs_data

    logger.info(f"Observational data loaded from {path}")
    return raw_np, raw_np_scaled


@torch.inference_mode()
def reconstruct_data(pl_model: PreTrainLightning, data: np.ndarray, scaled: bool = False) -> np.ndarray:
    """
    Reconstruct the data using the model.

    Args:
        pl_model (PreTrainLightning): The pre-trained model.
        data (np.ndarray): The data to reconstruct.
        scaled (bool): Whether the data is already scaled.

    Returns:
        np.ndarray: The reconstructed data.
    """
    if not scaled:
        data = data / np.abs(data).mean(axis=0)

    with torch.no_grad():
        _, _, _, param_est, reconstruction = pl_model(
            torch.tensor(data).float().to(pl_model.device).unsqueeze(0), mask_ratio=0
        )
        reconstruction = reconstruction.squeeze(0).cpu().numpy()

    logger.info("Data reconstruction completed.")
    return reconstruction


def plot_reconstructions(
    prediction: np.ndarray, ground_truth: np.ndarray, observational: np.ndarray, path: Path, scaled: bool = False) -> None:
    """
    Plot the reconstructions against ground truth and observational data.

    Args:
        prediction (np.ndarray): The predicted data.
        ground_truth (np.ndarray): The ground truth data.
        observational (np.ndarray): The observational data.
        system (str): The name of the system.
    """
    n = prediction.shape[1]

    fig, ax = plt.subplots(1, n, figsize=(20, 5))
    
    for i in range(n):
        ax[i].plot(prediction[:, i], label='Prediction')
        ax[i].plot(ground_truth[:, i], label='Ground Truth')
        ax[i].plot(observational[:, i], label='Noisy')
        ax[i].legend()

    if not scaled:
        plt.savefig(path / "reconstructions.png")
    else:
        plt.savefig(path / "reconstructions_scaled.png")
    logger.info("Reconstruction plots saved.")


def run_abc(
    system: viaABC,
    model: PreTrainLightning,
    tolerance_levels: List[float],
    num_particles: int,
    output_dir: Path,
    finetune: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the ABC-SMC algorithm.

    Args:
        system (LotkaVolterra): The system to run the algorithm on.
        model (PreTrainLightning): The model to use.
        tolerance_levels (List[float]): The tolerance levels for the algorithm.
        num_particles (int): The number of particles to use.
        output_dir (Path): The directory to save the output.
        finetune (bool): Whether to finetune the model.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Particles and weights from the ABC-SMC algorithm.
    """
    system.update_model(model)
    system.run(num_particles=num_particles)
    system.compute_statistics()

    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / ("latent_abc_smc_finetune.npz" if finetune else "latent_abc_smc_mtm.npz")
    np.savez(output_file, particles=particles, weights=weights)
    logger.info(f"ABC-SMC results saved to {output_file}")

    return particles, weights

def plot_particles(particles: np.ndarray, weights: np.ndarray, output_dir: Path, finetune: bool = False) -> None:
    """
    Plot the particles and save the plots.

    Args:
        particles (np.ndarray): The particles to plot.
        weights (np.ndarray): The weights of the particles.
        output_dir (Path): The directory to save the plots.
        finetune (bool): Whether the particles are from a finetuned model.
    """
    plot_dir = output_dir / ("figures_finetune" if finetune else "figures_mtm")
    plot_dir.mkdir(exist_ok=True)

    num_generations = particles.shape[0]
    num_parameters = particles.shape[-1]
    for i in range(num_generations):
        fig, ax = plt.subplots(1, num_parameters, figsize=(6, 4))
        for j in range(num_parameters):
            ax[j].hist(particles[i,:,j], bins=20, alpha=0.7, label="Posterior", weights=weights[i])
            
            # Set parameter title based on index
            if j == 0:
                ax[j].set_title(r'$\beta$')
            elif j == 1:
                ax[j].set_title(r'$\gamma$')
            else:
                ax[j].set_title(f'Parameter {j}')
            
            ax[j].axvline(x=1, color='r', linestyle='--', label='Ground Truth') # fixme
            ax[j].axvline(x=particles[i,:,j].mean(), color='g', linestyle='--', label='Mean')
            ax[j].set_xlim(0, 5)
            ax[j].legend()

        fig.suptitle(f"Generation {i+1}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"generation_{i+1}.png", dpi=100)
        
    plt.close()
    logger.info(f"Particle plots saved to {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ABC-SMC algorithm.")
    parser.add_argument("--path", type=str, required=True, help="Path to the model and data directory.")
    parser.add_argument("--system", type=str, default="lotka_volterra", help="Name of the system to use.")
    parser.add_argument("--tolerance_levels", type=float, nargs="+", required=True, help="Tolerance levels for ABC-SMC.")
    parser.add_argument("--num_particles", type=int, required=True, help="Number of particles for ABC-SMC.")
    parser.add_argument("--finetune", action="store_true", help="Whether the model is finetuned or not.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the data directory.")
    args = parser.parse_args()

    # Validate system and set number of parameters
    if args.system == "lotka_volterra":
        num_parameters = 2
    elif args.system == "sir":
        num_parameters = 3
    else:
        raise ValueError(f"System {args.system} not found.")

    # Convert path to Path object
    path = Path(args.path)
    output_dir = path / "output"
    output_dir.mkdir(exist_ok=True)

    # Load model, system, and data
    model = load_model(path, args.finetune, num_parameters)
    system = load_system(args.system, args.data_dir)

    if not args.finetune:
        raw_data, scaled_data = load_observational_data(Path("data"), system=args.system)
        ground_truth, _ = system.simulate([1, 1])
        # ground_truth_scaled = (ground_truth - ground_truth.mean(axis=0)) / ground_truth.std(axis=0)
        ground_truth_scaled = ground_truth / ground_truth.mean(axis=0)
        reconstructed_data_scaled = reconstruct_data(model, raw_data)

        reconstructed_data = (reconstructed_data_scaled * ground_truth.mean(axis=0))

        plot_reconstructions(reconstructed_data, ground_truth, raw_data, output_dir)
        plot_reconstructions(reconstructed_data_scaled, ground_truth_scaled, scaled_data, output_dir, scaled=True)

    # Run ABC-SMC and plot results
    particles, weights = run_abc(system, model, args.tolerance_levels, args.num_particles, output_dir, args.finetune)
    plot_particles(particles, weights, output_dir, args.finetune)