"""
latent_abc_smc.py

This script implements the Sequential Monte Carlo (SMC) algorithm for Approximate Bayesian Computation (ABC).
Some code is referenced from https://github.com/Pat-Laub/approxbayescomp/blob/master/src/approxbayescomp/smc.py#L73
"""

# Standard library imports
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Union, List

# Third-party scientific computing
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde, qmc

# Machine learning and visualization
import torch
import umap
#from cuml.manifold.umap import UMAP as cuUMAP
from tqdm import tqdm
import matplotlib.pyplot as plt

class LatentABCSMC:
    @torch.inference_mode()
    def __init__(self,
                num_parameters: int, 
                lower_bounds: np.ndarray, 
                upper_bounds: np.ndarray, 
                perturbation_kernels: np.ndarray, 
                observational_data: np.ndarray, 
                model: Union[torch.nn.Module, None],
                state0: Union[np.ndarray, None],
                t0: Union[int, None], 
                tmax: Union[int, None], 
                time_space: np.ndarray,
                batch_effect: bool = False):
        """
        Initialize the Latent-ABC algorithm.

        Parameters:
        model: The latent encoding model to be used in the algorithm.
        num_parameters (int): Number of parameters in the ODE system.
        TODO: fix to take in non-uniform priors
        lower_bounds (np.ndarray): Lower bounds for uniform priors.
        upper_bounds (np.ndarray): Upper bounds for uniform priors.
        perturbation_kernels (np.ndarray): Values used to perturb the particles using Uniform distribution.
        TODO: handle multiple data points at a single time point, possibly Bootstrap Aggregation (Bagging)?
        observational_data (np.ndarray): Observational/reference data to compare against. 
            Shape must be T x d where d is the dimension of the data and T is the number of time points.
        state0 (np.ndarray): Initial state of the system. Can be None if you want to sample the initial state.
        t0 (int): Initial time. Defaults to the first time point of time_space.
        tmax (int): Maximum time. Defaults to the last time point + 1 of time_space.
        time_space (np.ndarray): Evaluation space, must be sorted in ascending order.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("Initializing LatentABCSMC class")
        
        self.model = model
        if model is not None:
            self.model.eval()

        self.raw_observational_data = observational_data

        self.state0 = state0
        
        if state0 is None:
            self.logger.warning("Initial state is not initialized. You only want to do this if you wish to sample initial condition.")
            self.logger.warning("You must also define custom `simulate`, and `sample_priors` methods.")

        if t0 is None:
            raise ValueError("t0 must be provided")
        else:
            self.t0 = t0

        if tmax is None:
            raise ValueError("tmax must be provided")
        else:
            self.tmax = tmax

        self.time_space = time_space

        assert self.t0 < self.tmax, "t0 must be less than tmax"
        assert self.tmax >= time_space[-1], "tmax must be less than or equal to the last element of time_space"
        assert self.t0 <= time_space[0], "t0 must be greater than or equal to the first element of time_space"

        # Ensure observational data and time space match
        assert observational_data.shape[0] == len(time_space), f"Observational data and time space must match, observational data shape: {observational_data.shape[0]}, time space length: {len(time_space)}"

        if model is None:
            # warning
            self.logger.warning("Model is None. The model must be provided to encode the data and run the algorithm.")
            self.logger.warning("The class can be initialized without a model, but it will not be able to run the algorithm.")
        else:
            self.encoded_observational_data = self.model(torch.tensor(self.raw_observational_data, dtype=torch.float32).unsqueeze(0))[1].cpu().numpy()

        self.num_parameters = num_parameters
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Ensure lower and upper bounds match the number of parameters
        assert len(lower_bounds) == len(upper_bounds) == num_parameters, "Lower and upper bounds must match the number of parameters"

        self.perturbation_kernels = perturbation_kernels

        assert (perturbation_kernels > 0).all(), "Perturbation kernels must be greater than 0"
        
        self.batch_effect = batch_effect # TODO: implement batch effect
        # Ensure perturbation kernels match the number of parameters
        assert len(perturbation_kernels) == num_parameters, "Perturbation kernels must match the number of parameters"
        self.logger.info("Initialization complete")
        self.logger.info("LatentABCSMC class initialized with the following parameters:")
        self.logger.info(f"num_parameters: {num_parameters}")
        self.logger.info(f"lower_bounds: {lower_bounds}")
        self.logger.info(f"upper_bounds: {upper_bounds}")
        self.logger.info(f"perturbation_kernels: {perturbation_kernels}")
        self.logger.info(f"t0: {t0}")
        self.logger.info(f"tmax: {tmax}")
        self.logger.info(f"time_space: {time_space}")

    def update_model(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.logger.info("Model updated")

    def ode_system(self, t: int, state: np.ndarray, parameters: np.ndarray):
        raise NotImplementedError

    def simulate(self, parameters: np.ndarray):
        # TODO: implement batch effect
        if self.state0 is not None:
            solution = solve_ivp(self.ode_system, [self.t0, self.tmax], y0=self.state0, t_eval=self.time_space, args=(parameters,))
            return solution.y.T, solution.status

    def sample_priors(self) -> np.ndarray:
        """
        Sample from the prior distributions.

        This method should be implemented to return a sample from the prior distributions
        of the parameters. The specific implementation will depend on the prior distribution
        used in the model.

        Returns:
            np.ndarray: Sampled parameters from the prior distributions.
        """
        raise NotImplementedError

    def calculate_prior_prob(self, parameters: np.ndarray) -> float:
        """
        Calculate the prior probability of the given parameters.

        Args:
            parameters (np.ndarray): Parameters to calculate the prior probability for.

        Returns:
            np.ndarray: Prior probability of the parameters.
        """
        raise NotImplementedError

    def perturb_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Perturb the given parameters.

        This method should be implemented to perturb the given parameters according to
        the perturbation kernel used in the model. The specific implementation will depend
        on the perturbation kernel used.

        Args:
            parameters (np.ndarray): Parameters to perturb.

        Returns:
            np.ndarray: Perturbed parameters.
        """
        raise NotImplementedError

    def calculate_distance(self, y: np.ndarray) -> float:
        """
        Calculate the distance between the simulated data and the observed data.

        Args:
            y (np.ndarray): Simulated data.

        Returns:
            float: Distance between the simulated data and the observed data.
        """
        raise NotImplementedError
    
    @torch.inference_mode()
    def encode_observational_data(self):
        mean = self.raw_observational_data.mean(0)
        std = self.raw_observational_data.std(0)
        scaled_data = self.raw_observational_data/mean
        self.encoded_observational_data = self.model.get_latent(torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.model.device)).cpu().numpy()
    
    @torch.inference_mode()
    def run(self, num_particles: int, tolerance_levels : List):
        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the algorithm.")
        
        if tolerance_levels is None:
            raise ValueError("Tolerance levels must be provided")
    
        self.num_generations = len(tolerance_levels)
        self.num_particles = num_particles
        total_num_simulations = 0
        self.encode_observational_data()
        particles = np.ones((self.num_generations, self.num_particles, self.num_parameters))
        weights = np.ones((self.num_generations, self.num_particles))
        self.logger.info(f"Tolerance levels: {tolerance_levels}")
        start_time = time.time()
        self.logger.info("Starting ABC SMC run")
        for t in range(self.num_generations):
            self.logger.info(f"Generation {t + 1} started")
            start_time_generation = time.time()
            generation_particles = []
            generation_weights = []
            accepted = 0
            running_num_simulations = 0
            epsilon = tolerance_levels[t]
        
            while accepted < self.num_particles:
                if t == 0:
                    # Step 4: Sample from prior in the first generation
                    perturbed_params = self.sample_priors()
                    new_weight = 1
                    prior_probability = 1
                else:
                    # Step 6: Sample from the previous generation's particles with weights
                    previous_particles = np.array(particles[t-1,:,:])
                    previous_weights = np.array(weights[t-1,:])

                    idx = np.random.choice(len(previous_particles), p=previous_weights)

                    # Step 7: Perturb parameters and calculate the prior probability
                    perturbed_params = self.perturb_parameters(previous_particles[idx], previous_particles)

                    # Step 8: If prior probability of the perturbed parameters is 0, resample
                    prior_probability = self.calculate_prior_prob(perturbed_params)
                    if prior_probability <= 0:
                        continue # Go back to sampling if prior probability is 0

                    # Convert to np.float128 for better numerical precision
                    thetaLogWeight = np.longdouble(np.log(prior_probability)) - np.longdouble(gaussian_kde(previous_particles.T, weights=previous_weights).logpdf(perturbed_params))
                    new_weight = np.exp(thetaLogWeight, dtype=np.longdouble)

                # Step 9: Simulate the ODE system and encode the data
                y, status = self.simulate(perturbed_params)

                total_num_simulations += 1
                running_num_simulations += 1

                if status != 0:
                    continue # Go back to sampling if simulation failed

                y_scaled = (y - y.mean(axis=0)) / y.std(axis=0)
                y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(0).to(self.model.device)
                y_latent_np = self.model.get_latent(y_tensor).cpu().numpy()

                # Step 10: Compute the distance and check if it's within the tolerance
                dist = self.calculate_distance(y_latent_np)

                if dist >= epsilon:
                    continue  # Go back to sampling if distance is not small enough
                
                # Step 11: Accept the particle and store it
                accepted += 1
                generation_particles.append(perturbed_params)
                generation_weights.append(new_weight)

            # Step 12: Normalize weights for the current generation
            generation_weights /= np.sum(generation_weights, axis=0)
            particles[t] = np.array(generation_particles)
            weights[t] = np.array(generation_weights).flatten()
            end_time_generation = time.time()
            duration_generation = end_time_generation - start_time_generation
            duration = end_time_generation - start_time_generation
            mean_est = np.average(particles[t], weights=weights[t], axis=0)
            self.logger.info(f"Generation {t + 1} Completed. Accepted {self.num_particles} particles in {duration_generation:.2f} seconds with {running_num_simulations} total simulations.")
            self.logger.info(f"Mean estimate: {mean_est}")

        self.particles = particles
        self.weights = weights

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f"ABC SMC run completed in {duration:.2f} seconds with {total_num_simulations} total simulations.")

        return particles, weights

    def compute_statistics(self, generation: int = 0, return_as_dataframe: bool = False):
        """ Compute statistics of the particles at the given generation.
        Args:
            generation (int): Generation to compute statistics for. Defaults to the last generation.
            return_as_dataframe (bool): Whether to return the statistics as a pandas DataFrame. Defaults to False.
        """

        if self.particles is None or self.weights is None:
            raise ValueError("Particles and weights must be computed before calculating statistics.")
        
        if generation > self.num_generations:
            raise ValueError("Generation must be less than or equal to the number of generations.")
        
        # compute statistics of the given generation
        particles = self.particles[generation-1]
        weights = self.weights[generation-1]
        mean = np.average(particles, weights=weights, axis=0)
        std = np.sqrt(np.average((particles - mean) ** 2, weights=weights, axis=0))
        median = np.median(particles, axis=0)

        self.mean = mean    
        self.std = std
        self.median = median

        if return_as_dataframe:
            return pd.DataFrame({
                'mean': mean,
                'median': median,
                'std': std,
            })
        else:
            self.logger.info(f"Mean: {mean}")
            self.logger.info(f"Median: {median}")
            self.logger.info(f"Std: {std}")
    
    def visualize_generation(self, generation: int = -1, save: bool = False):
        fig, ax = plt.subplots(1, self.num_parameters, figsize=(6, 4))

        # Plot histograms for Beta and Alpha
        for i in range(self.num_parameters):
            ax[i].hist(self.particles[generation][:, i], bins=20, alpha=0.7, label="Posterior")
            ax[i].set_title(f'Parameter {i+1}')
            ax[i].axvline(x=self.particles[generation][:, i].mean(), color='g', linestyle='--', label='Mean')
            ax[i].legend()

        # Set a title for the entire figure
        fig.suptitle(f"Generation {i+1}")
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
        
        # Save the figure as a PNG file
        if save:
            if generation == -1:
                generation = self.num_generations

            os.makedirs("figures", exist_ok=True)
            plt.savefig(f"figures/generation_{i+1}.png", dpi=100)
        plt.close()

    @torch.inference_mode()
    def visualize_latent_space(self, dataloader, accelerator="cpu"):
        """Implement UMAP to visualize the latent space with support for GPU (cuML) or CPU (UMAP)."""
        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the algorithm.")
        
        # Set the model to evaluation mode
        self.model.eval()
        
        self.encode_observational_data()
        # List to store latent representations
        latent_representations = []
        # labels = []  # Optional: if you want to color points by class
        
        # Iterate over the dataloader to get latent representations
        for batch in tqdm(dataloader, desc="Extracting latent representations"):
            x, y = batch 
            y = y.to(self.model.device).float()
            # Get latent representation
            latent = self.model.get_latent(y)
            latent_representations.append(latent.cpu())  # Move to CPU for UMAP
            # labels.append(y.cpu())  # Optional: store labels for coloring
        
        # Concatenate all latent representations and labels
        latent_representations = torch.cat(latent_representations, dim=0).numpy()
        # labels = torch.cat(labels, dim=0).numpy()  # Optional: concatenate labels
        
        # Apply UMAP based on the accelerator parameter
        if accelerator == "gpu":
            print("Using cuML's UMAP (GPU accelerated)...")
            reducer = cuUMAP(n_components=2, random_state=42)
        elif accelerator == "cpu":
            print("Using original UMAP (CPU)...")
            reducer = umap.UMAP(n_components=2, random_state=42)
        
        # Fit and transform the latent representations
        umap_embedding = reducer.fit_transform(latent_representations)
        test_embedding = reducer.transform(self.encoded_observational_data)

        # Plot the UMAP embedding
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], color='blue', s=5, label='Train Data')
        plt.scatter(test_embedding[:, 0], test_embedding[:, 1], color='red', s=10, label='Observational Data')
        plt.title(f'UMAP Visualization of Latent Space ({accelerator.upper()})')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend()
        plt.show()

    @torch.inference_mode()
    def visualize_latent_space_generation(self, generation: int = -1):
        if generation == -1:
            generation = self.num_generations - 1
        
        particles = self.particles[generation]
        latent_representations = []

        for params in particles:
            y, status = self.simulate(params)
            if status == 0:
                y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self.model.device)
                latent = self.model.get_latent(y_tensor)
                latent_representations.append(latent.cpu().numpy())

        latent_representations = np.array(latent_representations)
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_embedding = reducer.fit_transform(latent_representations)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], cmap='Spectral', s=5)
        plt.colorbar(scatter, label='Class')
        plt.title(f'UMAP Visualization of Latent Space for Generation {generation + 1}')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.show()

    def __sample_priors(self, n: int = 1):
        """Sample from prior distribution using Latin Hypercube Sampling"""
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=self.num_parameters)
        
        # Generate samples in [0,1] range
        samples = sampler.random(n=n)
        
        # Scale samples to parameter bounds
        scaled_samples = qmc.scale(samples, self.lower_bounds, self.upper_bounds)
        print(scaled_samples.shape)
        return scaled_samples

    def __batch_simulations(self, num_simulations: int, prefix: str = "train", num_threads: int = 2):
        """ This method should never be called directly; it is only used by the generate_training_data method. """
        parameters = self.__sample_priors(n=num_simulations)
        valid_params = []
        valid_simulations = []

        os.makedirs("data", exist_ok=True)  # Create directory if it doesn't exist

        def run_simulation(i, param):
            try:
                simulation, status = self.simulate(param)
                if status == 0:
                    return i, simulation, param
                else:
                    self.logger.error(f"Simulation {i} failed with status: {status}")
            except Exception as e:
                self.logger.error(f"Simulation {i} failed with error: {e}")
            return None

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_simulation, i, param) for i, param in enumerate(parameters)]
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    i, simulation, param = result
                    valid_simulations.append(simulation)
                    valid_params.append(param)

        if valid_simulations:
            valid_simulations = np.array(valid_simulations)
            valid_params = np.array(valid_params)

            self.logger.info(f"Saving {len(valid_simulations)} simulations to data.")
            np.savez(f"data/{prefix}_data.npz", params=valid_params, simulations=valid_simulations)
        else:
            self.logger.warning("No valid simulations to save.")

    def generate_training_data(self, num_simulations: list = [50000, 10000, 10000], seed: int = 1234):
        np.random.seed(seed)
        self.logger.info(f"Generating training data for training with seed {seed}")

        prefix = ["train", "val", "test"]
        total_time = 0  # Initialize the total time counter
        
        for i, num_simulation in enumerate(num_simulations):
            self.logger.info(f"Generating {num_simulation} simulations for training data")
            start = time.time()
            self.__batch_simulations(num_simulation, prefix=prefix[i])
            end = time.time()
            elapsed = end - start
            total_time += elapsed
            self.logger.info(f"Generated {num_simulation} simulations for training data in {elapsed:.2f} seconds")

        self.logger.info(f"Training data generation completed and saved. Total time taken: {total_time:.2f} seconds")