"""
latent_abc_smc.py

This script implements the Sequential Monte Carlo (SMC) algorithm for Approximate Bayesian Computation (ABC).

Some code is referenced from https://github.com/Pat-Laub/approxbayescomp/blob/master/src/approxbayescomp/smc.py#L73
Some code is referenced from https://github.com/jakeret/abcpmc/blob/master/abcpmc/sampler.py
"""

# Standard library imports
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Union, List

from metrics import *
# Third-party scientific computing
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde, qmc, uniform, multivariate_normal, norm
from densratio import densratio

# Machine learning and visualization
import torch
import umap

#from cuml.manifold.umap import UMAP as cuUMAP
from tqdm import tqdm
import matplotlib.pyplot as plt

from tqdm import tqdm

class GaussianPrior(object):
    """
    Normal gaussian prior
     
    :param mu: scalar or vector of means
    :param sigma: scalar variance or covariance matrix
    """
    
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self._random = np.random.mtrand.RandomState()
        
    def __call__(self, theta=None):
        if theta is None:
            return self._random.multivariate_normal(self.mu, self.sigma)
        else:
            return multivariate_normal.pdf(theta, self.mu, self.sigma)
        
class TophatPrior(object):
    """
    Tophat prior
    
    :param min: scalar or array of min values
    :param max: scalar or array of max values
    """
    
    def __init__(self, min, max):
        self.min = np.atleast_1d(min)
        self.max = np.atleast_1d(max)
        self._random = np.random.mtrand.RandomState()
        assert self.min.shape == self.max.shape
        assert np.all(self.min < self.max)
        
    def __call__(self, theta=None):
        if theta is None:
            return np.array([self._random.uniform(mi, ma) for (mi, ma) in zip(self.min, self.max)])
        else:
            return 1 if np.all(theta < self.max) and np.all(theta >= self.min) else 0
        
class _WeightWrapper(object):  # @DontTrace
    """
    Wraps the computation of new particle weights.
    Allows for pickling the functionality.
    """
    
    def __init__(self, prior, sigma, ws, thetas):
        self.prior = prior
        self.sigma = sigma
        self.ws = ws
        self.thetas = thetas
    
    def __call__(self, theta):
        kernel = multivariate_normal(theta, self.sigma).pdf
        w = self.prior(theta) / np.sum(self.ws * kernel(self.thetas))
        return w

class viaABC:
    @torch.inference_mode()
    def __init__(self,
                num_parameters: int, 
                #lower_bounds: np.ndarray, 
                #upper_bounds: np.ndarray, 
                mu: np.ndarray,
                sigma: np.ndarray,
                observational_data: np.ndarray, 
                model: Union[torch.nn.Module, None],
                state0: Union[np.ndarray, None],
                t0: Union[int, None], 
                tmax: Union[int, None], 
                time_space: np.ndarray,
                pooling_method: str,
                metric: str = "l2",
                # batch_effect: bool = False,
                ):
        """
        Initialize the viaABC algorithm.

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
        self.logger.info("Initializing viaABC class")
        
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
        self.mu = mu
        self.sigma = sigma

        # Ensure lower and upper bounds match the number of parameters
        assert len(mu) == len(sigma) == num_parameters, "Length of mu and sigma must match the number of parameters"

        if pooling_method is None:
            raise ValueError("Pooling method must be provided.")
        else:
            self.pooling_method = pooling_method

        if metric is None:
            raise ValueError("Metric must be provided.")
        else:
            #TODO: check if provided metric is a valid metric
            self.metric = metric

        self.logger.info("Initialization complete")
        self.logger.info("LatentABCSMC class initialized with the following parameters:")
        self.logger.info(f"num_parameters: {num_parameters}")
        self.logger.info(f"Mu: {mu}")
        self.logger.info(f"Sigma: {sigma}")
        self.logger.info(f"t0: {t0}")
        self.logger.info(f"tmax: {tmax}")
        self.logger.info(f"time_space: {time_space}")
        self.logger.info(f"pooling_method: {pooling_method}")
        self.logger.info(f"metric: {metric}")

    def update_model(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.logger.info("Model updated")
        self.encode_observational_data()

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

        # Sample from the prior distribution
        return np.random.multivariate_normal(self.mu, self.sigma)

    def calculate_prior_prob(self, theta: np.ndarray) -> float:
        """
        Calculate the prior probability of the given parameters.

        Args:
            parameters (np.ndarray): Parameters to calculate the prior probability for.

        Returns:
            np.ndarray: Prior probability of the parameters.
        """

        return multivariate_normal.pdf(theta, self.mu, self.sigma)


    def perturb_parameters(self, theta: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Perturb the given parameters, following (Beaumont et al. 2009)

        Args:
            parameters (np.ndarray): Parameters to perturb.

        Returns:
            np.ndarray: Perturbed parameters.
        """

        proposed_theta = multivariate_normal.rvs(mean=theta, cov=cov)
        return proposed_theta

    def calculate_distance(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate distances between encoded observational data and input vectors in y in a vectorized manner.
        
        Args:
            y: Batched input vectors to compare against (shape: [batch_size, vector_size])
            
        Returns:
            np.ndarray: Distance measures between 0 and 2 for each item in the batch
        """
        x = self.encoded_observational_data  # Encoded data (single reference vector)

        # safe-guard
        if y.shape[0] == 1:
            y = y.squeeze(0)
        
        if self.metric == "cosine":
            # Compute cosine similarity for all items in the batch
            cos_sim_values = cosine_similarity(x, y)
            distances = 1-cos_sim_values
        elif self.metric == "l2":
            # Compute L2 distance for all items in the batch
            distances = l2_distance(x, y)
        elif self.metric == "bertscore":
            # # Compute similarity for all items in the batch using bert_score (assuming bert_score can handle batched inputs)
            _, _, f1_scores = bert_score(x, y)
            
            # Calculate distances for all items in the batch
            distances = 1-f1_scores
            
        return distances
    
    @torch.inference_mode()
    def encode_observational_data(self):
        scaled_data = self.preprocess(self.raw_observational_data)
        tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.model.device) # [1, T, d]
        self.encoded_observational_data = self.model.get_latent(tensor, pooling_method=self.pooling_method).cpu().numpy().squeeze(0)
    
    def _log_generation_stats(self, t: int, particles: np.ndarray, weights: np.ndarray, start_time: float, num_simulations: int, epsilon: float):
        duration = time.time() - start_time
        self.logger.info(f"Generation {t + 1} Completed. Accepted {self.num_particles} particles in {duration:.2f} seconds with {num_simulations} total simulations.")
        self.logger.info(f"Epsilon: {epsilon}")

        mean = np.average(particles, weights=weights, axis=0)
        var = np.average((particles - mean) ** 2, weights=weights, axis=0)
        median = np.median(particles, axis=0)

        self.logger.info(f"Mean: {mean}")
        self.logger.info(f"Median: {median}")
        self.logger.info(f"Variance: {var}")

    @torch.inference_mode()
    def run_init(self, num_particles: int, k: int):
        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the algorithm.")
    
        num_particles
        total_num_simulations = 0
        accepted = 0

        self.encode_observational_data()
        particles = np.ones((k * num_particles, self.num_parameters))
        weights = np.ones((k * self.num_particles))
        dists = np.zeros((k * num_particles))

        while accepted < self.num_particles:
            perturbed_params = self.sample_priors()
            prior_probability = 1.0
            new_weight = 1.0
        
            prior_probability = self.calculate_prior_prob(perturbed_params)

            if prior_probability <= 0:
                continue

            y, status = self.simulate(perturbed_params)
            running_num_simulations += 1

            if status != 0:
                continue

            y_scaled = self.preprocess(y)
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(0).to(self.model.device)
            y_latent_np = self.model.get_latent(y_tensor, pooling_method=self.pooling_method).cpu().numpy().squeeze(0)
            dist = self.calculate_distance(y_latent_np)

            particles[accepted] = perturbed_params
            weights[accepted] = new_weight
            dists[accepted] = dist
            accepted += 1

        # Sort by distance
        sorted_indices = np.argsort(dists)
        particles = particles[sorted_indices]
        weights = weights[sorted_indices]
        dists = dists[sorted_indices]

        # Select the top particles
        particles = particles[:num_particles]
        weights = weights[:num_particles]

        # Normalize weights
        weights /= np.sum(weights)
        epsilon = dists[:num_particles][-1] # initial epsilon
        
        return particles, weights, epsilon, total_num_simulations
    
    def _get_sigma(self, theta, weights):
        """
        Computes a weighted covariance matrix
        
        :param theta: the array of values
        :param weights: array of weights for each entry of the values
        
        :returns sigma: the weighted covariance matrix
        """
        n = theta.shape[1]
        sigma = np.empty((n, n))
        w = weights.sum() / (weights.sum()**2 - (weights**2).sum()) 
        average = np.average(theta, axis=0, weights=weights)
        for j in range(n):
            for k in range(n):
                sigma[j, k] = w * np.sum(weights * ((theta[:, j] - average[j]) * (theta[:, k] - average[k])))
        return sigma

    @torch.inference_mode()
    def run(self, num_particles: int, q_threshold=0.99):
        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the algorithm.")

        self.num_particles = num_particles
        total_num_simulations = 0

        self.encode_observational_data()
        
        # Use lists for dynamic growth during the run
        all_particles = []
        all_weights = []
        epsilons = []

        self.logger.info(f"Q Threshold: {q_threshold}")
        start_time = time.time()
        self.logger.info("Starting ABC SMC run")

        # Initial generation
        init_start_time = time.time()
        generation_particles, generation_weights, init_epsilon, running_num_simulations = self.run_init(num_particles, k=5)
        init_end_time = time.time()
        self.logger.info(f"Initialization completed in {init_end_time - init_start_time:.2f} seconds")

        # Store first generation
        all_particles.append(np.array(generation_particles))
        normalized_weights = np.array(generation_weights).reshape(num_particles) / np.sum(generation_weights)
        all_weights.append(normalized_weights)
        epsilons.append(init_epsilon)

        self._log_generation_stats(0, all_particles[0], all_weights[0], start_time, running_num_simulations, init_epsilon)

        t = 1  # Start from generation 1 (0 was initialization)
        while True:
            self.logger.info(f"Generation {t} started")
            start_time_generation = time.time()

            generation_particles = []
            generation_weights = []
            generation_dists = []
            accepted = 0
            running_num_simulations = 0

            previous_particles = all_particles[t - 1]
            previous_weights = all_weights[t - 1]

            while accepted < self.num_particles:
                idx = np.random.choice(len(previous_particles), p=previous_weights)
                theta = previous_particles[idx]
                sigma = self._get_sigma(theta, previous_weights)
                sigma = np.atleast_2d(sigma)

                # Perturb the parameters
                perturbed_params = self.perturb_parameters(theta, sigma)
                prior_probability = self.calculate_prior_prob(perturbed_params)

                if prior_probability <= 0:
                    continue
                
                # TODO: FIX THIS
                # theta_log_weight = np.log(prior_probability) - gaussian_kde(previous_particles.T, weights=previous_weights).logpdf(perturbed_params)
                # new_weight = np.exp(theta_log_weight)

                phi = np.array([np.sum(norm.logpdf(x, loc=perturbed_params, scale=sigma)) for x in previous_particles])
                log_weights = np.log(previous_weights) + phi
                lse = np.log(np.sum(np.exp(log_weights)))
                new_weight = np.exp(prior_probability - lse)

                if new_weight <= 0:
                    continue

                y, status = self.simulate(perturbed_params)
                running_num_simulations += 1

                if status != 0:
                    continue

                y_scaled = self.preprocess(y)
                y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(0).to(self.model.device)
                y_latent_np = self.model.get_latent(y_tensor, pooling_method=self.pooling_method).cpu().numpy().squeeze(0)
                dist = self.calculate_distance(y_latent_np)
                if dist >= epsilons[t-1]:
                    continue

                accepted += 1
                generation_particles.append(perturbed_params)
                generation_weights.append(new_weight)
                generation_dists.append(dist)

            total_num_simulations += running_num_simulations
            
            # Store current generation
            current_particles = np.array(generation_particles)
            current_weights = np.array(generation_weights).reshape(num_particles) / np.sum(generation_weights)
            all_particles.append(current_particles)
            all_weights.append(current_weights)

            # Calculate qt and new epsilon
            dens = densratio(generation_particles, previous_particles, kernel_num=100, verbose=False)
            density_ratios = dens.compute_density_ratio(generation_particles)
            ct = max(np.max(density_ratios), 1.0)
            qt = 1.0 / ct
            eps = np.quantile(generation_dists, qt)
            epsilons.append(eps)

            self._log_generation_stats(t, current_particles, current_weights, start_time_generation, running_num_simulations, eps)

            # Check stop condition
            if qt >= q_threshold:
                self.logger.info(f"Stopping criterion met (qt = {qt:.3f} >= {q_threshold})")
                break
                
            # Optional: Add other stopping criteria (e.g., minimum epsilon reached)
            if t >= 100:  # Absolute maximum generations
                self.logger.warning(f"Reached maximum generations (100) without meeting stop criterion")
                break
                
            t += 1

        # Convert lists to numpy arrays at the end
        self.particles = np.array(all_particles)
        self.weights = np.array(all_weights)

        duration = time.time() - start_time
        num_generations = t + 1  # +1 because we count generation 0
        self.logger.info(f"ABC SMC run completed in {duration:.2f} seconds with {total_num_simulations} total simulations over {num_generations} generations.")

        return self.particles, self.weights

    def compute_statistics(self, generation: int = 0):
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
        var = np.average((particles - mean) ** 2, weights=weights, axis=0)
        median = np.median(particles, axis=0)

        self.mean = mean    
        self.var = var
        self.median = median

        self.logger.info(f"Mean: {mean}")
        self.logger.info(f"Median: {median}")
        self.logger.info(f"Variance: {var}")

        return mean, median, var
    
    def visualize_generation(self, generation: int = -1, save: bool = False):
        fig, ax = plt.subplots(1, self.num_parameters, figsize=(6, 4))

        # Plot histograms for Beta and Alpha
        for i in range(self.num_parameters):
            ax[i].hist(self.particles[generation][:, i], bins=20, alpha=0.7, label="Posterior")
            ax[i].set_title(f'Parameter {i+1}')
            ax[i].axvline(x=self.particles[generation][:, i].mean(), color='g', linestyle='--', label='Mean')
            ax[i].legend()

        # Set a title for the entire figure
        fig.suptitle(f"Generation {generation+1}")
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
        
        # Save the figure as a PNG file
        if save:
            if generation == -1:
                generation = self.num_generations-1
 
            os.makedirs("figures", exist_ok=True)
            plt.savefig(f"figures/generation_{generation+1}.png", dpi=100)
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
            # TODO: implement cuML UMAP
            # reducer = cuUMAP(n_components=2, random_state=42)
            raise NotImplementedError("cuML UMAP is not implemented yet.")
        elif accelerator == "cpu":
            print("Using original UMAP (CPU)...")
            reducer = umap.UMAP(n_components=2, random_state=42)
        
        # Fit and transform the latent representations
        umap_embedding = reducer.fit_transform(latent_representations)
        test_embedding = reducer.transform(self.encoded_observational_data.reshape(1, -1))

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
            futures = {executor.submit(run_simulation, i, param): i for i, param in enumerate(parameters)}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        _, simulation, param = result
                        valid_simulations.append(simulation)
                        valid_params.append(param)
                except Exception as e:
                    self.logger.error(f"Error processing future: {e}")

        if valid_simulations:
            valid_simulations = np.array(valid_simulations)
            valid_params = np.array(valid_params)

            self.logger.info(f"Saving {len(valid_simulations)} simulations to data.")
            np.savez(f"data/{prefix}_data.npz", params=valid_params, simulations=valid_simulations)
        else:
            self.logger.warning("No valid simulations to save.")

    def generate_training_data(self, num_simulations: list = [50000, 10000, 10000], seed: int = 1234, num_workers: int = 1):
        np.random.seed(seed)
        self.logger.info(f"Generating training data for training with seed {seed}")

        prefix = ["train", "val", "test"]
        total_time = 0  # Initialize the total time counter
        
        for i, num_simulation in enumerate(num_simulations):
            self.logger.info(f"Generating {num_simulation} simulations for training data")
            start = time.time()
            self.__batch_simulations(num_simulation, prefix=prefix[i], num_threads=num_workers * 2)
            end = time.time()
            elapsed = end - start
            total_time += elapsed
            self.logger.info(f"Generated {num_simulation} simulations for training data in {elapsed:.2f} seconds")

        self.logger.info(f"Training data generation completed and saved. Total time taken: {total_time:.2f} seconds")

    def load_particles_weights(self, particles_path: str, weights_path: str):
        try:
            self.particles = np.load(particles_path)
            self.weights = np.load(weights_path)
            self.logger.info("Particles and weights loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading particles and weights: {e}")

    def update_train_dataset(self, train_dataset):
        # check if train_dataset is a torch Dataset
        if not isinstance(train_dataset, torch.utils.data.Dataset):
            raise ValueError("train_dataset must be a torch Dataset.")
        
        self.train_dataset = train_dataset
        self.logger.info("Training dataset updated.")

    @torch.inference_mode()
    def initialize_particles(self, num_particles: int, epsilon: float):
        if self.train_dataset is None:
            raise ValueError("Training dataset must be provided to initialize particles.")
        
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1000, shuffle=False)
        candidates = {'parameters': [], 'distances': []}
        
        # Iterate through the data loader
        for i, batch in enumerate(train_loader):
            x, y = batch
            y = y.to(self.model.device).float()
            latent = self.model.get_latent(y).cpu().numpy()
            
            # Ensure the latent is a tensor before appending
            candidates['parameters'].append(x.numpy() * 10)
            candidates['distances'].append(self.calculate_distance(latent))

        # Concatenate parameters and distances (all are tensors, so use torch.cat and torch.stack)
        candidates['parameters'] = np.concatenate(candidates['parameters'], axis=0)
        candidates['distances'] = np.concatenate(candidates['distances'], axis=0)

        # Sort the candidates by distance
        sorted_indices = np.argsort(candidates['distances'])
        sorted_candidates = candidates['parameters'][sorted_indices]
        sorted_distances = candidates['distances'][sorted_indices]

        # Filter candidates by epsilon
        valid_indices = sorted_distances < epsilon
        sorted_candidates = sorted_candidates[valid_indices]

        # Randomly sample num_particles from the valid candidates
        if len(sorted_candidates) < num_particles:
            raise ValueError("Not enough valid candidates to sample from.")
        
        np.random.seed(1234)
        np.random.shuffle(sorted_candidates)

        # Sample
        indices = np.random.choice(len(sorted_candidates), num_particles, replace=False)

        # Grab the first num_particles candidates
        # particles = sorted_candidates[:num_particles]
        particles = sorted_candidates[indices]
        weights = [1.0] * num_particles

        return particles, weights
    
    @torch.inference_mode()
    def get_latent(self, x):
        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the method.")
        
        # if x is numpy convert to tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.model.device)

        x = self.model.get_latent(x, self.pooling_method)

        # if x is tensor convert to numpy, safeguard
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        return x
    
    def preprocess(self, x):
        raise NotImplementedError
        
    
    