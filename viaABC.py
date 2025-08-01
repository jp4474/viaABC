"""
viaABC.py

This script implements the Adaptive Population Monte Carlo (APMC) algorithm for Approximate Bayesian Computation (ABC) [Simola et al.].

Some code is referenced from https://github.com/Pat-Laub/approxbayescomp/blob/master/src/approxbayescomp/smc.py#L73
Some code is referenced from https://rdrr.io/github/TimeWz667/odin2data/src/R/abcpmc.R
Some code is referenced from https://github.com/elfi-dev/elfi
"""

# Standard library imports
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Union

from metrics import *
# Third-party scientific computing
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc, multivariate_normal

# Machine learning and visualization
import torch

from tqdm import tqdm
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import *

class viaABC:
    """Implementation of the viaABC algorithm for approximate Bayesian computation."""

    @torch.inference_mode()
    def __init__(
        self,
        num_parameters: int,
        mu: np.ndarray,
        sigma: np.ndarray,
        observational_data: np.ndarray,
        model: Union[torch.nn.Module, None],
        state0: Union[np.ndarray, None],
        t0: Union[int, None],
        tmax: Union[int, None],
        time_space: np.ndarray,
        pooling_method: str = "cls",
        metric: str = "cosine",
    ) -> None:
        """Initialize the viaABC algorithm.

        Args:
            num_parameters: Number of parameters in the ODE system.
            mu: Mean values for normal priors.
            sigma: Standard deviation values for normal priors.
            observational_data: Observational/reference data to compare against.
                Shape must be T x d where d is the dimension of the data and
                T is the number of time points.
            model: The latent encoding model to be used in the algorithm.
            state0: Initial state of the system. Can be None if you want to sample
                the initial state.
            t0: Initial time. Must be provided.
            tmax: Maximum time. Must be provided.
            time_space: Evaluation space, must be sorted in ascending order.
            pooling_method: Method for pooling data. Must be provided.
            metric: Distance metric to use. Defaults to "cosine".

        Raises:
            ValueError: If required parameters are not provided.
            AssertionError: If various dimension and value checks fail.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("Initializing ViaABC class")

        self.raw_observational_data = observational_data
        self.state0 = state0

        if state0 is None:
            self.logger.warning(
                "Initial state is not initialized. Only do this if you wish to "
                "sample initial condition."
            )
            self.logger.warning(
                "You must also define custom `simulate` and `sample_priors` methods."
            )

        if t0 is None:
            raise ValueError("t0 must be provided")
        self.t0 = t0

        if tmax is None:
            raise ValueError("tmax must be provided")
        self.tmax = tmax

        self.time_space = time_space

        # Validate time parameters
        assert self.t0 < self.tmax, "t0 must be less than tmax"
        assert (
            self.tmax >= time_space[-1]
        ), "tmax must be greater than or equal to the last element of time_space"
        assert (
            self.t0 <= time_space[0]
        ), "t0 must be less than or equal to the first element of time_space"

        # Validate data dimensions
        assert observational_data.shape[0] == len(
            time_space
        ), (
            f"Observational data and time space must match. "
            f"Got {observational_data.shape[0]} time points in data but "
            f"{len(time_space)} in time space."
        )

        if model is None:
            self.logger.warning(
                "Model is None. The model must be provided to encode the data and run the algorithm.\n" \
                "The class can be initialized without a model, but it will not\nbe able to run the algorithm."
            )
        else:
            self.model = model
            self.model.eval()
            obs_tensor = torch.tensor(self.raw_observational_data, dtype=torch.float32)
            self.encoded_observational_data = self.model.get_latent(obs_tensor, pooling_method=pooling_method)

        self.num_parameters = num_parameters
        self.mu = mu
        self.sigma = sigma

        # Validate parameter dimensions
        assert (
            len(mu) == len(sigma) == num_parameters
        ), "Length of mu and sigma must match the number of parameters"

        if pooling_method is None:
            raise ValueError("Pooling method must be provided.")
        
        self.pooling_method = pooling_method

        if metric is None:
            raise ValueError("Metric must be provided.")
        
        valid_metrics = ['cosine', 'l1', 'l2', 'bertscore', 'pairwise_cosine', 'bertscore_batch', 'maxSim']

        if metric in valid_metrics:
            self.metric = metric
        else:
            raise ValueError(f"Metric must be one of {valid_metrics}")

        self.densratio = DensityRatioEstimation(n=self.num_particles, epsilon=0.01, max_iter=200, abs_tol=0.01, fold=5, optimize=False)
        self.generations = []

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
        self._encode_observational_data()

    def ode_system(self, t: int, state: np.ndarray, parameters: np.ndarray):
        raise NotImplementedError

    def simulate(self, parameters: np.ndarray):
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

    def calculate_prior_prob(self, theta: np.ndarray) -> float:
        """
        Calculate the prior probability of the given parameters.

        Args:
            parameters (np.ndarray): Parameters to calculate the prior probability for.

        Returns:
            np.ndarray: Prior probability of the parameters.
        """
        raise NotImplementedError

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
        #if y.shape[0] == 1:
        #    y = y.squeeze(0)
        
        if self.metric == "cosine":
            # Compute cosine similarity for all items in the batch
            cos_sim_values = cosine_similarity(x, y) # calculating the cosine similarity between z_sim and z_obs, -1 and 1 
            distances = 1-cos_sim_values # 
        elif self.metric == "l1":
            # Compute L1 distance for all items in the batch
            distances = l1_distance(x, y)
        elif self.metric == "l2":
            # Compute L2 distance for all items in the batch
            distances = l2_distance(x, y)
        elif self.metric == "bertscore":
            # # Compute similarity for all items in the batch using bert_score (assuming bert_score can handle batched inputs)
            _, _, f1_scores = bert_score(x, y)
            
            # Calculate distances for all items in the batch
            distances = 1-f1_scores

        elif self.metric == "pairwise_cosine":
            distances = 1 - pairwise_cosine(x, y)
        
        elif self.metric == "bertscore_batch":
            # Compute bert_score for all items in the batch
            mean_f1_score = bert_score_batch(x, y)

            distances = 1 - mean_f1_score

        elif self.metric == "maxSim":
            # Compute maxSim for all items in the batch
            distances = maxSim(x, y)

        return distances
    
    @torch.inference_mode()
    def _encode_observational_data(self):
        scaled_data = self.preprocess(self.raw_observational_data)
        self.encoded_observational_data = self.get_latent(scaled_data)
    
    def _log_generation_stats(self, t: int, particles: np.ndarray, weights: np.ndarray, start_time: float, num_simulations: int, epsilon: float, quantile: Union[float, None] = None):
        duration = time.time() - start_time
        self.logger.info(f"Generation {t} Completed. Accepted {self.num_particles} particles in {duration:.2f} seconds with {num_simulations} total simulations.")
        self.logger.info(f"Epsilon: {epsilon}")

        if quantile is not None:
            self.logger.info(f"Quantile: {quantile}")

        mean = np.average(particles, weights=weights, axis=0)
        var = np.average((particles - mean) ** 2, weights=weights, axis=0)
        median = np.median(particles, axis=0)

        self.logger.info(f"Mean: {mean}")
        self.logger.info(f"Median: {median}")
        self.logger.info(f"Variance: {var}")

    def update_train_dataset(self, train_dataset):
        # check if train_dataset is a torch Dataset
        if not isinstance(train_dataset, torch.utils.data.Dataset):
            raise ValueError("train_dataset must be a torch Dataset.")
        
        self.train_dataset = train_dataset
        self.logger.info("Training dataset updated.")

    @torch.inference_mode()
    def _run_init(self, num_particles: int, k: int = 5):
        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the algorithm.")
    
        total_num_simulations = 0
        accepted = 0

        self.encode_observational_data()
        particles = np.ones((k * num_particles, self.num_parameters))
        weights = np.ones((k * self.num_particles))
        dists = np.zeros((k * num_particles))

        while accepted < (k * self.num_particles):
            perturbed_params = self.sample_priors()
            prior_probability = 1.0
            new_weight = 1.0
        
            prior_probability = self.calculate_prior_prob(perturbed_params)

            if prior_probability <= 0:
                continue

            y, status = self.simulate(perturbed_params)
            total_num_simulations += 1

            if status != 0:
                continue

            y_scaled = self.preprocess(y)
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.model.device)
            y_latent_np = self.get_latent(y_tensor)
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

        sample_cov = np.atleast_2d(np.cov(particles.reshape(self.num_particles, -1), rowvar=False))
        sigma_max = np.min(np.sqrt(np.diag(sample_cov)))
        
        cov = 2 * np.diag(sample_cov)

        # Store first generation
        self.generations.append({
            't' : 0,
            'particles': np.array(particles),
            'weights': np.array(weights),
            'epsilon': epsilon,
            'cov' : sample_cov,
            'sigma_max': sigma_max,
            'simulations': total_num_simulations,
            'meta' :{
                'cov': cov
            }
        })
    
    @torch.inference_mode()
    def _run_init_v2(self, num_particles: int):
        if self.train_dataset is None:
            raise ValueError("Training dataset must be provided to initialize particles.")
        
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1000, shuffle=False)
        candidates = {'parameters': [], 'distances': []}
        
        # Iterate through the data loader
        for i, batch in enumerate(train_loader):
            x, y = batch
            y = y.to(self.model.device).float()
            latent = self.get_latent(y)
            # Ensure the latent is a tensor before appending
            candidates['parameters'].append(x.numpy())
            candidates['distances'].append(self.calculate_distance(latent))

        # Concatenate parameters and distances (all are tensors, so use torch.cat and torch.stack)
        candidates['parameters'] = np.concatenate(candidates['parameters'], axis=0)
        candidates['distances'] = np.concatenate(candidates['distances'], axis=0)

        # Sort the candidates by distance
        sorted_indices = np.argsort(candidates['distances'])
        sorted_candidates = candidates['parameters'][sorted_indices]
        sorted_distances = candidates['distances'][sorted_indices]

        particles = sorted_candidates[:num_particles]
        distances = sorted_distances[:num_particles]

        weights = np.ones(num_particles)/num_particles
        epsilon = distances[-1]

        total_num_simulations = len(train_loader) * 1000
        
        sample_cov = np.atleast_2d(np.cov(particles.reshape(self.num_particles, -1), rowvar=False))
        sigma_max = np.min(np.sqrt(np.diag(sample_cov)))
        
        cov = 2 * np.diag(sample_cov)

        # Store first generation
        self.generations.append({
            't' : 0,
            'particles': np.array(particles),
            'weights': np.array(weights),
            'distances' : distances,
            'epsilon': epsilon,
            'cov' : sample_cov,
            'sigma_max': sigma_max,
            'simulations': total_num_simulations,
            'meta' :{
                'cov': cov
            }
        })

    def _initialize_first_generation(self, k: int) -> None:
        """Initialize the first generation of particles."""
        self.logger.info("Initialization (generation 0) started")
        init_start_time = time.time()
        self._run_init(self.num_particles, k)
        #self._run_init_v2(self.num_particles)
        init_end_time = time.time()
        self.logger.info(
            f"Initialization completed in {init_end_time - init_start_time:.2f} seconds"
        )

    @torch.inference_mode()
    def run(
        self,
        num_particles: int,
        q_threshold: float = 0.99,
        max_generations: int = 20,
        k: int = 5,
    ) -> None:
        """Run the ABC PMC algorithm.
        
        Args:
            num_particles: Number of particles to maintain per generation
            q_threshold: Threshold for stopping criterion (qt >= q_threshold)
            k: Number of nearest neighbors for distance calculation
            max_generations: Maximum number of generations to run
        """
        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the algorithm.")
        
        self.generations = []

        self.num_particles = num_particles
        self.max_generations = max_generations - 1
        total_num_simulations = 0
        
        self.logger.info(f"Starting ABC PMC run with Q Threshold: {q_threshold}")
        start_time = time.time()

        # Initial generation
        self._initialize_first_generation(k = k)
        self._compute_statistics(0)
        
        # Run subsequent generations
        for generation_num in range(1, max_generations + 1):
            generation_start_time = time.time()
            self.logger.info(f"Generation {generation_num} started")
            
            # Unpack previous generation values
            prev_particles = self.generations[generation_num - 1]['particles']
            prev_weights = self.generations[generation_num - 1]['weights']
            prev_sigma_max = self.generations[generation_num - 1]['sigma_max']
            prev_epsilon = self.generations[generation_num - 1]['epsilon']
            prev_meta_cov = self.generations[generation_num - 1]['meta']['cov']
            
            # Run generation
            particles, weights, distances, simulations = self._run_generation(
                prev_particles=prev_particles,
                prev_weights=prev_weights,
                prev_cov=prev_meta_cov,
                epsilon=prev_epsilon
            )
            total_num_simulations += simulations
            
            # Process results
            current_gen = self._process_generation_results(
                generation_num=generation_num,
                particles=particles,
                weights=weights,
                distances=distances,
                prev_particles=prev_particles,
                prev_weights=prev_weights,
                prev_sigma_max=prev_sigma_max,
                simulations=simulations,
            )
            self.generations.append(current_gen)
            self._compute_statistics(generation_num)

            geneneration_end_time = time.time()
            self.logger.info(f"Generation {generation_num} completed in {geneneration_end_time - generation_start_time:.2f} seconds")
            
            # Check stopping conditions
            if self._should_stop(generation_num, current_gen['qt'], q_threshold, current_gen['epsilon']):
                break

        self._log_final_results(start_time, total_num_simulations)

    def _run_generation(
        self,
        prev_particles: np.ndarray,
        prev_weights: np.ndarray,
        prev_cov: np.ndarray,
        epsilon: float
    ) -> tuple:
        """Run a single generation of the ABC PMC algorithm.
        
        Args:
            prev_particles: Particles from previous generation
            prev_weights: Weights from previous generation
            prev_cov: Covariance matrix from previous generation
            epsilon: Epsilon threshold from previous generation
            
        Returns:
            Tuple of (particles, weights, distances, num_simulations)
        """
        particles = []
        weights = []
        distances = []
        accepted = 0
        running_num_simulations = 0
        
        with tqdm(total=self.num_particles) as pbar:
            while accepted < self.num_particles:

                theta, new_weight = self._propose_particle(
                    prev_particles=prev_particles,
                    prev_weights=prev_weights,
                    prev_cov=prev_cov
                )
                if theta is None:
                    continue
                    
                y, status = self.simulate(theta)
                running_num_simulations += 1
                
                if status != 0:
                    continue
                    
                dist = self._calculate_particle_distance(y)
                if dist >= epsilon:
                    continue
                    
                accepted += 1
                pbar.update(1)
                particles.append(theta)
                weights.append(new_weight)
                distances.append(dist)
        
        return np.array(particles), np.array(weights), np.array(distances), running_num_simulations

    def _propose_particle(
        self,
        prev_particles: np.ndarray,
        prev_weights: np.ndarray,
        prev_cov: np.ndarray
    ) -> tuple:
        """Propose a new particle and calculate its weight.
        
        Args:
            prev_particles: Particles from previous generation
            prev_weights: Weights from previous generation
            prev_cov: Covariance matrix from previous generation
            
        Returns:
            Tuple of (particle_params, weight) or (None, None) if invalid
        """
        idx = np.random.choice(self.num_particles, p=prev_weights)
        theta = prev_particles[idx]
        theta = np.atleast_2d(theta)
        
        # Perturb the parameters
        perturbed_params = self.perturb_parameters(theta[0], prev_cov)
        prior_probability = self.calculate_prior_prob(perturbed_params)
        
        if prior_probability <= 0:
            return None, None
            
        phi = np.array([
            np.sum(multivariate_normal.logpdf(perturbed_params, mean=x, cov=prev_cov)) 
            for x in prev_particles
        ])
        log_weights = np.log(prev_weights) + phi
        lse = np.log(np.sum(np.exp(log_weights)))
        new_weight = np.exp(prior_probability - lse)
        
        return (perturbed_params, new_weight) if new_weight > 0 else (None, None)

    def _process_generation_results(
        self,
        generation_num: int,
        particles: np.ndarray,
        weights: np.ndarray,
        distances: np.ndarray,
        prev_particles: np.ndarray,
        prev_weights: np.ndarray,
        prev_sigma_max: float,
        simulations: int
    ) -> dict:
        """Process and package generation results.
        
        Args:
            generation_num: Current generation number
            particles: Array of particle parameters
            weights: Array of particle weights
            distances: List of particle distances
            prev_particles: Particles from previous generation
            prev_weights: Weights from previous generation
            prev_sigma_max: Sigma max from previous generation
            
        Returns:
            Dictionary containing generation results
        """
        # Normalize weights
        weights_normalized = weights / np.sum(weights)
        
        # Calculate statistics
        sample_cov = np.atleast_2d(np.diag(weighted_var(particles, weights=weights_normalized)))
        sample_sigma = np.sqrt(np.diag(sample_cov))
        sigma_max = np.min(sample_sigma)
        meta_cov = 2 * np.diag(sample_cov)
        
        # Calculate qt and epsilon
        if self.densratio.optimize:
            sigma = list(10.0 ** np.arange(-1, 6))
        else:
            sigma = calculate_densratio_basis_sigma(sigma_max, prev_sigma_max)
        
        self.densratio.fit(
            x=prev_particles,
            y=particles,
            weights_x=prev_weights,
            weights_y=weights_normalized,
            sigma=sigma
        )
        
        max_value = max(self.densratio.max_ratio(), 1.0)
        qt = max(1 / max_value, 0.05)
        epsilon = weighted_sample_quantile(distances, qt, weights=weights_normalized)

        self.logger.info(f'ABC-SMC: Epsilon : {epsilon:.5f}')
        self.logger.info(f'ABC-SMC: Quantile : {qt:.5f}')
        self.logger.info(f'ABC-SMC: Simulations : {simulations}')
        
        return {
            't': generation_num,
            'particles': particles,
            'weights': weights_normalized,
            'epsilon': epsilon,
            'distances' : distances,
            'cov': sample_cov,
            'sigma_max': sigma_max,
            'simulations': simulations,
            'qt': qt,
            'meta': {
                'cov': meta_cov
            }
        }
    
    def _calculate_particle_distance(self, y: np.ndarray) -> float:
        """Calculate distance for a simulated particle.
        
        Args:
            y: Simulated observation
            
        Returns:
            Distance metric
        """
        y_scaled = self.preprocess(y) 
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.model.device)
        y_latent_np = self.get_latent(y_tensor) # z_sim
        return self.calculate_distance(y_latent_np)

    def _should_stop(self, generation_num: int, qt: float, q_threshold: float, epsilon: float) -> bool:
        """Determine if stopping conditions are met.
        
        Args:
            generation_num: Current generation number
            qt: Current qt value
            q_threshold: Threshold for qt
            
        Returns:
            True if should stop, False otherwise
        """
        
        if generation_num >= self.max_generations:
            self.logger.info("Stopping criterion met (max generations reached)")
            return True
        
        if qt >= q_threshold and generation_num >= 3:
            self.logger.info(f"Stopping criterion met (qt = {qt:.3f} >= {q_threshold})")
            return True
        
        return False

    def _log_final_results(self, start_time: float, total_simulations: int) -> None:
        """Log final results of the ABC PMC run.
        
        Args:
            start_time: Start time of the run
            total_simulations: Total number of simulations performed
        """
        duration = time.time() - start_time
        num_generations = len(self.generations)
        self.logger.info(
            f"ABC SMC run completed in {duration:.2f} seconds with "
            f"{total_simulations} total simulations over {num_generations} generations."
        )

    def _compute_statistics(self, generation: int = 0):
        """ Compute statistics of the particles at the given generation.
        Args:
            generation (int): Generation to compute statistics for. Defaults to the last generation.
            return_as_dataframe (bool): Whether to return the statistics as a pandas DataFrame. Defaults to False.
        """

        if generation >= len(self.generations):
            raise ValueError("Generation must be less than or equal to the number of generations.")
        
        # compute statistics of the given generation
        particles = self.generations[generation]['particles']
        weights = self.generations[generation]['weights']
        mean = np.average(particles, weights=weights, axis=0)
        var = np.average((particles - mean) ** 2, weights=weights, axis=0)
        median = np.median(particles, axis=0)

        self.mean = mean    
        self.var = var
        self.median = median

        self.logger.info(f"Mean: {mean}")
        self.logger.info(f"Median: {median}")
        self.logger.info(f"Variance: {var}")

        # TODO: calculate 95% HDI
        # hdi = hdi_of_grid(particles, weights, 0.95)
        # self.logger.info(f"95% HDI: {hdi}")

        return mean, median, var
    
    def __sample_priors(self, n: int = 1):
        """Sample from prior distribution using Latin Hypercube Sampling"""
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=self.num_parameters)
        
        # Generate samples in [0,1] range
        samples = sampler.random(n=n)
        
        # Scale samples to parameter bounds
        scaled_samples = qmc.scale(samples, self.lower_bounds, self.upper_bounds)
        return scaled_samples

    def __batch_simulations(self, num_simulations: int, save_dir: str = "data", prefix: str = "train", num_threads: int = 2):
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

            self.logger.info(f"Saving {len(valid_simulations)} simulations to {save_dir}.")
            np.savez(f"{os.path.join(os.getcwd(), save_dir)}/{prefix}_data.npz", params=valid_params, simulations=valid_simulations)
        else:
            self.logger.warning("No valid simulations to save.")

    def generate_training_data(self, num_simulations: list = [50000, 10000, 10000], save_dir: str = "data", seed: int = 1234, num_workers: int = 1):
        np.random.seed(seed)
        self.logger.info(f"Generating training data for training with seed {seed}")

        prefix = ["train", "val", "test"]
        total_time = 0  # Initialize the total time counter
        
        for i, num_simulation in enumerate(num_simulations):
            self.logger.info(f"Generating {num_simulation} simulations for training data")
            start = time.time()
            self.__batch_simulations(num_simulation, save_dir, prefix=prefix[i], num_threads=num_workers * 2)
            end = time.time()
            elapsed = end - start
            total_time += elapsed
            self.logger.info(f"Generated {num_simulation} simulations for training data in {elapsed:.2f} seconds")

        self.logger.info(f"Training data generation completed and saved. Total time taken: {total_time:.2f} seconds")

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
            latent = self.get_latent(y)
            
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
            x = torch.tensor(x, dtype=torch.float32).to(self.model.device)

        x = self.model.get_latent(x, self.pooling_method)

        # if x is tensor convert to numpy, safeguard
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        return x
    
    def preprocess(self, x):
        raise NotImplementedError
        
    
    