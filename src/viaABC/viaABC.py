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
import tempfile
import time
import numpy as np
import torch
from typing import Any, Callable, Optional, Union

from scipy.integrate import solve_ivp
from scipy.stats import qmc, multivariate_normal
from scipy.special import logsumexp

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from tqdm import tqdm

from src.viaABC.metrics import *
from src.utils.viaABC_utils.DensityRatioEstimation import dre_cpp
from src.utils.viaABC_utils.utils import weighted_var, calculate_densratio_basis_sigma, weighted_sample_quantile
from src.utils.viaABC_utils.ParticleWeight import particle_weight_cpp

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
        time_space: Optional[np.ndarray],
        pooling_method: str = "cls",
        metric: str = "cosine",
        transform: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("Initializing viaABC class")

        self.raw_observational_data = observational_data
        self.state0 = state0
        self.model: Optional[torch.nn.Module] = None
        self.time_space: Optional[np.ndarray] = None
        self.train_dataset: Optional[torch.utils.data.Dataset[Any]] = None
        self.num_particles: int = 0
        self.max_generations: int = 0
        self.encoded_observational_data: np.ndarray = np.array([])

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

        # Validate time parameters
        assert self.t0 < self.tmax, "t0 must be less than tmax"

        if time_space is not None:
            self.time_space = time_space
       
            assert (
                self.tmax >= self.time_space[-1]
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
        
        if model is None:
            self.logger.warning(
                "Model is None. The model must be provided to encode the data and run the algorithm.\n" \
                "The class can be initialized without a model, but it will not\nbe able to run the algorithm."
            )
        else:
            self.update_model(model)

        self.generations = []

        self.logger.info("viaABC class initialized with the following parameters:")
        self.logger.info(f"num_parameters: {num_parameters}")
        self.logger.info(f"Mu: {mu}")
        self.logger.info(f"Sigma: {sigma}")
        self.logger.info(f"t0: {t0}")
        self.logger.info(f"tmax: {tmax}")
        self.logger.info(f"time_space: {time_space}")
        self.logger.info(f"pooling_method: {pooling_method}")
        self.logger.info(f"metric: {metric}")

    def update_model(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model.eval()
        self.logger.info("Model updated")
        self._encode_observational_data()

    def ode_system(self, t: int, state: np.ndarray, parameters: np.ndarray) -> Any:
        raise NotImplementedError

    def simulate(self, parameters: np.ndarray, time_space: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
        if self.time_space is None and time_space is None:
            raise ValueError("time_space must be provided either during initialization or simulation.")
        if self.state0 is not None:
            if time_space is not None:
                solution = solve_ivp(self.ode_system, [self.t0, self.tmax], y0=self.state0, t_eval=time_space, args=(parameters,))
            else:
                solution = solve_ivp(self.ode_system, [self.t0, self.tmax], y0=self.state0, t_eval=self.time_space, args=(parameters,))
            return solution.y.T, solution.status
        else:
            raise ValueError("Initial state (state0) must be set before simulation.")

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

    def calculate_prior_log_prob(self, parameters: np.ndarray) -> float:
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

    def calculate_distance(self, y: np.ndarray) -> float | np.ndarray:
        """
        Calculate distances between encoded observational data and input vectors in y in a vectorized manner.
        
        Args:
            y: Batched input vectors to compare against (shape: [batch_size, vector_size])
            
        Returns:
            np.ndarray: Distance measures between 0 and 2 for each item in the batch
        """
        x = np.asarray(self.encoded_observational_data)
        y = np.asarray(y)

        single_input = y.shape[0] == 1 if y.ndim > 0 else True
        x_batch = np.broadcast_to(x, y.shape) if y.shape != x.shape else x
        
        if self.metric == "cosine":
            x_flat = x_batch.reshape(x_batch.shape[0], -1)
            y_flat = y.reshape(y.shape[0], -1)
            denom = np.linalg.norm(x_flat, axis=1) * np.linalg.norm(y_flat, axis=1)
            distances = 1 - (np.sum(x_flat * y_flat, axis=1) / (denom + 1e-8))
        elif self.metric == "l1":
            distances = np.abs(y - x_batch).reshape(y.shape[0], -1).mean(axis=1)
        elif self.metric == "l2":
            distances = np.linalg.norm((y - x_batch).reshape(y.shape[0], -1), axis=1)
        elif self.metric == "bertscore":
            distances = np.array([1 - bert_score(x, sample[np.newaxis, ...])[2] for sample in y], dtype=np.float32)
        elif self.metric == "pairwise_cosine":
            distances = np.array([1 - pairwise_cosine(x, sample[np.newaxis, ...]) for sample in y], dtype=np.float32)
        elif self.metric == "bertscore_batch":
            distances = np.array([1 - bert_score_batch(x, sample[np.newaxis, ...]) for sample in y], dtype=np.float32)
        elif self.metric == "maxSim":
            distances = np.array([maxSim(x, sample[np.newaxis, ...]) for sample in y], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        if single_input:
            return float(np.asarray(distances).reshape(-1)[0])
        return np.asarray(distances, dtype=np.float32)
    
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
        
        self.logger.info(f"Mean: {mean:.4f}")
        self.logger.info(f"Median: {median:.4f}")
        self.logger.info(f"Variance: {var:.4f}")

    def update_train_dataset(self, train_dataset: torch.utils.data.Dataset[Any]) -> None:
        # check if train_dataset is a torch Dataset
        if not isinstance(train_dataset, torch.utils.data.Dataset):
            raise ValueError("train_dataset must be a torch Dataset.")
        
        self.train_dataset = train_dataset
        self.logger.info("Training dataset updated.")

    @torch.inference_mode()
    def _run_init(self, num_particles: int, k: int = 5):
        self.densratio = dre_cpp.DensityRatioEstimation(n=100, epsilon=0.001, max_iter=1000, abs_tol=1e-4, fold=5, optimize=False)

        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the algorithm.")

        total_num_simulations = 0
        accepted = 0
        target_particles = k * num_particles

        self._encode_observational_data()
        
        # Pre-allocate arrays with exact size needed
        particles = np.empty((target_particles, self.num_parameters))
        weights = np.ones(target_particles, dtype=np.float32)  # Use float32 if precision allows
        dists = np.empty(target_particles, dtype=np.float32)
        
        # Get device once to avoid repeated attribute access
        device = self.model.device
        max_workers = max(1, min(os.cpu_count() or 1, target_particles))
        batch_size = max(max_workers, min(256, num_particles))

        while accepted < target_particles:
            remaining = target_particles - accepted
            current_batch_size = min(batch_size, remaining * 2)

            candidate_params = []
            while len(candidate_params) < current_batch_size:
                perturbed_params = self.sample_priors()
                prior_log_pdf = self.calculate_prior_log_prob(perturbed_params)
                if np.isfinite(prior_log_pdf):
                    candidate_params.append(np.asarray(perturbed_params))

            with ThreadPoolExecutor(max_workers=min(max_workers, len(candidate_params))) as executor:
                futures = [executor.submit(self.simulate, params) for params in candidate_params]
                simulation_results = [future.result() for future in futures]

            total_num_simulations += len(candidate_params)

            successful_params = []
            successful_scaled = []
            for params, (y, status) in zip(candidate_params, simulation_results):
                if status != 0:
                    continue
                successful_params.append(params)
                successful_scaled.append(self.preprocess(y))

            if not successful_scaled:
                continue

            y_batch = torch.as_tensor(
                np.stack(successful_scaled, axis=0),
                dtype=torch.float32,
                device=device,
            )
            y_latent_np = self.get_latent(y_batch)
            batch_dist_values = self.calculate_distance(y_latent_np)
            batch_dists = np.atleast_1d(np.asarray(batch_dist_values, dtype=np.float32))

            write_count = min(remaining, len(successful_params))
            particles[accepted:accepted + write_count] = np.asarray(successful_params[:write_count])
            dists[accepted:accepted + write_count] = batch_dists[:write_count]
            accepted += write_count

        sorted_indices = np.argsort(dists)
        particles = particles[sorted_indices]
        weights = weights[sorted_indices]
        dists = dists[sorted_indices]

        # Select the top particles
        particles = particles[:num_particles]
        weights = weights[:num_particles]

        # Weights are already normalized (all ones, sum = num_particles)
        weights = weights / np.sum(weights)
        epsilon = dists[num_particles-1]

        # Optimized covariance calculation
        sample_cov = np.atleast_2d(np.cov(particles.reshape(num_particles, -1), rowvar=False))
        sigma_max = np.min(np.sqrt(np.diag(sample_cov)))
        
        # More efficient diagonal operations
        cov = 2 * np.diag(sample_cov)

        # Store first generation
        self.generations.append({
            't': 0,
            'particles': particles.copy(),  # Only copy if needed elsewhere
            'weights': weights.copy(),
            'epsilon': epsilon,
            'cov': sample_cov,
            'sigma_max': sigma_max,
            'simulations': total_num_simulations,
            'meta': {
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
        """Run the viaABC algorithm.
        
        Args:
            num_particles: Number of particles to maintain per generation
            q_threshold: Threshold for stopping criterion (qt >= q_threshold)
            k: Number of nearest neighbors for distance calculation
            max_generations: Maximum number of generations to run
        """

        # TODO: This can be optimized by concurrently simulating particles in batches. 
        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the algorithm.")
        
        self.generations = []
        self.num_particles = num_particles
        self.max_generations = max_generations - 1
        total_num_simulations = 0
        
        # Cache logger and reduce string formatting overhead
        logger = self.logger
        logger.info(f"Starting viaABC run with Q Threshold: {q_threshold}")
        start_time = time.perf_counter()  # More precise timing

        # Initial generation
        self._initialize_first_generation(k=k)
        self._compute_statistics(0)
        
        # Run subsequent generations
        for generation_num in range(1, max_generations + 1):
            generation_start_time = time.perf_counter()
            logger.info(f"Generation {generation_num} started")
            prev_gen = self.generations[-1]
            
            # Direct unpacking from cached reference (avoid repeated dict lookups)
            prev_particles = prev_gen['particles']
            prev_weights = prev_gen['weights'] 
            prev_sigma_max = prev_gen['sigma_max']
            prev_epsilon = prev_gen['epsilon']
            prev_meta_cov = prev_gen['meta']['cov']
            
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

            generation_end_time = time.perf_counter()
            logger.info(f"Generation {generation_num} completed in {generation_end_time - generation_start_time:.2f} seconds")
            
            # Early termination check with cached values
            if self._should_stop(generation_num, current_gen['qt'], q_threshold):
                break

        end_time = time.perf_counter()
        duration = end_time - start_time
        self._log_final_results(duration, total_num_simulations)

    def _run_generation(
        self,
        prev_particles: np.ndarray,
        prev_weights: np.ndarray,
        prev_cov: np.ndarray,
        epsilon: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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

        with tqdm(total=self.num_particles, miniters=self.num_particles // 10, maxinterval=float('inf')) as pbar:
            while accepted < self.num_particles:

                theta = self._propose_particle_fast(
                    prev_particles=prev_particles,
                    prev_weights=prev_weights,
                    prev_cov=prev_cov
                )
                    
                y, status = self.simulate(theta)
                running_num_simulations += 1
                
                if status != 0:
                    continue
                    
                dist = self._calculate_particle_distance(y)
                if dist >= epsilon:
                    continue
                    
                # Only calculate weight for accepted particles
                new_weight = self._calculate_particle_weight(
                    theta, prev_particles, prev_weights, prev_cov
                )
                
                accepted += 1
                pbar.update(1)
                particles.append(theta)
                weights.append(new_weight)
                distances.append(dist)
        
        return np.array(particles), np.array(weights), np.array(distances), running_num_simulations

    def _propose_particle_fast(
        self,
        prev_particles: np.ndarray,
        prev_weights: np.ndarray,
        prev_cov: np.ndarray,
        batch_size: int = 4,
        max_attempts: int = 250
    ) -> np.ndarray:
        """
        Propose a new particle efficiently using batch proposals.
        
        Args:
            prev_particles: Particles from previous generation
            prev_weights: Weights from previous generation
            prev_cov: Covariance matrix from previous generation
            batch_size: Number of candidates to propose per attempt
            max_attempts: Maximum number of batch attempts
        
        Returns:
            Particle parameters
        
        Raises:
            RuntimeError: If unable to generate a valid particle after max_attempts
        """
        for _ in range(max_attempts):
            # Sample a batch of particle indices based on weights
            idxs = np.random.choice(len(prev_particles), size=batch_size, p=prev_weights)
            candidates = prev_particles[idxs]

            # Perturb all candidates
            perturbed = np.array([self.perturb_parameters(theta, prev_cov) for theta in candidates])

            # Evaluate prior log-probabilities
            priors = np.array([self.calculate_prior_log_prob(p) for p in perturbed])

            # Select first valid particle
            valid = perturbed[np.isfinite(priors)]
            if len(valid) > 0:
                return valid[0]

        # Failed to propose a valid particle
        raise RuntimeError(
            f"Unable to generate valid particle after {max_attempts * batch_size} proposals. "
            "Consider adjusting prior bounds or covariance matrix."
        )

    def _calculate_particle_weight(
        self,
        theta: np.ndarray,
        prev_particles: np.ndarray,
        prev_weights: np.ndarray,
        prev_cov: np.ndarray
    ) -> float:
        """Calculate weight for an accepted particle."""
        
        prev_particles = np.atleast_2d(prev_particles)
        prev_cov = np.atleast_2d(prev_cov)
        prev_cov += 1e-6 * np.eye(prev_cov.shape[0])

        new_weight = particle_weight_cpp.calculate_particle_weight(
            theta,
            prev_particles,
            prev_weights,
            prev_cov,
            self.calculate_prior_log_prob
        )

        return new_weight

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
    ) -> dict[str, Any]:
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
        epsilon = np.float32(epsilon)

        self.logger.info(f'Epsilon : {epsilon:.7f}')       # float32 ~7 decimals
        self.logger.info(f'Quantile : {qt:.7f}')
        self.logger.info(f'Simulations : {simulations}')

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
        y_tensor = torch.as_tensor(
            y_scaled,
            dtype=torch.float32,
            device=self.model.device,
        )
        y_latent_np = self.get_latent(y_tensor) # z_sim
        return float(self.calculate_distance(y_latent_np))

    def _should_stop(self, generation_num: int, qt: float, q_threshold: float) -> bool:
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
            self.logger.info(f"Stopping criterion met (qt = {qt:.7f} >= {q_threshold})")
            return True
        
        return False

    def _log_final_results(self, total_time: float, total_simulations: int) -> None:
        """Log final results of the viaABC run.

        Args:
            total_time: Total time taken for the run
            total_simulations: Total number of simulations performed
        """
        num_generations = len(self.generations)
        self.logger.info(
            f"viaABC run completed in {total_time:.2f} seconds with "
            f"{total_simulations} total simulations over {num_generations} generations."
        )

    def _compute_statistics(self, generation: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        fmt = lambda arr: np.array2string(np.asarray(arr), formatter={'float_kind': lambda x: f"{x:.7f}"})
        self.logger.info(f"Mean: {fmt(mean)}")
        self.logger.info(f"Median: {fmt(median)}")
        self.logger.info(f"Variance: {fmt(var)}")

        # TODO: calculate 95% HDI
        # hdi = hdi_of_grid(particles, weights, 0.95)
        # self.logger.info(f"95% HDI: {hdi}")

        return mean, median, var
    
    def __sample_priors(self, n: int = 1) -> np.ndarray:
        """Sample from prior distribution using Latin Hypercube Sampling"""
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=self.num_parameters)
        
        # Generate samples in [0,1] range
        samples = sampler.random(n=n)
        
        # Scale samples to parameter bounds
        scaled_samples = qmc.scale(samples, self.lower_bounds, self.upper_bounds)
        return scaled_samples

    def __batch_simulations(
        self, 
        num_simulations: int, 
        save_dir: str = "data", 
        prefix: str = "train", 
        num_threads: int = 2
    ) -> None:
        """
        Run batch simulations in parallel and save valid results.
        
        This method should never be called directly; it is only used by 
        the generate_training_data method.
        
        Args:
            num_simulations: Number of simulations to run
            save_dir: Directory to save results
            prefix: Prefix for output filename
            num_threads: Number of parallel threads
        """
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Sample parameters for all simulations
        parameters = self.__sample_priors(n=num_simulations)

        if num_threads <= 0:
            num_threads = min(32, (os.cpu_count() or 1) + 4, max(1, num_simulations))
        else:
            num_threads = min(num_threads, max(1, num_simulations))

        batch_size = min(256, max(32, num_threads * 8))
        
        def run_simulation(i: int, param: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
            """Run a single simulation and return results or None on failure."""
            try:
                simulation, status = self.simulate(parameters = param)
                if status == 0:
                    return simulation, param
                else:
                    self.logger.error(f"Simulation {i} failed with status: {status}")
            except Exception as e:
                self.logger.error(f"Simulation {i} failed with error: {e}")
            return None
        
        shard_paths: list[tuple[str, str]] = []
        buffered_simulations: list[np.ndarray] = []
        buffered_params: list[np.ndarray] = []
        total_valid = 0
        simulation_shape: Optional[tuple[int, ...]] = None
        simulation_dtype: Optional[np.dtype[Any]] = None
        param_shape: Optional[tuple[int, ...]] = None
        param_dtype: Optional[np.dtype[Any]] = None

        def flush_batch() -> None:
            nonlocal total_valid
            if not buffered_simulations:
                return

            batch_index = len(shard_paths)
            sim_batch = np.asarray(buffered_simulations)
            param_batch = np.asarray(buffered_params)
            sim_path = os.path.join(save_dir, f".{prefix}_simulations_{batch_index}.npy")
            param_path = os.path.join(save_dir, f".{prefix}_params_{batch_index}.npy")
            np.save(sim_path, sim_batch)
            np.save(param_path, param_batch)
            shard_paths.append((sim_path, param_path))
            total_valid += sim_batch.shape[0]
            buffered_simulations.clear()
            buffered_params.clear()

        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_index = {}
                next_index = 0
                max_pending = max(num_threads * 4, batch_size)

                while next_index < num_simulations and len(future_to_index) < max_pending:
                    future = executor.submit(run_simulation, next_index, parameters[next_index])
                    future_to_index[future] = next_index
                    next_index += 1

                while future_to_index:
                    done, _ = wait(future_to_index, return_when=FIRST_COMPLETED)
                    for future in done:
                        idx = future_to_index.pop(future)
                        try:
                            result = future.result()
                            if result is not None:
                                simulation, param = result
                                simulation = np.asarray(simulation)
                                param = np.asarray(param)

                                if simulation_shape is None:
                                    simulation_shape = simulation.shape
                                    simulation_dtype = simulation.dtype
                                    param_shape = param.shape
                                    param_dtype = param.dtype

                                buffered_simulations.append(simulation)
                                buffered_params.append(param)
                                if len(buffered_simulations) >= batch_size:
                                    flush_batch()
                        except Exception as e:
                            self.logger.error(f"Error processing simulation {idx}: {e}")

                        if next_index < num_simulations:
                            new_future = executor.submit(run_simulation, next_index, parameters[next_index])
                            future_to_index[new_future] = next_index
                            next_index += 1

            flush_batch()

            if total_valid == 0 or simulation_shape is None or simulation_dtype is None or param_shape is None or param_dtype is None:
                self.logger.warning("No valid simulations to save.")
                return

            with tempfile.TemporaryDirectory(dir=save_dir) as tmpdir:
                sim_merged_path = os.path.join(tmpdir, f"{prefix}_simulations.npy")
                param_merged_path = os.path.join(tmpdir, f"{prefix}_params.npy")

                merged_simulations = np.lib.format.open_memmap(
                    sim_merged_path,
                    mode="w+",
                    dtype=simulation_dtype,
                    shape=(total_valid, *simulation_shape),
                )
                merged_params = np.lib.format.open_memmap(
                    param_merged_path,
                    mode="w+",
                    dtype=param_dtype,
                    shape=(total_valid, *param_shape),
                )

                start = 0
                for sim_path, param_path in shard_paths:
                    sim_batch = np.load(sim_path, mmap_mode="r")
                    param_batch = np.load(param_path, mmap_mode="r")
                    end = start + sim_batch.shape[0]
                    merged_simulations[start:end] = sim_batch
                    merged_params[start:end] = param_batch
                    start = end

                output_path = os.path.join(save_dir, f"{prefix}_data.npz")
                np.savez(output_path, params=merged_params, simulations=merged_simulations)

            self.logger.info(
                f"Successfully saved {total_valid} simulations to {output_path} "
                f"using {num_threads} threads and batch size {batch_size}"
            )
        finally:
            for sim_path, param_path in shard_paths:
                if os.path.exists(sim_path):
                    os.remove(sim_path)
                if os.path.exists(param_path):
                    os.remove(param_path)

    def generate_training_data(self, num_simulations: Union[int, list[int], tuple[int, int, int]] = 50000, save_dir: str = "data", seed: int = 1234, num_workers: int = 1) -> None:
        np.random.seed(seed)
        self.logger.info(f"Generating training data for training with seed {seed}")

        total_time = 0
        if isinstance(num_simulations, int):
            simulation_plan = (("train", num_simulations),)
        elif isinstance(num_simulations, (list, tuple)) and len(num_simulations) == 3:
            simulation_plan = (
                ("train", int(num_simulations[0])),
                ("val", int(num_simulations[1])),
                ("test", int(num_simulations[2])),
            )
        else:
            raise ValueError(
                "num_simulations must be an int or a list/tuple of three ints: [train, val, test]."
            )

        for prefix, count in simulation_plan:
            if count < 0:
                raise ValueError(f"num_simulations for {prefix} must be non-negative, got {count}.")

            if count == 0:
                self.logger.info(f"Skipping {prefix} data generation because num_simulations is 0")
                continue

            # check if training_dataset is already set
            if os.path.exists(os.path.join(save_dir, f"{prefix}_data.npz")):
                try:
                    dataset = np.load(os.path.join(save_dir, f"{prefix}_data.npz"))
                    # Check if the shapes of the loaded data match the expected shapes based on current configuration
                    if count == dataset["params"].shape[0] and count == dataset["simulations"].shape[0]:
                        self.logger.info(f"{prefix}_data.npz already exists in {save_dir} and matches configuration. Skipping generation.")
                        return
                    else:
                        self.logger.warning(f"{prefix}_data.npz exists but shapes mismatch. Regenerating...")

                except Exception as e:
                    self.logger.error(f"Failed to load existing {os.path.join(save_dir, f'{prefix}_data.npz')}: {e}. Regenerating...")

            self.logger.info(f"Generating {count} simulations for {prefix} data")
            start = time.time()
            self.__batch_simulations(count, save_dir, prefix=prefix, num_threads=num_workers * 2)
            elapsed = time.time() - start
            total_time += elapsed
            self.logger.info(f"Generated {count} simulations for {prefix} data in {elapsed:.2f} seconds")

        self.logger.info(f"Training data generation completed and saved. Total time taken: {total_time:.2f} seconds")

    def update_train_dataset(self, train_dataset: torch.utils.data.Dataset[Any]) -> None:
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
    def get_latent(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the method.")
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.model.device)
        elif isinstance(x, torch.Tensor):
            x = x.float().to(self.model.device)
        else:
            raise TypeError(f"Unsupported type for x: {type(x)}")

        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = self.model.get_latent(x, self.pooling_method)

        # if x is tensor convert to numpy, safeguard
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        return x
    
    def preprocess(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    
    
