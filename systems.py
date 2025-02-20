from latent_abc_smc import LatentABCSMC
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde, norm, uniform
import numpy as np
import torch
import logging
from typing import Union
import pandas as pd
import time
import math
import os
import warnings
from tempfile import TemporaryFile

from scipy.stats import qmc
from concurrent.futures import ThreadPoolExecutor, as_completed

class LotkaVolterra(LatentABCSMC):
    def __init__(self,
        #num_particles = 1000, 
        #num_generations = 5, 
        num_parameters = 2, 
        lower_bounds = np.array([0, 0]), 
        upper_bounds = np.array([10, 10]), 
        perturbation_kernels = np.array([0.1, 0.1]), 
        # tolerance_levels = np.array([0.2, 0.1, 0.05, 0.01, 0.005]), 
        model = None, 
        observational_data = np.array([[1.87, 0.65, 0.22, 0.31, 1.64, 1.15, 0.24, 2.91],
                                        [0.49, 2.62, 1.54, 0.02, 1.14, 1.68, 1.07, 0.88]]).T, 
        state0 = np.array([1, 0.5]),
        t0 = 0,
        tmax = 15, 
        time_space = np.array([1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4])):
        #time_space = np.array([0.5, 1.1, 2.4, 3.1, 3.9, 5.1, 5.6, 7.1, 7.5, 9.0, 9.6, 11.0, 11.9, 13.0, 14.4, 14.7])),
        super().__init__(num_parameters, lower_bounds, upper_bounds, perturbation_kernels, observational_data, model, state0, t0, tmax, time_space)

    def ode_system(self, t, state, parameters):
        # Lotka-Volterra equations
        alpha, delta = parameters
        beta, gamma = 1, 1
        prey, predator = state
        dprey = prey * (alpha - beta * predator)
        dpredator = predator * (-gamma + delta * prey)
        return [dprey, dpredator]
    
    def calculate_distance(self, y: np.ndarray, norm: int = 2) -> float:
        """
        Calculate distance between encoded observational data and input vector y.
        
        Args:
            y: Input vector to compare against
            norm: Type of norm to use (1=Manhattan, 2=Euclidean, inf=Chebyshev)
        
        Returns:
            float: Distance measure between 0 and 2
        """
        x = self.encoded_observational_data.flatten()
        y = y.flatten()
        
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
            
        cosine_similarity = np.dot(x, y) / (norm_x * norm_y)
        # Clip to handle numerical errors
        return 1-cosine_similarity

    def sample_priors(self):
        # Sample from the prior distribution
        priors = np.random.uniform(self.lower_bounds, self.upper_bounds, self.num_parameters)
        return priors
    
    def calculate_prior_prob(self, parameters):
        probabilities = uniform.cdf(parameters, loc=self.lower_bounds, scale=self.upper_bounds)
        return np.prod(probabilities)
    
    def perturb_parameters(self, parameters, previous_particles):
        # Perturb the parameters
        perturbations = np.random.uniform(-self.perturbation_kernels, self.perturbation_kernels)
        parameters += perturbations
        return parameters

class MZB(LatentABCSMC):
    def __init__(self,
        #num_particles = 1000, 
        #num_generations = 5, 
        num_parameters = 6, 
        lower_bounds = np.array([0.5, 14, 0.1, -5, 3.5, -4]), # mean
        upper_bounds = np.array([0.25, 2, 0.15, 1.2, 0.8, 1.2]),  # std
        perturbation_kernels = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 
        tolerance_levels = np.array([30, 20, 10, 5, 1]), 
        model = None, 
        observational_data = None,
        state0 = None,
        t0 = 40,
        tmax = 732,
        time_space = np.array([59, 69, 76, 88, 95, 102, 108, 109, 113, 119, 122, 123, 124, 141, 156, 158, 183, 212, 217, 219, 235, 261, 270, 289, 291, 306, 442, 524, 563, 566, 731])):
        super().__init__(num_parameters, lower_bounds, upper_bounds, perturbation_kernels, observational_data, tolerance_levels, model, state0, t0, tmax, time_space)

    def sample_priors(self):
        while True:
            # Sample from the prior distribution
            priors = np.random.normal(self.lower_bounds, self.upper_bounds, self.num_parameters)
            
            # Extract individual parameters
            phi, y0_Log, kappa_0, rho_Log, beta, delta_Log = priors
            
            # Apply constraints
            if (0 < phi < 1 and 
                0 < kappa_0 < 1 and 
                beta > 0):
                # If all constraints are satisfied, return the priors
                return priors
            # If any constraint is violated, the loop will continue and resample
    
    def perturb_parameters(self, parameters, previous_particles):
        loc = parameters
        scale = np.sqrt(2 * np.var(previous_particles, axis=0))
        assert loc.shape == scale.shape, "loc and scale must have the same shape"

        while True:
            perturbed_parameters = np.random.normal(loc, scale)
            # Apply constraints
            phi, y0_Log, kappa_0, rho_Log, beta, delta_Log = perturbed_parameters
            if (0 < phi < 1 and
                0 < kappa_0 < 1 and 
                beta > 0):
                # If all constraints are satisfied, return the perturbed parameters
                return perturbed_parameters
    
    # Define the ODE system
    def ode_system(self, t, state, parameters):
        # parameters
        phi, rho, delta, beta = parameters
        
        # fixed parameters
        P = 1118208  # mean of the precursor counts 
        chi_source = 0.78
        eps_donor = 0.99
        eps_host = 0.95
        
        # the system of ODEs
        dXdt = np.zeros(4)
        # donor Ki67+ cells
        dXdt[0] = phi * eps_donor * chi_source * P + rho * (state[0] + 2 * state[1]) - ((1 / beta) + delta) * state[0]
        # donor Ki67- cells
        dXdt[1] = phi * (1 - eps_donor) * chi_source * P + (1 / beta) * state[0] - (rho + delta) * state[1]
        # host Ki67+ cells
        dXdt[2] = phi * eps_host * (1 - chi_source) * P + rho * (state[2] + 2 * state[3]) - ((1 / beta) + delta) * state[2]
        # host Ki67- cells
        dXdt[3] = phi * (1 - eps_host) * (1 - chi_source) * P + (1 / beta) * state[2] - (rho + delta) * state[3]
        
        return dXdt

    def simulate(self, parameters: np.ndarray):
        phi, y0_Log, kappa_0, rho_Log, beta, delta_Log = parameters 

        y0 = np.exp(y0_Log)
        rho = np.exp(rho_Log)
        delta = np.exp(delta_Log)
        kappa_0_value = kappa_0  # Assuming kappa_0 is also a frozen distribution

        y0 = [0.0, 0.0, y0 * kappa_0_value, y0 * (1 - kappa_0_value)]
        parameters = [phi, rho, delta, beta]

        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            try:
                solution = solve_ivp(self.ode_system, [self.t0, self.tmax], y0=y0, t_eval=self.time_space, args=(parameters,), method="BDF")
                status = solution.status

                if status != 0:
                    return np.zeros((self.time_space.shape[0], 4)), status
                
                k_hat = solution.y.T
                # Check if all values in k_hat are positive
                if np.any(k_hat < 0):
                    raise RuntimeWarning("All values in k_hat must be positive.")
                
                if np.any(k_hat > 1e8):
                    raise RuntimeWarning("All values in k_hat must be less than 1e8.")

                numsolve = len(k_hat)
                MZ_counts_mean = np.zeros(numsolve)
                donor_fractions_mean = np.zeros(numsolve)
                donor_ki_mean = np.zeros(numsolve)
                host_ki_mean = np.zeros(numsolve)
                Nfd_mean = np.zeros(numsolve)

                # Vectorized computation for efficiency
                MZ_counts_mean = k_hat[:, 0] + k_hat[:, 1] + k_hat[:, 2] + k_hat[:, 3]
                donor_fractions_mean = (k_hat[:, 0] + k_hat[:, 1]) / MZ_counts_mean

                # Calculate donor_ki_mean
                donor_ki_mean = np.where(
                    k_hat[:, 0] <= 1e-8,  # Condition: if k_hat[:, 0] <= 1e-8
                    0,  # Value if condition is True
                    np.divide(  # Value if condition is False
                        k_hat[:, 0],
                        (k_hat[:, 0] + k_hat[:, 1]),
                        out=np.zeros_like(k_hat[:, 0]),
                        where=((k_hat[:, 0] + k_hat[:, 1]) > 0)
                    )
                )

                # Calculate host_ki_mean
                host_ki_mean = np.where(
                    k_hat[:, 2] <= 1e-8,  # Condition: if k_hat[:, 2] <= 1e-8
                    0,  # Value if condition is True
                    np.divide(  # Value if condition is False
                        k_hat[:, 2],
                        (k_hat[:, 2] + k_hat[:, 3]),
                        out=np.zeros_like(k_hat[:, 2]),
                        where=((k_hat[:, 2] + k_hat[:, 3]) > 0)  # Fixed: Use k_hat[:, 2] + k_hat[:, 3] instead of k_hat[:, 0] + k_hat[:, 1]
                    )
                )
                Nfd_mean = donor_fractions_mean / 0.78

                data = np.array([MZ_counts_mean, donor_ki_mean, host_ki_mean, Nfd_mean]).T
                # scale = np.mean(np.abs(data), axis=0)
                return data, status
            except RuntimeWarning:
                return np.zeros((self.time_space.shape[0], 4)), -1
    
    def __sample_priors(self, n: int = 1):
        valid_samples = []

        sampler = qmc.LatinHypercube(d=6, seed=0)
        while len(valid_samples) < n:
            samples = sampler.random(n=n)
            lower_bound = [0, 8, 0, -9, 1, -6]
            upper_bound = [1, 20, 0.6, -1, 6.5, 0.5]
            scaled_samples = qmc.scale(samples, lower_bound, upper_bound)
            # priors = np.random.normal(self.lower_bounds, self.upper_bounds, (n, self.num_parameters))
            priors = scaled_samples
            for sample in priors:
                phi, y0_Log, kappa_0, rho_Log, beta, delta_Log = sample
                if (0 < phi < 1 and 
                    0 < kappa_0 < 1 and 
                    beta > 0):
                    valid_samples.append(sample)
            if len(valid_samples) >= n:
                break
        return np.array(valid_samples[:n])

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

    def calculate_distance(self, y: np.ndarray, norm: int = 2) -> float:
        """
        Calculate distance between encoded observational data and input vector y.
        
        Args:
            y: Input vector to compare against
            norm: Type of norm to use (1=Manhattan, 2=Euclidean, inf=Chebyshev)
        
        Returns:
            float: Distance measure between 0 and 2
        """
        x = self.encoded_observational_data.flatten()
        y = y.flatten()
        
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
            
        cosine_similarity = np.dot(x, y) / (norm_x * norm_y)
        # Clip to handle numerical errors
        return 1-cosine_similarity
    
    def calculate_prior_prob(self, parameters):
        # Calculate the prior probability using log-sum-exp trick for numerical stability
        # Calculate log probabilities
        log_probs = norm.logpdf(parameters, loc=self.lower_bounds, scale=self.upper_bounds)
        
        # Sum the log probabilities (equivalent to multiplying in normal space)  
        log_sum = np.sum(log_probs)
        
        # Convert back to probability space
        return np.exp(log_sum)