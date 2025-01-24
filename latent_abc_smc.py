"""
latent_abc_smc.py

This script implements the Sequential Monte Carlo (SMC) algorithm for Approximate Bayesian Computation (ABC).
Some code is referenced from https://github.com/Pat-Laub/approxbayescomp/blob/master/src/approxbayescomp/smc.py#L73
"""

from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
import numpy as np
import torch
import logging
from typing import Union
import pandas as pd
import time
import os

class LatentABCSMC:
    @torch.inference_mode()
    def __init__(self,
                num_particles: int, 
                num_generations: int, 
                num_parameters: int, 
                lower_bounds: np.ndarray, 
                upper_bounds: np.ndarray, 
                perturbation_kernels: np.ndarray, 
                observational_data: np.ndarray, 
                tolerance_levels: np.ndarray, 
                model: Union[torch.nn.Module, None],
                state0: Union[np.ndarray, None],
                t0: Union[int, None], 
                tmax: Union[int, None], 
                time_space: np.ndarray,
                batch_effect: bool = False):
        """
        Initialize the Latent-ABC algorithm.

        Parameters:
        num_particles (int): Number of particles to use in the algorithm.
        num_generations (int): Number of generations to run the algorithm.
        tolerance_levels (np.ndarray): Array of tolerance levels for each generation.
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
            self.t0 = time_space[0]
        else:
            self.t0 = t0

        if tmax is None:
            self.tmax = time_space[-1] + 1
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

        self.num_particles = num_particles
        self.num_generations = num_generations

        assert num_particles > 0, "Number of particles must be greater than 0"
        assert num_generations > 0, "Number of generations must be greater than 0"
        
        # Validate tolerance
        self.tolerance_levels = tolerance_levels
        assert len(tolerance_levels) == num_generations, "Tolerance levels must match the number of generations"
        
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
        self.logger.info(f"num_particles: {num_particles}")
        self.logger.info(f"num_generations: {num_generations}")
        self.logger.info(f"num_parameters: {num_parameters}")
        self.logger.info(f"lower_bounds: {lower_bounds}")
        self.logger.info(f"upper_bounds: {upper_bounds}")
        self.logger.info(f"tolerance_levels: {tolerance_levels}")
        self.logger.info(f"perturbation_kernels: {perturbation_kernels}")

    def update_model(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.logger.info("Model updated")

    def ode_system(self, t: int, state: np.ndarray, parameters: np.ndarray):
        raise NotImplementedError

    def simulate(self, parameters: np.ndarray):
        # TODO: implement batch effect
        # if self.batch_effect:
        #     time_pairs = [(self.time_space[i], self.time_space[i + 1]) for i in range(len(self.time_space) - 1)]
        #     solution = np.ones((self.time_space.shape[0], self.state0.shape[0]))
        #     for (t0, tmax) in time_pairs:
        #         y0 = self.raw_observational_data[t0,:]
        #         interval_solution = solve_ivp(self.ode_system, [t0, tmax], y0=y0, t_eval=[t0, tmax], args=(parameters,))
        #         solution[:,t0,:] = interval_solution.y.T
        # else:
        if self.state0 is not None:
            solution = solve_ivp(self.ode_system, [self.t0, self.tmax], y0=self.state0, t_eval=self.time_space, args=(parameters,))
        return solution.y.T

    def sample_priors(self):
        """
        Sample from the prior distributions.

        This method should be implemented to return a sample from the prior distributions
        of the parameters. The specific implementation will depend on the prior distribution
        used in the model.

        Returns:
            np.ndarray: Sampled parameters from the prior distributions.
        """
        raise NotImplementedError

    def calculate_prior_prob(self, parameters: np.ndarray):
        """
        Calculate the prior probability of the given parameters.

        Args:
            parameters (np.ndarray): Parameters to calculate the prior probability for.

        Returns:
            np.ndarray: Prior probability of the parameters.
        """
        raise NotImplementedError

    def perturb_parameters(self, parameters: np.ndarray):
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

    def calculate_distance(self, y: np.ndarray):
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
        self.encoded_observational_data = self.model(torch.tensor(self.raw_observational_data, dtype=torch.float32).unsqueeze(0))[1].cpu().numpy()

    @torch.inference_mode()
    def run(self):
        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the algorithm.")
        
        self.logger.info("Starting ABC SMC run")
        start_time = time.time()
        self.encode_observational_data()
        # numpy array of size num_particles x num_parameters
        particles = np.ones((self.num_generations, self.num_particles, self.num_parameters))
        weights = np.ones((self.num_generations, self.num_particles, self.num_parameters))

        for t in range(self.num_generations):
            self.logger.info(f"Generation {t + 1} started")
            start_time_generation = time.time()
            generation_particles = []
            generation_weights = []
            accepted = 0
            epsilon = self.tolerance_levels[t]
        
            while accepted < self.num_particles:
                if t == 0:
                    # Step 4: Sample from prior in the first generation
                    perturbed_params = self.sample_priors()
                    new_weight = 1
                    prior_probability = 1
                else:
                    # Step 6: Sample from the previous generation's particles with weights
                    previous_particles = np.array(particles[t-1,:,:])
                    previous_weights = np.array(weights[t-1,:,:])

                    # params = np.ones((self.num_parameters,)) * 0
                    # for j in range(self.num_parameters):
                    #     idx = np.random.choice(len(previous_particles[:,j]), p=previous_weights[:,j])
                    #     params[j] = previous_particles[idx][j]

                    idx = np.random.choice(len(previous_particles), p=previous_weights)

                    # Step 7: Perturb parameters and calculate the prior probability
                    perturbed_params = self.perturb_parameters(previous_particles[idx])

                    # Step 8: If prior probability of the perturbed parameters is 0, resample
                    prior_probability = self.calculate_prior_prob(perturbed_params)
                    if prior_probability <= 0:
                        continue # Go back to sampling if prior probability is 0

                    thetaLogWeight = np.log(prior_probability) - gaussian_kde(previous_particles, weights=previous_weights).log_pdf(perturbed_params)
                    new_weight = np.exp(thetaLogWeight)

                assert new_weight >= 0, "Weight must be non-negative" # TODO: unnecessary since exp(x) >= 0 for all x?

                # Step 9: Simulate the ODE system and encode the data
                y = self.simulate(perturbed_params)
                y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
                y_latent_np = self.model(y_tensor)[1].cpu().numpy()

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
            weights[t] = np.array(generation_weights)
            end_time_generation = time.time()
            duration_generation = end_time_generation - start_time_generation
            duration = end_time_generation - start_time_generation
            self.logger.info(f"Generation {t + 1} Completed. Accepted {self.num_particles} particles in {duration_generation:.2f} seconds")

        self.particles = particles
        self.weights = weights

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f"ABC SMC run completed in {duration:.2f} seconds")

        return particles, weights

    def compute_statistics(self, return_as_dataframe: bool = False):
        if self.particles is None or self.weights is None:
            raise ValueError("Particles and weights must be computed before calculating statistics.")
        
        # compute statistics of the last generation
        particles = self.particles[-1]
        weights = self.weights[-1]
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
    
    def visualize_generation(self, generation: int = -1):
        # TODO: visualize the particles (parameter space) for the given generation
        pass

    def visualize_latent_space(self):
        # TODO: implement umap to visualize latent space of the latent space of the training data and the observed data
        pass

    def visualize_latent_space_generation(self, generation: int = -1):
        # TODO: impelemt umap to take in the particles at the given generation, simulate the ODE system, and visualize the latent space
        # call visualize_latent_space method to reduce redundancy
        pass

    def __batch_simulations(self, num_simulations: int, prefix: str = "train"):
        """ This method should never be called directly; it is only used by the generate_training_data method. """
        simulations = np.ones((num_simulations, self.raw_observational_data.shape[0], self.raw_observational_data.shape[1]))
        params = np.ones((num_simulations, self.num_parameters))

        for i in range(num_simulations):
            parameter = self.sample_priors()
            simulation = self.simulate(parameter)
            simulations[i] = simulation
            params[i] = parameter

        if not os.path.exists("data"):
            os.makedirs("data")

        np.save(f"data/{prefix}_params.npy", params)
        np.save(f"data/{prefix}_simulations.npy", simulations)

    def generate_training_data(self, num_simulations: list = [50000, 10000, 10000], seed: int = 1234):
        np.random.seed(seed)
        self.logger.info(f"Generating training data for training with seed {seed}")
        self.__batch_simulations(num_simulations=num_simulations[0], prefix="train")
        self.__batch_simulations(num_simulations=num_simulations[1], prefix="val")
        self.__batch_simulations(num_simulations=num_simulations[2], prefix="test")
        self.logger.info("Training data generation completed and saved.")

class LotkaVolterra(LatentABCSMC):
    def __init__(self,
        num_particles = 1000, 
        num_generations = 5, 
        num_parameters = 2, 
        lower_bounds = np.array([1e-4, 1e-4]), 
        upper_bounds = np.array([10, 10]), 
        perturbation_kernels = np.array([0.1, 0.1]), 
        tolerance_levels = np.array([30, 20, 10, 5, 1]), 
        model = None, 
        observational_data = np.array([[1.87, 0.65, 0.22, 0.31, 1.64, 1.15, 0.24, 2.91],
                                        [0.49, 2.62, 1.54, 0.02, 1.14, 1.68, 1.07, 0.88]]).T, 
        state0 = np.array([1, 0.5]),
        t0 = None,
        tmax = 15, time_space = np.array([1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4])):
        super().__init__(num_particles, num_generations, num_parameters, lower_bounds, upper_bounds, perturbation_kernels, observational_data, tolerance_levels, model, state0, t0, tmax, time_space)

    def ode_system(self, t, state, parameters):
        # Lotka-Volterra equations
        alpha, delta = parameters
        beta, gamma = 1, 1
        prey, predator = state
        dprey = prey * (alpha - beta * predator)
        dpredator = predator * (-gamma + delta * prey)
        return [dprey, dpredator]
    
    def calculate_distance(self, y: np.ndarray, norm: int = 2):
        return np.linalg.norm(y - self.encoded_observational_data, ord=norm)

    def sample_priors(self):
        # Sample from the prior distribution
        priors = np.random.uniform(self.lower_bounds, self.upper_bounds, self.num_parameters)
        return priors
    
    def prior_prob(self, parameters):
        # Calculate the prior probability
        mask = (parameters > self.lower_bounds) & (parameters < self.upper_bounds)
        assert mask.all(), "Parameters must be within bounds"
        probabilities = (parameters - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
        return np.prod(probabilities)
    
    def perturb_parameters(self, parameters):
        # Perturb the parameters
        perturbations = np.random.uniform(-self.perturbation_kernels, self.perturbation_kernels)
        parameters += perturbations
        return parameters

class MZB(LatentABCSMC):
    def __init__(self,
        num_particles = 1000, 
        num_generations = 5, 
        num_parameters = 6, 
        lower_bounds = np.array([0.5, 14, 0.1, -5, 3.5, -4]), 
        upper_bounds = np.array([0.25, 2, 0.15, 1.2, 0.8, 1.2]), 
        perturbation_kernels = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 
        tolerance_levels = np.array([30, 20, 10, 5, 1]), 
        model = None, 
        observational_data = None,
        state0 = None,
        t0 = None,
        tmax = None,
        time_space = np.array([59, 69, 76, 88, 95, 102, 108, 109, 113, 119, 122, 123, 124, 141, 156, 158, 183, 212, 217, 219, 235, 261, 270, 289, 291, 306, 442, 524, 563, 566, 731])):
        super().__init__(num_particles, num_generations, num_parameters, lower_bounds, upper_bounds, perturbation_kernels, observational_data, tolerance_levels, model, state0, t0, tmax, time_space)

    def sample_priors(self):
        # Sample from the prior distribution
        priors = np.random.normal(self.lower_bounds, self.upper_bounds, self.num_parameters)
        return priors
    
    def perturb_parameters(self, parameters):
        # Perturb the parameters
        perturbations = np.random.uniform(-self.perturbation_kernels, self.perturbation_kernels)
        parameters += perturbations
        return parameters
    
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

        solution = solve_ivp(self.ode_system, [self.t0, self.tmax], y0=y0, t_eval=self.time_space, args=(parameters,))
        
        k_hat = solution.y.T
        numsolve = len(k_hat)
        MZ_counts_mean = np.zeros(numsolve)
        donor_fractions_mean = np.zeros(numsolve)
        donor_ki_mean = np.zeros(numsolve)
        host_ki_mean = np.zeros(numsolve)
        Nfd_mean = np.zeros(numsolve)

        for i in range(numsolve):
            # total counts
            MZ_counts_mean[i] = k_hat[i, 0] + k_hat[i, 1] + k_hat[i, 2] + k_hat[i, 3]

            # donor fractions normalised with chimerism in the source
            donor_fractions_mean[i] = (k_hat[i, 0] + k_hat[i, 1]) / MZ_counts_mean[i]

            # fractions of ki67 positive cells in the donor compartment
            donor_ki_mean[i] = 0 if k_hat[i, 0] == 0 else k_hat[i, 0] / (k_hat[i, 0] + k_hat[i, 1])

            # fractions of ki67 positive cells in the host compartment
            host_ki_mean[i] = 0 if k_hat[i, 2] == 0 else k_hat[i, 2] / (k_hat[i, 2] + k_hat[i, 3])

            Nfd_mean[i] = donor_fractions_mean[i] / 0.78

        return np.array([MZ_counts_mean, donor_ki_mean, host_ki_mean, Nfd_mean]).T
