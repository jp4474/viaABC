from latent_abc_smc import LatentABCSMC
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
import numpy as np
import torch
import logging
from typing import Union
import pandas as pd
import time
import math
import os
from scipy.stats import qmc
from concurrent.futures import ThreadPoolExecutor

class LotkaVolterra(LatentABCSMC):
    def __init__(self,
        num_particles = 1000, 
        num_generations = 5, 
        num_parameters = 2, 
        lower_bounds = np.array([1e-4, 1e-4]), 
        upper_bounds = np.array([10, 10]), 
        perturbation_kernels = np.array([0.1, 0.1]), 
        tolerance_levels = np.array([0.2, 0.1, 0.05, 0.01, 0.005]), 
        model = None, 
        observational_data = np.array([[1.87, 0.65, 0.22, 0.31, 1.64, 1.15, 0.24, 2.91],
                                        [0.49, 2.62, 1.54, 0.02, 1.14, 1.68, 1.07, 0.88]]).T, 
        state0 = np.array([1, 0.5]),
        t0 = 0,
        tmax = 15, 
        time_space = np.array([1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4])):
        #time_space = np.array([0.5, 1.1, 2.4, 3.1, 3.9, 5.1, 5.6, 7.1, 7.5, 9.0, 9.6, 11.0, 11.9, 13.0, 14.4, 14.7])),
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
        # calcuclate cosine similarity
        y = y.mean(1)
        cosine_similarity = np.dot(self.encoded_observational_data.mean(1).flatten(), y.flatten()) / (np.linalg.norm(self.encoded_observational_data.mean(1).flatten()) * np.linalg.norm(y.flatten()))
        return 1 - cosine_similarity

    def sample_priors(self):
        # Sample from the prior distribution
        priors = np.random.uniform(self.lower_bounds, self.upper_bounds, self.num_parameters)
        return priors
    
    def calculate_prior_prob(self, parameters):
        # Calculate the prior probability
        # mask = (parameters > self.lower_bounds) & (parameters < self.upper_bounds)
        # assert mask.all(), "Parameters must be within bounds"
        probabilities = (parameters - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
        return np.prod(probabilities)
    
    def perturb_parameters(self, parameters):
        # Perturb the parameters
        perturbations = np.random.uniform(-self.perturbation_kernels, self.perturbation_kernels)
        parameters += perturbations
        return parameters

class BCell(LatentABCSMC):
    def __init__(self,
        num_particles = 1000, 
        num_generations = 5, 
        num_parameters = 7, 
        lower_bounds = np.array([0.01, 0.01, 0.01, 0.8, 0.1, 0.8, 8]), 
        upper_bounds = np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 1.5]), 
        perturbation_kernels = np.array([0.005, 0.005, 0.005, 0.1, 0.05, 0.1, 0.5]),
        tolerance_levels = None,
        model = None, 
        observational_data = None,
        state0 = None,
        t0 = 4,
        tmax = 34, 
        time_space = None):
        super().__init__(num_particles, num_generations, num_parameters, lower_bounds, upper_bounds, perturbation_kernels, observational_data, tolerance_levels, model, state0, t0, tmax, time_space)

    def ode_system(self, t, state, parameters):
        def CAR_negative_MZB(time):
            M0 = math.exp(14.06)
            nu = 0.0033
            b0 = 20.58
            return M0 * (1 + math.exp(-nu * (time - b0) ** 2))

        def Total_FoB(time):
            M0 = math.exp(16.7)
            nu = 0.004
            b0 = 20
            return M0 * (1 + math.exp(-nu * (time - b0) ** 2))

        alpha, beta, mu, delta, lambda_WT, lambda_N2KO = parameters

        dydt = [
            alpha * Total_FoB(t) - delta * state[0],  # CAR+ GCB (WT)
            mu * Total_FoB(t) + beta * CAR_negative_MZB(t) - lambda_WT * state[1],  # CAR+ MZB (WT)
            beta * CAR_negative_MZB(t) - lambda_N2KO * state[2]  # CAR+ MZB (N2KO)
        ]

        return dydt
    
    def __sample_priors(self, n = 1):
        # Sample from the prior distribution
        priors = np.random.uniform(self.lower_bounds, self.upper_bounds, (n, self.num_parameters))
        return priors

    def sample_priors(self):
        # Sample from the prior distribution
        priors = np.random.normal(self.lower_bounds, self.upper_bounds, self.num_parameters)
        return priors
    
    def calculate_prior_prob(self, parameters):
        # calculate pdf of normal distribution
        pass


class MZB(LatentABCSMC):
    def __init__(self,
        num_particles = 1000, 
        num_generations = 5, 
        num_parameters = 6, 
        lower_bounds = np.array([0.5, 14, 0.1, -5, 3.5, -4]), # mean
        upper_bounds = np.array([0.25, 2, 0.15, 1.2, 0.8, 1.2]),  # std
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
