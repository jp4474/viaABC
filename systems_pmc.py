from latent_abc_smc import LatentABCSMC
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde, norm, uniform, multivariate_normal
import numpy as np
import torch
import logging
from typing import Union, List
import pandas as pd
import time
import math
import os
import warnings
from systems import *
from tempfile import TemporaryFile
from metrics import *

from scipy.stats import qmc
from concurrent.futures import ThreadPoolExecutor, as_completed
    
class LotkaVolterra(LatentABCSMC):
    def __init__(self,
        num_parameters = 2, 
        lower_bounds = np.array([0, 0]), 
        upper_bounds = np.array([10, 10]), 
        model = None, 
        observational_data = np.array([[1.87, 0.65, 0.22, 0.31, 1.64, 1.15, 0.24, 2.91],
                                        [0.49, 2.62, 1.54, 0.02, 1.14, 1.68, 1.07, 0.88]]).T, 
        state0 = np.array([1, 0.5]),
        t0 = 0,
        tmax = 15, 
        time_space = np.array([1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4]),
        pooling_method = "cls",
        metric = "cosine"):
        super().__init__(num_parameters, lower_bounds, upper_bounds, observational_data, model, state0, t0, tmax, time_space, pooling_method, metric)

    def ode_system(self, t, state, parameters):
        # Lotka-Volterra equations
        alpha, delta = parameters
        beta, gamma = 1, 1
        prey, predator = state
        dprey = prey * (alpha - beta * predator)
        dpredator = predator * (-gamma + delta * prey)
        return [dprey, dpredator]
        
    def sample_priors(self):
        # Sample from the prior distribution
        priors = np.random.uniform(self.lower_bounds, self.upper_bounds, self.num_parameters)
        return priors
    
    def calculate_prior_prob(self, parameters):
        probabilities = uniform.pdf(parameters, loc=self.lower_bounds, scale=self.upper_bounds - self.lower_bounds) 
        return np.prod(probabilities)
    
    def perturb_parameters(self, parameters, previous_particles, sigma = 0.1):
        # Perturb the parameters
        perturbations = sigma * np.random.uniform(-1, 1, self.num_parameters)
        parameters += perturbations
        return parameters
    
    def preprocess(self, x):
        mean = np.mean(np.abs(x), 0)
        x = x / mean
        return x

class MZB(LatentABCSMC):
    def __init__(self,
        num_parameters = 5, 
        lower_bounds = np.array( [0, 10, -0.2, -7.4, -6.4]), # mean
        upper_bounds = np.array([1, 18, 0.4, -2.6, -1.6]),  # std
        model = None, 
        observational_data = None,
        state0 = None,
        t0 = 40,
        tmax = 732,
        time_space = np.array([59, 69, 76, 88, 95, 102, 108, 109, 113, 119, 122, 124, 141, 156, 158, 183, 212, 217, 219, 235, 261, 270, 289, 291, 306, 442, 524, 563, 566, 731]),
        pooling_method = "cls",
        metric = "cosine"):
        super().__init__(num_parameters, lower_bounds, upper_bounds, observational_data, model, state0, t0, tmax, time_space, pooling_method, metric)

    def sample_priors(self):
        # Sample from the prior distribution
        priors = np.random.uniform(self.lower_bounds, self.upper_bounds, self.num_parameters)
        return priors
    
    def calculate_prior_prob(self, parameters):
        probabilities = uniform.pdf(parameters, loc=self.lower_bounds, scale=self.upper_bounds - self.lower_bounds)
        return np.prod(probabilities)

    def perturb_parameters(self, theta, sigma):
        return multivariate_normal(theta, sigma)
    
    # Define the ODE system
    def ode_system(self, t, state, parameters):
        # parameters
        phi, rho, delta = parameters
        beta = 4
        
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
        phi, y0_Log, kappa_0, rho_Log, delta_Log = parameters 

        y0 = np.exp(y0_Log)
        rho = np.exp(rho_Log)
        delta = np.exp(delta_Log)

        y0 = [0.0, 0.0, y0 * kappa_0, y0 * (1 - kappa_0)]
        parameters = [phi, rho, delta]

        solution = solve_ivp(self.ode_system, [self.t0, self.tmax], y0=y0, t_eval=self.time_space, args=(parameters,))
        status = solution.status

        if status != 0:
            return np.zeros((self.time_space.shape[0], 4)), status
                
        k_hat = solution.y.T

        numsolve = len(k_hat)
        MZ_counts_mean = np.zeros(numsolve)
        donor_fractions_mean = np.zeros(numsolve)
        donor_ki_mean = np.zeros(numsolve)
        host_ki_mean = np.zeros(numsolve)
        Nfd_mean = np.zeros(numsolve)

        # Compute MZ_counts_mean safely
        MZ_counts_mean = k_hat[:, 0] + k_hat[:, 1] + k_hat[:, 2] + k_hat[:, 3]

        # Avoid division by zero in donor_fractions_mean
        donor_fractions_mean = np.where(
            MZ_counts_mean > 0,
            (k_hat[:, 0] + k_hat[:, 1]) / MZ_counts_mean,
            0
        )

        # Avoid division by zero in donor_ki_mean
        donor_ki_mean = np.where(
            k_hat[:, 0] > 1e-8,
            k_hat[:, 0] / np.maximum(k_hat[:, 0] + k_hat[:, 1], 1e-8),
            0
        )

        # Avoid division by zero in host_ki_mean
        host_ki_mean = np.where(
            k_hat[:, 2] > 1e-8,
            k_hat[:, 2] / np.maximum(k_hat[:, 2] + k_hat[:, 3], 1e-8),
            0
        )

        # Avoid division by zero in Nfd_mean
        Nfd_mean = np.where(
            0.78 > 0,
            donor_fractions_mean / 0.78,
            0
        )

        data = np.array([MZ_counts_mean, donor_ki_mean, host_ki_mean, Nfd_mean]).T

        if np.all(data >= 0) and np.all(data < 5e6):
            return data, status
        else:
            return data, -1


    def preprocess(self, x):
        mzb, donor_ki67, host_ki67, nfd = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        # Normalize mzb
        mzb = mzb/3e6

        # Transform donor_ki67 and host_ki67
        donor_ki67 = np.arcsin(np.sqrt(donor_ki67))

        host_ki67 = np.arcsin(np.sqrt(host_ki67))

        # No transformation for nfd
        nfd = nfd

        # Stack the transformed data
        x = np.array([mzb, donor_ki67, host_ki67, nfd]).T
        
        return x
    
