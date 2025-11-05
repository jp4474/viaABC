from viaABC import viaABC
from scipy.stats import uniform, truncnorm
import numpy as np
from typing import Union, List
from systems import *
from metrics import *
from scipy.ndimage import convolve
from scipy.integrate import solve_ivp
from functools import lru_cache

class LotkaVolterra(viaABC):
    def __init__(self,
        num_parameters = 2, 
        mu = np.array([0, 0]),
        sigma = np.array([10, 10]),
        model = None, 
        observational_data = np.array([[1.87, 0.65, 0.22, 0.31, 1.64, 1.15, 0.24, 2.91],
                                        [0.49, 2.62, 1.54, 0.02, 1.14, 1.68, 1.07, 0.88]]).T, 
        state0 = np.array([1, 0.5]), 
        t0 = 0,
        tmax = 15, 
        time_space = np.array([1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4]),
        pooling_method = "no_cls",
        metric = "pairwise_cosine"):
        super().__init__(num_parameters, mu, sigma, observational_data, model, state0, t0, tmax, time_space, pooling_method, metric)
        self.lower_bounds = mu 
        self.upper_bounds = sigma

    def ode_system(self, t, state, parameters):
        # Lotka-Volterra equations
        alpha, delta = parameters
        beta, gamma = 1, 1
        prey, predator = state # x, y
        dprey = prey * (alpha - beta * predator)
        dpredator = predator * (-gamma + delta * prey)
        return [dprey, dpredator]

    def sample_priors(self):
        # Sample from the prior distribution
        priors = np.random.uniform(self.lower_bounds, self.upper_bounds, self.num_parameters)
        return priors
    
    def calculate_prior_log_prob(self, parameters):
        probabilities = uniform.logpdf(parameters, loc=self.lower_bounds, scale=self.upper_bounds - self.lower_bounds)
        probabilities = np.sum(probabilities)
        return probabilities
    
    def preprocess(self, x):
        mean = np.mean(np.abs(x), 0)
        x = x / mean
        return x

class SpatialSIR3D(viaABC):
    def __init__(self,
        num_parameters = 2, 
        mu = np.array( [0.2, 0.2]), # Lower Bound
        sigma = np.array([4.5, 4.5]),
        model = None, 
        observational_data = None,
        state0 = None,
        t0 = 0,
        tmax = 16,
        interval = 1,
        time_space = np.arange(1, 16, 1),
        pooling_method = "no_cls",
        metric = "pairwise_cosine",
        grid_size = 80,
        initial_infected = 5,
        radius = 5):

        observational_data = self.labels2map(np.load('/home/jp4474/viaABC/data/SPATIAL/data.npy'))

        super().__init__(num_parameters, mu, sigma, observational_data, model, state0, t0, tmax, time_space, pooling_method, metric)
        self.grid_size = grid_size
        self.initial_infected = initial_infected
        self.radius = radius
        self.time_steps = int((tmax - t0)/interval)
        self.lower_bounds = mu
        self.upper_bounds = sigma

    def simulate(self, parameters: np.ndarray):
        SUSCEPTIBLE, INFECTED, RECOVERED = 0, 1, 2

        beta, tau_I = parameters
        dt = .05                   # time step, small to simulate continuous time
        I = tau_I                 # infection duration (τ_I in paper, fixed time)    <---------------  0.2 - 4.0
        R = 1.0                    # resistance duration (τ_R = 1.0, fixed in paper) 
        steps = int(np.round(np.max(self.time_space) / dt))

        # Initialize the grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        grid_shape = grid.shape
        infection_timer = np.zeros(grid_shape)
        recovery_timer = np.zeros(grid_shape)
        susceptible_timer = np.zeros(grid_shape)

        centers = np.array([[44, 67], [24, 67], [64, 73], [3, 55], [12, 20]])

        for x, y in centers:
            dx, dy = np.random.randint(-self.radius, self.radius + 1, 2)
            xi, yi = np.clip([x + dx, y + dy], 0, self.grid_size - 1)
            grid[xi, yi] = INFECTED

        kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])

        # Prepare to store grid states
        frames = []
        frames.append(grid.copy())

        # Run the simulation
        frames = []
        for t in range(steps):
            # Count infected neighbors
            infected_neighbors = convolve((grid == INFECTED).astype(np.uint8), kernel, mode='constant')

            # Calculate infection probability based on PNAS paper formula
            p_inf = 1 - np.exp(-beta * infected_neighbors * dt)

            # Infect susceptible cells
            rand_vals = np.random.rand(*grid_shape)
            new_infections = (grid == SUSCEPTIBLE) & (rand_vals < p_inf) 
            grid[new_infections] = INFECTED

            recovery_timer[new_infections] = 0  # Reset recovery timer when infected
            infection_timer[new_infections] = 0
            susceptible_timer[new_infections] = 0  # Reset susceptible timer when infected

            # Update timers and state transitions
            infection_timer[grid == INFECTED] += dt
            to_recover = (grid == INFECTED) & (infection_timer >= I)
            grid[to_recover] = RECOVERED

            susceptible_timer[to_recover] = 0  # Reset susceptible timer when recovering
            recovery_timer[to_recover] = 0
            infection_timer[to_recover] = 0  # Optional: reset infection timer

            recovery_timer[grid == RECOVERED] += dt
            to_reset = (grid == RECOVERED) & (recovery_timer >= R)
            grid[to_reset] = SUSCEPTIBLE

            infection_timer[to_reset] = 0  # Reset infection timer on return to susceptible
            susceptible_timer[to_reset] = 0  # Reset timer on return to susceptible
            recovery_timer[to_reset] = 0     # Optional: reset recovery timer

            # Increment susceptible timer for susceptible cells
            susceptible_timer[grid == SUSCEPTIBLE] += dt

            # Store timers as a 3D tensor for visualization/ML
            # x = np.stack((susceptible_timer, infection_timer, recovery_timer), axis=-1)
            # x = np.stack((susceptible_timer, infection_timer, recovery_timer), axis=-1)
            susceptible = (grid == SUSCEPTIBLE).astype(np.float32)
            infected = (grid == INFECTED).astype(np.float32)
            recovered = (grid == RECOVERED).astype(np.float32)
            x = np.stack((susceptible, infected, recovered), axis=-1)
            # add along the time dimension
            # x = x.sum(axis=-1)
            frames.append(x.copy())

        # 15 x 80 x 80 x 3
        # Convert to one-hot encoded 3D array for visualization/ML
        frames = np.array(frames)
        frames_idx = (self.time_space / dt).astype(int) - 1
        output = frames[frames_idx].transpose(3, 0, 1, 2) 

        # TODO: use try and catch
        return output, 0
    
    def sample_priors(self):
        # Sample from the prior distribution
        priors = np.random.uniform(self.lower_bounds, self.upper_bounds, self.num_parameters)
        return priors
            
    def calculate_prior_log_prob(self, parameters):
        # Calculate the prior log probability of the parameters
        # This must match the prior distribution used in sampling
        log_probabilities = uniform.logpdf(parameters, loc=self.lower_bounds, scale=self.upper_bounds - self.lower_bounds) 
        return np.sum(log_probabilities)

    def labels2map(self, y):
        susceptible = (y == 0)
        infected = (y == 1)
        resistant = (y == 2)

        y_onehot = np.stack([susceptible, infected, resistant], axis=1)  # Shape: (3, H, W)

        return y_onehot
    
    def preprocess(self, x):
        # add a channel dimension at the beginning in numpy
        if x.shape[0] == 15:
            x = x.transpose(1, 0, 2, 3)

        if x.ndim == 4:
            x = np.expand_dims(x, axis=0)

        return x