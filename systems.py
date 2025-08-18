from viaABC import viaABC
from scipy.stats import uniform, truncnorm
import numpy as np
from typing import Union, List, Optional
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
    
class CARModel(viaABC):
    def __init__(self,
        num_parameters=8, 
        mu=np.array([0.01, 0.01, 0.01, 0.01, 0.8, 0.1, 0.8, 8]),
        sigma=np.array([0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 1.5]),
        lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        upper_bounds = np.array([1, 1, 1, 1, 1, 1, 1, 12]),
        model=None,
        t0=4,
        tmax=30, 
        state0=np.array([np.exp(11.5), np.exp(10.8)]),
        time_space=np.array([4, 7, 9, 14, 17, 22, 26, 30]),
        pooling_method="no_cls",
        metric="pairwise_cosine"):
            
        # Load data once and store
        self.observational_data = np.load('/home/jp4474/CAR/viaABC/data/BCELL/noisy_data.npy')

        self.mu = mu
        self.sigma = sigma
        # Pre-compute bounds arrays
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Call parent constructor
        super().__init__(num_parameters, self.mu, self.sigma, self.observational_data, model, 
                        state0, t0, tmax, time_space, pooling_method, metric)

        assert np.all(self.lower_bounds < self.upper_bounds), "Lower bounds must be less than upper bounds"
        assert np.all(self.mu >= self.lower_bounds) and np.all(self.mu <= self.upper_bounds), "Mu must be within bounds"

        # Pre-compute constants for efficiency
        self._setup_constants()
        
        # Pre-compute truncnorm parameters for prior sampling
        self._precompute_truncnorm_params()

    def _setup_constants(self):
        """Pre-compute constants used in calculations"""
        # Constants for _CAR_positive_FOB
        self.F0 = np.exp(11.722278)
        self.B0 = np.exp(4.475064)
        self.n = 4.781548
        self.X = 6.943644
        self.q = 5
        self.X_q = self.X**self.q  # Pre-compute X^q
        
        # Constants for _CAR_negative_MZB
        self.M0_neg = np.exp(14.06)
        self.nu_neg = 0.0033
        self.b0_neg = 20.58
        
        # Constants for _Total_FoB
        self.M0_total = np.exp(16.7)
        self.nu_total = 0.004
        self.b0_total = 20
        
        # ODE constants
        self.t0_ode = 4.0

    def _precompute_truncnorm_params(self):
        """Pre-compute truncated normal parameters for efficiency"""
        self.truncnorm_a = (self.lower_bounds - self.mu) / self.sigma
        self.truncnorm_b = (self.upper_bounds - self.mu) / self.sigma

    @lru_cache(maxsize=1000)
    def _CAR_positive_FOB(self, t):
        """Cached version of CAR positive FOB calculation"""
        t_q = t**self.q
        return self.F0 + (self.B0 * t**self.n) * (1 - t_q / (self.X_q + t_q))

    @lru_cache(maxsize=1000)
    def _CAR_negative_MZB(self, t):
        """Cached version of CAR negative MZB calculation"""
        return self.M0_neg * (1 + np.exp(-self.nu_neg * (t - self.b0_neg)**2))

    @lru_cache(maxsize=1000)
    def _Total_FoB(self, t):
        """Cached version of Total FoB calculation"""
        return self.M0_total * (1 + np.exp(-self.nu_total * (t - self.b0_total)**2))
    
    def simulate(self, parameters: np.ndarray, time_space: Optional[np.ndarray] = None) -> tuple:
        """
        Simulate the ODE system with given parameters.
        
        Args:
            parameters: Array of parameters, last element is M0N2 (log-transformed)
            time_space: Optional time evaluation points. If None, uses self.time_space
            
        Returns:
            tuple: (solution array, solver status)
        """
        # Extract and transform the last parameter
        m0n2_log = parameters[-1]
        car_mz0n2k0 = np.exp(m0n2_log)
        
        # Prepare initial state
        initial_state = np.array([self.state0[0], self.state0[1], car_mz0n2k0])
        
        # Set time evaluation points
        t_eval = self.time_space if time_space is None else time_space
        
        # Solve ODE system
        solution = solve_ivp(
            fun=self.ode_system,
            t_span=[self.t0, self.tmax],
            y0=initial_state,
            t_eval=t_eval,
            args=(parameters[:-1],)
        )
        
        # Process solution based on whether default time_space was used
        if time_space is None:
            # Apply NaN masking for default time_space
            solution_array = solution.y.T.copy()
            nan_mask = (np.array([0, 4, 5, 6, 7]), np.array([2, 2, 2, 2, 2]))
            solution_array[nan_mask] = 0
            return solution_array, solution.status
        else:
            # Return solution directly for custom time_space
            return solution.y.T, solution.status


    def ode_system(self, t, state, parameters):
        """
        Optimized ODE system derived from Stan code.

        Parameters:
        - t: time (scalar)
        - state: [y1, y2]
        - parameters: [alpha, beta, mu, delta, lambda_WT, nu]
        """
        y1, y2, y3 = state

        alpha, beta, mu, nu, delta, lambda_WT, lambda_N2KO = parameters

        # Time-dependent modulation centered at t0
        mod = 1 + np.exp(nu * (t - self.t0_ode)**2)

        # Time-modulated parameters (vectorized division)
        alpha_tau = alpha / mod
        beta_tau = beta / mod
        mu_tau = mu # / mod

        # External inputs (using cached functions)
        total_fob = self._Total_FoB(t)
        car_neg_mzb = self._CAR_negative_MZB(t)

        # ODEs
        d_y1 = alpha_tau * total_fob - delta * y1  # CAR positive GCB cells in WT
        d_y2 = mu_tau * total_fob + beta_tau * car_neg_mzb - lambda_WT * y2  # CAR positive MZB cells in WT
        d_y3 = beta_tau * car_neg_mzb - lambda_N2KO * y3  # CAR positive MZB cells in N2KO

        return np.array([d_y1, d_y2, d_y3])

    def sample_priors(self):
        """Optimized prior sampling using pre-computed parameters"""
        # Vectorized sampling using pre-computed truncnorm parameters
        priors = truncnorm.rvs(
            a=self.truncnorm_a, 
            b=self.truncnorm_b, 
            loc=self.mu, 
            scale=self.sigma, 
            size=self.num_parameters
        )
        return priors

    def calculate_prior_log_prob(self, parameters):
        """Optimized prior probability calculation"""
        # Vectorized log probability calculation
        log_probs = truncnorm.logpdf(
            parameters, 
            a=self.truncnorm_a, 
            b=self.truncnorm_b, 
            loc=self.mu, 
            scale=self.sigma
        )
        return np.sum(log_probs)
    
    def preprocess(self, x):
        x = np.abs(x)
        s = np.mean(x, axis=0)
        return x / s
