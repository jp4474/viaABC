from src.viaABC.viaABC import viaABC
from scipy.stats import uniform
import numpy as np
from src.viaABC.systems import *
from src.viaABC.metrics import *
from scipy.ndimage import convolve
from PIL import Image
from typing import Optional
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

import sys
import rootutils
import hydra

# Setup project root and add to PYTHONPATH
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Now add the build folder relative to root
from pathlib import Path
PROJECT_ROOT = Path(rootutils.find_root())
sys.path.append(str(PROJECT_ROOT / "src" / "viaABC" / "spatial2D" / "build"))

import spatial2D_cpp as cpp

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
        metric = "pairwise_cosine",
        transform = None):
        self.transform = transform
        super().__init__(num_parameters, mu, sigma, observational_data, model, state0, t0, tmax, time_space, pooling_method, metric,)
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
        if self.transform is not None:
            x = self.transform(x)
        return x
    
class Spatial2D(viaABC):
    """
    Represents a 2D spatial simulation model for parameter inference and
    data assimilation. Loads observational data from images or text files,
    processes it into grid and one-hot formats, and supports simulation,
    prior sampling, and prior probability calculation. Integrates with C++
    extensions for efficient simulation.
    """
    def __init__(self,
        num_parameters = 3, 
        mu = np.array( [0., 0., 0.]), # Lower Bound
        sigma = np.array([1., 1., 1.]),
        model = None, 
        observational_data: Optional[np.ndarray] = None,
        state0 = None,
        t0 = 0,
        tmax = 24,
        dt = 0.1,
        time_space = None,
        pooling_method = "no_cls",
        metric = "pairwise_cosine",
        sample_id: str = "sample_1",):

        sample_paths = self._load_spatial2d_samples()
        if sample_id not in sample_paths:
            raise ValueError(f"Unknown sample_id={sample_id!r}.")

        sample_cfg = sample_paths[sample_id]
        image_path = sample_cfg.get("image")
        txt_path = sample_cfg.get("txt")

        if txt_path is None:
            raise ValueError(f"Sample {sample_id!r} is missing a txt path.")

        txt_path = Path(hydra.utils.to_absolute_path(txt_path))
        img = None

        if image_path is not None:
            try:
                img = self.read_image_as_matrix(Path(hydra.utils.to_absolute_path(image_path)))
            except FileNotFoundError:
                img = None

        if img is None:
            grid = self.read_txt_as_matrix(txt_path)
        else:
            grid = self.image_to_grid(img)

        self.observational_data = self.labels2map(grid)  # Shape: (num_classes, height, width) == (6, 1200, 1200)

        super().__init__(num_parameters, mu, sigma, self.observational_data, model, state0, t0, tmax, time_space, pooling_method, metric)
        self.lower_bounds = mu
        self.upper_bounds = sigma
        self.dt = dt
        self.observational_data_flattened = grid.astype(int).tolist()

    @staticmethod
    def _load_spatial2d_samples():
        data_name = "spatial2D"
        if HydraConfig.initialized():
            data_name = HydraConfig.get().runtime.choices.get("data", data_name)

        overrides = [f"data={data_name}", "model=spatial2D"]

        if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            cfg = compose(config_name="train", overrides=overrides)
            return cfg.data.observation_samples

        with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
            cfg = compose(config_name="train", overrides=overrides)
            return cfg.data.observation_samples

    def read_txt_as_matrix(self, txt_path: str) -> np.ndarray:
    # converts a txt file to a numpy array
        return np.loadtxt(txt_path, dtype=np.uint8)

    def read_image_as_matrix(self, image_path: str) -> np.ndarray:
    # converts an image to a numpy array
        img = Image.open(image_path)
        # Convert to numpy array
        img_array = np.array(img)       
        return img_array

    def image_to_grid(self, img: np.ndarray) -> np.ndarray:
        # Threshold an RGB segmentation image into simulator state IDs.
        red_threshold = 40
        green_threshold = 40
        blue_threshold = 50

        yellow_mask = (img[:, :, 0] > red_threshold) & (img[:, :, 1] > green_threshold) & (img[:, :, 2] < blue_threshold)
        no_color_mask = (img[:, :, 0] < red_threshold) & (img[:, :, 1] < green_threshold) & (img[:, :, 2] < blue_threshold)
        red_mask = (img[:, :, 0] > red_threshold) & (img[:, :, 1] < green_threshold) & (img[:, :, 2] < blue_threshold)
        green_mask = (img[:, :, 0] < red_threshold) & (img[:, :, 1] > green_threshold) & (img[:, :, 2] < blue_threshold)
        blue_mask = (img[:, :, 0] < red_threshold) & (img[:, :, 1] < green_threshold) & (img[:, :, 2] > blue_threshold)

        grid = np.zeros(img.shape[:2], dtype=np.uint8)
        grid[red_mask] = 0
        grid[yellow_mask] = 1
        grid[blue_mask] = 2
        grid[no_color_mask] = 3
        grid[green_mask] = 4
        return grid
        
    def simulate(self, parameters: np.ndarray) -> tuple[np.ndarray, int]:
        """ Simulate the spatial 2D model using C++ extension. Output numpy array of shape (num_classes, height, width) and boolean indicating success. 0 means success, 1 means failure."""
        params = cpp.Parameters()
        params.alpha = parameters[0]
        params.beta = parameters[1]
        params.gamma = parameters[2]
        params.dt = self.dt
        params.t0 = self.t0
        params.t_end = self.tmax

        #TODO: replace cpp with cython extension when ready

        g = cpp.Grid(self.observational_data_flattened, params)
        g.simulate()

        return g.numpy(), 0
    
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
        # (1200, 1200) to (6, 1200, 1200) one-hot encoding
        return np.eye(6, dtype=np.float32)[y].transpose(2, 0, 1)
    
    def preprocess(self, x):
        if x.ndim == 2:
            x = x[None, ...]
        return x

class SpatialSIR3D(viaABC):
    def __init__(self,
        num_parameters = 2, 
        mu = np.array( [0.2, 0.2]), # Lower Bound
        sigma = np.array([4.5, 4.5]),
        model = None, 
        observational_data: Optional[np.ndarray] = None,
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

        observational_data = self.labels2map(observational_data) # Your observational data may not require this step
        super().__init__(num_parameters, mu, sigma, observational_data, model, state0, t0, tmax, time_space, pooling_method, metric)
        # observational_data = self.labels2map(observational_data) # Your observational data may not require this step

        self.logger.info("Your observational data shape: %s", observational_data.shape)
        self.logger.info("Converted labels to one-hot encoded maps. Remove this step if not needed.")

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
