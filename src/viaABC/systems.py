from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import logging
import numpy as np
import torch
from PIL import Image
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
import sys

import rootutils
import hydra
from scipy.ndimage import convolve
from scipy.stats import uniform

from src.viaABC.metrics import bert_score, bert_score_batch, cosine_similarity, l1_distance, l2_distance, maxSim, pairwise_cosine
from src.viaABC.viaABC import viaABC

log = logging.getLogger(__name__)

# Setup project root and add to PYTHONPATH
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

PROJECT_ROOT = Path(rootutils.find_root())
sys.path.append(str(PROJECT_ROOT / "src" / "viaABC" / "spatial2D" / "build"))

# import spatial2D_cpp as cpp
from src.viaABC.spatial2D import _grid_core as cpp

class LotkaVolterra(viaABC):
    def __init__(self,
        num_parameters: int = 2,
        mu: np.ndarray = np.array([0, 0]),
        sigma: np.ndarray = np.array([10, 10]),
        model: Optional[torch.nn.Module] = None,
        observational_data: np.ndarray = np.array([[1.87, 0.65, 0.22, 0.31, 1.64, 1.15, 0.24, 2.91],
                                        [0.49, 2.62, 1.54, 0.02, 1.14, 1.68, 1.07, 0.88]]).T,
        state0: np.ndarray = np.array([1, 0.5]),
        t0: int = 0,
        tmax: int = 15,
        time_space: np.ndarray = np.array([1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4]),
        pooling_method: str = "no_cls",
        metric: str = "pairwise_cosine",
        transform: Any = None) -> None:
        self.transform = transform
        super().__init__(num_parameters, mu, sigma, observational_data, model, state0, t0, tmax, time_space, pooling_method, metric,)
        self.lower_bounds = mu 
        self.upper_bounds = sigma

    def ode_system(self, t: float, state: np.ndarray, parameters: np.ndarray) -> list[float]:
        # Lotka-Volterra equations
        alpha, delta = parameters
        beta, gamma = 1, 1
        prey, predator = state # x, y
        dprey = prey * (alpha - beta * predator)
        dpredator = predator * (-gamma + delta * prey)
        return [dprey, dpredator]

    def sample_priors(self) -> np.ndarray:
        # Sample from the prior distribution
        priors = np.random.uniform(self.lower_bounds, self.upper_bounds, self.num_parameters)
        return priors
    
    def calculate_prior_log_prob(self, parameters: np.ndarray) -> float:
        probabilities = uniform.logpdf(parameters, loc=self.lower_bounds, scale=self.upper_bounds - self.lower_bounds)
        probabilities = np.sum(probabilities)
        return probabilities
    
    def preprocess(self, x: np.ndarray) -> np.ndarray:
        if self.transform is not None:
            x = self.transform(x)
        return x
    
class Spatial2D(viaABC):
    """
    Represents a 2D spatial simulation model for parameter inference and
    data assimilation. Loads observational data from images or text files,
    processes it into grid and one-hot formats, and supports simulation,
    prior sampling, and prior probability calculation.

    Architecture note:
        `viaABC` defines the generic inference loop, while this subclass owns
        domain-specific concerns: how to load one observation, how to run one
        spatial simulation, and how to compare multiple reference samples when
        the experiment uses more than one observed grid.
    """
    def __init__(self,
        num_parameters: int = 2,
        mu: np.ndarray = np.array([0.0, 0.0]), # Lower Bound
        sigma: np.ndarray = np.array([0.1, 0.1]), # Upper Bound
        model: Optional[torch.nn.Module] = None,
        state0: Optional[np.ndarray] = None,
        t0: int = 0,
        tmax: int = 24,
        dt: float = 0.4,
        time_space: Optional[np.ndarray] = None,
        pooling_method: str = "no_cls",
        metric: str = "pairwise_cosine",
        transform: Any = None,
        use_time_series: bool = True,
        num_frames: int | None = None,
        sample_id: str | Sequence[str] | None = ["sample_1", "sample_2", "sample_3", "sample_4"]) -> None:

        self.transform = transform
        self.use_time_series = use_time_series
        self.num_frames = num_frames
        # Build the Cython simulator objects lazily so object construction stays
        # cheap unless we actually run simulations.
        self._cython_cores: list[Any] | None = None
        self._last_simulation_sample_index = 0
        self._sample_source_info: list[dict[str, Any]] = []
        sample_paths = self._load_spatial2d_samples()
        sample_ids = [sample_id] if isinstance(sample_id, str) else list(sample_id or [])
        if not sample_ids:
            raise ValueError("sample_id must be a sample name or a non-empty list of sample names.")

        sample_pairs = [
            self._load_sample_grids(sample_paths, current_sample_id)
            for current_sample_id in sample_ids
        ]
        initial_grids = [initial_grid for initial_grid, _ in sample_pairs]
        observation_grids = [observation_grid for _, observation_grid in sample_pairs]

        # Keep simulator initial states and observed final states separate.
        # The simulator evolves the TXT grid forward, while the encoder compares
        # against the image-derived observation grid.
        self._initial_grids = np.stack(initial_grids, axis=0)
        self._observation_grids = np.stack(observation_grids, axis=0)
        self._multiple_samples = len(sample_pairs) > 1

        if self._multiple_samples:
            # Multi-sample Spatial2D stays a subclass concern: we store one
            # observation tensor per sample and later aggregate distances here
            # instead of teaching the generic viaABC base class about sample-wise
            # simulator semantics.
            self.observational_data = np.stack(
                [self.labels2map(grid) for grid in observation_grids],
                axis=0,
            )
            self.observational_data_flattened = [
                grid.astype(int).tolist() for grid in initial_grids
            ]
        else:
            self.observational_data = self.labels2map(observation_grids[0])
            self.observational_data_flattened = initial_grids[0].astype(int).tolist()

        super().__init__(
            num_parameters,
            mu,
            sigma,
            self.observational_data,
            model,
            state0,
            t0,
            tmax,
            None,
            pooling_method,
            metric,
            transform,
            encode_observational_data=False,
        )
        self.lower_bounds = mu
        self.upper_bounds = sigma
        self.dt = dt
        if time_space is not None:
            self.time_space = np.asarray(time_space, dtype=np.float64)
            if self.num_frames is not None and len(self.time_space) != self.num_frames:
                raise ValueError("system.num_frames must match len(system.time_space).")

        if len(self.lower_bounds) != self.num_parameters or len(self.upper_bounds) != self.num_parameters:
            raise ValueError(
                "Spatial2D prior bounds must match num_parameters. "
                f"Received num_parameters={self.num_parameters}, "
                f"len(mu)={len(self.lower_bounds)}, len(sigma)={len(self.upper_bounds)}."
            )

        # Spatial2D owns time-series preprocessing, so encode observations only
        # after subclass-specific fields such as `time_space` are initialized.
        if model is not None:
            self.update_model(model)

    @staticmethod
    def _load_spatial2d_samples() -> Mapping[str, Mapping[str, Any]]:
        # Sample metadata lives in Hydra config so experiments can switch
        # observations without hard-coding file paths in the system class.
        data_name = "spatial2D"
        if HydraConfig.initialized():
            data_name = str(HydraConfig.get().runtime.choices.get("data", data_name))

        overrides = [f"data={data_name}", "model=spatial2D"]

        if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            cfg = compose(config_name="train", overrides=overrides)
            return cfg.data.observation_samples

        with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
            cfg = compose(config_name="train", overrides=overrides)
            return cfg.data.observation_samples

    def read_txt_as_matrix(self, txt_path: str | Path) -> np.ndarray:
    # converts a txt file to a numpy array
        return np.loadtxt(txt_path, dtype=np.uint8)

    def read_image_as_matrix(self, image_path: str | Path) -> np.ndarray:
    # converts an image to a numpy array
        img = Image.open(image_path)
        # Convert to numpy array
        img_array = np.array(img)       
        return img_array

    def _load_sample_grids(
        self,
        sample_paths: Mapping[str, Mapping[str, Any]],
        sample_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if sample_id not in sample_paths:
            raise ValueError(f"Unknown sample_id={sample_id!r}.")

        sample_cfg = sample_paths[sample_id]
        image_path = sample_cfg.get("image")
        txt_path = sample_cfg.get("txt")

        if txt_path is None:
            raise ValueError(f"Sample {sample_id!r} is missing a txt path.")

        txt_path = Path(hydra.utils.to_absolute_path(txt_path))
        initial_grid = self.read_txt_as_matrix(txt_path)
        resolved_image_path = None
        img = None

        if image_path is not None:
            resolved_image_path = Path(hydra.utils.to_absolute_path(image_path))
            try:
                img = self.read_image_as_matrix(resolved_image_path)
            except FileNotFoundError:
                img = None

        if img is None:
            # TXT is the simulator-native representation and therefore the
            # reliable fallback when the processed image is unavailable.
            self._sample_source_info.append(
                {
                    "sample_id": sample_id,
                    "initial_txt_path": str(txt_path),
                    "observed_raw_image_path": str(resolved_image_path) if resolved_image_path is not None else None,
                    "observed_grid_source": "txt_fallback",
                    "initial_grid_shape": tuple(initial_grid.shape),
                    "observed_grid_shape": tuple(initial_grid.shape),
                }
            )
            log.info(
                "Spatial2D sample %s: initial txt=%s, observed raw image=%s, "
                "observed final grid source=txt fallback, shape=%s",
                sample_id,
                txt_path,
                resolved_image_path,
                initial_grid.shape,
            )
            return initial_grid, initial_grid.copy()

        observed_grid = self.image_to_grid(img)
        if observed_grid.shape != initial_grid.shape:
            if (
                observed_grid.shape[0] < initial_grid.shape[0]
                or observed_grid.shape[1] < initial_grid.shape[1]
            ):
                raise ValueError(
                    f"Sample {sample_id!r} observed grid shape {observed_grid.shape} "
                    f"is smaller than initial grid shape {initial_grid.shape}."
                )
            observed_grid = observed_grid[: initial_grid.shape[0], : initial_grid.shape[1]]
        self._sample_source_info.append(
            {
                "sample_id": sample_id,
                "initial_txt_path": str(txt_path),
                "observed_raw_image_path": str(resolved_image_path),
                "observed_grid_source": "raw_image",
                "initial_grid_shape": tuple(initial_grid.shape),
                "raw_image_shape": tuple(img.shape),
                "observed_grid_shape": tuple(observed_grid.shape),
            }
        )
        log.info(
            "Spatial2D sample %s: initial txt=%s, observed raw image=%s, "
            "observed final grid source=raw image -> grid, raw image shape=%s, grid shape=%s",
            sample_id,
            txt_path,
            resolved_image_path,
            img.shape,
            observed_grid.shape,
        )
        return initial_grid, observed_grid

    def image_to_grid(self,img: np.ndarray) -> np.ndarray:   
        # Threshold an RGB segmentation image into simulator state IDs.
        red_threshold = 50
        green_threshold = 45
        blue_threshold = 50
        hotspot_green_threshold = 120


        yellow_mask = (img[:, :, 0] > red_threshold) & (img[:, :, 1] > green_threshold) & (img[:, :, 2] < blue_threshold)
        no_color_mask = (img[:, :, 0] < red_threshold) & (img[:, :, 1] < green_threshold) & (img[:, :, 2] < blue_threshold)
        red_mask = (img[:, :, 0] > red_threshold) & (img[:, :, 1] < green_threshold) & (img[:, :, 2] < blue_threshold)
        green_mask = (img[:, :, 0] < red_threshold) & (img[:, :, 1] > green_threshold) & (img[:, :, 2] < blue_threshold)
        blue_mask = (img[:, :, 0] < red_threshold) & (img[:, :, 1] < green_threshold) & (img[:, :, 2] > blue_threshold)
        # hotspot mask
        hotspot_mask = (img[:, :, 0] < red_threshold) & (img[:, :, 1] > hotspot_green_threshold) & (img[:, :, 2] < blue_threshold)

        grid = np.zeros(img.shape[:2], dtype=np.uint8)
        grid[red_mask] = 0
        grid[yellow_mask] = 1
        grid[blue_mask] = 2
        grid[no_color_mask] = 3
        grid[green_mask] = 4
        grid[hotspot_mask] = 5

        return grid
        
    def simulate(self, parameters: np.ndarray, time_space: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """ 
        Simulate the spatial 2D model using C++ extension. Output numpy array of
        shape (height, width), or (time, height, width) when time_space is set.
        Status 0 means success, 1 means failure.
        """

        # params = cpp.Parameters()
        # params.alpha = parameters[0]
        # params.beta = parameters[1]
        # params.dt = self.dt
        # params.t0 = self.t0
        # params.t_end = self.tmax

        # g = cpp.Grid(self.observational_data_flattened, params)
        # g.simulate()

        # return g.numpy(), 0
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape[0] != 2:
            raise ValueError(
                "Spatial2D simulator expects exactly 2 parameters: "
                "[alpha, beta]. "
                f"Received shape {parameters.shape} with values {parameters!r}."
            )

        if self._cython_cores is None:
            # Reuse the compiled grid cores across calls; constructing them
            # repeatedly adds overhead but does not change simulation results.
            self._cython_cores = [
                cpp.GridCore(np.asarray(grid, dtype=np.int32))
                for grid in self._initial_grids
            ]

        if self._multiple_samples:
            core_index = int(np.random.randint(len(self._cython_cores)))
        else:
            core_index = 0
        self._last_simulation_sample_index = core_index

        core = self._cython_cores[core_index]
        alpha = float(parameters[0])
        beta = float(parameters[1])
        dt = float(self.dt)
        t0 = float(self.t0)
        target_times = self.time_space if time_space is None else np.asarray(time_space, dtype=np.float64)

        if not self.use_time_series or target_times is None:
            return core.simulation(alpha, beta, dt, t0, float(self.tmax)), 0

        return np.stack([
            core.simulation(alpha, beta, dt, t0, float(t_end))
            for t_end in target_times
        ], axis=0), 0
    
    def simulate_for_inference(self, parameters: np.ndarray) -> tuple[np.ndarray, int]:
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape[0] != 2:
            raise ValueError(
                "Spatial2D simulator expects exactly 2 parameters: "
                "[alpha, beta]. "
                f"Received shape {parameters.shape} with values {parameters!r}."
            )

        if not self._multiple_samples:
            return self.simulate(parameters)

        alpha = float(parameters[0])
        beta = float(parameters[1])
        dt = float(self.dt)
        t0 = float(self.t0)
        target_times = self.time_space

        if self._cython_cores is None:
            # Reuse the compiled grid cores across calls; constructing them
            # repeatedly adds overhead but does not change simulation results.
            self._cython_cores = [
                cpp.GridCore(np.asarray(grid, dtype=np.int32))
                for grid in self._initial_grids
            ]

        if not self.use_time_series or target_times is None:
            return np.stack(
                [
                    core.simulation(alpha, beta, dt, t0, float(self.tmax))
                    for core in self._cython_cores
                ],
                axis=0,
            ), 0

        simulations = [
            np.stack(
                [
                    core.simulation(alpha, beta, dt, t0, float(t_end))
                    for t_end in target_times
                ],
                axis=0,
            )
            for core in self._cython_cores
        ]
        return np.stack(simulations, axis=0), 0



    def sample_priors(self) -> np.ndarray:
        # Sample from the uniform prior distribution
        priors = np.random.uniform(
            self.lower_bounds,
            self.upper_bounds,
            self.num_parameters,
        )
        return priors
            
    def calculate_prior_log_prob(self, parameters: np.ndarray) -> float:
        # Calculate the prior log probability of the parameters
        # This must match the prior distribution used in sampling
        log_probabilities = uniform.logpdf(
            parameters,
            loc=self.lower_bounds,
            scale=self.upper_bounds - self.lower_bounds,
        )
        return np.sum(log_probabilities)

    def labels2map(self, y: np.ndarray) -> np.ndarray:
        # (1200, 1200) to (6, 1200, 1200) one-hot encoding
        return np.eye(6, dtype=np.float32)[y].transpose(2, 0, 1)

    def _uses_temporal_encoder(self) -> bool:
        if self.model is None:
            return False
        model = getattr(self.model, "model", self.model)
        patch_embed = getattr(model, "patch_embed", None)
        return patch_embed is not None and hasattr(patch_embed, "frames")

    @torch.inference_mode()
    def _encode_observational_data(self):
        observation_input = self._observation_input()
        scaled_data = self.preprocess(observation_input)
        self.encoded_observational_data = self.get_latent(scaled_data)

    def _observation_input(self) -> np.ndarray:
        if not self._uses_temporal_encoder():
            return self._observation_grids if self._multiple_samples else self._observation_grids[0]

        if self._temporal_frame_count() != 2:
            raise ValueError("Spatial2D temporal observations currently require exactly two frames.")

        observations = np.stack([self._initial_grids, self._observation_grids], axis=1)
        return observations if self._multiple_samples else observations[0]

    @torch.inference_mode()
    def get_latent(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be provided to encode the data and run the method.")

        x = torch.as_tensor(x, dtype=torch.float32, device=self.model.device)
        expected_ndim = 5 if self._uses_temporal_encoder() else 4

        sample_batch_shape: tuple[int, int] | None = None
        if self._multiple_samples and x.ndim == expected_ndim + 1:
            sample_batch_shape = (int(x.shape[0]), int(x.shape[1]))
            x = x.reshape(-1, *x.shape[2:])
        elif x.ndim == expected_ndim - 1:
            x = x.unsqueeze(0)
        elif x.ndim != expected_ndim:
            raise ValueError(f"Expected Spatial2D model input with {expected_ndim - 1} or {expected_ndim} dims, got {tuple(x.shape)}")

        x = self.model.get_latent(x, self.pooling_method)
        if sample_batch_shape is not None:
            x = x.reshape(*sample_batch_shape, *x.shape[1:])
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def _temporal_frame_count(self) -> int:
        if self.num_frames is not None:
            return int(self.num_frames)
        if self.time_space is not None:
            return len(self.time_space)
        return 1

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if self._uses_temporal_encoder():
            if x.ndim == 3 and x.shape[0] == self._temporal_frame_count():
                x = np.eye(6, dtype=np.float32)[x].transpose(3, 0, 1, 2)
            elif x.ndim == 4:
                if x.shape[:2] == (6, self._temporal_frame_count()):
                    pass
                elif x.shape[1] == self._temporal_frame_count():
                    batch_size, time_steps = x.shape[:2]
                    x = np.stack(
                        [self.labels2map(grid) for grid in x.reshape(-1, *x.shape[2:])],
                        axis=0,
                    ).reshape(batch_size, time_steps, 6, *x.shape[2:]).transpose(0, 2, 1, 3, 4)
                else:
                    raise ValueError(f"Temporal Spatial2D input must be [T,H,W], [B,T,H,W], [6,T,H,W], or [B,6,T,H,W], got {x.shape}.")
            elif x.ndim != 5 or x.shape[2] != self._temporal_frame_count():
                raise ValueError(f"Temporal Spatial2D input must include a time dimension, got {x.shape}.")
            return x.astype(np.float32)

        if x.ndim == 2:
            x = self.labels2map(x)
        elif x.ndim == 3:
            x = np.stack([self.labels2map(grid) for grid in x], axis=0)
        elif x.ndim != 4:
            raise ValueError(f"Non-temporal Spatial2D input must be [H,W], [B,H,W], [6,H,W], or [B,6,H,W], got {x.shape}.")
        if self.transform is not None:
            x = self.transform(x)
        return x

    def calculate_distance(self, y: np.ndarray) -> float | np.ndarray:
        if not self._multiple_samples:
            return super().calculate_distance(y)

        x = self.encoded_observational_data
        y = np.asarray(y)
        if y.ndim >= 3 and y.shape[1] == x.shape[0]:
            return np.asarray(
                [
                    np.mean(
                        [
                            self._calculate_sample_distance(x[i:i + 1], y[j, i:i + 1])
                            for i in range(x.shape[0])
                        ]
                    )
                    for j in range(y.shape[0])
                ],
                dtype=np.float32,
            )

        if y.shape[0] != x.shape[0]:
            sample_index = self._last_simulation_sample_index
            return self._calculate_sample_distance(
                x[sample_index:sample_index + 1],
                y[:1],
            )

        # For multi-sample observations we score each sample independently and
        # average. This keeps the external ABC objective as "one scalar distance
        # per parameter proposal", even though internally we now compare several
        # observation/simulation pairs.
        return float(np.mean([
            self._calculate_sample_distance(x[i:i + 1], y[i:i + 1])
            for i in range(x.shape[0])
        ]))

    def _calculate_sample_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        # We keep the metric dispatch local so multi-sample aggregation can score
        # one observation/simulation pair at a time without mutating shared base
        # state like `self.encoded_observational_data`.
        if self.metric == "cosine":
            return float(1 - cosine_similarity(x, y))
        if self.metric == "l1":
            return float(l1_distance(x, y))
        if self.metric == "l2":
            return float(l2_distance(x, y))
        if self.metric == "bertscore":
            _, _, f1_scores = bert_score(x, y)
            return float(1 - f1_scores)
        if self.metric == "pairwise_cosine":
            return float(1 - pairwise_cosine(x, y))
        if self.metric == "bertscore_batch":
            return float(1 - bert_score_batch(x, y))
        if self.metric == "maxSim":
            return float(maxSim(x, y))
        raise ValueError(f"Unsupported metric: {self.metric}")
    
class SpatialSIR3D(viaABC):
    def __init__(self,
        num_parameters: int = 2,
        mu: np.ndarray = np.array( [0.2, 0.2]), # Lower Bound
        sigma: np.ndarray = np.array([4.5, 4.5]),
        model: Optional[torch.nn.Module] = None,
        observational_data: Optional[np.ndarray] = None,
        state0: Optional[np.ndarray] = None,
        t0: int = 0,
        tmax: int = 16,
        interval: int = 1,
        time_space: np.ndarray = np.arange(1, 16, 1),
        pooling_method: str = "no_cls",
        metric: str = "pairwise_cosine",
        grid_size: int = 80,
        initial_infected: int = 5,
        radius: int = 5) -> None:

        if observational_data is None:
            raise ValueError("observational_data must be provided for SpatialSIR3D.")
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

    def simulate(self, parameters: np.ndarray) -> tuple[np.ndarray, int]:
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
    
    def sample_priors(self) -> np.ndarray:
        # Sample from the prior distribution
        priors = np.random.uniform(self.lower_bounds, self.upper_bounds, self.num_parameters)
        return priors
            
    def calculate_prior_log_prob(self, parameters: np.ndarray) -> float:
        # Calculate the prior log probability of the parameters
        # This must match the prior distribution used in sampling
        log_probabilities = uniform.logpdf(parameters, loc=self.lower_bounds, scale=self.upper_bounds - self.lower_bounds) 
        return np.sum(log_probabilities)

    def labels2map(self, y: np.ndarray) -> np.ndarray:
        susceptible = (y == 0)
        infected = (y == 1)
        resistant = (y == 2)

        y_onehot = np.stack([susceptible, infected, resistant], axis=1)  # Shape: (3, H, W)

        return y_onehot
    
    def preprocess(self, x: np.ndarray) -> np.ndarray:
        # add a channel dimension at the beginning in numpy
        if x.shape[0] == 15:
            x = x.transpose(1, 0, 2, 3)

        if x.ndim == 4:
            x = np.expand_dims(x, axis=0)

        return x
