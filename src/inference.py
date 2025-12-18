import logging
from pathlib import Path

import hydra
import rootutils
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import namer

# -----------------------------------------------------------------------------
# Setup project root
# -----------------------------------------------------------------------------
rootutils.setup_root(Path.cwd(), indicator=".project-root", pythonpath=True)

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
from src.viaABC.systems import *
from src.models.lightning_module import PreTrainLightning

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Global logging setup (Hydra-safe, propagation-safe)
# -----------------------------------------------------------------------------
def setup_file_logging(save_dir: Path) -> None:
    log_path = save_dir / "inference.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing file handlers (Hydra may add its own)
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] - %(message)s"
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)

    # Ensure child loggers propagate
    logging.captureWarnings(True)

    log.info(f"File logging initialized at: {log_path}")


# -----------------------------------------------------------------------------
# Model + Transform Loader
# -----------------------------------------------------------------------------
def load_model_and_transform(cfg: DictConfig):
    """
    Load the model and transform EXACTLY as defined during training
    from `.hydra/config.yaml`, then restore weights from checkpoint.
    """

    run_dir = Path(cfg.run_folder_path)

    # -------------------------------------------------------------------------
    # 1. Load training Hydra config
    # -------------------------------------------------------------------------
    train_cfg_path = run_dir / ".hydra" / "config.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Training Hydra config not found: {train_cfg_path}")

    train_cfg = OmegaConf.load(train_cfg_path)

    if "model" not in train_cfg:
        raise KeyError("`model` not found in training config")

    model_cfg = train_cfg.model
    transform_cfg = model_cfg.get("transform", None)

    log.info("Loaded model + transform config from training run")

    # -------------------------------------------------------------------------
    # 2. Locate checkpoint
    # -------------------------------------------------------------------------
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Find checkpoints containing the substring
    matches = sorted(
        ckpt_dir.glob(f"*{cfg.checkpoint_substr}*.ckpt"),
        key=lambda p: p.stat().st_mtime,  # newest last
    )

    if not matches:
        raise FileNotFoundError(
            f"No checkpoint containing '{cfg.checkpoint_substr}' found in {ckpt_dir}"
        )

    # Pick the newest matching checkpoint
    ckpt_path = matches[-1]

    log.info(f"Loading checkpoint: {ckpt_path}")

    # -------------------------------------------------------------------------
    # 3. Instantiate model
    # -------------------------------------------------------------------------
    model: PreTrainLightning = instantiate(model_cfg)

    # -------------------------------------------------------------------------
    # 4. Load checkpoint weights (Loading lightning module)
    # -------------------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    fixed_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        fixed_state[k] = v

    missing, unexpected = model.model.load_state_dict(fixed_state, strict=False)

    if missing:
        log.warning(f"Missing keys (non-strict load): {missing}")
    if unexpected:
        log.warning(f"Unexpected keys (non-strict load): {unexpected}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    log.info(f"Using device: {device}")
    model.eval()

    # -------------------------------------------------------------------------
    # 5. Instantiate transform (if present)
    # -------------------------------------------------------------------------
    transform = instantiate(transform_cfg) if transform_cfg is not None else None

    log.info(
        f"Model loaded successfully | Transform: "
        f"{transform.__class__.__name__ if transform else 'None'}"
    )

    return model, transform


# -----------------------------------------------------------------------------
# Inference Runner
# -----------------------------------------------------------------------------
def run_inference(cfg: DictConfig) -> None:
    """
    Run viaABC inference using a pretrained encoder and its
    training-time transform.
    """

    # -------------------------------------------------------------------------
    # 0. Create save dir + setup logging FIRST
    # -------------------------------------------------------------------------
    save_dir = Path(cfg.folder_name) / namer.generate()
    save_dir.mkdir(parents=True, exist_ok=True)

    setup_file_logging(save_dir)

    log.info("Starting inference run")

    # -------------------------------------------------------------------------
    # 1. Load pretrained model + transform
    # -------------------------------------------------------------------------
    model, transform = load_model_and_transform(cfg)

    # -------------------------------------------------------------------------
    # 2. Instantiate viaABC system
    # -------------------------------------------------------------------------
    kwargs = {"model": model}

    if transform is not None:
        kwargs["transform"] = transform

    system: BaseSystem = instantiate(cfg.system, **kwargs)


    log.info(f"viaABC System initialized: {system.__class__.__name__}")
    log.info(f"Metric: {system.metric}")

    # -------------------------------------------------------------------------
    # 3. Run viaABC (ALL logs captured)
    # -------------------------------------------------------------------------
    log.info(
        f"Starting viaABC | particles={cfg.abc.num_particles}, "
        f"k={cfg.abc.k}, q={cfg.abc.q_threshold}"
    )

    abc_results = system.run(
        num_particles=cfg.abc.num_particles,
        k=cfg.abc.k,
        q_threshold=cfg.abc.q_threshold,
        max_generations=cfg.abc.max_generations,
    )

    # -------------------------------------------------------------------------
    # 4. Save results
    # -------------------------------------------------------------------------
    generations_path = save_dir / "abc_generations.npy"
    np.save(generations_path, system.generations, allow_pickle=True)

    log.info(f"viaABC generations saved to: {generations_path}")
    log.info("Inference completed successfully")


# -----------------------------------------------------------------------------
# Hydra Entry Point
# -----------------------------------------------------------------------------
@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="inference.yaml",
)
def main(cfg: DictConfig) -> None:
    run_inference(cfg)


if __name__ == "__main__":
    main()
