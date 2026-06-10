#!/usr/bin/env python
"""Save a Spatial2D observation reconstruction figure for one sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf


rootutils.setup_root(Path(__file__).resolve().parents[1], indicator=".project-root", pythonpath=True)

from src.inference import load_model_and_transform, load_training_config
from src.viaABC.systems import Spatial2D


STATE_NAMES = {
    0: "red",
    1: "yellow",
    2: "blue",
    3: "background",
    4: "green",
    5: "hotspot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-substr", default="last")
    parser.add_argument("--sample-id", default="sample_2")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def spatial2d_cmap() -> ListedColormap:
    colors = np.array(
        [
            [90, 15, 10],
            [120, 120, 10],
            [10, 10, 90],
            [0, 0, 0],
            [10, 120, 10],
            [0, 255, 0],
        ],
        dtype=np.float32,
    )
    return ListedColormap(colors / 255.0)


def system_kwargs_from_train_cfg(train_cfg, sample_id: str) -> dict[str, object]:
    if "system" in train_cfg:
        system_cfg = OmegaConf.to_container(train_cfg.system, resolve=True)
    else:
        system_cfg = {}

    kwargs = {
        "num_parameters": system_cfg.get("num_parameters", 2),
        "pooling_method": system_cfg.get("pooling_method", "no_cls"),
        "metric": system_cfg.get("metric", "pairwise_cosine"),
        "use_time_series": system_cfg.get("use_time_series", True),
        "time_space": system_cfg.get("time_space"),
        "num_frames": system_cfg.get("num_frames"),
        "sample_id": sample_id,
    }

    model_cfg = train_cfg.get("model", {}).get("net", {})
    if kwargs["num_frames"] is None:
        kwargs["num_frames"] = model_cfg.get("num_frames")
    if kwargs["time_space"] is None and kwargs["num_frames"] == 2:
        kwargs["time_space"] = [0, 24]

    return kwargs


def reconstruct(model, system: Spatial2D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_grid = system._observation_grids[0]
    obs_input = system._observation_input()
    if getattr(system, "_multiple_samples", False):
        obs_input = obs_input[0]

    model_input = system.preprocess(obs_input)
    tensor = torch.from_numpy(model_input).to(model.device).unsqueeze(0).float()

    with torch.inference_mode():
        pred_tokens = model.model.forward(tensor, mask_ratio=0.0)[-1]
        recon = model.model.unpatchify(pred_tokens).detach().cpu().numpy()[0].argmax(axis=0)

    initial_grid = system._initial_grids[0]
    obs_plot = obs_grid[-1] if obs_grid.ndim == 3 else obs_grid
    recon_plot = recon[-1] if recon.ndim == 3 else recon
    return initial_grid, obs_plot, recon_plot


def save_figure(
    initial_grid: np.ndarray,
    obs_grid: np.ndarray,
    recon_grid: np.ndarray,
    sample_id: str,
    checkpoint: str,
    output_path: Path,
) -> None:
    cmap = spatial2d_cmap()
    mismatch = obs_grid != recon_grid
    match_fraction = float((~mismatch).mean())

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    panels = [
        (initial_grid, "Initial"),
        (obs_grid, "Observation"),
        (recon_grid, "Reconstruction"),
        (mismatch.astype(np.uint8), f"Mismatch ({1.0 - match_fraction:.3%})"),
    ]

    for ax, (grid, title) in zip(axes, panels):
        if title.startswith("Mismatch"):
            ax.imshow(grid, cmap="magma", vmin=0, vmax=1)
        else:
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=5)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{sample_id} reconstruction | checkpoint={checkpoint} | pixel match={match_fraction:.4f}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create(
        {
            "run_folder_path": str(args.run_dir),
            "checkpoint_substr": args.checkpoint_substr,
        }
    )
    train_cfg = load_training_config(cfg)
    model, transform = load_model_and_transform(cfg, train_cfg)

    requested_device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model.to(requested_device)
    model.eval()

    system = Spatial2D(
        model=model,
        transform=transform,
        **system_kwargs_from_train_cfg(train_cfg, args.sample_id),
    )

    initial_grid, obs_grid, recon_grid = reconstruct(model, system)
    image_path = args.output_dir / f"{args.sample_id}_reconstruction.png"
    save_figure(initial_grid, obs_grid, recon_grid, args.sample_id, args.checkpoint_substr, image_path)

    metadata = {
        "sample_id": args.sample_id,
        "run_dir": str(args.run_dir),
        "checkpoint_substr": args.checkpoint_substr,
        "output_image": str(image_path),
        "pixel_match_fraction": float((obs_grid == recon_grid).mean()),
        "observation_counts": {
            STATE_NAMES[idx]: int((obs_grid == idx).sum()) for idx in sorted(STATE_NAMES)
        },
        "reconstruction_counts": {
            STATE_NAMES[idx]: int((recon_grid == idx).sum()) for idx in sorted(STATE_NAMES)
        },
        "sample_source_info": getattr(system, "_sample_source_info", []),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
