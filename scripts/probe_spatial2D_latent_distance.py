#!/usr/bin/env python
"""Probe Spatial2D latent distances without running ABC.

This script diagnoses whether ABC distance plateaus are caused mainly by
shared-initial-frame dominance or by an insensitive latent metric.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rootutils
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


PROJECT_ROOT = rootutils.setup_root(
    Path(__file__).resolve().parents[1],
    indicator=".project-root",
    pythonpath=True,
)

from src.viaABC.metrics import l2_distance, pairwise_cosine  # noqa: E402
from src.viaABC.systems import Spatial2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-substr", default="last")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-theta", default=24, type=int)
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--pooling-method", default="no_cls")
    parser.add_argument("--metric", default="pairwise_cosine")
    parser.add_argument("--prior-low", default="0,0")
    parser.add_argument("--prior-high", default="1,1")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_vector(value: str) -> np.ndarray:
    parsed = np.array([float(part) for part in value.split(",")], dtype=np.float64)
    if parsed.shape != (2,):
        raise ValueError(f"Expected two comma-separated values, got {value!r}.")
    return parsed


def load_training_config(run_dir: Path):
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def load_model_and_transform(
    run_dir: Path,
    checkpoint_substr: str,
    device: torch.device,
):
    train_cfg = load_training_config(run_dir)
    model = instantiate(train_cfg.model)

    ckpt_dir = run_dir / "checkpoints"
    matches = sorted(
        ckpt_dir.glob(f"*{checkpoint_substr}*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint containing {checkpoint_substr!r} in {ckpt_dir}"
        )
    ckpt_path = matches[-1]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    fixed_state = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("model."):
            key = key[len("model.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        fixed_state[key] = value

    missing, unexpected = model.model.load_state_dict(fixed_state, strict=False)
    if missing:
        print(f"WARNING missing keys: {missing}", flush=True)
    if unexpected:
        print(f"WARNING unexpected keys: {unexpected}", flush=True)

    model.to(device)
    model.eval()
    transform_cfg = train_cfg.data.get("transform", None)
    transform = instantiate(transform_cfg) if transform_cfg is not None else None
    return train_cfg, model, transform, ckpt_path


@contextmanager
def use_training_observation_samples(train_cfg: Any):
    samples = OmegaConf.to_container(train_cfg.data.observation_samples, resolve=True)
    original = Spatial2D._load_spatial2d_samples
    Spatial2D._load_spatial2d_samples = staticmethod(lambda: samples)
    try:
        yield
    finally:
        Spatial2D._load_spatial2d_samples = original


def build_system(
    train_cfg: Any,
    model: torch.nn.Module,
    transform: Any,
    pooling_method: str,
    metric: str,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
) -> Spatial2D:
    system_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg.system, resolve=True))
    system_cfg.pooling_method = pooling_method
    system_cfg.metric = metric
    system_cfg.mu = prior_low.tolist()
    system_cfg.sigma = prior_high.tolist()

    with use_training_observation_samples(train_cfg):
        return instantiate(system_cfg, model=model, transform=transform)


def cosine_distance_tokens(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    x_norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    y_norm = y / (np.linalg.norm(y, axis=-1, keepdims=True) + eps)
    return float(1.0 - np.sum(x_norm * y_norm, axis=-1).mean())


def split_frame_tokens(system: Spatial2D, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    inner_model = system.model.model
    frames = int(inner_model.patch_embed.t_grid_size)
    grid = int(inner_model.patch_embed.grid_size)
    spatial_tokens = grid * grid
    expected_tokens = frames * spatial_tokens
    tokens = z
    if tokens.shape[1] == expected_tokens + 1:
        tokens = tokens[:, 1:, :]
    if tokens.shape[1] != expected_tokens:
        raise ValueError(
            f"Cannot split {tokens.shape[1]} tokens into "
            f"{frames} frames x {spatial_tokens} spatial tokens."
        )
    return tokens.reshape(tokens.shape[0], frames, spatial_tokens, tokens.shape[-1])


def latent_distance_breakdown(
    system: Spatial2D,
    obs_z: np.ndarray,
    sim_z: np.ndarray,
) -> dict[str, float]:
    obs_z = np.asarray(obs_z)
    sim_z = np.asarray(sim_z)
    obs_frames = split_frame_tokens(system, obs_z)
    sim_frames = split_frame_tokens(system, sim_z)
    return {
        "latent_pairwise_cosine": float(1.0 - pairwise_cosine(obs_z, sim_z)),
        "latent_token_cosine": cosine_distance_tokens(obs_z, sim_z),
        "latent_l2": float(l2_distance(obs_z, sim_z)),
        "latent_frame0_cosine": cosine_distance_tokens(obs_frames[:, 0], sim_frames[:, 0]),
        "latent_frame1_cosine": cosine_distance_tokens(obs_frames[:, 1], sim_frames[:, 1]),
    }


def raw_grid_metrics(
    initial: np.ndarray,
    observed: np.ndarray,
    simulated: np.ndarray,
) -> dict[str, float]:
    initial = np.asarray(initial)
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)
    obs_changed = observed != initial
    sim_changed = simulated != initial
    intersection = np.logical_and(obs_changed, sim_changed).sum()
    union = np.logical_or(obs_changed, sim_changed).sum()
    obs_hist = np.bincount(observed.reshape(-1), minlength=6) / observed.size
    sim_hist = np.bincount(simulated.reshape(-1), minlength=6) / simulated.size
    return {
        "raw_final_mismatch": float(np.mean(observed != simulated)),
        "raw_obs_changed_frac": float(np.mean(obs_changed)),
        "raw_sim_changed_frac": float(np.mean(sim_changed)),
        "raw_changed_iou": float(intersection / union) if union else math.nan,
        "raw_class_hist_l1": float(np.abs(obs_hist - sim_hist).sum()),
    }


def encode_label_pairs(system: Spatial2D, pairs: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return system.get_latent(system.preprocess(pairs))


def compare_one_pair(
    system: Spatial2D,
    sample_index: int,
    pair: np.ndarray,
    label: str,
    theta: np.ndarray | None = None,
) -> dict[str, object]:
    obs_z = system.encoded_observational_data[sample_index : sample_index + 1]
    sim_z = encode_label_pairs(system, pair[np.newaxis, ...])
    row: dict[str, object] = {
        "label": label,
        "theta_alpha": float(theta[0]) if theta is not None else math.nan,
        "theta_beta": float(theta[1]) if theta is not None else math.nan,
        "sample_index": sample_index,
    }
    row.update(latent_distance_breakdown(system, obs_z, sim_z))
    row.update(
        raw_grid_metrics(
            system._initial_grids[sample_index],
            system._observation_grids[sample_index],
            pair[-1],
        )
    )
    return row


def one_hot_temporal_pair(pair: np.ndarray) -> np.ndarray:
    return np.eye(6, dtype=np.float32)[pair].transpose(3, 0, 1, 2)


def compare_one_preprocessed_pair(
    system: Spatial2D,
    sample_index: int,
    preprocessed_pair: np.ndarray,
    label: str,
    simulated_final: np.ndarray | None = None,
) -> dict[str, object]:
    obs_z = system.encoded_observational_data[sample_index : sample_index + 1]
    with torch.inference_mode():
        sim_z = system.get_latent(preprocessed_pair[np.newaxis, ...])
    row: dict[str, object] = {
        "label": label,
        "theta_alpha": math.nan,
        "theta_beta": math.nan,
        "sample_index": sample_index,
    }
    row.update(latent_distance_breakdown(system, obs_z, sim_z))
    if simulated_final is None:
        row.update(
            {
                "raw_final_mismatch": math.nan,
                "raw_obs_changed_frac": math.nan,
                "raw_sim_changed_frac": math.nan,
                "raw_changed_iou": math.nan,
                "raw_class_hist_l1": math.nan,
            }
        )
    else:
        row.update(
            raw_grid_metrics(
                system._initial_grids[sample_index],
                system._observation_grids[sample_index],
                simulated_final,
            )
        )
    return row


def run_controls(system: Spatial2D, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    num_samples = system._initial_grids.shape[0]
    for i in range(num_samples):
        initial = system._initial_grids[i]
        observed = system._observation_grids[i]
        j = (i + 1) % num_samples
        base_pair = np.stack([initial, observed], axis=0)
        zero_initial = one_hot_temporal_pair(base_pair)
        zero_initial[:, 0] = 0.0
        zero_final = one_hot_temporal_pair(base_pair)
        zero_final[:, 1] = 0.0
        control_pairs = {
            "identity_obs_vs_obs": np.stack([initial, observed], axis=0),
            "null_final_obs_vs_initial": np.stack([initial, initial], axis=0),
            "fix_initial_vary_final": np.stack([initial, system._observation_grids[j]], axis=0),
            "fix_final_vary_initial": np.stack([system._initial_grids[j], observed], axis=0),
            "cross_initial_same_final": np.stack([system._initial_grids[j], observed], axis=0),
            "same_initial_cross_final": np.stack([initial, system._observation_grids[j]], axis=0),
            "same_initial_random_final": np.stack(
                [initial, rng.integers(0, 6, size=initial.shape, dtype=np.uint8)],
                axis=0,
            ),
        }
        for label, pair in control_pairs.items():
            rows.append(compare_one_pair(system, i, pair, label))
        rows.append(compare_one_preprocessed_pair(system, i, zero_initial, "mask_zero_initial"))
        rows.append(compare_one_preprocessed_pair(system, i, zero_final, "mask_zero_final"))
    return pd.DataFrame(rows)


def capture_last_encoder_qk(
    system: Spatial2D,
    preprocessed_pairs: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    inner_model = system.model.model
    attn = inner_model.blocks[-1].attn
    captured: dict[str, torch.Tensor] = {}

    def save_q(_module, _inputs, output):
        captured["q"] = output.detach()

    def save_k(_module, _inputs, output):
        captured["k"] = output.detach()

    q_handle = attn.q.register_forward_hook(save_q)
    k_handle = attn.k.register_forward_hook(save_k)
    try:
        with torch.inference_mode():
            x = torch.as_tensor(
                preprocessed_pairs,
                dtype=torch.float32,
                device=system.model.device,
            )
            system.model.get_latent(x, system.pooling_method)
    finally:
        q_handle.remove()
        k_handle.remove()

    q = captured["q"]
    k = captured["k"]
    batch_size, num_tokens, dim = q.shape
    num_heads = int(attn.num_heads)
    head_dim = dim // num_heads
    q = q.reshape(batch_size, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.reshape(batch_size, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
    return q, k


def attention_final_to_initial_summary(
    system: Spatial2D,
    chunk_size: int = 128,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    observation_pairs = system._observation_input()
    preprocessed = system.preprocess(observation_pairs)
    q, k = capture_last_encoder_qk(system, preprocessed)
    attn = system.model.model.blocks[-1].attn
    grid = int(system.model.model.patch_embed.grid_size)
    spatial_tokens = grid * grid
    total_tokens = q.shape[2]
    if total_tokens != spatial_tokens * 2:
        raise ValueError(
            f"Expected two temporal frames in attention tokens, got {total_tokens} tokens."
        )

    initial_key_slice = slice(0, spatial_tokens)
    final_key_slice = slice(spatial_tokens, spatial_tokens * 2)
    final_query_start = spatial_tokens
    rows: list[dict[str, float | int]] = []
    maps = {
        "final_to_initial_ratio": np.empty((q.shape[0], grid, grid), dtype=np.float32),
        "final_to_final_ratio": np.empty((q.shape[0], grid, grid), dtype=np.float32),
    }

    for sample_index in range(q.shape[0]):
        sample_initial_ratios = []
        sample_final_ratios = []
        for start in range(0, spatial_tokens, chunk_size):
            end = min(start + chunk_size, spatial_tokens)
            query_indices = slice(final_query_start + start, final_query_start + end)
            q_chunk = q[sample_index : sample_index + 1, :, query_indices, :]
            scores = torch.matmul(
                q_chunk,
                k[sample_index : sample_index + 1].transpose(-2, -1),
            ) * float(attn.scale)
            probs = torch.softmax(scores, dim=-1)
            init_ratio = probs[..., initial_key_slice].sum(dim=-1).mean(dim=1)
            final_ratio = probs[..., final_key_slice].sum(dim=-1).mean(dim=1)
            sample_initial_ratios.append(init_ratio.squeeze(0).detach().cpu().numpy())
            sample_final_ratios.append(final_ratio.squeeze(0).detach().cpu().numpy())

        initial_ratio_map = np.concatenate(sample_initial_ratios, axis=0).reshape(grid, grid)
        final_ratio_map = np.concatenate(sample_final_ratios, axis=0).reshape(grid, grid)
        maps["final_to_initial_ratio"][sample_index] = initial_ratio_map
        maps["final_to_final_ratio"][sample_index] = final_ratio_map
        rows.append(
            {
                "sample_index": sample_index,
                "final_queries_to_initial_keys_mean": float(initial_ratio_map.mean()),
                "final_queries_to_initial_keys_std": float(initial_ratio_map.std()),
                "final_queries_to_initial_keys_min": float(initial_ratio_map.min()),
                "final_queries_to_initial_keys_max": float(initial_ratio_map.max()),
                "final_queries_to_final_keys_mean": float(final_ratio_map.mean()),
                "final_queries_to_final_keys_std": float(final_ratio_map.std()),
                "final_queries_to_final_keys_min": float(final_ratio_map.min()),
                "final_queries_to_final_keys_max": float(final_ratio_map.max()),
            }
        )

    return pd.DataFrame(rows), maps


def run_theta_probe(
    system: Spatial2D,
    n_theta: int,
    seed: int,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random_thetas = rng.uniform(prior_low, prior_high, size=(n_theta, 2))
    anchor_thetas = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.25, 0.25],
            [0.5, 0.5],
            [0.75, 0.75],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    thetas = np.vstack([anchor_thetas, random_thetas])
    rows = []
    num_samples = system._initial_grids.shape[0]

    for theta_index, theta in enumerate(thetas):
        sims, status = system.simulate_for_inference(theta)
        if status != 0:
            print(f"theta {theta_index} failed: {theta}", flush=True)
            continue
        with torch.inference_mode():
            sim_z_all = system.get_latent(system.preprocess(sims))
        for sample_index in range(num_samples):
            obs_z = system.encoded_observational_data[
                sample_index : sample_index + 1
            ]
            sim_z = sim_z_all[sample_index : sample_index + 1]
            row: dict[str, object] = {
                "label": "theta_probe",
                "theta_index": theta_index,
                "theta_alpha": float(theta[0]),
                "theta_beta": float(theta[1]),
                "sample_index": sample_index,
            }
            row.update(latent_distance_breakdown(system, obs_z, sim_z))
            row.update(
                raw_grid_metrics(
                    system._initial_grids[sample_index],
                    system._observation_grids[sample_index],
                    sims[sample_index, -1],
                )
            )
            rows.append(row)

        if (theta_index + 1) % 5 == 0:
            print(f"processed {theta_index + 1}/{len(thetas)} theta", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def summarize(probe_df: pd.DataFrame, controls_df: pd.DataFrame) -> dict[str, Any]:
    summary_cols = [
        "latent_pairwise_cosine",
        "latent_frame0_cosine",
        "latent_frame1_cosine",
        "latent_l2",
        "raw_final_mismatch",
        "raw_changed_iou",
        "raw_class_hist_l1",
    ]
    per_theta = probe_df.groupby("theta_index", as_index=False).agg(
        {
            "theta_alpha": "first",
            "theta_beta": "first",
            "latent_pairwise_cosine": "mean",
            "latent_frame0_cosine": "mean",
            "latent_frame1_cosine": "mean",
            "latent_l2": "mean",
            "raw_final_mismatch": "mean",
            "raw_changed_iou": "mean",
            "raw_class_hist_l1": "mean",
        }
    )
    corr = probe_df[summary_cols].corr(method="spearman")
    control_summary = controls_df.groupby("label")[summary_cols].agg(
        ["mean", "std", "min", "max"]
    )
    describe = probe_df[summary_cols].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    return {
        "summary_cols": summary_cols,
        "per_theta": per_theta,
        "corr": corr,
        "control_summary": control_summary,
        "describe": describe,
    }


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    ckpt_path: Path,
    controls_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    summaries: dict[str, Any],
    attention_df: pd.DataFrame | None = None,
    attention_maps: dict[str, np.ndarray] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    controls_df.to_csv(output_dir / "controls.csv", index=False)
    probe_df.to_csv(output_dir / "theta_probe.csv", index=False)
    summaries["per_theta"].to_csv(output_dir / "theta_probe_aggregated.csv", index=False)
    summaries["corr"].to_csv(output_dir / "spearman_correlation.csv")
    summaries["control_summary"].to_csv(output_dir / "control_summary.csv")
    summaries["describe"].to_csv(output_dir / "theta_probe_describe.csv")
    if attention_df is not None:
        attention_df.to_csv(output_dir / "attention_summary.csv", index=False)
    if attention_maps is not None:
        np.savez_compressed(output_dir / "attention_patch_maps.npz", **attention_maps)

    manifest = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(ckpt_path),
        "n_theta_random": int(args.n_theta),
        "n_theta_total": int(summaries["per_theta"].shape[0]),
        "pooling_method": args.pooling_method,
        "metric": args.metric,
        "prior_low": args.prior_low,
        "prior_high": args.prior_high,
        "outputs": {
            "controls": str(output_dir / "controls.csv"),
            "control_summary": str(output_dir / "control_summary.csv"),
            "theta_probe": str(output_dir / "theta_probe.csv"),
            "theta_probe_aggregated": str(output_dir / "theta_probe_aggregated.csv"),
            "theta_probe_describe": str(output_dir / "theta_probe_describe.csv"),
            "spearman_correlation": str(output_dir / "spearman_correlation.csv"),
            "attention_summary": str(output_dir / "attention_summary.csv"),
            "attention_patch_maps": str(output_dir / "attention_patch_maps.npz"),
        },
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_low = parse_vector(args.prior_low)
    prior_high = parse_vector(args.prior_high)

    requested_device = args.device
    use_cuda = requested_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"project root: {PROJECT_ROOT}", flush=True)
    print(f"run dir: {args.run_dir}", flush=True)
    print(f"output dir: {args.output_dir}", flush=True)
    print(f"device: {device}", flush=True)

    train_cfg, model, transform, ckpt_path = load_model_and_transform(
        args.run_dir,
        args.checkpoint_substr,
        device,
    )
    system = build_system(
        train_cfg=train_cfg,
        model=model,
        transform=transform,
        pooling_method=args.pooling_method,
        metric=args.metric,
        prior_low=prior_low,
        prior_high=prior_high,
    )
    print(f"checkpoint: {ckpt_path}", flush=True)
    print(f"initial grids: {system._initial_grids.shape}", flush=True)
    print(f"observation grids: {system._observation_grids.shape}", flush=True)
    print(f"encoded observation: {system.encoded_observational_data.shape}", flush=True)

    print("running controls...", flush=True)
    controls_df = run_controls(system, args.seed)
    print("running theta probe...", flush=True)
    probe_df = run_theta_probe(system, args.n_theta, args.seed, prior_low, prior_high)
    print("running last-layer attention attribution...", flush=True)
    attention_df, attention_maps = attention_final_to_initial_summary(system)
    summaries = summarize(probe_df, controls_df)
    write_outputs(
        args.output_dir,
        args,
        ckpt_path,
        controls_df,
        probe_df,
        summaries,
        attention_df=attention_df,
        attention_maps=attention_maps,
    )

    print("control summary:", flush=True)
    print(summaries["control_summary"].to_string(), flush=True)
    print("spearman correlation:", flush=True)
    print(summaries["corr"].to_string(), flush=True)
    print("attention summary:", flush=True)
    print(attention_df.to_string(index=False), flush=True)
    print(f"wrote outputs to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
