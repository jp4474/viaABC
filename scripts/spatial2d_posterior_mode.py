#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import rootutils
from PIL import Image
from tqdm.auto import tqdm


rootutils.setup_root(Path.cwd(), indicator=".project-root", pythonpath=True)

from src.viaABC.systems import Spatial2D  # noqa: E402


N_STATES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Spatial2D posterior predictive mode grids from a completed "
            "viaABC inference run."
        )
    )
    parser.add_argument(
        "--abc-generations-path",
        type=Path,
        default=Path(
            "/insomnia001/depts/iicd/users/kz2537/viaABC/run/train/spatial2D/"
            "2026-06-03_11-34-14_bs10_acc2_nw2/inference_output/"
            "2026-06-04_12-42-01_56900436/abc_generations.npy"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/insomnia001/depts/iicd/users/kz2537/viaABC/run/train/spatial2D/"
            "2026-06-03_11-34-14_bs10_acc2_nw2/posterior_mode"
        ),
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=["sample_1", "sample_2", "sample_3", "sample_4"],
    )
    parser.add_argument("--num-posterior-samples", type=int, default=100)
    parser.add_argument("--simulations-per-parameter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10216370)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even when DONE and posterior_mode_results.npz already exist.",
    )
    return parser.parse_args()


def final_frame(labels: np.ndarray) -> np.ndarray:
    return labels[-1] if labels.ndim == 3 else labels


def raw_observed_grid(system: Spatial2D) -> np.ndarray:
    info = system._sample_source_info[0]
    image_path = info.get("observed_raw_image_path")
    if image_path is None:
        return system._observation_grids[0]

    grid = system.image_to_grid(np.array(Image.open(image_path)))
    height, width = system._initial_grids[0].shape
    return grid[:height, :width]


def split_indices(n_items: int, n_chunks: int) -> list[np.ndarray]:
    n_chunks = max(1, min(n_chunks, n_items))
    return [chunk for chunk in np.array_split(np.arange(n_items), n_chunks) if len(chunk)]


def simulate_count_chunk(
    sample_id: str,
    parameter_chunk: np.ndarray,
    simulations_per_parameter: int,
) -> np.ndarray:
    system = Spatial2D(
        model=None,
        transform=None,
        sample_id=sample_id,
        use_time_series=False,
    )
    height, width = system._initial_grids[0].shape
    counts = np.zeros((N_STATES, height, width), dtype=np.uint16)

    for parameters in parameter_chunk:
        for _ in range(simulations_per_parameter):
            simulated, status = system.simulate(parameters)
            if status != 0:
                raise RuntimeError(f"Simulation failed for parameters={parameters!r}")
            labels = final_frame(simulated).astype(np.uint8, copy=False)
            for state in range(N_STATES):
                counts[state] += labels == state

    return counts


def mode_simulated_result(
    sample_id: str,
    posterior_samples: np.ndarray,
    simulations_per_parameter: int,
    workers: int,
) -> np.ndarray:
    chunks = split_indices(len(posterior_samples), workers)
    counts = None

    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                simulate_count_chunk,
                sample_id,
                posterior_samples[chunk],
                simulations_per_parameter,
            )
            for chunk in chunks
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{sample_id} chunks",
        ):
            chunk_counts = future.result()
            if counts is None:
                counts = chunk_counts
            else:
                counts += chunk_counts

    if counts is None:
        raise RuntimeError(f"No simulations were run for {sample_id}.")

    return counts.argmax(axis=0).astype(np.uint8)


def build_static_grids(sample_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    initial_grids = []
    observed_grids = []
    for sample_id in sample_ids:
        system = Spatial2D(
            model=None,
            transform=None,
            sample_id=sample_id,
            use_time_series=False,
        )
        initial_grids.append(system._initial_grids[0].astype(np.uint8, copy=False))
        observed_grids.append(raw_observed_grid(system).astype(np.uint8, copy=False))

    return np.stack(initial_grids, axis=0), np.stack(observed_grids, axis=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "posterior_mode_results.npz"
    manifest_path = args.output_dir / "manifest.json"
    done_path = args.output_dir / "DONE"

    if done_path.exists() and results_path.exists() and not args.overwrite:
        print(f"Existing completed result found: {results_path}")
        return

    abc_generations = np.load(args.abc_generations_path, allow_pickle=True)
    posterior = abc_generations[-1]
    particles = np.asarray(posterior["particles"], dtype=np.float64)
    weights = np.asarray(posterior["weights"], dtype=np.float64)
    weights = weights / weights.sum()

    rng = np.random.default_rng(args.seed)
    posterior_indices = rng.choice(
        len(particles),
        size=args.num_posterior_samples,
        replace=True,
        p=weights,
    )
    posterior_samples = particles[posterior_indices]

    sample_ids = list(args.sample_ids)
    initial_grids, observed_grids = build_static_grids(sample_ids)
    simulated_results = []
    workers_per_sample = max(1, args.workers)

    for sample_id in sample_ids:
        simulated_results.append(
            mode_simulated_result(
                sample_id=sample_id,
                posterior_samples=posterior_samples,
                simulations_per_parameter=args.simulations_per_parameter,
                workers=workers_per_sample,
            )
        )

    tmp_results_path = results_path.with_suffix(".npz.tmp")
    with tmp_results_path.open("wb") as f:
        np.savez_compressed(
            f,
            sample_ids=np.asarray(sample_ids),
            initial_grids=initial_grids,
            observed_grids=observed_grids,
            simulated_results=np.stack(simulated_results, axis=0),
            posterior_samples=posterior_samples,
            posterior_indices=posterior_indices,
            posterior_weights=weights,
            abc_generations_path=str(args.abc_generations_path),
        )
    tmp_results_path.replace(results_path)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "abc_generations_path": str(args.abc_generations_path),
        "results_path": str(results_path),
        "sample_ids": sample_ids,
        "num_posterior_samples": args.num_posterior_samples,
        "simulations_per_parameter": args.simulations_per_parameter,
        "seed": args.seed,
        "workers": args.workers,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    done_path.write_text(datetime.now().isoformat(timespec="seconds") + "\n")
    print(f"Saved completed posterior mode results: {results_path}")


if __name__ == "__main__":
    main()
