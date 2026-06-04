#!/usr/bin/env python
"""Evaluate sample-level Spatial2D terminal-color dominance in reconstructions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rootutils
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


rootutils.setup_root(Path(__file__).resolve().parents[1], indicator=".project-root", pythonpath=True)


TARGET_CLASSES = (1, 4, 5)
CLASS_NAMES = {1: "yellow", 4: "green", 5: "hot_spot", -1: "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-substr", default="last")
    parser.add_argument("--samples-per-class", default=200, type=int)
    parser.add_argument(
        "--target-scope",
        choices=("terminal", "transition"),
        default="terminal",
        help="Use terminal-frame majority, or changed-pixel transition majority.",
    )
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def dominance(labels: np.ndarray) -> int:
    if labels.size == 0:
        return -1
    counts = np.array([(labels == cls).sum() for cls in TARGET_CLASSES])
    if counts.max() == 0:
        return -1
    return TARGET_CLASSES[int(counts.argmax())]


def classify_terminal_sample(sample: np.ndarray) -> tuple[int, dict[str, int]]:
    final = sample[-1]
    dom = dominance(final.reshape(-1))
    counts = {CLASS_NAMES[cls]: int((final == cls).sum()) for cls in TARGET_CLASSES}
    counts["pixels"] = int(final.size)
    return dom, counts


def classify_transition_sample(sample: np.ndarray) -> tuple[int, dict[str, int]]:
    initial = sample[0]
    final = sample[-1]
    transition_mask = initial != final
    changed_final = final[transition_mask]
    dom = dominance(changed_final)
    counts = {CLASS_NAMES[cls]: int((changed_final == cls).sum()) for cls in TARGET_CLASSES}
    counts["transition_pixels"] = int(transition_mask.sum())
    return dom, counts


def one_hot_spatial2d(sample: np.ndarray) -> torch.Tensor:
    x = np.eye(6, dtype=np.float32)[sample].transpose(3, 0, 1, 2)
    return torch.from_numpy(x).unsqueeze(0)


def load_model(run_dir: Path, checkpoint_substr: str, device: torch.device):
    train_cfg_path = run_dir / ".hydra" / "config.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_cfg_path}")
    train_cfg = OmegaConf.load(train_cfg_path)
    model = instantiate(train_cfg.model)

    ckpt_dir = run_dir / "checkpoints"
    matches = sorted(ckpt_dir.glob(f"*{checkpoint_substr}*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No checkpoint containing {checkpoint_substr!r} in {ckpt_dir}")
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
        print(f"WARNING missing keys: {missing}")
    if unexpected:
        print(f"WARNING unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model, ckpt_path


def choose_indices(labels: np.ndarray, samples_per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cls in TARGET_CLASSES:
        cls_indices = np.flatnonzero(labels == cls)
        if samples_per_class > 0 and len(cls_indices) > samples_per_class:
            cls_indices = rng.choice(cls_indices, size=samples_per_class, replace=False)
        selected.extend(int(i) for i in cls_indices)
    selected.sort()
    return selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    simulations_path = args.data_dir / ".train_data_cache" / "simulations.npy"
    if not simulations_path.exists():
        simulations_path = args.data_dir / "train_data.npz"

    if simulations_path.suffix == ".npy":
        simulations = np.load(simulations_path, mmap_mode="r")
    else:
        simulations = np.load(simulations_path, allow_pickle=True)["simulations"]

    target_labels = np.empty(len(simulations), dtype=np.int16)
    target_counts_by_class: dict[int, Counter] = defaultdict(Counter)
    per_sample_counts: list[dict[str, int]] = []
    for idx in range(len(simulations)):
        if args.target_scope == "terminal":
            dom, counts = classify_terminal_sample(simulations[idx])
        else:
            dom, counts = classify_transition_sample(simulations[idx])
        target_labels[idx] = dom
        target_counts_by_class[dom].update(counts)
        per_sample_counts.append(counts)
        if (idx + 1) % 1000 == 0:
            print(f"classified target dominance for {idx + 1}/{len(simulations)} samples", flush=True)

    selected_indices = choose_indices(target_labels, args.samples_per_class, args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model, ckpt_path = load_model(args.run_dir, args.checkpoint_substr, device)

    confusion_terminal: dict[str, Counter] = defaultdict(Counter)
    confusion_target_mask: dict[str, Counter] = defaultdict(Counter)
    confusion_pred_mask: dict[str, Counter] = defaultdict(Counter)
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for n, idx in enumerate(selected_indices, start=1):
            sample = simulations[idx]
            initial = sample[0]
            final = sample[-1]
            target_transition_mask = initial != final

            x = one_hot_spatial2d(sample).to(device)
            pred_tokens = model.model.forward(x, mask_ratio=0.0)[-1]
            pred = model.model.unpatchify(pred_tokens).detach().cpu().numpy()[0].argmax(axis=0)
            pred_final = pred[-1]

            target_dom = int(target_labels[idx])
            pred_dom_terminal = dominance(pred_final.reshape(-1))
            pred_dom_on_target_mask = dominance(pred_final[target_transition_mask])
            pred_transition_mask = initial != pred_final
            pred_dom_on_pred_mask = dominance(pred_final[pred_transition_mask])

            target_name = CLASS_NAMES[target_dom]
            pred_terminal_name = CLASS_NAMES[pred_dom_terminal]
            pred_target_mask_name = CLASS_NAMES[pred_dom_on_target_mask]
            pred_pred_mask_name = CLASS_NAMES[pred_dom_on_pred_mask]
            confusion_terminal[target_name][pred_terminal_name] += 1
            confusion_target_mask[target_name][pred_target_mask_name] += 1
            confusion_pred_mask[target_name][pred_pred_mask_name] += 1

            row = {
                "index": idx,
                "target_dominance": target_name,
                "pred_terminal_dominance": pred_terminal_name,
                "pred_dominance_on_target_transition_mask": pred_target_mask_name,
                "pred_dominance_on_pred_transition_mask": pred_pred_mask_name,
                "target_transition_pixels": int(target_transition_mask.sum()),
                "pred_transition_pixels": int(pred_transition_mask.sum()),
            }
            for cls in TARGET_CLASSES:
                row[f"pred_terminal_{CLASS_NAMES[cls]}_pixels"] = int((pred_final == cls).sum())
                row[f"pred_on_target_mask_{CLASS_NAMES[cls]}_pixels"] = int((pred_final[target_transition_mask] == cls).sum())
                row[f"pred_on_pred_mask_{CLASS_NAMES[cls]}_pixels"] = int((pred_final[pred_transition_mask] == cls).sum())
            rows.append(row)

            if n % 25 == 0:
                print(f"evaluated reconstruction dominance for {n}/{len(selected_indices)} samples", flush=True)

    summary = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(ckpt_path),
        "data_dir": str(args.data_dir),
        "target_scope": args.target_scope,
        "total_samples": int(len(simulations)),
        "target_dominance_counts": {
            CLASS_NAMES.get(int(cls), str(cls)): int((target_labels == cls).sum())
            for cls in sorted(set(target_labels.tolist()))
        },
        "selected_counts": {
            CLASS_NAMES[cls]: int(sum(1 for idx in selected_indices if target_labels[idx] == cls))
            for cls in TARGET_CLASSES
        },
        "confusion_pred_terminal": {
            key: dict(counter) for key, counter in confusion_terminal.items()
        },
        "confusion_pred_on_target_transition_mask": {
            key: dict(counter) for key, counter in confusion_target_mask.items()
        },
        "confusion_pred_on_pred_transition_mask": {
            key: dict(counter) for key, counter in confusion_pred_mask.items()
        },
    }

    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (args.output_dir / "per_sample.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
