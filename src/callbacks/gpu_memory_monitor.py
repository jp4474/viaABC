from typing import Dict

import torch
from lightning import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class GPUMemoryMonitor(Callback):
    """Log CUDA memory usage for single-GPU and DDP training.

    In distributed runs, metrics from all ranks are gathered and logged by rank 0
    so rank-zero-only experiment loggers still receive one value per GPU.
    """

    def __init__(
        self,
        every_n_train_steps: int = 100,
        log_peak: bool = True,
        reset_peak_stats: bool = False,
        synchronize: bool = False,
    ) -> None:
        if every_n_train_steps <= 0:
            raise ValueError("every_n_train_steps must be a positive integer.")

        self.every_n_train_steps = every_n_train_steps
        self.log_peak = log_peak
        self.reset_peak_stats = reset_peak_stats
        self.synchronize = synchronize

    def on_train_start(self, trainer, pl_module) -> None:
        if self.reset_peak_stats and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not torch.cuda.is_available():
            return
        if trainer.global_step == 0 or trainer.global_step % self.every_n_train_steps != 0:
            return

        metrics = self._collect_memory_metrics(trainer, pl_module)
        if metrics:
            self._log_metrics(trainer, metrics)

        if self.reset_peak_stats:
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def _collect_memory_metrics(self, trainer, pl_module) -> Dict[str, float]:
        device = pl_module.device
        if device.type != "cuda":
            return {}

        if self.synchronize:
            torch.cuda.synchronize(device)

        local_stats = self._current_device_stats(trainer, device)
        all_stats = self._gather_stats(trainer, local_stats, device)

        metrics: Dict[str, float] = {}
        for stats in all_stats:
            rank = int(stats[0].item())
            device_index = int(stats[1].item())
            prefix = f"gpu_memory/rank_{rank}/cuda_{device_index}"
            metrics[f"{prefix}/allocated_gb"] = stats[2].item()
            metrics[f"{prefix}/reserved_gb"] = stats[3].item()
            if self.log_peak:
                metrics[f"{prefix}/max_allocated_gb"] = stats[4].item()

        return metrics

    def _current_device_stats(self, trainer, device: torch.device) -> torch.Tensor:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        values = [
            float(getattr(trainer, "global_rank", 0)),
            float(device_index),
            self._bytes_to_gib(torch.cuda.memory_allocated(device)),
            self._bytes_to_gib(torch.cuda.memory_reserved(device)),
            self._bytes_to_gib(torch.cuda.max_memory_allocated(device)),
        ]
        return torch.tensor(values, device=device, dtype=torch.float64)

    def _gather_stats(self, trainer, local_stats: torch.Tensor, device: torch.device):
        if getattr(trainer, "world_size", 1) <= 1:
            return [local_stats]
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return [local_stats]

        gathered = [torch.zeros_like(local_stats, device=device) for _ in range(trainer.world_size)]
        torch.distributed.all_gather(gathered, local_stats)
        return gathered

    @rank_zero_only
    def _log_metrics(self, trainer, metrics: Dict[str, float]) -> None:
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

    @staticmethod
    def _bytes_to_gib(value: int) -> float:
        return float(value) / 1024**3

