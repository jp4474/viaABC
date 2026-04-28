from types import SimpleNamespace

import pytest
import torch

from src.callbacks.gpu_memory_monitor import GPUMemoryMonitor


class DummyLogger:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step=None):
        self.calls.append((metrics, step))


def _trainer(global_step=100, world_size=1, global_rank=0, logger=None):
    return SimpleNamespace(
        global_step=global_step,
        world_size=world_size,
        global_rank=global_rank,
        logger=logger,
    )


def _module(device="cuda:0"):
    return SimpleNamespace(device=torch.device(device))


def test_raises_for_non_positive_interval():
    with pytest.raises(ValueError, match="every_n_train_steps"):
        GPUMemoryMonitor(every_n_train_steps=0)


def test_skips_when_cuda_is_unavailable(monkeypatch):
    callback = GPUMemoryMonitor(every_n_train_steps=1)
    logger = DummyLogger()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    callback.on_train_batch_end(_trainer(logger=logger), _module(), None, None, 0)

    assert logger.calls == []


def test_respects_logging_interval(monkeypatch):
    callback = GPUMemoryMonitor(every_n_train_steps=10)
    logger = DummyLogger()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        callback,
        "_collect_memory_metrics",
        lambda trainer, pl_module: {"gpu_memory/rank_0/cuda_0/allocated_gb": 1.0},
    )

    callback.on_train_batch_end(_trainer(global_step=9, logger=logger), _module(), None, None, 0)
    callback.on_train_batch_end(_trainer(global_step=10, logger=logger), _module(), None, None, 0)

    assert len(logger.calls) == 1
    assert logger.calls[0] == ({"gpu_memory/rank_0/cuda_0/allocated_gb": 1.0}, 10)


def test_collects_rank_and_device_scoped_metric_names(monkeypatch):
    callback = GPUMemoryMonitor(every_n_train_steps=1, log_peak=True)
    local_stats = torch.tensor([0.0, 0.0, 1.5, 2.5, 3.5], dtype=torch.float64)
    other_rank_stats = torch.tensor([1.0, 1.0, 4.5, 5.5, 6.5], dtype=torch.float64)

    monkeypatch.setattr(callback, "_current_device_stats", lambda trainer, device: local_stats)
    monkeypatch.setattr(
        callback,
        "_gather_stats",
        lambda trainer, stats, device: [stats, other_rank_stats],
    )

    metrics = callback._collect_memory_metrics(_trainer(world_size=2), _module("cuda:0"))

    assert metrics == {
        "gpu_memory/rank_0/cuda_0/allocated_gb": 1.5,
        "gpu_memory/rank_0/cuda_0/reserved_gb": 2.5,
        "gpu_memory/rank_0/cuda_0/max_allocated_gb": 3.5,
        "gpu_memory/rank_1/cuda_1/allocated_gb": 4.5,
        "gpu_memory/rank_1/cuda_1/reserved_gb": 5.5,
        "gpu_memory/rank_1/cuda_1/max_allocated_gb": 6.5,
    }


def test_peak_metric_can_be_disabled(monkeypatch):
    callback = GPUMemoryMonitor(every_n_train_steps=1, log_peak=False)
    local_stats = torch.tensor([0.0, 0.0, 1.5, 2.5, 3.5], dtype=torch.float64)

    monkeypatch.setattr(callback, "_current_device_stats", lambda trainer, device: local_stats)
    monkeypatch.setattr(callback, "_gather_stats", lambda trainer, stats, device: [stats])

    metrics = callback._collect_memory_metrics(_trainer(), _module("cuda:0"))

    assert metrics == {
        "gpu_memory/rank_0/cuda_0/allocated_gb": 1.5,
        "gpu_memory/rank_0/cuda_0/reserved_gb": 2.5,
    }


def test_gather_stats_uses_distributed_all_gather(monkeypatch):
    callback = GPUMemoryMonitor(every_n_train_steps=1)
    local_stats = torch.tensor([1.0, 0.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    peer_stats = torch.tensor([0.0, 0.0, 5.0, 6.0, 7.0], dtype=torch.float64)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_gather(gathered, stats):
        gathered[0].copy_(peer_stats)
        gathered[1].copy_(stats)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    gathered = callback._gather_stats(
        _trainer(world_size=2),
        local_stats,
        torch.device("cpu"),
    )

    assert torch.equal(gathered[0], peer_stats)
    assert torch.equal(gathered[1], local_stats)


def test_reset_peak_stats_on_train_start(monkeypatch):
    callback = GPUMemoryMonitor(reset_peak_stats=True)
    calls = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: calls.append(device))

    callback.on_train_start(_trainer(), _module("cuda:0"))

    assert calls == [torch.device("cuda:0")]

