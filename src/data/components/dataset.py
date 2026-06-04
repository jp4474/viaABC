# from sklearn.preprocessing import MinMaxScaler
import os
import shutil
import zipfile
from abc import ABC, abstractmethod
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path
import fcntl

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Literal

MMAP_CACHE_THRESHOLD_BYTES = 1 << 30


class _IndexedAccessor:
    """Array-like proxy so existing dataset subclasses can keep using self.simulations[idx]."""

    def __init__(self, storage: "BaseArrayStorage", kind: str):
        self.storage = storage
        self.kind = kind

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, idx):
        if self.kind == "simulation":
            return self.storage.simulation_at(idx)
        return self.storage.param_at(idx)


class BaseArrayStorage(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def simulation_at(self, idx):
        raise NotImplementedError

    @abstractmethod
    def param_at(self, idx):
        raise NotImplementedError


class InMemoryArrayStorage(BaseArrayStorage):
    def __init__(self, simulations, params):
        self._simulations = simulations
        self._params = params

    def __len__(self):
        return len(self._simulations)

    def simulation_at(self, idx):
        return self._simulations[idx]

    def param_at(self, idx):
        return self._params[idx]


class MemmapArrayStorage(BaseArrayStorage):
    # Back large arrays with npy memmaps so we can index samples without loading
    # the whole dataset into host RAM.
    def __init__(self, simulation_path: str, param_path: str):
        self._simulations = np.load(simulation_path, mmap_mode="r")
        self._params = np.load(param_path, mmap_mode="r")

    def __len__(self):
        return len(self._simulations)

    def simulation_at(self, idx):
        return self._simulations[idx]

    def param_at(self, idx):
        return self._params[idx]


class ShardedArrayStorage(BaseArrayStorage):
    # Read pre-generated shard files lazily and resolve a global dataset index
    # to the corresponding shard/local offset on demand.
    def __init__(self, simulation_paths: list[str], param_paths: list[str]):
        if len(simulation_paths) != len(param_paths):
            raise ValueError("Simulation shard count must match parameter shard count.")

        self._simulation_shards = [np.load(path, mmap_mode="r") for path in simulation_paths]
        self._param_shards = [np.load(path, mmap_mode="r") for path in param_paths]
        self._offsets = []

        total = 0
        for sim_shard, param_shard in zip(self._simulation_shards, self._param_shards):
            if len(sim_shard) != len(param_shard):
                raise ValueError("Simulation and parameter shard lengths must match.")
            total += len(sim_shard)
            self._offsets.append(total)

    def __len__(self):
        return self._offsets[-1] if self._offsets else 0

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        shard_idx = bisect_right(self._offsets, idx)
        shard_start = 0 if shard_idx == 0 else self._offsets[shard_idx - 1]
        local_idx = idx - shard_start
        return shard_idx, local_idx

    def simulation_at(self, idx):
        shard_idx, local_idx = self._locate(idx)
        return self._simulation_shards[shard_idx][local_idx]

    def param_at(self, idx):
        shard_idx, local_idx = self._locate(idx)
        return self._param_shards[shard_idx][local_idx]

class BaseNumpyDataset(Dataset):
    def __init__(self, data_dir, prefix="train", transform=None):
        self.data_dir = data_dir
        self.prefix = prefix
        self.transform = transform
        self.data = None

        storage = self._build_storage()
        self.simulations = _IndexedAccessor(storage, "simulation")
        self.params = _IndexedAccessor(storage, "param")

    def _build_storage(self) -> BaseArrayStorage:
        shard_storage = self._build_sharded_storage()
        if shard_storage is not None:
            return shard_storage

        npz_file = Path(self.data_dir) / f"{self.prefix}_data.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {npz_file}")

        if npz_file.stat().st_size >= MMAP_CACHE_THRESHOLD_BYTES:
            # Large npz files are zip archives, which are awkward to random-read
            # efficiently. Materialize them once into sidecar npy files, then
            # reopen those arrays as memmaps for cheap indexed access.
            simulation_path, param_path = self._materialize_npz_members(npz_file)
            return MemmapArrayStorage(simulation_path, param_path)

        self.data = np.load(npz_file, allow_pickle=True)
        return InMemoryArrayStorage(self.data["simulations"], self.data["params"])

    def _build_sharded_storage(self) -> BaseArrayStorage | None:
        data_dir = Path(self.data_dir)
        simulation_paths = sorted(str(path) for path in data_dir.glob(f".{self.prefix}_simulations_*.npy"))
        param_paths = sorted(str(path) for path in data_dir.glob(f".{self.prefix}_params_*.npy"))

        if not simulation_paths and not param_paths:
            return None

        # Prefer shards when they are present so multi-process jobs can start
        # reading immediately without waiting for a monolithic npz cache step.
        return ShardedArrayStorage(simulation_paths, param_paths)

    def _materialize_npz_members(self, npz_file: Path) -> tuple[str, str]:
        cache_dir = npz_file.parent / f".{npz_file.stem}_cache"
        simulation_path = cache_dir / "simulations.npy"
        param_path = cache_dir / "params.npy"
        lock_path = cache_dir / ".cache.lock"

        if self._cache_matches_npz(npz_file, simulation_path, param_path):
            return str(simulation_path), str(param_path)

        cache_dir.mkdir(parents=True, exist_ok=True)
        with self._cache_lock(lock_path):
            # Another process may have completed the cache while we were waiting
            # on the lock, so re-check inside the critical section.
            if self._cache_matches_npz(npz_file, simulation_path, param_path):
                return str(simulation_path), str(param_path)

            self._clear_stale_cache(simulation_path, param_path)
            self._extract_npz_member(npz_file, "simulations.npy", simulation_path)
            self._extract_npz_member(npz_file, "params.npy", param_path)

        return str(simulation_path), str(param_path)

    def _cache_matches_npz(self, npz_file: Path, simulation_path: Path, param_path: Path) -> bool:
        if not simulation_path.exists() or not param_path.exists():
            return False

        npz_mtime = npz_file.stat().st_mtime
        if simulation_path.stat().st_mtime < npz_mtime or param_path.stat().st_mtime < npz_mtime:
            return False

        try:
            expected = self._npz_member_headers(npz_file)
            actual = {
                "simulations.npy": self._npy_header(simulation_path),
                "params.npy": self._npy_header(param_path),
            }
        except Exception:
            return False

        return expected == actual

    def _npz_member_headers(self, npz_file: Path) -> dict[str, tuple[tuple[int, ...], bool, str]]:
        headers = {}
        with zipfile.ZipFile(npz_file) as archive:
            for name in ("simulations.npy", "params.npy"):
                with archive.open(name) as member:
                    headers[name] = self._read_npy_header(member)
        return headers

    def _npy_header(self, path: Path) -> tuple[tuple[int, ...], bool, str]:
        with open(path, "rb") as file:
            return self._read_npy_header(file)

    def _read_npy_header(self, file_obj) -> tuple[tuple[int, ...], bool, str]:
        version = np.lib.format.read_magic(file_obj)
        shape, fortran_order, dtype = np.lib.format._read_array_header(file_obj, version)
        return tuple(shape), bool(fortran_order), np.dtype(dtype).str

    def _clear_stale_cache(self, simulation_path: Path, param_path: Path) -> None:
        for path in (simulation_path, param_path):
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            if tmp_path.exists():
                os.remove(tmp_path)
            if path.exists():
                os.remove(path)

    def _extract_npz_member(self, npz_file: Path, member_name: str, target_path: Path) -> None:
        if target_path.exists():
            return

        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        if tmp_path.exists():
            # Clean up a stale partial file left by an interrupted previous run
            # before writing a new temp file.
            os.remove(tmp_path)

        with zipfile.ZipFile(npz_file) as archive:
            with archive.open(member_name) as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)

        # Atomic replace prevents readers from observing a half-written cache file.
        os.replace(tmp_path, target_path)

    @contextmanager
    def _cache_lock(self, lock_path: Path):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as lock_file:
            # Serialize first-time cache creation across concurrent processes.
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def __len__(self):
        return len(self.simulations)

    def _apply_transform(self, x):
        if self.transform is not None:
            return self.transform(x)
        return x
    
class LotkaVolterraDataset(BaseNumpyDataset):
    def __init__(self, data_dir, prefix="train", transform=None):
        super().__init__(data_dir, prefix, transform)

    def __getitem__(self, idx):
        x = self.simulations[idx]
        x = self._apply_transform(x)
        x = torch.as_tensor(x, dtype=torch.float32)
        return x
    
class Spatial2DDataset(BaseNumpyDataset):
    def __init__(self, data_dir, prefix="train", temporal=True, expected_num_frames=None):
        super().__init__(data_dir, prefix,)
        self.temporal = temporal
        self.expected_num_frames = expected_num_frames

    def _transform(self, x):
        x = np.asarray(x)
        if x.ndim == 2:
            if self.temporal and self.expected_num_frames is not None:
                raise ValueError(f"Expected {self.expected_num_frames} Spatial2D frames, got a single grid.")
            return np.eye(6, dtype=np.float32)[x].transpose(2, 0, 1)
        if x.ndim == 3:
            if self.expected_num_frames is not None and x.shape[0] != self.expected_num_frames:
                raise ValueError(f"Expected {self.expected_num_frames} Spatial2D frames, got {x.shape[0]}.")
            if not self.temporal:
                raise ValueError(f"Non-temporal Spatial2D data must be [H,W], got {x.shape}.")
            return np.eye(6, dtype=np.float32)[x].transpose(3, 0, 1, 2)
        raise ValueError(f"Unsupported Spatial2D simulation shape: {x.shape}")

    def __getitem__(self, idx):
        x = self.simulations[idx]
        x = self._transform(x)
        x = torch.as_tensor(x, dtype=torch.float32)
        return x
    
class SpatialSIRDataset(BaseNumpyDataset):
    def __init__(self, data_dir, prefix="train", transform=None):
        super().__init__(data_dir, prefix, transform)

    def __getitem__(self, idx):
        x = self.simulations[idx]
        x = self._apply_transform(x)
        x = torch.as_tensor(x, dtype=torch.float32)
        return x
