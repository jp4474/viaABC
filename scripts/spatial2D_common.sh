#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PROJECT_ROOT
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

default_spatial2d_storage_root() {
  local cluster_root="/insomnia001/depts/iicd/users/${USER}/viaABC"

  if [[ -d "/insomnia001/depts/iicd/users/${USER}" ]] || [[ -d "/insomnia001/depts/iicd" ]]; then
    printf '%s\n' "${cluster_root}"
    return
  fi

  printf '%s\n' "${PROJECT_ROOT}"
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

activate_env() {
  local venv_activate="${VENV_ACTIVATE:-}"
  local conda_env_name="${CONDA_ENV_NAME:-}"

  if [[ -n "${venv_activate}" ]]; then
    [[ -f "${venv_activate}" ]] || die "VENV_ACTIVATE points to a missing file: ${venv_activate}"
    # shellcheck disable=SC1090
    source "${venv_activate}"
    return
  fi

  if [[ -n "${conda_env_name}" ]]; then
    command -v conda >/dev/null 2>&1 || die "CONDA_ENV_NAME is set but conda is unavailable."
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${conda_env_name}"
    return
  fi

  if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.venv/bin/activate"
    return
  fi

  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    return
  fi

  log "No environment auto-activated. Assuming the Slurm job environment is already prepared."
}

assert_python_stack() {
  command -v python >/dev/null 2>&1 || die "python is not available in PATH."

  python - <<'PY'
required = ["torch", "lightning", "hydra", "rootutils", "numpy", "Cython", "pybind11"]

try:
    import importlib.util as importlib_util
    find_spec = importlib_util.find_spec
except Exception:
    from pkgutil import find_loader

    def find_spec(name):
        return find_loader(name)

missing = [name for name in required if find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {', '.join(missing)}")
print("Python stack check passed.")
PY
}

assert_spatial2d_data() {
  local spatial2d_data_dir="${SPATIAL2D_SOURCE_DATA_DIR:-${SPATIAL2D_DATA_DIR:-${PROJECT_ROOT}/data/spatial2D}}"
  [[ -d "${spatial2d_data_dir}" ]] || die "Spatial2D data directory not found: ${spatial2d_data_dir}"

  local required_files=(
    "initial_grid1_cpp.txt"
    "initial_grid2_cpp.txt"
    "initial_grid3_cpp.txt"
    "initial_grid4_cpp.txt"
  )

  local name
  for name in "${required_files[@]}"; do
    [[ -f "${spatial2d_data_dir}/${name}" ]] || die "Missing required spatial2D input: ${spatial2d_data_dir}/${name}"
  done
}

ensure_spatial2d_extension() {
  if python - <<'PY'
from src.viaABC.spatial2D import GridCore
raise SystemExit(0 if GridCore is not None else 1)
PY
  then
    log "Spatial2D extension already importable."
    return
  fi

  log "Spatial2D extension missing. Building in place."
  (
    cd "${PROJECT_ROOT}"
    python src/viaABC/spatial2D/setup.py build_ext --inplace
  )

  python - <<'PY'
from src.viaABC.spatial2D import GridCore
if GridCore is None:
    raise SystemExit("Spatial2D extension build finished, but import still failed.")
print("Spatial2D extension build check passed.")
PY
}

assert_gpu_memory() {
  local min_gpu_mem_mib="${MIN_GPU_MEM_MIB:-80000}"

  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable. This job requires a GPU node."

  local gpu_info
  gpu_info="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits)"
  [[ -n "${gpu_info}" ]] || die "No GPUs detected by nvidia-smi."

  log "Allocated GPU(s):"
  printf '%s\n' "${gpu_info}"

  local max_mem
  max_mem="$(printf '%s\n' "${gpu_info}" | awk -F',' 'BEGIN{max=0} {gsub(/ /,"",$3); if ($3+0 > max) max=$3+0} END{print max}')"
  [[ -n "${max_mem}" ]] || die "Failed to parse GPU memory from nvidia-smi."

  if (( max_mem < min_gpu_mem_mib )); then
    die "Largest visible GPU has ${max_mem} MiB, but spatial2D training requires at least ${min_gpu_mem_mib} MiB. Submit to an 80GB GPU partition or add the correct cluster-specific constraint."
  fi
}

print_runtime_context() {
  log "Project root: ${PROJECT_ROOT}"
  log "Working directory: $(pwd)"
  log "Python: $(command -v python)"
  if [[ -n "${SPATIAL2D_SOURCE_DATA_DIR:-}" ]]; then
    log "Spatial2D source data dir: ${SPATIAL2D_SOURCE_DATA_DIR}"
  fi
  if [[ -n "${SPATIAL2D_DATA_DIR:-}" ]]; then
    log "Spatial2D generated data dir: ${SPATIAL2D_DATA_DIR}"
  fi
  if [[ -n "${TRAIN_RUN_DIR:-}" ]]; then
    log "Training run dir: ${TRAIN_RUN_DIR}"
  fi
  python --version
}
