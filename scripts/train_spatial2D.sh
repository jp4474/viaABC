#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/spatial2D_common.sh"

cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-8}}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${MKL_NUM_THREADS:-8}}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${NUMEXPR_NUM_THREADS:-8}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SPATIAL2D_STORAGE_ROOT="${SPATIAL2D_STORAGE_ROOT:-$(default_spatial2d_storage_root)}"
SPATIAL2D_SOURCE_DATA_DIR="${SPATIAL2D_SOURCE_DATA_DIR:-${PROJECT_ROOT}/data/spatial2D}"
SPATIAL2D_DATA_DIR="${SPATIAL2D_DATA_DIR:-${SPATIAL2D_STORAGE_ROOT}/data/spatial2D}"
LOGGER_CONFIG="${LOGGER_CONFIG:-csv}"
TRAIN_RUN_STAMP="${TRAIN_RUN_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)_job${SLURM_JOB_ID:-manual}}"
TRAIN_RUN_BASE="${TRAIN_RUN_BASE:-${SPATIAL2D_STORAGE_ROOT}/run/train/spatial2D}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${TRAIN_RUN_BASE}/${TRAIN_RUN_STAMP}}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
ENABLE_NVIDIA_SMI_MONITOR="${ENABLE_NVIDIA_SMI_MONITOR:-true}"
NVIDIA_SMI_MONITOR_INTERVAL="${NVIDIA_SMI_MONITOR_INTERVAL:-10}"

activate_env
assert_python_stack
assert_spatial2d_data
assert_gpu_memory
ensure_spatial2d_extension
print_runtime_context

mkdir -p "${TRAIN_RUN_DIR}" "${TRAIN_RUN_BASE}"
reuse_spatial2d_generated_data

start_nvidia_smi_monitor() {
  [[ "${ENABLE_NVIDIA_SMI_MONITOR}" == "true" ]] || return 0
  [[ "${SLURM_PROCID:-0}" == "0" ]] || return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0

  local output_path="${TRAIN_RUN_DIR}/nvidia_smi_memory.csv"
  local interval="${NVIDIA_SMI_MONITOR_INTERVAL}"

  printf 'timestamp,index,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w\n' > "${output_path}"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits >> "${output_path}" 2>/dev/null || true
      sleep "${interval}"
    done
  ) &
  NVIDIA_SMI_MONITOR_PID=$!
  log "nvidia-smi monitor writing to ${output_path} every ${interval}s."
}

stop_nvidia_smi_monitor() {
  if [[ -n "${NVIDIA_SMI_MONITOR_PID:-}" ]]; then
    kill "${NVIDIA_SMI_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${NVIDIA_SMI_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}

trap stop_nvidia_smi_monitor EXIT INT TERM
start_nvidia_smi_monitor

cmd=(
  python src/train.py
  experiment=spatial2D
  logger="${LOGGER_CONFIG}"
  hydra.run.dir="${TRAIN_RUN_DIR}"
  trainer.default_root_dir="${TRAIN_RUN_DIR}"
  data.datamodule.data_dir="${SPATIAL2D_DATA_DIR}"
  data.datamodule.dataset.data_dir="${SPATIAL2D_DATA_DIR}"
  data.num_workers="${SPATIAL2D_TRAIN_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
)

if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # Intentional word splitting for Hydra override arguments supplied by the submitter.
  # shellcheck disable=SC2206
  extra_train_args=( ${EXTRA_TRAIN_ARGS} )
  cmd+=("${extra_train_args[@]}")
fi

log "Launching spatial2D training through the project Hydra interface."
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

log "spatial2D training completed successfully."
