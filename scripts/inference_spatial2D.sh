#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/spatial2D_common.sh"

cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-8}}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${MKL_NUM_THREADS:-8}}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${NUMEXPR_NUM_THREADS:-8}}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OPENBLAS_NUM_THREADS:-8}}"

SPATIAL2D_STORAGE_ROOT="${SPATIAL2D_STORAGE_ROOT:-$(default_spatial2d_storage_root)}"
SPATIAL2D_SOURCE_DATA_DIR="${SPATIAL2D_SOURCE_DATA_DIR:-${PROJECT_ROOT}/data/spatial2D}"
SPATIAL2D_DATA_DIR="${SPATIAL2D_DATA_DIR:-${SPATIAL2D_STORAGE_ROOT}/data/spatial2D}"
TRAIN_RUN_BASE="${TRAIN_RUN_BASE:-${SPATIAL2D_STORAGE_ROOT}/run/train/spatial2D}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-}"
if [[ -z "${INFER_RUN_FOLDER_PATH:-}" ]]; then
  if [[ -n "${TRAIN_RUN_DIR}" ]]; then
    INFER_RUN_FOLDER_PATH="${TRAIN_RUN_DIR}"
  elif completed_train_run="$(latest_completed_train_run "${TRAIN_RUN_BASE}")"; then
    INFER_RUN_FOLDER_PATH="${completed_train_run}"
  else
    INFER_RUN_FOLDER_PATH=""
  fi
fi
INFER_OUTPUT_BASE="${INFER_OUTPUT_BASE:-${INFER_RUN_FOLDER_PATH}/inference_output}"
EXTRA_INFER_ARGS="${EXTRA_INFER_ARGS:-}"

[[ -n "${INFER_RUN_FOLDER_PATH}" ]] || die "INFER_RUN_FOLDER_PATH is empty. Point it at a completed spatial2D training run directory."

activate_env
assert_python_stack
assert_spatial2d_data
assert_gpu_memory
ensure_spatial2d_extension
print_runtime_context

mkdir -p "${INFER_OUTPUT_BASE}"
reuse_spatial2d_generated_data

cmd=(
  python src/inference.py
  inference=spatial2D
  run_folder_path="${INFER_RUN_FOLDER_PATH}"
  folder_name="${INFER_OUTPUT_BASE}"
  abc.num_workers="${SLURM_CPUS_PER_TASK:-8}"
)

if [[ -n "${EXTRA_INFER_ARGS}" ]]; then
  # Intentional word splitting for Hydra override arguments supplied by the submitter.
  # shellcheck disable=SC2206
  extra_infer_args=( ${EXTRA_INFER_ARGS} )
  cmd+=("${extra_infer_args[@]}")
fi

log "Launching spatial2D inference through the project Hydra interface."
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

log "spatial2D inference completed successfully."

# Modify configs/inference/spatial2D.yaml or pass EXTRA_INFER_ARGS to change parameters as needed.
