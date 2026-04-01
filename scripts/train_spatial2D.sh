#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/spatial2D_common.sh"

cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-8}}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${MKL_NUM_THREADS:-8}}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${NUMEXPR_NUM_THREADS:-8}}"

SPATIAL2D_DATA_DIR="${SPATIAL2D_DATA_DIR:-${PROJECT_ROOT}/data/spatial2D}"
LOGGER_CONFIG="${LOGGER_CONFIG:-csv}"
TRAIN_RUN_STAMP="${TRAIN_RUN_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)_job${SLURM_JOB_ID:-manual}}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${PROJECT_ROOT}/run/train/spatial2D/${TRAIN_RUN_STAMP}}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

activate_env
assert_python_stack
assert_spatial2d_data
assert_gpu_memory
ensure_spatial2d_extension
print_runtime_context

mkdir -p "${TRAIN_RUN_DIR}" "${PROJECT_ROOT}/run/train/spatial2D"

cmd=(
  python src/train.py
  experiment=spatial2D
  logger="${LOGGER_CONFIG}"
  hydra.run.dir="${TRAIN_RUN_DIR}"
  trainer.default_root_dir="${TRAIN_RUN_DIR}"
  data.datamodule.data_dir="${SPATIAL2D_DATA_DIR}"
  data.dataset.data_dir="${SPATIAL2D_DATA_DIR}"
  data.num_workers="${SLURM_CPUS_PER_TASK:-8}"
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
