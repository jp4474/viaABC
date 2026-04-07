#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/spatial2D_common.sh"

cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-8}}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${MKL_NUM_THREADS:-8}}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${NUMEXPR_NUM_THREADS:-8}}"

SPATIAL2D_STORAGE_ROOT="${SPATIAL2D_STORAGE_ROOT:-$(default_spatial2d_storage_root)}"
SPATIAL2D_SOURCE_DATA_DIR="${SPATIAL2D_SOURCE_DATA_DIR:-${PROJECT_ROOT}/data/spatial2D}"
SPATIAL2D_DATA_DIR="${SPATIAL2D_DATA_DIR:-${SPATIAL2D_STORAGE_ROOT}/data/spatial2D}"
DATA_RUN_DIR="${DATA_RUN_DIR:-${SPATIAL2D_STORAGE_ROOT}/run/data/spatial2D}"
TRAIN_SIZES="${TRAIN_SIZES:-50000}"
DATA_SEED="${DATA_SEED:-42}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
EXTRA_DATA_ARGS="${EXTRA_DATA_ARGS:-}"

activate_env
assert_python_stack
assert_spatial2d_data
ensure_spatial2d_extension
print_runtime_context

mkdir -p "${SPATIAL2D_DATA_DIR}" "${DATA_RUN_DIR}"
reuse_spatial2d_generated_data

cmd=(
  python src/generate_training_data.py
  --train_sizes "${TRAIN_SIZES}"
  --seed "${DATA_SEED}"
  --num_workers "${DATA_NUM_WORKERS}"
  --save_dir "${SPATIAL2D_DATA_DIR}"
)

if [[ -n "${EXTRA_DATA_ARGS}" ]]; then
  # Intentional word splitting for additional CLI flags supplied by the submitter.
  # shellcheck disable=SC2206
  extra_data_args=( ${EXTRA_DATA_ARGS} )
  cmd+=("${extra_data_args[@]}")
fi

log "Launching spatial2D data generation."
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

log "spatial2D data generation completed successfully."
