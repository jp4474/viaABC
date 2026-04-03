#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ACCOUNT="${ACCOUNT:-iicd}"
PARTITION="${PARTITION:-iicd1}"

DATA_CPUS="${DATA_CPUS:-16}"
DATA_MEM="${DATA_MEM:-48G}"
DATA_TIME="${DATA_TIME:-24:00:00}"

TRAIN_GPUS="${TRAIN_GPUS:-1}"
TRAIN_CPUS="${TRAIN_CPUS:-16}"
TRAIN_MEM="${TRAIN_MEM:-96G}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"

SBATCH_EXPORTS="${SBATCH_EXPORTS:-ALL}"
SPATIAL2D_STORAGE_ROOT="${SPATIAL2D_STORAGE_ROOT:-/insomnia001/depts/iicd/users/${USER}/viaABC}"
SPATIAL2D_SOURCE_DATA_DIR="${SPATIAL2D_SOURCE_DATA_DIR:-${PROJECT_ROOT}/data/spatial2D}"
SPATIAL2D_DATA_DIR="${SPATIAL2D_DATA_DIR:-${SPATIAL2D_STORAGE_ROOT}/data/spatial2D}"
TRAIN_RUN_BASE="${TRAIN_RUN_BASE:-${SPATIAL2D_STORAGE_ROOT}/run/train/spatial2D}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SPATIAL2D_STORAGE_ROOT}/run/slurm}"

DATA_OMP_NUM_THREADS="${DATA_OMP_NUM_THREADS:-${DATA_CPUS}}"
DATA_MKL_NUM_THREADS="${DATA_MKL_NUM_THREADS:-${DATA_CPUS}}"
DATA_NUMEXPR_NUM_THREADS="${DATA_NUMEXPR_NUM_THREADS:-${DATA_CPUS}}"
DATA_OPENBLAS_NUM_THREADS="${DATA_OPENBLAS_NUM_THREADS:-${DATA_CPUS}}"

TRAIN_OMP_NUM_THREADS="${TRAIN_OMP_NUM_THREADS:-${TRAIN_CPUS}}"
TRAIN_MKL_NUM_THREADS="${TRAIN_MKL_NUM_THREADS:-${TRAIN_CPUS}}"
TRAIN_NUMEXPR_NUM_THREADS="${TRAIN_NUMEXPR_NUM_THREADS:-${TRAIN_CPUS}}"
TRAIN_OPENBLAS_NUM_THREADS="${TRAIN_OPENBLAS_NUM_THREADS:-${TRAIN_CPUS}}"

mkdir -p "${SLURM_LOG_DIR}" "${SPATIAL2D_DATA_DIR}" "${TRAIN_RUN_BASE}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

submit_job() {
  local parsable
  parsable="$(sbatch "$@")"
  printf '%s\n' "${parsable}"
}

log "Submitting spatial2D data job."
log "Spatial2D source data dir: ${SPATIAL2D_SOURCE_DATA_DIR}"
log "Spatial2D generated data dir: ${SPATIAL2D_DATA_DIR}"
log "Train run base dir: ${TRAIN_RUN_BASE}"
log "Slurm log dir: ${SLURM_LOG_DIR}"
log "Data job thread config: cpus=${DATA_CPUS}, OMP=${DATA_OMP_NUM_THREADS}, MKL=${DATA_MKL_NUM_THREADS}, NUMEXPR=${DATA_NUMEXPR_NUM_THREADS}, OPENBLAS=${DATA_OPENBLAS_NUM_THREADS}"
data_job_id="$(
  submit_job \
    --parsable \
    --job-name viaabc-spatial2d-data \
    --account "${ACCOUNT}" \
    --partition "${PARTITION}" \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task "${DATA_CPUS}" \
    --mem "${DATA_MEM}" \
    --time "${DATA_TIME}" \
    --output "${SLURM_LOG_DIR}/%x-%j.out" \
    --error "${SLURM_LOG_DIR}/%x-%j.err" \
    --export "${SBATCH_EXPORTS},SPATIAL2D_STORAGE_ROOT=${SPATIAL2D_STORAGE_ROOT},SPATIAL2D_SOURCE_DATA_DIR=${SPATIAL2D_SOURCE_DATA_DIR},SPATIAL2D_DATA_DIR=${SPATIAL2D_DATA_DIR},TRAIN_RUN_BASE=${TRAIN_RUN_BASE},SLURM_LOG_DIR=${SLURM_LOG_DIR},OMP_NUM_THREADS=${DATA_OMP_NUM_THREADS},MKL_NUM_THREADS=${DATA_MKL_NUM_THREADS},NUMEXPR_NUM_THREADS=${DATA_NUMEXPR_NUM_THREADS},OPENBLAS_NUM_THREADS=${DATA_OPENBLAS_NUM_THREADS}" \
    --wrap "cd '${PROJECT_ROOT}' && srun bash scripts/data_spatial2D.sh"
)"

log "Data job submitted: ${data_job_id}"
log "Submitting dependent spatial2D training job."
log "Training job thread config: cpus=${TRAIN_CPUS}, OMP=${TRAIN_OMP_NUM_THREADS}, MKL=${TRAIN_MKL_NUM_THREADS}, NUMEXPR=${TRAIN_NUMEXPR_NUM_THREADS}, OPENBLAS=${TRAIN_OPENBLAS_NUM_THREADS}"

train_job_id="$(
  submit_job \
    --parsable \
    --job-name viaabc-spatial2d-train \
    --account "${ACCOUNT}" \
    --partition "${PARTITION}" \
    --nodes 1 \
    --ntasks 1 \
    --gpus "${TRAIN_GPUS}" \
    --cpus-per-task "${TRAIN_CPUS}" \
    --mem "${TRAIN_MEM}" \
    --time "${TRAIN_TIME}" \
    --dependency "afterok:${data_job_id}" \
    --output "${SLURM_LOG_DIR}/%x-%j.out" \
    --error "${SLURM_LOG_DIR}/%x-%j.err" \
    --export "${SBATCH_EXPORTS},SPATIAL2D_STORAGE_ROOT=${SPATIAL2D_STORAGE_ROOT},SPATIAL2D_SOURCE_DATA_DIR=${SPATIAL2D_SOURCE_DATA_DIR},SPATIAL2D_DATA_DIR=${SPATIAL2D_DATA_DIR},TRAIN_RUN_BASE=${TRAIN_RUN_BASE},SLURM_LOG_DIR=${SLURM_LOG_DIR},OMP_NUM_THREADS=${TRAIN_OMP_NUM_THREADS},MKL_NUM_THREADS=${TRAIN_MKL_NUM_THREADS},NUMEXPR_NUM_THREADS=${TRAIN_NUMEXPR_NUM_THREADS},OPENBLAS_NUM_THREADS=${TRAIN_OPENBLAS_NUM_THREADS}" \
    --wrap "cd '${PROJECT_ROOT}' && srun bash scripts/train_spatial2D.sh"
)"

log "Training job submitted: ${train_job_id}"
log "Submission chain complete. Training will start only after data generation succeeds."
