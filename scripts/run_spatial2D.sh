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

mkdir -p run/slurm

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

submit_job() {
  local parsable
  parsable="$(sbatch "$@")"
  printf '%s\n' "${parsable}"
}

log "Submitting spatial2D data job."
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
    --output "run/slurm/%x-%j.out" \
    --error "run/slurm/%x-%j.err" \
    --export "${SBATCH_EXPORTS}" \
    --wrap "cd '${PROJECT_ROOT}' && srun bash scripts/data_spatial2D.sh"
)"

log "Data job submitted: ${data_job_id}"
log "Submitting dependent spatial2D training job."

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
    --output "run/slurm/%x-%j.out" \
    --error "run/slurm/%x-%j.err" \
    --export "${SBATCH_EXPORTS}" \
    --wrap "cd '${PROJECT_ROOT}' && srun bash scripts/train_spatial2D.sh"
)"

log "Training job submitted: ${train_job_id}"
log "Submission chain complete. Training will start only after data generation succeeds."
