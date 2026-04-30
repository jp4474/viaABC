#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

latest_completed_train_run() {
  local base_dir="$1"
  local candidate
  local newest=""
  local newest_mtime=0
  local mtime

  [[ -d "${base_dir}" ]] || return 1

  for candidate in "${base_dir}"/*; do
    [[ -d "${candidate}" ]] || continue
    [[ -f "${candidate}/.hydra/config.yaml" ]] || continue
    [[ -d "${candidate}/checkpoints" ]] || continue
    compgen -G "${candidate}/checkpoints/*.ckpt" >/dev/null || continue

    mtime="$(stat -c %Y "${candidate}" 2>/dev/null || printf '0')"
    if (( mtime >= newest_mtime )); then
      newest_mtime="${mtime}"
      newest="${candidate}"
    fi
  done

  [[ -n "${newest}" ]] || return 1
  printf '%s\n' "${newest}"
}

ACCOUNT="${ACCOUNT:-iicd}"
PARTITION="${PARTITION:-iicd1}"

DATA_CPUS="${DATA_CPUS:-16}"
DATA_MEM="${DATA_MEM:-48G}"
DATA_TIME="${DATA_TIME:-24:00:00}"

TRAIN_GPUS="${TRAIN_GPUS:-1}"
TRAIN_CPUS="${TRAIN_CPUS:-16}"
TRAIN_MEM="${TRAIN_MEM:-96G}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"

INFER_GPUS="${INFER_GPUS:-1}"
INFER_GPU_TYPE="${INFER_GPU_TYPE:-l40}"
INFER_CPUS="${INFER_CPUS:-64}"
INFER_MEM="${INFER_MEM:-192G}"
INFER_TIME="${INFER_TIME:-48:00:00}"
INFER_MIN_GPU_MEM_MIB="${INFER_MIN_GPU_MEM_MIB:-40000}"

SBATCH_EXPORTS="${SBATCH_EXPORTS:-ALL}"
SPATIAL2D_STORAGE_ROOT="${SPATIAL2D_STORAGE_ROOT:-/insomnia001/depts/iicd/users/${USER}/viaABC}"
SPATIAL2D_SOURCE_DATA_DIR="${SPATIAL2D_SOURCE_DATA_DIR:-${PROJECT_ROOT}/data/spatial2D}"
SPATIAL2D_DATA_DIR="${SPATIAL2D_DATA_DIR:-${SPATIAL2D_STORAGE_ROOT}/data/spatial2D}"
TRAIN_RUN_BASE="${TRAIN_RUN_BASE:-${SPATIAL2D_STORAGE_ROOT}/run/train/spatial2D}"
TRAIN_RUN_STAMP="${TRAIN_RUN_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)_submit}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${TRAIN_RUN_BASE}/${TRAIN_RUN_STAMP}}"
if [[ -z "${INFER_RUN_FOLDER_PATH:-}" ]]; then
  if completed_train_run="$(latest_completed_train_run "${TRAIN_RUN_BASE}")"; then
    INFER_RUN_FOLDER_PATH="${completed_train_run}"
  else
    INFER_RUN_FOLDER_PATH="${TRAIN_RUN_DIR}"
  fi
fi
INFER_OUTPUT_BASE="${INFER_OUTPUT_BASE:-${INFER_RUN_FOLDER_PATH}/inference_output}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SPATIAL2D_STORAGE_ROOT}/run/slurm}"

DATA_OMP_NUM_THREADS="${DATA_OMP_NUM_THREADS:-${DATA_CPUS}}"
DATA_MKL_NUM_THREADS="${DATA_MKL_NUM_THREADS:-${DATA_CPUS}}"
DATA_NUMEXPR_NUM_THREADS="${DATA_NUMEXPR_NUM_THREADS:-${DATA_CPUS}}"
DATA_OPENBLAS_NUM_THREADS="${DATA_OPENBLAS_NUM_THREADS:-${DATA_CPUS}}"

TRAIN_OMP_NUM_THREADS="${TRAIN_OMP_NUM_THREADS:-${TRAIN_CPUS}}"
TRAIN_MKL_NUM_THREADS="${TRAIN_MKL_NUM_THREADS:-${TRAIN_CPUS}}"
TRAIN_NUMEXPR_NUM_THREADS="${TRAIN_NUMEXPR_NUM_THREADS:-${TRAIN_CPUS}}"
TRAIN_OPENBLAS_NUM_THREADS="${TRAIN_OPENBLAS_NUM_THREADS:-${TRAIN_CPUS}}"

INFER_OMP_NUM_THREADS="${INFER_OMP_NUM_THREADS:-${INFER_CPUS}}"
INFER_MKL_NUM_THREADS="${INFER_MKL_NUM_THREADS:-${INFER_CPUS}}"
INFER_NUMEXPR_NUM_THREADS="${INFER_NUMEXPR_NUM_THREADS:-${INFER_CPUS}}"
INFER_OPENBLAS_NUM_THREADS="${INFER_OPENBLAS_NUM_THREADS:-${INFER_CPUS}}"

mkdir -p "${SLURM_LOG_DIR}" "${SPATIAL2D_DATA_DIR}" "${TRAIN_RUN_BASE}" "${INFER_OUTPUT_BASE}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

submit_job() {
  local parsable
  parsable="$(sbatch "$@")"
  printf '%s\n' "${parsable}"
}

log "Preparing spatial2D Slurm submission."
log "Spatial2D source data dir: ${SPATIAL2D_SOURCE_DATA_DIR}"
log "Spatial2D generated data dir: ${SPATIAL2D_DATA_DIR}"
log "Train run base dir: ${TRAIN_RUN_BASE}"
log "Train run dir: ${TRAIN_RUN_DIR}"
log "Inference output base: ${INFER_OUTPUT_BASE}"
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
log "Submitting spatial2D training job."
log "Training job resources: gpus=${TRAIN_GPUS}, cpus=${TRAIN_CPUS}, mem=${TRAIN_MEM}, time=${TRAIN_TIME}"

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
    --output "${SLURM_LOG_DIR}/%x-%j.out" \
    --error "${SLURM_LOG_DIR}/%x-%j.err" \
    --dependency "afterok:${data_job_id}" \
    --export "${SBATCH_EXPORTS},SPATIAL2D_STORAGE_ROOT=${SPATIAL2D_STORAGE_ROOT},SPATIAL2D_SOURCE_DATA_DIR=${SPATIAL2D_SOURCE_DATA_DIR},SPATIAL2D_DATA_DIR=${SPATIAL2D_DATA_DIR},TRAIN_RUN_BASE=${TRAIN_RUN_BASE},TRAIN_RUN_DIR=${TRAIN_RUN_DIR},SLURM_LOG_DIR=${SLURM_LOG_DIR},OMP_NUM_THREADS=${TRAIN_OMP_NUM_THREADS},MKL_NUM_THREADS=${TRAIN_MKL_NUM_THREADS},NUMEXPR_NUM_THREADS=${TRAIN_NUMEXPR_NUM_THREADS},OPENBLAS_NUM_THREADS=${TRAIN_OPENBLAS_NUM_THREADS}" \
    --wrap "cd '${PROJECT_ROOT}' && srun bash scripts/train_spatial2D.sh"
)"

# log "Training job submitted: ${train_job_id}"
log "Submitting spatial2D inference job."
log "Inference job resources: gpu_type=${INFER_GPU_TYPE:-any}, gpus=${INFER_GPUS}, cpus=${INFER_CPUS}, mem=${INFER_MEM}, time=${INFER_TIME}"

infer_gpu_request="${INFER_GPUS}"
if [[ -n "${INFER_GPU_TYPE}" ]]; then
  infer_gpu_request="${INFER_GPU_TYPE}:${INFER_GPUS}"
fi


infer_job_id="$(
  submit_job \
    --parsable \
    --job-name viaabc-spatial2d-infer \
    --account "${ACCOUNT}" \
    --partition "${PARTITION}" \
    --nodes 1 \
    --ntasks 1 \
    --gpus "${infer_gpu_request}" \
    --cpus-per-task "${INFER_CPUS}" \
    --mem "${INFER_MEM}" \
    --time "${INFER_TIME}" \
    --output "${SLURM_LOG_DIR}/%x-%j.out" \
    --error "${SLURM_LOG_DIR}/%x-%j.err" \
    --dependency "afterok:${train_job_id}" \
    --export "${SBATCH_EXPORTS},SPATIAL2D_STORAGE_ROOT=${SPATIAL2D_STORAGE_ROOT},SPATIAL2D_SOURCE_DATA_DIR=${SPATIAL2D_SOURCE_DATA_DIR},SPATIAL2D_DATA_DIR=${SPATIAL2D_DATA_DIR},TRAIN_RUN_BASE=${TRAIN_RUN_BASE},TRAIN_RUN_DIR=${TRAIN_RUN_DIR},INFER_RUN_FOLDER_PATH=${INFER_RUN_FOLDER_PATH},INFER_OUTPUT_BASE=${INFER_OUTPUT_BASE},SLURM_LOG_DIR=${SLURM_LOG_DIR},MIN_GPU_MEM_MIB=${INFER_MIN_GPU_MEM_MIB},OMP_NUM_THREADS=${INFER_OMP_NUM_THREADS},MKL_NUM_THREADS=${INFER_MKL_NUM_THREADS},NUMEXPR_NUM_THREADS=${INFER_NUMEXPR_NUM_THREADS},OPENBLAS_NUM_THREADS=${INFER_OPENBLAS_NUM_THREADS}" \
    --wrap "cd '${PROJECT_ROOT}' && srun bash scripts/inference_spatial2D.sh"
)"

log "Inference job submitted: ${infer_job_id}"
log "Submission chain complete."
