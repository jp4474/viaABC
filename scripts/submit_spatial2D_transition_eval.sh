#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ACCOUNT="${ACCOUNT:-iicd}"
PARTITION="${PARTITION:-burst}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_GPU_TYPE="${EVAL_GPU_TYPE:-l40s}"
EVAL_CPUS="${EVAL_CPUS:-4}"
EVAL_MEM="${EVAL_MEM:-64G}"
EVAL_TIME="${EVAL_TIME:-12:00:00}"
MICROMAMBA_ENV_PATH="${MICROMAMBA_ENV_PATH:-/insomnia001/home/kz2537/micromamba/envs/viaabc310}"
SPATIAL2D_STORAGE_ROOT="${SPATIAL2D_STORAGE_ROOT:-/insomnia001/depts/iicd/users/${USER}/viaABC}"
SPATIAL2D_DATA_DIR="${SPATIAL2D_DATA_DIR:-${SPATIAL2D_STORAGE_ROOT}/data/spatial2D}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${SPATIAL2D_STORAGE_ROOT}/run/train/spatial2D/2026-06-03_11-34-14_bs10_acc2_nw2}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-200}"
TARGET_SCOPE="${TARGET_SCOPE:-terminal}"
CHECKPOINT_SUBSTR="${CHECKPOINT_SUBSTR:-last}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SPATIAL2D_STORAGE_ROOT}/run/slurm}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${TRAIN_RUN_DIR}/transition_dominance_eval/${TARGET_SCOPE}_$(date +%Y-%m-%d_%H-%M-%S)}"

mkdir -p "${SLURM_LOG_DIR}" "${EVAL_OUTPUT_DIR}"

gpu_request="${EVAL_GPUS}"
if [[ -n "${EVAL_GPU_TYPE}" ]]; then
  gpu_request="${EVAL_GPU_TYPE}:${EVAL_GPUS}"
fi

job_id="$(
  sbatch \
    --parsable \
    --job-name viaabc-spatial2d-dom-eval \
    --account "${ACCOUNT}" \
    --partition "${PARTITION}" \
    --nodes 1 \
    --ntasks 1 \
    --gpus "${gpu_request}" \
    --cpus-per-task "${EVAL_CPUS}" \
    --mem "${EVAL_MEM}" \
    --time "${EVAL_TIME}" \
    --output "${SLURM_LOG_DIR}/%x-%j.out" \
    --error "${SLURM_LOG_DIR}/%x-%j.err" \
    --export "ALL,MICROMAMBA_ENV_PATH=${MICROMAMBA_ENV_PATH},PROJECT_ROOT=${PROJECT_ROOT},TRAIN_RUN_DIR=${TRAIN_RUN_DIR},SPATIAL2D_DATA_DIR=${SPATIAL2D_DATA_DIR},EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR},SAMPLES_PER_CLASS=${SAMPLES_PER_CLASS},TARGET_SCOPE=${TARGET_SCOPE},CHECKPOINT_SUBSTR=${CHECKPOINT_SUBSTR}" \
    --wrap "source '${SCRIPT_DIR}/spatial2D_common.sh' && activate_env && cd '${PROJECT_ROOT}' && python scripts/evaluate_spatial2D_transition_dominance.py --run-dir \"\${TRAIN_RUN_DIR}\" --data-dir \"\${SPATIAL2D_DATA_DIR}\" --checkpoint-substr \"\${CHECKPOINT_SUBSTR}\" --samples-per-class \"\${SAMPLES_PER_CLASS}\" --target-scope \"\${TARGET_SCOPE}\" --output-dir \"\${EVAL_OUTPUT_DIR}\""
)"

printf 'Submitted transition dominance eval job: %s\n' "${job_id}"
printf 'Eval output dir: %s\n' "${EVAL_OUTPUT_DIR}"
