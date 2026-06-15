#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ACCOUNT="${ACCOUNT:-iicd}"
PARTITION="${PARTITION:-burst}"
PROBE_GPUS="${PROBE_GPUS:-1}"
PROBE_GPU_TYPE="${PROBE_GPU_TYPE:-l40s}"
PROBE_CPUS="${PROBE_CPUS:-8}"
PROBE_MEM="${PROBE_MEM:-64G}"
PROBE_TIME="${PROBE_TIME:-04:00:00}"
SPATIAL2D_STORAGE_ROOT="${SPATIAL2D_STORAGE_ROOT:-/insomnia001/depts/iicd/users/${USER}/viaABC}"
MICROMAMBA_ENV_PATH="${MICROMAMBA_ENV_PATH:-/insomnia001/depts/iicd/users/${USER}/micromamba/envs/viaabc310}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${SPATIAL2D_STORAGE_ROOT}/run/train/spatial2D/2026-06-03_11-34-14_bs10_acc2_nw2}"
CHECKPOINT_SUBSTR="${CHECKPOINT_SUBSTR:-last}"
N_THETA="${N_THETA:-24}"
POOLING_METHOD="${POOLING_METHOD:-no_cls}"
METRIC="${METRIC:-pairwise_cosine}"
PRIOR_LOW="${PRIOR_LOW:-0,0}"
PRIOR_HIGH="${PRIOR_HIGH:-1,1}"
SEED="${SEED:-12345}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SPATIAL2D_STORAGE_ROOT}/run/slurm}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-${TRAIN_RUN_DIR}/latent_distance_probe/$(date +%Y-%m-%d_%H-%M-%S)}"

IFS=',' read -r PRIOR_LOW_0 PRIOR_LOW_1 <<< "${PRIOR_LOW}"
IFS=',' read -r PRIOR_HIGH_0 PRIOR_HIGH_1 <<< "${PRIOR_HIGH}"
[[ -n "${PRIOR_LOW_0}" && -n "${PRIOR_LOW_1}" && -n "${PRIOR_HIGH_0}" && -n "${PRIOR_HIGH_1}" ]] || {
  printf 'Expected PRIOR_LOW and PRIOR_HIGH as two comma-separated values, got PRIOR_LOW=%s PRIOR_HIGH=%s\n' "${PRIOR_LOW}" "${PRIOR_HIGH}" >&2
  exit 1
}

mkdir -p "${SLURM_LOG_DIR}" "${PROBE_OUTPUT_DIR}"

gpu_request="${PROBE_GPUS}"
if [[ -n "${PROBE_GPU_TYPE}" ]]; then
  gpu_request="${PROBE_GPU_TYPE}:${PROBE_GPUS}"
fi

job_id="$(
  sbatch \
    --parsable \
    --job-name viaabc-spatial2d-latent-probe \
    --account "${ACCOUNT}" \
    --partition "${PARTITION}" \
    --nodes 1 \
    --ntasks 1 \
    --gpus "${gpu_request}" \
    --cpus-per-task "${PROBE_CPUS}" \
    --mem "${PROBE_MEM}" \
    --time "${PROBE_TIME}" \
    --output "${SLURM_LOG_DIR}/%x-%j.out" \
    --error "${SLURM_LOG_DIR}/%x-%j.err" \
    --export "ALL,MICROMAMBA_ENV_PATH=${MICROMAMBA_ENV_PATH},PROJECT_ROOT=${PROJECT_ROOT},TRAIN_RUN_DIR=${TRAIN_RUN_DIR},CHECKPOINT_SUBSTR=${CHECKPOINT_SUBSTR},PROBE_OUTPUT_DIR=${PROBE_OUTPUT_DIR},N_THETA=${N_THETA},POOLING_METHOD=${POOLING_METHOD},METRIC=${METRIC},PRIOR_LOW_0=${PRIOR_LOW_0},PRIOR_LOW_1=${PRIOR_LOW_1},PRIOR_HIGH_0=${PRIOR_HIGH_0},PRIOR_HIGH_1=${PRIOR_HIGH_1},SEED=${SEED},OMP_NUM_THREADS=${PROBE_CPUS},MKL_NUM_THREADS=${PROBE_CPUS},NUMEXPR_NUM_THREADS=${PROBE_CPUS},OPENBLAS_NUM_THREADS=${PROBE_CPUS}" \
    --wrap "source '${SCRIPT_DIR}/spatial2D_common.sh' && activate_env && cd '${PROJECT_ROOT}' && python scripts/probe_spatial2D_latent_distance.py --run-dir \"\${TRAIN_RUN_DIR}\" --checkpoint-substr \"\${CHECKPOINT_SUBSTR}\" --output-dir \"\${PROBE_OUTPUT_DIR}\" --n-theta \"\${N_THETA}\" --pooling-method \"\${POOLING_METHOD}\" --metric \"\${METRIC}\" --prior-low \"\${PRIOR_LOW_0},\${PRIOR_LOW_1}\" --prior-high \"\${PRIOR_HIGH_0},\${PRIOR_HIGH_1}\" --seed \"\${SEED}\""
)"

printf 'Submitted latent distance probe job: %s\n' "${job_id}"
printf 'Probe output dir: %s\n' "${PROBE_OUTPUT_DIR}"
