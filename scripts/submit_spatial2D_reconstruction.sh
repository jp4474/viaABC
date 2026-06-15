#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Submit a lightweight Spatial2D reconstruction visualization job.

Environment overrides:
  TRAIN_RUN_DIR       Completed Spatial2D training run directory.
  SAMPLE_ID           Spatial2D sample id, default sample_2.
  CHECKPOINT_SUBSTR   Checkpoint filename substring, default last.
  RECON_OUTPUT_DIR    Output directory for PNG + metadata.
  PARTITION           Slurm partition, default burst.
  RECON_GPU_TYPE      GPU type, default A6000.
  RECON_CPUS          CPUs per task, default 4.
  RECON_MEM           Memory, default 48G.
  RECON_TIME          Wall time, default 02:00:00.
EOF
  exit 0
fi

ACCOUNT="${ACCOUNT:-iicd}"
PARTITION="${PARTITION:-burst}"
RECON_GPUS="${RECON_GPUS:-1}"
RECON_GPU_TYPE="${RECON_GPU_TYPE:-A6000}"
RECON_CPUS="${RECON_CPUS:-4}"
RECON_MEM="${RECON_MEM:-48G}"
RECON_TIME="${RECON_TIME:-02:00:00}"
SPATIAL2D_STORAGE_ROOT="${SPATIAL2D_STORAGE_ROOT:-/insomnia001/depts/iicd/users/${USER}/viaABC}"
MICROMAMBA_ENV_PATH="${MICROMAMBA_ENV_PATH:-/insomnia001/depts/iicd/users/${USER}/micromamba/envs/viaabc310}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${SPATIAL2D_STORAGE_ROOT}/run/train/spatial2D/2026-06-03_11-34-14_bs10_acc2_nw2}"
SAMPLE_ID="${SAMPLE_ID:-sample_2}"
CHECKPOINT_SUBSTR="${CHECKPOINT_SUBSTR:-last}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SPATIAL2D_STORAGE_ROOT}/run/slurm}"
RECON_OUTPUT_DIR="${RECON_OUTPUT_DIR:-${TRAIN_RUN_DIR}/reconstruction_visualizations/${SAMPLE_ID}_$(date +%Y-%m-%d_%H-%M-%S)}"

mkdir -p "${SLURM_LOG_DIR}" "${RECON_OUTPUT_DIR}"

gpu_args=(--gpus "${RECON_GPUS}")
if [[ -n "${RECON_GPU_TYPE}" ]]; then
  gpu_args=(--gres "gpu:${RECON_GPU_TYPE}:${RECON_GPUS}")
fi

job_id="$(
  sbatch \
    --parsable \
    --job-name viaabc-spatial2d-recon \
    --account "${ACCOUNT}" \
    --partition "${PARTITION}" \
    --nodes 1 \
    --ntasks 1 \
    "${gpu_args[@]}" \
    --cpus-per-task "${RECON_CPUS}" \
    --mem "${RECON_MEM}" \
    --time "${RECON_TIME}" \
    --output "${SLURM_LOG_DIR}/%x-%j.out" \
    --error "${SLURM_LOG_DIR}/%x-%j.err" \
    --export "ALL,MICROMAMBA_ENV_PATH=${MICROMAMBA_ENV_PATH},PROJECT_ROOT=${PROJECT_ROOT},TRAIN_RUN_DIR=${TRAIN_RUN_DIR},SAMPLE_ID=${SAMPLE_ID},CHECKPOINT_SUBSTR=${CHECKPOINT_SUBSTR},RECON_OUTPUT_DIR=${RECON_OUTPUT_DIR}" \
    --wrap "source '${SCRIPT_DIR}/spatial2D_common.sh' && activate_env && cd '${PROJECT_ROOT}' && python scripts/visualize_spatial2D_reconstruction.py --run-dir \"\${TRAIN_RUN_DIR}\" --checkpoint-substr \"\${CHECKPOINT_SUBSTR}\" --sample-id \"\${SAMPLE_ID}\" --output-dir \"\${RECON_OUTPUT_DIR}\""
)"

printf 'Submitted Spatial2D reconstruction job: %s\n' "${job_id}"
printf 'Reconstruction output dir: %s\n' "${RECON_OUTPUT_DIR}"
printf 'Expected image: %s/%s_reconstruction.png\n' "${RECON_OUTPUT_DIR}" "${SAMPLE_ID}"
