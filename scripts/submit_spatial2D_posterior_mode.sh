#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Submit a CPU Spatial2D posterior predictive mode job.

Environment overrides:
  ABC_GENERATIONS_PATH       Completed viaABC abc_generations.npy.
  POSTERIOR_MODE_OUTPUT_DIR  Output directory for posterior_mode_results.npz.
  POSTERIOR_MODE_CPUS        CPUs per task, default 16.
  POSTERIOR_MODE_MEM         Memory, default 96G.
  POSTERIOR_MODE_TIME        Wall time, default 24:00:00.
  PARTITION                  Slurm partition, default burst.
  ACCOUNT                    Slurm account, default iicd.
EOF
  exit 0
fi

ACCOUNT="${ACCOUNT:-iicd}"
PARTITION="${PARTITION:-burst}"
POSTERIOR_MODE_CPUS="${POSTERIOR_MODE_CPUS:-16}"
POSTERIOR_MODE_MEM="${POSTERIOR_MODE_MEM:-96G}"
POSTERIOR_MODE_TIME="${POSTERIOR_MODE_TIME:-24:00:00}"
SPATIAL2D_STORAGE_ROOT="${SPATIAL2D_STORAGE_ROOT:-/insomnia001/depts/iicd/users/${USER}/viaABC}"
MICROMAMBA_ENV_PATH="${MICROMAMBA_ENV_PATH:-/insomnia001/depts/iicd/users/${USER}/micromamba/envs/viaabc310}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SPATIAL2D_STORAGE_ROOT}/run/slurm}"
ABC_GENERATIONS_PATH="${ABC_GENERATIONS_PATH:-${SPATIAL2D_STORAGE_ROOT}/run/train/spatial2D/2026-06-03_11-34-14_bs10_acc2_nw2/inference_output/2026-06-04_12-42-01_56900436/abc_generations.npy}"
POSTERIOR_MODE_OUTPUT_DIR="${POSTERIOR_MODE_OUTPUT_DIR:-${SPATIAL2D_STORAGE_ROOT}/run/train/spatial2D/2026-06-03_11-34-14_bs10_acc2_nw2/posterior_mode/job_${SLURM_JOB_ID:-submit}_$(date +%Y-%m-%d_%H-%M-%S)}"

mkdir -p "${SLURM_LOG_DIR}" "${POSTERIOR_MODE_OUTPUT_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

job_id="$(
  sbatch \
    --parsable \
    --job-name viaabc-spatial2d-post-mode \
    --account "${ACCOUNT}" \
    --partition "${PARTITION}" \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task "${POSTERIOR_MODE_CPUS}" \
    --mem "${POSTERIOR_MODE_MEM}" \
    --time "${POSTERIOR_MODE_TIME}" \
    --output "${SLURM_LOG_DIR}/%x-%j.out" \
    --error "${SLURM_LOG_DIR}/%x-%j.err" \
    --export "ALL,MICROMAMBA_ENV_PATH=${MICROMAMBA_ENV_PATH},PROJECT_ROOT=${PROJECT_ROOT},ABC_GENERATIONS_PATH=${ABC_GENERATIONS_PATH},POSTERIOR_MODE_OUTPUT_DIR=${POSTERIOR_MODE_OUTPUT_DIR},POSTERIOR_MODE_CPUS=${POSTERIOR_MODE_CPUS},OMP_NUM_THREADS=${OMP_NUM_THREADS},MKL_NUM_THREADS=${MKL_NUM_THREADS},NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS},OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}" \
    --wrap "source '${SCRIPT_DIR}/spatial2D_common.sh' && activate_env && assert_python_stack && ensure_spatial2d_extension && cd '${PROJECT_ROOT}' && python scripts/spatial2d_posterior_mode.py --abc-generations-path \"\${ABC_GENERATIONS_PATH}\" --output-dir \"\${POSTERIOR_MODE_OUTPUT_DIR}\" --workers \"\${POSTERIOR_MODE_CPUS}\""
)"

printf 'Submitted Spatial2D posterior mode job: %s\n' "${job_id}"
printf 'Posterior mode output dir: %s\n' "${POSTERIOR_MODE_OUTPUT_DIR}"
printf 'Expected result: %s/posterior_mode_results.npz\n' "${POSTERIOR_MODE_OUTPUT_DIR}"
