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
ENABLE_INFER_RESOURCE_MONITOR="${ENABLE_INFER_RESOURCE_MONITOR:-true}"
INFER_RESOURCE_MONITOR_INTERVAL="${INFER_RESOURCE_MONITOR_INTERVAL:-${NVIDIA_SMI_MONITOR_INTERVAL:-10}}"

[[ -n "${INFER_RUN_FOLDER_PATH}" ]] || die "INFER_RUN_FOLDER_PATH is empty. Point it at a completed spatial2D training run directory."

activate_env
assert_python_stack
assert_spatial2d_data

assert_gpu_memory
ensure_spatial2d_extension
print_runtime_context

mkdir -p "${INFER_OUTPUT_BASE}"
reuse_spatial2d_generated_data

start_infer_resource_monitor() {
  [[ "${ENABLE_INFER_RESOURCE_MONITOR}" == "true" ]] || return 0

  local root_pid="$1"
  local output_path="${INFER_OUTPUT_BASE}/resource_usage_${SLURM_JOB_ID:-manual}.csv"
  local interval="${INFER_RESOURCE_MONITOR_INTERVAL}"

  printf 'timestamp,node_cpu_utilization_pct,process_cpu_pct,process_rss_mib,process_vsz_mib,mem_used_mib,mem_total_mib,mem_available_mib,gpu_index,gpu_name,gpu_utilization_gpu_pct,gpu_memory_used_mib,gpu_memory_total_mib,gpu_power_draw_w\n' > "${output_path}"

  (
    local prev_total=0
    local prev_idle=0
    local user nice system idle iowait irq softirq steal guest guest_nice
    local total idle_all total_delta idle_delta cpu_pct
    local mem_total_kib mem_available_kib mem_used_mib mem_total_mib mem_available_mib
    local process_stats process_cpu process_rss_kib process_vsz_kib process_rss_mib process_vsz_mib
    local timestamp gpu_rows

    while true; do
      read -r _ user nice system idle iowait irq softirq steal guest guest_nice < /proc/stat
      idle_all=$((idle + iowait))
      total=$((user + nice + system + idle + iowait + irq + softirq + steal + guest + guest_nice))

      if (( prev_total == 0 )); then
        cpu_pct="0.00"
      else
        total_delta=$((total - prev_total))
        idle_delta=$((idle_all - prev_idle))
        if (( total_delta > 0 )); then
          cpu_pct="$(awk -v total="${total_delta}" -v idle="${idle_delta}" 'BEGIN { printf "%.2f", 100 * (total - idle) / total }')"
        else
          cpu_pct="0.00"
        fi
      fi
      prev_total="${total}"
      prev_idle="${idle_all}"

      mem_total_kib="$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)"
      mem_available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
      mem_total_mib=$((mem_total_kib / 1024))
      mem_available_mib=$((mem_available_kib / 1024))
      mem_used_mib=$(((mem_total_kib - mem_available_kib) / 1024))
      if ps -p "${root_pid}" >/dev/null 2>&1; then
        process_stats="$(ps -p "${root_pid}" -o %cpu=,rss=,vsz= 2>/dev/null || true)"
        if [[ -n "${process_stats}" ]]; then
          read -r process_cpu process_rss_kib process_vsz_kib <<< "${process_stats}"
          process_rss_mib=$((process_rss_kib / 1024))
          process_vsz_mib=$((process_vsz_kib / 1024))
        else
          process_cpu="0.0"
          process_rss_mib=0
          process_vsz_mib=0
        fi
      else
        process_cpu="0.0"
        process_rss_mib=0
        process_vsz_mib=0
      fi

      timestamp="$(date '+%F %T')"
      if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_rows="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null || true)"
      else
        gpu_rows=""
      fi

      if [[ -n "${gpu_rows}" ]]; then
        while IFS= read -r gpu_row; do
          printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "${timestamp}" "${cpu_pct}" "${process_cpu}" "${process_rss_mib}" "${process_vsz_mib}" "${mem_used_mib}" "${mem_total_mib}" "${mem_available_mib}" "${gpu_row}"
        done <<< "${gpu_rows}" >> "${output_path}"
      else
        printf '%s,%s,%s,%s,%s,%s,%s,%s,,,,,,\n' "${timestamp}" "${cpu_pct}" "${process_cpu}" "${process_rss_mib}" "${process_vsz_mib}" "${mem_used_mib}" "${mem_total_mib}" "${mem_available_mib}" >> "${output_path}"
      fi

      sleep "${interval}"
    done
  ) &
  INFER_RESOURCE_MONITOR_PID=$!
  log "resource monitor writing to ${output_path} every ${interval}s."
}

stop_infer_resource_monitor() {
  if [[ -n "${INFER_RESOURCE_MONITOR_PID:-}" ]]; then
    kill "${INFER_RESOURCE_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${INFER_RESOURCE_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}

trap stop_infer_resource_monitor EXIT INT TERM

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

"${cmd[@]}" &
INFER_MAIN_PID=$!
start_infer_resource_monitor "${INFER_MAIN_PID}"
wait "${INFER_MAIN_PID}"

log "spatial2D inference completed successfully."

# Modify configs/inference/spatial2D.yaml or pass EXTRA_INFER_ARGS to change parameters as needed.
