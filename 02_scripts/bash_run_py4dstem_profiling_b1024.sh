#!/bin/bash
# Run this bash script with `sudo bash ./02_scripts/bash_run_py4dstem_profiling.sh`
# Note that the GPU metrics in nsys requires sudo permission

# === LOCAL RUN: py4dstem profiling script ===

# Set job name and identifiers manually if needed
JOB_NAME="py4dstem_profiling_tBL_WSe2_b1024_p12_6slice"
JOB_ID=$(date +"%Y%m%d_%H%M%S")  # Replace SLURM_JOB_ID with timestamp
LOG_DIR="03_output/logs"
PROFILE_DIR="03_output/profiling_5000Ada"
mkdir -p "$LOG_DIR" "$PROFILE_DIR"

LOG_FILE="${LOG_DIR}/log_job_${JOB_ID}_${JOB_NAME}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==== JOB START ===="
pwd
hostname
date

echo "Activating Conda environment..."
# Set up conda in a sudo shell
CONDA_BASE="/home/cl2696/miniforge3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py4DSTEM

# Set the params_path variable
PARAMS_PATH="01_params/py4dstem_profiling_tBL_WSe2_b1024_p12_6slice.yml"
echo "params_path = ${PARAMS_PATH}"

# Use GPUID
GPU_IDS=1  # default to 1, the none display GPU
export CUDA_VISIBLE_DEVICES=$GPU_IDS # This has no effect to nvidia-smi and nvidia-smi dmon, but can mask the nsys and following python/Matlab

# --- Log initial GPU info
echo "====== Initial GPU Info ======"
nvidia-smi

# Start nvidia-smi dmon logging
DMON_LOG="${PROFILE_DIR}/py4dstem_gpu_dmon_${JOB_NAME}_${JOB_ID}.csv"
nvidia-smi dmon -i $GPU_IDS -s pucvmet -o TD -d 1 -f "$DMON_LOG" &
DMON_PID=$!

# Start nvidia-smi smi logging
SMI_LOG="${PROFILE_DIR}/py4dstem_gpu_smi_${JOB_NAME}_${JOB_ID}.csv"
nvidia-smi --id=$GPU_IDS \
  --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,temperature.gpu,power.draw \
  --format=csv -l 1 -f "$SMI_LOG" &
SMI_PID=$!

# Run nsys profiling, note that we have to manually restict the exposed GPU since py4DSTEM can't select GPU ID
NSYS_OUT="${PROFILE_DIR}/py4dstem_gpu_nsys_${JOB_NAME}_${JOB_ID}.nsys-rep"
echo "====== Starting Nsight Systems Profiling ======"
nsys profile -t cuda,nvtx,osrt,cudnn,cublas \
    --cudabacktrace=true \
    --gpu-metrics-devices=0 \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -x true \
    --delay 300 \
    --duration 300 \
    --output "$NSYS_OUT" \
    python -u ./02_scripts/run_py4dstem_profiling.py --params_path "${PARAMS_PATH}"

# Kill background monitoring
kill $DMON_PID $SMI_PID || true

date
echo "==== JOB END ===="