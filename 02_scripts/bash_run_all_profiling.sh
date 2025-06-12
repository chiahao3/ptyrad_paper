#!/bin/bash
# Run this script with `sudo bash ./02_scripts/bash_run_all_profiling.sh`

set -e  # Exit on any error
set -x  # Echo each command (optional for debugging)

# List of all your individual profiling scripts
SCRIPT_LIST=(
    "02_scripts/bash_run_ptyrad_profiling_b1024.sh"
    "02_scripts/bash_run_ptyshv_profiling_b1024.sh"
    "02_scripts/bash_run_py4dstem_profiling_b1024.sh"
    "02_scripts/bash_run_ptyrad_profiling_b256.sh"
    "02_scripts/bash_run_ptyshv_profiling_b256.sh"
    "02_scripts/bash_run_py4dstem_profiling_b256.sh"
    "02_scripts/bash_run_ptyrad_profiling_b64.sh"
    "02_scripts/bash_run_ptyshv_profiling_b64.sh"
    "02_scripts/bash_run_py4dstem_profiling_b64.sh"
    "02_scripts/bash_run_ptyrad_profiling_b16.sh"
    "02_scripts/bash_run_ptyshv_profiling_b16.sh"
    "02_scripts/bash_run_py4dstem_profiling_b16.sh"
)

for SCRIPT in "${SCRIPT_LIST[@]}"; do
    echo "==== Running $SCRIPT ===="
    sudo bash "$SCRIPT"
    echo "==== Finished $SCRIPT ===="
    echo ""
done

echo "==== All profiling jobs completed ===="