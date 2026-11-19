#!/bin/bash
#SBATCH --job-name=cosilab_cpu
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cosilab_cpu_%j.out
# Any CPU-only COSI-Lab script in the OnlineHMR image (no GPU), with the dataset mounted read-only like the
# other jobs. Arguments: the script and its arguments. Submit from the repo root:
#   sbatch --time=01:00:00 scripts/cosilab_cpu.sbatch scripts/cosilab_refit_extrinsic.py --cam_batch cam12_batch01 ...
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
[ -x apptainer/run.sh ] || { echo "Submit from the repo root"; exit 1; }

DATASET=/tudelft.net/staff-umbrella/neon/cosilab_project/4TU_upload
PROJECT=$(dirname "$DATASET")  # side cameras: raw files and calibration outside the 4TU upload
export APPTAINER_BIND="$DATASET:$DATASET:ro,$PROJECT/raw_sensor_data:$PROJECT/raw_sensor_data:ro,$PROJECT/processed_data:$PROJECT/processed_data:ro"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
exec apptainer/run.sh python "$@"
