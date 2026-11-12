#!/bin/bash
#SBATCH --job-name=cosilab_multiview
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/cosilab_multiview_%j.out
# Multi-view SMPL refinement of the cam-batches of one session. Needs their results of
# scripts/cosilab_smpl.sbatch. --time fits 600-frame clips only. Arguments go to scripts/run_cosilab_multiview.py:
#   sbatch scripts/cosilab_multiview.sbatch --cam_batches cam01_batch01 cam03_batch01
# Results: /tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_multiview/<cam_batch>/
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
[ -x apptainer/run.sh ] || { echo "Submit from the repo root"; exit 1; }
PROJECT=/tudelft.net/staff-umbrella/neon/cosilab_project  # videos and calibration for the overlays, read-only
export APPTAINER_BIND="$PROJECT/4TU_upload:$PROJECT/4TU_upload:ro,$PROJECT/raw_sensor_data:$PROJECT/raw_sensor_data:ro,$PROJECT/processed_data:$PROJECT/processed_data:ro"
nvidia-smi -L
exec apptainer/run.sh python scripts/run_cosilab_multiview.py "$@"
