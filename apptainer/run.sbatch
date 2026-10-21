#!/bin/bash
#SBATCH --job-name=onlinehmr
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=04:00:00
#SBATCH --output=onlinehmr_%j.out
# Runs a command in the OnlineHMR image on an A40. The image has no sm_120 kernels, so never
# request an RTX PRO 6000; for an L40 pass --gres=gpu:nvidia_l40:1. Submit from the repo root:
#   sbatch apptainer/run.sbatch python scripts/run_custom_mt.py --video <VIDEO>.mp4 --no-viz --calib false --depth-mask
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
[ -x apptainer/run.sh ] || { echo "Submit from the repo root"; exit 1; }
nvidia-smi -L
exec apptainer/run.sh "$@"
