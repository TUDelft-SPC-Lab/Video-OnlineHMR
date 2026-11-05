#!/bin/bash
#SBATCH --job-name=cosilab_detect_vitpose
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=00:15:00
#SBATCH --array=0-1
#SBATCH --output=logs/cosilab_detect_vitpose_%A_%a.out
# ViTDet-H person boxes + VIT003 keypoints for the COSI-Lab cameras without dataset keypoints, read from the raw
# GoPro files. Needs <out_dir>/<cam_batch>/proposals.npz of scripts/cosilab_project_boxes.py first.
# Array index -> CAM_BATCHES below (or set CAM_BATCH). Extra arguments go to scripts/run_cosilab_detect_vitpose.py:
#   sbatch scripts/cosilab_detect_vitpose.sbatch --start 6000 --num_frames 600
#   CAM_BATCH_LIST="cam12_batch01 cam13_batch01 cam14_batch01 cam15_batch01" sbatch --array=0-3 scripts/cosilab_detect_vitpose.sbatch ...
# --time fits the 600-frame clips only. Results: .../Video-OnlineHMR/cosilab_vit003_extra/<cam_batch>/detections.npz
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
[ -x apptainer/run.sh ] || { echo "Submit from the repo root"; exit 1; }

DATASET=/tudelft.net/staff-umbrella/neon/cosilab_project/4TU_upload
MODELS=/tudelft.net/staff-umbrella/neon/experiments/VIT003
CAM_BATCHES=(${CAM_BATCH_LIST:-cam02_batch01 cam04_batch01})  # CAM_BATCH_LIST="cam12_batch01 ..." overrides
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CAM_BATCH=${CAM_BATCH:-${CAM_BATCHES[$TASK_ID]}}
echo "Array task $TASK_ID: $CAM_BATCH"
nvidia-smi -L

PROJECT=$(dirname "$DATASET")
export APPTAINER_BIND="$DATASET:$DATASET:ro,$MODELS:$MODELS:ro,$PROJECT/raw_sensor_data:$PROJECT/raw_sensor_data:ro,$PROJECT/processed_data:$PROJECT/processed_data:ro"
exec apptainer/run.sh python scripts/run_cosilab_detect_vitpose.py --cam_batch "$CAM_BATCH" "$@"
