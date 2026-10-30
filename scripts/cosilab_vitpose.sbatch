#!/bin/bash
#SBATCH --job-name=cosilab_vitpose
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=00:10:00
#SBATCH --array=0-41
#SBATCH --output=logs/cosilab_vitpose_%A_%a.out
# Custom ViTPose (default: VIT003) on the COSI-Lab videos, on the boxes of the dataset ViTPose results.
# Array index = position in the sorted folder list, as in scripts/cosilab_smpl.sbatch. Extra arguments go
# to scripts/run_cosilab_vitpose.py. --time fits the 600-frame clips only; raise it for full batches.
#   sbatch --array=0,7 scripts/cosilab_vitpose.sbatch --start 6000 --num_frames 600
# Results: /tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_vitpose_vit003/<cam_batch>/
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
[ -x apptainer/run.sh ] || { echo "Submit from the repo root"; exit 1; }

DATASET=/tudelft.net/staff-umbrella/neon/cosilab_project/4TU_upload
MODELS=/tudelft.net/staff-umbrella/neon/experiments/VIT003
CAM_BATCHES=($(ls "$DATASET/processed/keypoints/vitpose_results"))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ -z "${CAM_BATCH:-}" ]; then
    [ "$TASK_ID" -lt "${#CAM_BATCHES[@]}" ] || { echo "No cam-batch for array index $TASK_ID"; exit 1; }
    CAM_BATCH=${CAM_BATCHES[$TASK_ID]}
fi
echo "Array task $TASK_ID: $CAM_BATCH"
nvidia-smi -L

# Mount the dataset and the checkpoint folder read-only, on top of the binds of apptainer/run.sh
export APPTAINER_BIND="$DATASET:$DATASET:ro,$MODELS:$MODELS:ro"
exec apptainer/run.sh python scripts/run_cosilab_vitpose.py --cam_batch "$CAM_BATCH" "$@"
