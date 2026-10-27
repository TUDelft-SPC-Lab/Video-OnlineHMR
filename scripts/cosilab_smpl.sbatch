#!/bin/bash
#SBATCH --job-name=cosilab_smpl
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=04:00:00
#SBATCH --array=0-41
#SBATCH --output=logs/cosilab_smpl_%A_%a.out
# OnlineHMR (SMPL only) on the COSI-Lab videos, one array task per camXX_batchYY folder of the
# ViTPose results (42 in total, index = position in the sorted folder list, so 0 = cam01_batch01,
# 7 = cam03_batch01, ...). Extra arguments go to scripts/run_cosilab_smpl.py. Submit from the repo root:
#   sbatch scripts/cosilab_smpl.sbatch                                   # all cam-batches
#   sbatch scripts/cosilab_smpl.sbatch --refine                          # also fit to the keypoints
#   sbatch --array=0 scripts/cosilab_smpl.sbatch --num_frames 600 --start 6000 --refine   # smoke test
#   CAM_BATCH=cam08_batch03 sbatch --array=0 scripts/cosilab_smpl.sbatch                  # one by name
# Results: /tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_smpl/<cam_batch>/
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
[ -x apptainer/run.sh ] || { echo "Submit from the repo root"; exit 1; }

DATASET=/tudelft.net/staff-umbrella/neon/cosilab_project/4TU_upload
CAM_BATCHES=($(ls "$DATASET/processed/keypoints/vitpose_results"))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ -z "${CAM_BATCH:-}" ]; then
    [ "$TASK_ID" -lt "${#CAM_BATCHES[@]}" ] || { echo "No cam-batch for array index $TASK_ID"; exit 1; }
    CAM_BATCH=${CAM_BATCHES[$TASK_ID]}
fi
echo "Array task $TASK_ID: $CAM_BATCH"
nvidia-smi -L

# Mount the dataset read-only; apptainer adds APPTAINER_BIND to the binds of apptainer/run.sh
export APPTAINER_BIND="$DATASET:$DATASET:ro"
exec apptainer/run.sh python scripts/run_cosilab_smpl.py --cam_batch "$CAM_BATCH" "$@"
