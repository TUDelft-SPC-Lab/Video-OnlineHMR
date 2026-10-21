#!/bin/bash
# Run a command inside the OnlineHMR image, from the repo root, with the GPU and the
# experiment folder (models, data, outputs, caches) mounted. Examples:
#   apptainer/run.sh python scripts/run_custom_mt.py --video <VIDEO>.mp4 --no-viz --calib false --depth-mask
#   apptainer/run.sh bash      # interactive shell
set -euo pipefail

EXP_DIR=/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR
SIF=${SIF:-$EXP_DIR/onlinehmr.sif}
REPO=$(cd "$(dirname "$0")/.." && pwd -P)

# Model downloads at runtime (ViTDet-H via detectron2, torch hub, HF) go to project storage, not home.
export APPTAINERENV_FVCORE_CACHE=$EXP_DIR/cache/iopath
export APPTAINERENV_TORCH_HOME=$EXP_DIR/cache/torch
export APPTAINERENV_HF_HOME=$EXP_DIR/cache/huggingface
mkdir -p "$EXP_DIR"/cache/{iopath,torch,huggingface}

# data/, checkpoints/, datasets/, logs/, results/, res_human_camera/ in the repo are symlinks
# into $EXP_DIR, so mounting $EXP_DIR at the same path makes them resolve inside the container.
exec apptainer exec --nv \
    --bind "$EXP_DIR" \
    --bind "$REPO" \
    --pwd "$REPO" \
    "$SIF" "$@"
