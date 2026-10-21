#!/bin/bash
#SBATCH --job-name=build-onlinehmr-sif
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=build_sif_%j.out
# Builds the OnlineHMR Apptainer image. No GPU needed. Submit from the repo root:
#   sbatch apptainer/build.sbatch
set -euo pipefail

EXP_DIR=/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR
SIF=${SIF:-$EXP_DIR/onlinehmr.sif}

# Build scratch on node-local disk, not in home.
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/apptainer-build.XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
export APPTAINER_TMPDIR=$SCRATCH/tmp APPTAINER_CACHEDIR=$SCRATCH/cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
export MAX_JOBS=${SLURM_CPUS_PER_TASK:-8}

cd "${SLURM_SUBMIT_DIR:-.}"
[ -f apptainer/onlinehmr.def ] || { echo "Submit from the repo root"; exit 1; }
apptainer build --force "$SCRATCH/onlinehmr.sif" apptainer/onlinehmr.def
mv "$SCRATCH/onlinehmr.sif" "$SIF"
echo "Image written to $SIF"
