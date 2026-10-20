#!/bin/bash
mkdir -p data/pretrain data/smpl

files=(
    "cascade_mask_rcnn_vitdet_h_75ep.py data/pretrain"
    "downsample_mat.pkl data/smpl"
    "J_regressor_extra.npy data/smpl"
    "J_regressor_h36m.npy data/smpl"
    "kintree_table.pkl data/smpl"
    "smpl_mean_params.npz data/smpl"
    "colors.txt data"
)

# Skip the download if every file is already in place
missing=0
for entry in "${files[@]}"; do
    read -r name dest <<< "$entry"
    [[ -s "$dest/$name" ]] || missing=1
done
if [[ $missing -eq 0 ]]; then
    echo "Extra files found, skipping"
    exit 0
fi

# -c resumes partial files and skips complete ones in tmp_download/ on re-runs
gdown -c --folder https://drive.google.com/drive/folders/1GWQKcrpJl84g9ouG5L3S1uy-eIloZJGI?usp=sharing -O tmp_download/

# gdown puts the files in a subfolder named after the Drive folder
failed=0
for entry in "${files[@]}"; do
    read -r name dest <<< "$entry"
    src=$(find tmp_download -type f -name "$name" 2>/dev/null | head -n 1)
    if [[ -n "$src" && -s "$src" ]]; then
        mv "$src" "$dest/"
    elif [[ ! -s "$dest/$name" ]]; then
        echo "Missing $name, re-run this script to retry"
        failed=1
    fi
done

# Keep tmp_download/ on failure so a re-run can resume
[[ $failed -eq 0 ]] && rm -rf tmp_download/
exit $failed
