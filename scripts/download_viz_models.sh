#!/bin/bash
# Download only the SMPL files needed by visualize_viser.py and scripts/visualize_smpl_projection.py
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

mkdir -p ./data/smpl

# SMPL Neutral model (skipped if already present)
if [[ ! -s data/smpl/SMPL_NEUTRAL.pkl ]]; then
    echo -e "\nYou need to register at https://smplify.is.tue.mpg.de"
    read -p "Username (SMPLify):" username
    read -s -p "Password (SMPLify):" password; echo
    username=$(urle $username)
    password=$(urle $password)

    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' -O './data/smpl/smplify.zip' --no-check-certificate --continue
    unzip data/smpl/smplify.zip -d data/smpl/smplify
    mv data/smpl/smplify/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl data/smpl/SMPL_NEUTRAL.pkl
    rm -rf data/smpl/smplify
    rm -rf data/smpl/smplify.zip
else
    echo "SMPL neutral model found, skipping"
fi

# Extra joint regressor, from the same Drive folder as download_extra.sh (skipped if already present)
if [[ ! -s data/smpl/J_regressor_extra.npy ]]; then
    gdown -c 1Qm67H7iPTuiuHj3pqgOqKf9WcKyhCDJU -O data/smpl/J_regressor_extra.npy
else
    echo "J_regressor_extra.npy found, skipping"
fi

# Fail if anything is still missing so a re-run can retry
failed=0
for f in data/smpl/SMPL_NEUTRAL.pkl data/smpl/J_regressor_extra.npy; do
    if [[ ! -s "$f" ]]; then
        echo "Missing $f, re-run this script to retry"
        failed=1
    fi
done
exit $failed
