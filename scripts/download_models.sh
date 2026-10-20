#!/bin/bash
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

# SMPL Male and Female model (skipped if already present)
if [[ ! -s data/smpl/SMPL_FEMALE.pkl || ! -s data/smpl/SMPL_MALE.pkl ]]; then
    echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
    read -p "Username (SMPL):" username
    read -s -p "Password (SMPL):" password; echo
    username=$(urle $username)
    password=$(urle $password)

    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' -O './data/smpl/smpl.zip' --no-check-certificate --continue
    unzip data/smpl/smpl.zip -d data/smpl/smpl
    mv data/smpl/smpl/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl data/smpl/SMPL_FEMALE.pkl
    mv data/smpl/smpl/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl data/smpl/SMPL_MALE.pkl
    rm -rf data/smpl/smpl
    rm -rf data/smpl/smpl.zip
else
    echo "SMPL male/female models found, skipping"
fi

# VIBE occlusion augmentation
mkdir -p ./data ./data/pretrain/hmr2b ./checkpoints
gdown -c -O ./data/pascal_occluders.pkl https://drive.google.com/file/d/1_Qv9eAKVkfvZjdl9qaRyxVrIeAnAwevE/view?usp=sharing

# Thirdparty checkpoints
gdown -c -O ./data/pretrain/hmr2b/epoch=35-step=1000000.ckpt https://drive.google.com/file/d/1W4fcp8mwS19Rg_A7MoTS1lc7JafqTGu-/view?usp=sharing
wget -c -O ./data/pretrain/DEVA-propagation.pth https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth
wget -c -O ./data/pretrain/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
gdown -c -O ./data/pretrain/camcalib_sa_biased_l2.ckpt https://drive.google.com/file/d/1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO/view?usp=sharing

wget -c -O ./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
wget -c -O ./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
wget -c -O ./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl

# MoGe-2 (direct download, avoids needing git-lfs)
wget -c -O ./data/pretrain/mogev2_model.pt https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt

# Our checkpoint
mkdir -p ./results/onlinehmr
gdown -c -O ./results/onlinehmr/checkpoint.pth.tar "https://drive.google.com/uc?id=1bbMhJyk9Qk5Ub01y7MJG5xFzRRkKB2Mm"

# World eval
gdown -c -O ./results/emdb2_results.zip "https://drive.google.com/uc?id=1XIoaUSTK_3QYgJuZXph_xIdrUxtii17n"
unzip results/emdb2_results.zip -d results/
rm -rf results/emdb2_results.zip 
