#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# Linear evaluation arguments; edit them here to change the run. Variables
# named after a transfer_linear.py argument set that argument; the others
# build paths.

# Pretrained encoder
vision_model=vit_small_patch16_224 # [resnet50, vit_small_patch16_224]
pretrain_views=2
pretrain_expt=LeJEPA_${vision_model}_V${pretrain_views}
pretrained_encoder=results/${pretrain_expt}/model_epoch_399.pth

# Feature extraction
bs=128
workers=6

# Data
datasets_folder=/path/to/datasets

# One dataset per line: data_tag data_path metric.
# Comment out a line to skip that dataset.
datasets=(
    "im100      ${datasets_folder}/ImageNet-100    micro"
    "cifar10    ${datasets_folder}/CIFAR10         micro"
    "cifar100   ${datasets_folder}/CIFAR100        micro"
    "food101    ${datasets_folder}/Food-101        micro"
    "pets       ${datasets_folder}/oxford_pets     macro"
    "flowers102 ${datasets_folder}/oxford_flowers  macro"
    "caltech101 ${datasets_folder}/caltech-101     macro"
    "cars       ${datasets_folder}/stanford_cars   micro"
    "aircraft   ${datasets_folder}/fgvc_aircraft   macro"
    "dtd        ${datasets_folder}/dtd             micro"
    "sun397     ${datasets_folder}/sun397          micro"
    "mit67      ${datasets_folder}/MIT67           micro"
    "cub200     ${datasets_folder}/CUB_200_2011    micro"
)

for entry in "${datasets[@]}"; do
    read -r data_tag data_path metric <<< "${entry}"
    echo "Linear evaluation: ${data_tag} (${metric})"
    python -u transfer_linear.py \
        --dataset "${data_tag}" \
        --data_path "${data_path}" \
        --metric "${metric}" \
        --pretrained_encoder "${pretrained_encoder}" \
        --vision_model "${vision_model}" \
        --bs "${bs}" \
        --workers "${workers}" \
        --save_dir "results/linear/${pretrain_expt}/${data_tag}/${data_tag}_lbfgs"
done
