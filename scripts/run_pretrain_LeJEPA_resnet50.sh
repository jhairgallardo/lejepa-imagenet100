#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# Pretraining arguments. Each variable sets the pretrain.py argument of the
# same name; edit them here to change the run.

# Data
dataset=im100
data_path=/path/to/ImageNet-100
probe_num_classes=100

# Model
vision_model=resnet50
num_views=2
proj_dim=512
lamb=0.01

# Optimization
epochs=400
warmup_epochs=10
bs=128 # per GPU
lr=2e-3
wd=1e-1
seed=0

# Hardware
devices=4
strategy=ddp
workers=6

# Output
save_dir=results/LeJEPA_${vision_model}_V${num_views}

python -u pretrain.py \
    --dataset "${dataset}" \
    --data_path "${data_path}" \
    --probe_num_classes "${probe_num_classes}" \
    --vision_model "${vision_model}" \
    --num_views "${num_views}" \
    --proj_dim "${proj_dim}" \
    --lamb "${lamb}" \
    --epochs "${epochs}" \
    --warmup_epochs "${warmup_epochs}" \
    --bs "${bs}" \
    --lr "${lr}" \
    --wd "${wd}" \
    --seed "${seed}" \
    --devices "${devices}" \
    --strategy "${strategy}" \
    --workers "${workers}" \
    --save_dir "${save_dir}"
