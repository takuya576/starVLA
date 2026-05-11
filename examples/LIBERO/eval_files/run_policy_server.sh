#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/home/takuya/starVLA
LIBERO_HOME=/home/takuya/LIBERO
STARVLA_PYTHON=/data/takuya/miniconda3/envs/starVLA/bin/python
LIBERO_PYTHON=/data/takuya/miniconda3/envs/libero/bin/python

# === Checkpoint ===
CKPT=${STARVLA_DIR}/results/Checkpoints/libero_all_CosmoPredict2GR00T_lora_false/final_model/libero_all_CosmoPredict2GR00T_lora_false.pt

export star_vla_python=${STARVLA_PYTHON}
your_ckpt=${CKPT}
gpu_id=0
port=6694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
