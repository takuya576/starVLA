#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/home/takuya/starVLA
STARVLA_PYTHON=/data/takuya/miniconda3/envs/starVLA/bin/python

# === Checkpoint ===
CKPT=${STARVLA_DIR}/results/Checkpoints/20260604_172117_libero_all_WanGR00T_lora_true/checkpoints/steps_10000_pytorch_model.pt

gpu_id=5

# === Generation knobs ===
LIBERO_SUITE=libero_spatial
EPISODE=0
NUM_FRAMES=17
NUM_INFERENCE_STEPS=4
GUIDANCE_SCALE=5.0
HEIGHT=224
WIDTH=224
FPS=8
SEED=42
OUT=results/imagined/wan_${LIBERO_SUITE}_ep${EPISODE}.mp4

################# Imagine future video ######################
CUDA_VISIBLE_DEVICES=$gpu_id ${STARVLA_PYTHON} examples/LIBERO/eval_files/generate_wan_video.py \
    --ckpt ${CKPT} \
    --libero-suite ${LIBERO_SUITE} \
    --episode ${EPISODE} \
    --num-frames ${NUM_FRAMES} \
    --num-inference-steps ${NUM_INFERENCE_STEPS} \
    --guidance-scale ${GUIDANCE_SCALE} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --fps ${FPS} \
    --seed ${SEED} \
    --out ${OUT}
# ###########################################################
