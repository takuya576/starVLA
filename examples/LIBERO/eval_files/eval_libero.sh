#!/bin/bash
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/home/takuya/starVLA

cd ${STARVLA_DIR}
conda activate starVLA
# === Checkpoint ===
CKPT=${STARVLA_DIR}/results/Checkpoints/20260604_172117_libero_all_WanGR00T_lora_true/checkpoints/steps_10000_pytorch_model.pt


###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/home/takuya/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/data/takuya/miniconda3/envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

host="127.0.0.1"
base_port=6694
unnorm_key="franka"
your_ckpt=${CKPT}

# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# model_root: <run_id> directory (parent of final_model/ or checkpoints/)
model_root=$(dirname "$(dirname "$your_ckpt")")
# === End of environment variable configuration ===
###########################################################################################

task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="${model_root}/results/${task_suite_name}/${folder_name}"

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
