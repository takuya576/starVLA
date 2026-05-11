

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4,5
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=WanGR00T
freeze_module_list='backbone'
lora_enable=false
# base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
base_wm=playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=${data_mix}_${Framework_name}_lora_${lora_enable}
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \
  --gpu_ids 4,5 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.action_model.future_action_window_size 7 \
  --framework.world_model.base_wm ${base_wm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.lora.enable ${lora_enable} \
  --trainer.max_train_steps 40000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity takuya50719-the-university-of-tokyo \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
