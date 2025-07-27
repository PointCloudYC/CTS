#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
DATA_DIR_V1="data/ArchiStyle-v1"
DATA_DIR_V2="data/ArchiStyle-v2"
MODELS_V1=("resnet50")
MODELS_V2=("alexnet")
GPU_ID=1

# --- Function to run evaluation ---
run_evaluation() {
  local data_dir=$1
  local model_name=$2
  local experiment_name=$3
  local model_dir="experiments-${experiment_name}/${model_name}_pretrained"
  echo "--- Evaluating ${model_name} from ${model_dir} on ${data_dir} ---"
  
  python function/evaluate.py \
    --data_dir "${data_dir}" \
    --model_name "${model_name}" \
    --model_dir "${model_dir}" \
    --gpu_id "${GPU_ID}" \
    --pretrained \
    --save_confusion_matrix
}

# --- Run for v1 experiments ---
echo "--- Starting v1 evaluations ---"
for model in "${MODELS_V1[@]}"; do
  echo "model: ${model}"
  echo "data_dir: ${DATA_DIR_V1}"
  echo "experiment_name: v1"
  echo "model_dir: experiments-v1/${model}_pretrained"
  echo "gpu_id: ${GPU_ID}"
  echo "pretrained: true"
  echo "save_confusion_matrix: true"
  echo "--------------------------------"
  run_evaluation "${DATA_DIR_V1}" "${model}" "v1"
done

# # --- Run for v2 experiments ---
# echo "--- Starting v2 evaluations ---"
# for model in "${MODELS_V2[@]}"; do
#   run_evaluation "${DATA_DIR_V2}" "${model}" "v2"
# done

echo "--- All evaluations complete ---"
