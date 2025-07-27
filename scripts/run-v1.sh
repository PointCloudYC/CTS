#!/bin/bash
# runs training and evaluation on the ArchiStyle-v1 dataset

DATA_DIR="data/ArchiStyle-v1"
MODELS=("alexnet" "resnet50" "densenet121")
# MODELS=("densenet121")
EXPERIMENT_NAME="v1"

for MODEL_NAME in "${MODELS[@]}"; do
  # Run model from scratch
  MODEL_DIR="experiments-${EXPERIMENT_NAME}/${MODEL_NAME}_from_scratch"
  echo "Running ${MODEL_NAME} (from scratch) on ${DATA_DIR}..."
#   echo "training..."
#   python function/train.py \
#   --data_dir ${DATA_DIR} \
#   --model_name ${MODEL_NAME} \
#   --model_dir ${MODEL_DIR}

  python function/evaluate.py \
  --data_dir ${DATA_DIR} \
  --model_name ${MODEL_NAME} \
  --model_dir ${MODEL_DIR} \
  --save_confusion_matrix

  # Run pre-trained model
  MODEL_DIR="experiments-${EXPERIMENT_NAME}/${MODEL_NAME}_pretrained"
  echo "Running ${MODEL_NAME} (pre-trained) on ${DATA_DIR}..."
  echo "training..."
#   python function/train.py \
#   --data_dir ${DATA_DIR} \
#   --model_name ${MODEL_NAME} \
#   --model_dir ${MODEL_DIR} \
#   --pretrained

  python function/evaluate.py \
  --data_dir ${DATA_DIR} \
  --model_name ${MODEL_NAME} \
  --model_dir ${MODEL_DIR} \
  --pretrained \
  --save_confusion_matrix
done
