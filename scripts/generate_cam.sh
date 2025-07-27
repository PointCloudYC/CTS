#!/bin/bash

# Generate CAMs for the best model on the ArchiStyle-v1 dataset or ArchiStyle-v2 dataset

echo "Generating CAMs for ArchiStyle-v1"
MODEL_DIR="experiments-v1/resnet50_pretrained"
DATA_DIR="data/ArchiStyle-v1"
MODEL_NAME="resnet50"

python function/generate_cam.py \
    --model_dir ${MODEL_DIR} \
    --data_dir ${DATA_DIR} \
    --model_name ${MODEL_NAME} \
    --pretrained    

echo "Generating CAMs for ArchiStyle-v2"
MODEL_DIR="experiments-v2/alexnet_pretrained"
DATA_DIR="data/ArchiStyle-v2"
MODEL_NAME="alexnet"

python function/generate_cam.py \
    --model_dir ${MODEL_DIR} \
    --data_dir ${DATA_DIR} \
    --model_name ${MODEL_NAME} \
    --pretrained    