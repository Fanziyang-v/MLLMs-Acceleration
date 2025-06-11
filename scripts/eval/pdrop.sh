#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL="llava_pdrop"
PRETRAINED="liuhaotian/llava-v1.5-7b"

# ! PyramidDrop parameters
PRUNING_LAYERS="2|8|16|24"
RETENTION_RATIO=0.5

# ! Evaluation Tasks
TASKS="mme,pope"

accelerate launch \
    --num_processes 8 \
    --main_process_port 12345 \
    -m lmms_eval \
    --model $MODEL \
    --model_args "pretrained=${PRETRAINED},pruning_layers=${PRUNING_LAYERS},retention_ratio=${RETENTION_RATIO}" \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${MODEL}_${TASKS}" \
    --output_path "./logs"
