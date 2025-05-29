#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL="llava_visionzip"
PRETRAINED="liuhaotian/llava-v1.5-7b"

# ! VisionZip parameters
DOMINANT=54
CONTEXTUAL=10

# ! Evaluation Tasks
TASKS="mme,pope"

accelerate launch \
    --num_processes 8 \
    --main_process_port 12345 \
    -m lmms_eval \
    --model $MODEL \
    --model_args "pretrained=${PRETRAINED},dominant=${DOMINANT},contextual=${CONTEXTUAL}" \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${MODEL}_${TASKS}" \
    --output_path "./logs"
