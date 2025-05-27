#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL="llava_onevision_prunevid"
TASKS="videomme"
PRETRAINED="lmms-lab/llava-onevision-qwen2-7b-ov"

# ! PruneVid parameters
TEMPORAL_SEGMENT_RATIO=0.25
K=7
THRESHOLD=0.8
CLUSTER_RATIO=0.5
RETENTION_RATIO=0.4
SELECTED_LAYER=10


accelerate launch \
    --num_processes 8 \
    --main_process_port 12345 \
    -m lmms_eval \
    --model $MODEL \
    --model_args "pretrained=${PRETRAINED},temporal_segment_ratio=${TEMPORAL_SEGMENT_RATIO},k=${K},cluster_ratio=${CLUSTER_RATIO},retention_ratio=${RETENTION_RATIO},selected_layer=${SELECTED_LAYER}" \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${MODEL}_${TASKS}" \
    --output_path "./logs"
