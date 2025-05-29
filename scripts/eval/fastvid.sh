#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL="llava_onevision_fastvid"
PRETRAINED="lmms-lab/llava-onevision-qwen2-7b-ov"

# ! FastVID Parameters
DYSEG_C=8
DYSEG_TAU=0.9
RETENTION_RATIO=0.10
STPRUNE_D=0.4
DTM_ALPHA=0.6
DTM_P=4
K=4

# ! Evaluation Tasks
TASKS="videomme,egoschema"

accelerate launch \
    --num_processes 8 \
    --main_process_port 12345 \
    -m lmms_eval \
    --model $MODEL \
    --model_args "pretrained=${PRETRAINED},dyseg_c=${DYSEG_C},dyseg_tau=${DYSEG_TAU},retention_ratio=${RETENTION_RATIO},stprune_d=${STPRUNE_D},dtm_alpha=${DTM_ALPHA},dtm_p=${DTM_P},k=${K}" \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${MODEL}_dyseg_c${DYSEG_C}_dyseg_tau${DYSEG_TAU}_retention_ratio${RETENTION_RATIO}_stprune_d${STPRUNE_D}_dtm_alpha${DTM_ALPHA}_dtm_p${DTM_P}_k${K}" \
    --output_path "./logs"
