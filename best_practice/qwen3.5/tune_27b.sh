#!/usr/bin/env bash
#
# MFU tuning for Qwen3.5-VL 27B (dense) — Megatron-LM-for-sail path
#
# Usage:
#   bash tune_27b.sh
#
# Two nodes, TP=2, DP=8, MBS 1 x GA 8 (global batch 64), with Megatron-FSDP
# sharding parameters and optimizer state across the DP ranks. TE CUDA graphs
# stay off by default (CUDA_GRAPH_MODULES=attn to opt in).
#
# Data shape: the mock provider does not resize, so 1920x1080 is snapped to
# 1920x1088 and two images need SEQ_LEN 16384; for a 4096-token cell shrink the
# images:
#   NUM_IMAGES=2 IMAGE_SIZE_W=448 IMAGE_SIZE_H=448 IMAGE_SEQ_LENGTH=1024 SEQ_LEN=4096
#
# Launch the same command on both nodes; only RANK differs.
# Results are saved under $OUTPUT_DIR/<tag>/train_node<RANK>.log

OUTPUT_DIR_DEFAULT="./mfu_tuning_27b_megatron"
NNODES_DEFAULT=2
MASTER_PORT_DEFAULT=29501
HF_MODEL_DEFAULT="Qwen/Qwen3.5-27B"

MODEL_VARIANT="${MODEL_VARIANT:-27b}"
TP="${TP:-2}"
EP="${EP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-1}"
GA="${GA:-8}"
NUM_IMAGES="${NUM_IMAGES:-2}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-1920}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-1088}"
IMAGE_SEQ_LENGTH="${IMAGE_SEQ_LENGTH:-8192}"
SEQ_LEN="${SEQ_LEN:-16384}"
ENABLE_VISION_CP="${ENABLE_VISION_CP:-0}"
CUDA_GRAPH_MODULES="${CUDA_GRAPH_MODULES-}"
USE_FSDP_DEFAULT=1
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"

source "$(dirname "${BASH_SOURCE[0]}")/internal/common.sh"
apply_megatron_env

export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1

run_megatron "27b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga_fsdp${TAG_SUFFIX:-}" \
    ${EXTRA_ARGS:-}
