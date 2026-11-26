#!/usr/bin/env bash
#
# MFU tuning for Qwen3.5-VL 4B (dense) — Megatron-LM-for-sail path
#
# Usage:
#   bash tune_4b.sh
#
# One node, TP=1, DP=8, MBS 1 x GA 8 (global batch 64). See tune_2b.sh for the
# mock-image geometry note (the mock provider does not resize; 1920x1080
# becomes 1920x1088).
#
# Results are saved under $OUTPUT_DIR/<tag>/train_node<RANK>.log

OUTPUT_DIR_DEFAULT="./mfu_tuning_4b_megatron"
NNODES_DEFAULT=1
MASTER_PORT_DEFAULT=29502
HF_MODEL_DEFAULT="Qwen/Qwen3.5-4B"

MODEL_VARIANT="${MODEL_VARIANT:-4b}"
TP="${TP:-1}"
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
ENABLE_VISION_CP="${ENABLE_VISION_CP:-1}"
CUDA_GRAPH_MODULES="${CUDA_GRAPH_MODULES-attn}"
USE_FSDP_DEFAULT=0
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"

source "$(dirname "${BASH_SOURCE[0]}")/internal/common.sh"
apply_megatron_env

run_megatron "4b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga${TAG_SUFFIX:-}" \
    ${EXTRA_ARGS:-}
