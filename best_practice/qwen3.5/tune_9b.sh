#!/usr/bin/env bash
#
# MFU tuning for Qwen3.5-VL 9B (dense) — Megatron-LM-for-sail path
#
# Usage:
#   bash tune_9b.sh
#
# One node, TP=1, DP=8, MBS 1 x GA 8 (global batch 64). The heavier 8192-token
# 4-image row runs TP=2 (DP=4) on this path as well:
#   TP=2 SEQ_LEN=8192 NUM_IMAGES=4 bash tune_9b.sh
# See tune_2b.sh for the mock-image geometry note.
#
# Results are saved under $OUTPUT_DIR/<tag>/train_node<RANK>.log

OUTPUT_DIR_DEFAULT="./mfu_tuning_9b_megatron"
NNODES_DEFAULT=1
MASTER_PORT_DEFAULT=29503
HF_MODEL_DEFAULT="Qwen/Qwen3.5-9B"

MODEL_VARIANT="${MODEL_VARIANT:-9b}"
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
SEQ_LEN="${SEQ_LEN:-4096}"
ENABLE_VISION_CP="${ENABLE_VISION_CP:-1}"
CUDA_GRAPH_MODULES="${CUDA_GRAPH_MODULES-attn}"
USE_FSDP_DEFAULT=0
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"

source "$(dirname "${BASH_SOURCE[0]}")/internal/common.sh"
apply_megatron_env
export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1

run_megatron "9b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga${TAG_SUFFIX:-}" \
    ${EXTRA_ARGS:-}
