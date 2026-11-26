#!/usr/bin/env bash
#
# End-to-end smoke test for the Megatron-LM-for-sail path.
#
# Exercises multi-image mock data, THD packing, context parallelism and vision
# context parallelism on a single 8-GPU node with the proxy variant (4 decoder
# blocks, 16 experts), so it needs no model checkpoint. HF_MODEL only supplies
# the Qwen3.5 tokenizer: point it at any local Qwen3.5 snapshot, or leave the
# default when the hub cache holds it.
#
# Usage:
#   bash smoke.sh
#
# Two iterations, no checkpoint writes.

OUTPUT_DIR_DEFAULT="./smoke_out"
NNODES_DEFAULT=1
MASTER_PORT_DEFAULT=29511
HF_MODEL_DEFAULT="Qwen/Qwen3.5-2B"

MODEL_VARIANT="${MODEL_VARIANT:-proxy}"
TP="${TP:-1}"
EP="${EP:-4}"
PP="${PP:-1}"
CP="${CP:-2}"
MBS="${MBS:-1}"
GA="${GA:-2}"
NUM_IMAGES="${NUM_IMAGES:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
IMAGE_SEQ_LENGTH="${IMAGE_SEQ_LENGTH:-1024}"
SEQ_LEN="${SEQ_LEN:-2048}"
ENABLE_VISION_CP="${ENABLE_VISION_CP:-1}"
USE_PACKED_SEQUENCE="${USE_PACKED_SEQUENCE:-1}"
USE_FSDP_DEFAULT=0
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"
TRAIN_ITERS="${TRAIN_ITERS:-2}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
apply_megatron_env

run_megatron "smoke_proxy_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE}_cp${CP}_thd${TAG_SUFFIX:-}" \
    ${EXTRA_ARGS:-}
