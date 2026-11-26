#!/usr/bin/env bash
#
# MFU tuning for Qwen3.5-VL 35B-A3B (MoE) — Megatron-LM-for-sail path
#
# Usage:
#   bash tune_35b_a3b.sh
#
# Two nodes, TP=1, EP=16, DP=16, MBS 1 x GA 8 (global batch 128), with the flex
# DeepEP dispatcher and forced load balancing for the perf rows. The vision
# tower runs on the language ranks and --enable-vision-cp shards the images
# across CP ranks (inert at CP=1).
#
# Not available on this path: the MIMO EP-overlap schedule
# (--enable-ep-overlap), DeepEP autotune, vision activation offload,
# MEGATRON_SWIGLU_USE_CUDAC / MEGATRON_MOE_TOPK_RECONSTRUCT, and the fused
# pre-GDR GDN kernel (--gdn-fused-pre-gdr).
#
# Data shape: the mock provider does not resize, so 1920x1080 becomes 1920x1088
# and two images need SEQ_LEN 16384. TP=1 keeps the whole 35B parameter set on
# every rank; at 16384 tokens that is the memory ceiling. For 4096-token
# text-only rows use NUM_IMAGES=0 SEQ_LEN=4096 (text-only also drops the vision
# tower from the model).
#
# Launch the same command on both nodes; only RANK differs.
# Results are saved under $OUTPUT_DIR/<tag>/train_node<RANK>.log

OUTPUT_DIR_DEFAULT="./mfu_tuning_35b_a3b_megatron"
NNODES_DEFAULT=2
MASTER_PORT_DEFAULT=29500
HF_MODEL_DEFAULT="Qwen/Qwen3.5-35B-A3B"
COPY_ENGINE_WRAPPER=1

MODEL_VARIANT="${MODEL_VARIANT:-35b_a3b}"
TP="${TP:-1}"
EP="${EP:-16}"
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
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}"
DISPATCHER="${DISPATCHER:-flex}"
MOE_DISPATCHER_BACKEND="${MOE_DISPATCHER_BACKEND:-deepep}"
FORCE_LOAD_BALANCING="${FORCE_LOAD_BALANCING:-1}"

source "$(dirname "${BASH_SOURCE[0]}")/internal/common.sh"
apply_megatron_env

# DeepEP-specific settings, only relevant for the 35B-A3B MoE.
export DEEPEP_NUM_MAX_ICN_PEERS=16
export DEEP_EP_WARMUP_ENABLED=1
export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1

run_megatron "35ba3b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga${TAG_SUFFIX:-}" \
    ${EXTRA_ARGS:-}
