#!/usr/bin/env bash
#
# MFU tuning for Qwen3.5-VL 122B-A10B (MoE, FSDP) — Megatron-LM-for-sail path
#
# Usage:
#   bash tune_122b_a10b.sh
#
# Eight nodes, TP=1, PP=1, EP=16, DP=64, MBS 1 x GA 2, so global batch 128 and
# DEEPEP_NUM_MAX_ICN_PEERS follows EP at CP=1. Megatron-FSDP shards parameters,
# gradients and optimizer state across the DP ranks.
#
# CP handling:
#   * --linear-cp-mode is chunkwise at CP=1 and headwise at CP>1: chunkwise
#     deadlocks at the first GDN layer on this model at CP>1.
#   * At CP>1 the allgather dispatcher replaces the DeepEP flex dispatcher and
#     forced routing is dropped; the grad-reduce overlap and DeepEP autotune
#     that the Bridge recipe switched off there do not exist on this path.
#   * The vision tower runs on the language ranks, so --enable-vision-cp
#     shards the images across CP ranks (inert at CP=1). There is no vision
#     activation offload on this path.
#
# Recompute follows SEQ_LEN (full at 8192 and above, off below); an explicit
# RECOMPUTE always wins.
#
# Launch the same command on all eight nodes; only RANK differs.
# Results are saved under $OUTPUT_DIR/<tag>/train_node<RANK>.log

OUTPUT_DIR_DEFAULT="./mfu_tuning_122b_megatron"
NNODES_DEFAULT=8
MASTER_PORT_DEFAULT=29624
HF_MODEL_DEFAULT="Qwen/Qwen3.5-122B-A10B"

MODEL_VARIANT="${MODEL_VARIANT:-122b_a10b}"
TP="${TP:-1}"
EP="${EP:-16}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-1}"
GA="${GA:-2}"
NUM_IMAGES="${NUM_IMAGES:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
IMAGE_SEQ_LENGTH="${IMAGE_SEQ_LENGTH:-1024}"
SEQ_LEN="${SEQ_LEN:-8192}"
ENABLE_VISION_CP="${ENABLE_VISION_CP:-1}"
CUDA_GRAPH_MODULES="${CUDA_GRAPH_MODULES-}"
USE_FSDP_DEFAULT=1
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}"

# GDN CP strategy. Chunkwise deadlocks on this model at CP>1 (py-spy: ranks
# split between build_cp_context and the MoE allgather), so CP>1 uses headwise.
if [[ "$CP" == "1" ]]; then
    LINEAR_CP_MODE="${LINEAR_CP_MODE:-chunkwise}"
else
    LINEAR_CP_MODE="${LINEAR_CP_MODE:-headwise}"
fi

# CP>1 cannot use the DeepEP flex dispatcher yet (its static NVL buffer overflows
# at 122B cp2/16k); the allgather dispatcher only needs allgather/reduce-scatter
# on the expert group. Those cells only have to complete, not win on throughput.
if [[ "$CP" == "1" ]]; then
    DISPATCHER="${DISPATCHER:-flex}"
    MOE_DISPATCHER_BACKEND="${MOE_DISPATCHER_BACKEND:-deepep}"
    FORCE_LOAD_BALANCING="${FORCE_LOAD_BALANCING:-1}"
else
    DISPATCHER="${DISPATCHER:-alltoall}"
    FORCE_LOAD_BALANCING="${FORCE_LOAD_BALANCING:-0}"
fi

# DeepEP only takes the fabric path when this equals the EP group size and the
# group spans more ranks than one node. CP>1 stays on the RDMA path: the fabric
# peer mapping does not cover the CP rank layout yet.
if [[ "$CP" == "1" ]]; then
    DEEPEP_NUM_MAX_ICN_PEERS="${DEEPEP_NUM_MAX_ICN_PEERS:-$EP}"
else
    DEEPEP_NUM_MAX_ICN_PEERS="${DEEPEP_NUM_MAX_ICN_PEERS:-1}"
fi

# Recompute follows SEQ_LEN (full at 8192 and above, off below); an explicit
# RECOMPUTE always wins.
if [[ -z "${RECOMPUTE:-}" ]]; then
    if (( SEQ_LEN >= 8192 )); then
        RECOMPUTE=1
    else
        RECOMPUTE=0
    fi
fi

source "$(dirname "${BASH_SOURCE[0]}")/internal/common.sh"
apply_megatron_env

# DeepEP-specific settings.
export DEEPEP_NUM_MAX_ICN_PEERS
export DEEP_EP_WARMUP_ENABLED=1
export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1

run_megatron "122ba10b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE}_${MBS}mbs_${GA}ga_cp${CP}_fsdp${TAG_SUFFIX:-}" \
    ${EXTRA_ARGS:-}
