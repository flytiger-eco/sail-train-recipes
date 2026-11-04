#!/usr/bin/env bash
#
# MFU tuning for Qwen3.5-VL 2B (dense) — MegatronMIMO with vision CP
#
# Usage:
#   bash tune_2b.sh
#
# Dense model: no MoE experts, no EP/ETP.
# 2b fits on 1 node 8 GPUs with TP=1, DP=8.
#
# Results are saved under $OUTPUT_DIR/<tag>/train_node<RANK>.log

OUTPUT_DIR_DEFAULT="./mfu_tuning_2b_mimo"
NNODES_DEFAULT=1
MASTER_PORT_DEFAULT=29501
HF_MODEL_DEFAULT="Qwen/Qwen3.5-2B"

source "$(dirname "${BASH_SOURCE[0]}")/tune_common.sh"
apply_common_env

export MEGATRON_VISION_FIXED_LENGTH_ATTENTION=1
# Emit per-step "GPU utilization: <N> MODEL_TFLOP/s/GPU" + FLOPS snapshots (parsed by parse_steady_state.py)
export MIMO_DEBUG="${MIMO_DEBUG:-1}"

# ─── Best configuration ─────────────────────────────────────────────────────
BEST_LANG="${BEST_LANG:-language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0}"
BEST_IMG="${BEST_IMG:-images=tp=1,cp=1,dp=8,rank_offset=0}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-1920}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-1080}"
NUM_IMAGES="${NUM_IMAGES:-2}"
SEQ_LEN="${SEQ_LEN:-4096}"
MBS="${MBS:-1}"
GA="${GA:-8}"

cuda_graph_args "attn"

run_exp "2b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga${TAG_SUFFIX:-}" \
    --hf-model "$HF_MODEL" \
    --component "$BEST_LANG" \
    --component "$BEST_IMG" \
    --linear-cp-mode chunkwise --gdn-fused-pre-gdr \
    --mbs $MBS --ga $GA \
    --seq-length $SEQ_LEN --num-images $NUM_IMAGES --image-size-w $IMAGE_SIZE_W --image-size-h $IMAGE_SIZE_H \
    --enable-grad-overlap --enable-sp --cross-entropy-fusion-impl te ${CUDA_GRAPH_ARGS} ${EXTRA_ARGS:-}
