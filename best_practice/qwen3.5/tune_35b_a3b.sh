#!/usr/bin/env bash
#
# MFU tuning for Qwen3.5-VL 35B-A3B (MoE) — MegatronMIMO with vision CP
#
# Usage:
#   bash tune_35b_a3b.sh
#
# MIMO uses disjoint rank groups for language and image encoder:
#   --component language=tp=N,etp=M,dp=K,rank_offset=0
#   --component images=tp=1,dp=<world>,rank_offset=0     (colocated)
#
# For MoE models, use etp (or ep) in the component spec for expert parallelism.
# In MIMO, etp maps to the EP dimension in HyperCommGrid.
#
# Results are saved under $OUTPUT_DIR/<tag>/train_node${RANK}.log

OUTPUT_DIR_DEFAULT="./mfu_tuning_35b_a3b_mimo_cp"
NNODES_DEFAULT=2
MASTER_PORT_DEFAULT=29500
HF_MODEL_DEFAULT="Qwen/Qwen3.5-35B-A3B"
COPY_ENGINE_WRAPPER=1

source "$(dirname "${BASH_SOURCE[0]}")/tune_common.sh"
apply_common_env

# DeepEP-specific settings, only relevant for 35B-A3B MoE.
export DEEPEP_NUM_MAX_ICN_PEERS=16
export DEEP_EP_WARMUP_ENABLED=1
export MEGATRON_VISION_FIXED_LENGTH_ATTENTION=1
export MIMO_DEBUG="${MIMO_DEBUG:-1}"
export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1
# Never write PCCL_MNNVL_ENABLE: the runtime enables MNNVL by default and the
# copy engine needs peer memory.

# ─── Best configuration ─────────────────────────────────────────────────────
BEST_LANG="${BEST_LANG:-language=tp=1,cp=1,ep=16,dp=16,etp=1,pp=1,rank_offset=0}"
BEST_IMG="${BEST_IMG:-images=tp=1,cp=1,dp=16,rank_offset=0}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-1920}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-1080}"
NUM_IMAGES="${NUM_IMAGES:-2}"
SEQ_LEN="${SEQ_LEN:-4096}"
MBS="${MBS:-1}"
GA="${GA:-8}"

# EP overlap is a MIMO schedule-plan path: pure text and language TP=CP=PP=1.
# Image-heavy and CP>1 rows must leave it off (the entrypoint rejects them).
ENABLE_EP_OVERLAP="${ENABLE_EP_OVERLAP:-0}"
DELAY_WGRAD="${DELAY_WGRAD:-0}"
OVERLAP_ARGS=""
if [[ "$ENABLE_EP_OVERLAP" == "1" ]]; then
    OVERLAP_ARGS="--enable-ep-overlap"
    [[ "$DELAY_WGRAD" == "1" ]] && OVERLAP_ARGS+=" --delay-wgrad"
fi
MOE_DEEPEP_NUM_SMS="${MOE_DEEPEP_NUM_SMS:-20}"

cuda_graph_args ""

run_exp "35ba3b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga${TAG_SUFFIX:-}" \
    --hf-model "$HF_MODEL" \
    --component "$BEST_LANG" \
    --component "$BEST_IMG" \
    --moe-dispatcher-type flex --moe-dispatcher-backend deepep \
    --moe-deepep-num-sms "$MOE_DEEPEP_NUM_SMS" \
    --linear-cp-mode chunkwise --gdn-fused-pre-gdr \
    --enable-grad-overlap --mbs $MBS --ga $GA \
    --seq-length $SEQ_LEN --num-images $NUM_IMAGES --image-size-w $IMAGE_SIZE_W --image-size-h $IMAGE_SIZE_H \
    --enable-sp --enable-vision-cp --moe-force-load-balancing ${CUDA_GRAPH_ARGS} ${OVERLAP_ARGS} --enable-deepep-autotune --enable-vision-activation-offload --vision-offload-num-layers 20 --cross-entropy-fusion-impl te ${EXTRA_ARGS:-}
