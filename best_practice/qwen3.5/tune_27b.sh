#!/usr/bin/env bash
#
# MFU tuning for Qwen3.5-VL 27B (dense) — standard VLM path
#
# Usage:
#   bash tune_27b.sh
#
# Uses the standard VLM training path (pretrain) instead of the MegatronMIMO
# path used by tune_2b.sh / tune_4b.sh / tune_9b.sh / tune_35b_a3b.sh.
# All features (recompute, CUDA graph, offload, etc.) are aligned;
# only the training path differs (pretrain vs pretrain_megatron_mimo).

OUTPUT_DIR_DEFAULT="./mfu_tuning_27b_vlm"
NNODES_DEFAULT=1
MASTER_PORT_DEFAULT=29501
HF_MODEL_DEFAULT="Qwen/Qwen3.5-27B"
SCRIPT="${SCRIPT:-train_qwen35_vl_vlm.py}"

source "$(dirname "${BASH_SOURCE[0]}")/tune_common.sh"
apply_common_env

export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1
export MEGATRON_VISION_FIXED_LENGTH_ATTENTION=1

# ─── Best configuration ─────────────────────────────────────────────────────
TP="${TP:-2}"
CP="${CP:-1}"
PP="${PP:-1}"
VPP="${VPP:-1}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-1920}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-1080}"
NUM_IMAGES="${NUM_IMAGES:-2}"
SEQ_LEN="${SEQ_LEN:-4096}"
MBS="${MBS:-1}"
GA="${GA:-8}"

run_exp "27b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga${TAG_SUFFIX:-}" \
    --model-size 27b \
    --hf-processor-path "$HF_MODEL" \
    --tp $TP --cp $CP \
    --linear-cp-mode chunkwise --gdn-fused-pre-gdr \
    --mbs $MBS --ga $GA \
    --seq-length $SEQ_LEN --num-images $NUM_IMAGES --image-size-w $IMAGE_SIZE_W --image-size-h $IMAGE_SIZE_H \
    --enable-sp --enable-fsdp --enable-grad-overlap --cross-entropy-fusion-impl te \
    ${EXTRA_ARGS:-}
