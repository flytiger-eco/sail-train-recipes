#!/usr/bin/env bash
# Rank entrypoint for the Qwen3.5 cluster training CI.
#
# The scheduler starts one container per node and runs this script once per rank.
# It provides NNODES / NODE_RANK / NPROC_PER_NODE / MASTER_ADDR / MASTER_PORT /
# TRAIN_ITERS / OUTPUT_DIR plus the matrix keys of the workflow declaration.

set -uo pipefail

SCRIPT="best_practice/qwen3.5/train_qwen35_vl_mimo.py"

MODEL="${MODEL:-}"
HF_MODEL="${HF_MODEL:?HF_MODEL must be provided by the CI declaration}"

DENSE_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0"
DENSE_IMG="images=tp=1,cp=1,dp=8,rank_offset=0"
DENSE_ARGS=(--enable-grad-overlap --enable-sp
            --enable-cuda-graph-te --cuda-graph-scope attn)

LANG_LAYOUT="$DENSE_LANG"
IMG_LAYOUT="$DENSE_IMG"
MODEL_ARGS=("${DENSE_ARGS[@]}")

case "$MODEL" in
    qwen35-2b|qwen35-4b)
        ;;
    qwen35-35b-a3b)
        LANG_LAYOUT="language=tp=1,cp=1,ep=16,dp=16,etp=1,pp=1,rank_offset=0"
        IMG_LAYOUT="images=tp=1,cp=1,dp=16,rank_offset=0"
        MODEL_ARGS=(--enable-grad-overlap --enable-sp --enable-vision-cp
                    --enable-cuda-graph-te
                    --moe-dispatcher-type flex --moe-dispatcher-backend deepep
                    --moe-deepep-num-sms 20 --moe-force-load-balancing
                    --enable-deepep-autotune
                    --enable-vision-activation-offload --vision-offload-num-layers 20)
        ;;
    *)
        echo "MODEL must be qwen35-2b, qwen35-4b or qwen35-35b-a3b (got '${MODEL}')" >&2
        exit 2
        ;;
esac

NPROC="${NPROC_PER_NODE:-8}"
TOTAL_NODES="${NNODES:-1}"
RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is required}"
MASTER_PORT="${MASTER_PORT:-29500}"
ITERS="${TRAIN_ITERS:-20}"
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_IMAGES="${NUM_IMAGES:-2}"
MBS="${MBS:-1}"
GA="${GA:-8}"
OUT="${OUTPUT_DIR:-$PWD/ci_output}"
mkdir -p "$OUT"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLA_CACHE_MODE="${FLA_CACHE_MODE:-default}"
export ENABLE_COPY_ENGINE=0
export PCCL_CTA_POLICY=0x1
export NCCL_CTA_POLICY=0x1
export PCCL_CE_ENABLE_RS=0
export NCCL_GRAPH_REGISTER=0
export PCCL_GRAPH_REGISTER=0
export MEGATRON_LOGGING_LEVEL=20
export MEGATRON_SWIGLU_USE_CUDAC=1
export MEGATRON_MOE_TOPK_RECONSTRUCT=triton
export NVTE_CPU_OFFLOAD_V1=1
export UB_SKIPMC=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=1
export TRITON_CACHE_DIR="/tmp/triton_cache_$(hostname)"
export MEGATRON_VISION_FIXED_LENGTH_ATTENTION=1
export MIMO_DEBUG="${MIMO_DEBUG:-1}"
export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1
export DEEPEP_NUM_MAX_ICN_PEERS=16
export DEEP_EP_WARMUP_ENABLED=1

echo "[ci] model=${MODEL} hf=${HF_MODEL} rank=${RANK}/${TOTAL_NODES}" \
     "nproc=${NPROC} seq=${SEQ_LEN} images=${NUM_IMAGES} iters=${ITERS}"

python -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --nnodes="$TOTAL_NODES" \
    --node-rank="$RANK" \
    --rdzv-backend=static \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --max-restarts=0 \
    "$SCRIPT" \
    --hf-model "$HF_MODEL" \
    --component "$LANG_LAYOUT" \
    --component "$IMG_LAYOUT" \
    --seq-length "$SEQ_LEN" \
    --num-images "$NUM_IMAGES" \
    --image-size-w 1920 --image-size-h 1080 \
    --mbs "$MBS" --ga "$GA" \
    --train-iters "$ITERS" --log-interval 1 \
    --output-dir "$OUT/tensorboard" \
    --linear-cp-mode chunkwise --gdn-fused-pre-gdr \
    --cross-entropy-fusion-impl te \
    "${MODEL_ARGS[@]}" \
    2>&1 | tee "$OUT/ci_train_rank${RANK}.log"
