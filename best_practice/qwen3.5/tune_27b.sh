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

set -uo pipefail

# CI mode: the scheduler exports NODE_RANK/NPROC_PER_NODE and expects a single
# run whose exit status is the run's status, with no profiler overhead.
CI_MODE="${CI_MODE:-0}"
if [[ "$CI_MODE" == "1" ]]; then
    HF_MODEL="${HF_MODEL:?HF_MODEL must be provided by the CI declaration}"
fi

# ─── User config ────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-./mfu_tuning_27b_vlm}"
NPROC="${NPROC_PER_NODE:-${NPROC:-8}}"
NNODES="${NNODES:-1}"
RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29501}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-27B}"

SCRIPT="${SCRIPT:-train_qwen35_vl_vlm.py}"
TORCHRUN="python -m torch.distributed.run"

mkdir -p "$OUTPUT_DIR"

# ─── Helper ─────────────────────────────────────────────────────────────────
run_exp() {
    local tag="$1"; shift
    local extra_args=("$@")

    local log_dir="${OUTPUT_DIR}/${tag}"
    mkdir -p "$log_dir"
    local log_file="${log_dir}/train_node${RANK}.log"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  EXP : ${tag}"
    echo "  LOG : ${log_file}"
    echo "════════════════════════════════════════════════════════"

    local profile_args=""
    if [[ "$CI_MODE" != "1" ]]; then
        profile_args="--enable-profiler --enable-memory-snapshot "
        profile_args+="--memory-snapshot-path ${log_dir}/memory_snapshot_node${RANK}.pickle "
        profile_args+="--profile-step-start 10 --profile-step-end 11"
    fi
    set +e
    $TORCHRUN \
        --nproc_per_node="$NPROC" \
        --nnodes="$NNODES" \
        --rdzv-backend=static \
        --node-rank="$RANK" \
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --max-restarts=0 \
        "$SCRIPT" \
        --model-size 27b \
        --hf-processor-path "$HF_MODEL" \
        --output-dir "${log_dir}/tensorboard_node${RANK}" \
        --train-iters "$TRAIN_ITERS" \
        --log-interval "$LOG_INTERVAL" \
        ${profile_args} \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -ne 0 ]]; then
        echo "[WARN] ${tag} FAILED (exit ${exit_code})" >&2
        echo "TUNE_STATUS: FAILED exit=${exit_code}" >> "$log_file"
    else
        echo "TUNE_STATUS: OK" >> "$log_file"
    fi

    if [[ "$CI_MODE" == "1" ]]; then
        # One CI cell is one run: its exit status must reach the scheduler.
        return "$exit_code"
    fi
    sleep 5
}

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLA_CACHE_MODE="${FLA_CACHE_MODE:-default}"
ENABLE_COPY_ENGINE="${ENABLE_COPY_ENGINE:-0}"
case "$ENABLE_COPY_ENGINE" in
    1)
        export NCCL_CTA_POLICY=2
        export PCCL_CTA_POLICY=2
        export PCCL_CE_ENABLE_RS=1
        ;;
    0)
        export NCCL_CTA_POLICY=0x1
        export PCCL_CTA_POLICY=0x1
        export PCCL_CE_ENABLE_RS=0
        ;;
    *)
        echo "ENABLE_COPY_ENGINE must be 0 or 1" >&2
        exit 2
        ;;
esac
export ENABLE_COPY_ENGINE
export NCCL_GRAPH_REGISTER=0
export PCCL_GRAPH_REGISTER=0
export MEGATRON_LOGGING_LEVEL=20
export MEGATRON_SWIGLU_USE_CUDAC="${MEGATRON_SWIGLU_USE_CUDAC:-1}"
export MEGATRON_MOE_TOPK_RECONSTRUCT="${MEGATRON_MOE_TOPK_RECONSTRUCT:-triton}"
export NVTE_CPU_OFFLOAD_V1=1 ; export UB_SKIPMC=1 ;
export NVTE_USE_CUTLASS_GROUPED_GEMM=1;
export TRITON_CACHE_DIR=/tmp/triton_cache_$(hostname)
export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1;
export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1;
export MEGATRON_VISION_FIXED_LENGTH_ATTENTION=1;

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
    --tp $TP --cp $CP  \
    --mbs $MBS --ga $GA \
    --seq-length $SEQ_LEN --num-images $NUM_IMAGES --image-size-w $IMAGE_SIZE_W --image-size-h $IMAGE_SIZE_H \
    --enable-sp --enable-fsdp --enable-grad-overlap --cross-entropy-fusion-impl te \
    ${EXTRA_ARGS:-}
