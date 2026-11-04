#!/usr/bin/env bash
#
# MFU tuning for Qwen3.5-VL 4B (dense) — MegatronMIMO with vision CP
#
# Usage:
#   bash tune_4b.sh
#
# Dense model: no MoE experts, no EP/ETP.
# 4B fits on 1 node 8 GPUs with TP=2, PP=1.
#
# Results are saved under $OUTPUT_DIR/<tag>/train_node<RANK>.log

set -uo pipefail

# CI mode: the scheduler exports NODE_RANK/NPROC_PER_NODE and expects a single
# run whose exit status is the run's status, with no profiler overhead.
CI_MODE="${CI_MODE:-0}"
if [[ "$CI_MODE" == "1" ]]; then
    HF_MODEL="${HF_MODEL:?HF_MODEL must be provided by the CI declaration}"
fi

# ─── User config ────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-./mfu_tuning_4b_mimo}"
NPROC="${NPROC_PER_NODE:-${NPROC:-8}}"
NNODES="${NNODES:-1}"
RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29501}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-4B}"
	
SCRIPT="train_qwen35_vl_mimo.py"
TORCHRUN="python -m torch.distributed.run"

mkdir -p "$OUTPUT_DIR"

# ─── Helper ─────────────────────────────────────────────────────────────────
# run_exp <tag> <lang_component_spec> <img_component_spec> [extra_args...]
run_exp() {
    local tag="$1"; shift
    local lang_comp="$1"; shift
    local img_comp="$1"; shift
    local extra_args=("$@")

    local log_dir="${OUTPUT_DIR}/${tag}"
    mkdir -p "$log_dir"
    local log_file="${log_dir}/train_node${RANK}.log"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  EXP : ${tag}"
    echo "  LANG: ${lang_comp}"
    echo "  IMG : ${img_comp}"
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
        --hf-model "$HF_MODEL" \
        --component "$lang_comp" \
        --component "$img_comp" \
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
export MEGATRON_VISION_FIXED_LENGTH_ATTENTION=1;
export MIMO_DEBUG="${MIMO_DEBUG:-1}";
export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1;

BEST_LANG="${BEST_LANG:-language=tp=2,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0}"
BEST_IMG="${BEST_IMG:-images=tp=1,cp=1,dp=8,rank_offset=0}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-1920}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-1080}"
NUM_IMAGES="${NUM_IMAGES:-2}"
SEQ_LEN="${SEQ_LEN:-4096}"
MBS="${MBS:-1}"
GA="${GA:-8}"
ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-1}"

CUDA_GRAPH_ARGS=()
if [[ "$ENABLE_CUDA_GRAPH" == "1" ]]; then
    CUDA_GRAPH_ARGS=(--enable-cuda-graph-te --cuda-graph-scope "attn")
fi

run_exp "4b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga${TAG_SUFFIX:-}" \
   "$BEST_LANG" "$BEST_IMG" \
    --mbs $MBS --ga $GA \
   --seq-length $SEQ_LEN --num-images $NUM_IMAGES --image-size-w $IMAGE_SIZE_W --image-size-h $IMAGE_SIZE_H \
   --enable-grad-overlap --enable-sp --cross-entropy-fusion-impl te "${CUDA_GRAPH_ARGS[@]}" ${EXTRA_ARGS:-}
