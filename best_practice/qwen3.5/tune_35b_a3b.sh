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
# Summary is printed at the end.

set -uo pipefail

# ─── User config ────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-./mfu_tuning_35b_a3b_mimo_cp}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-2}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-35B-A3B}"

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

    set +e
    # MODEL_PROFILE_RANGE=10,11 prof.sh "$log_dir/${tag}_$(hostname)"  
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
        --enable-profiler \
        --enable-memory-snapshot \
        --memory-snapshot-path "${log_dir}/memory_snapshot_node${RANK}.pickle" \
        --profile-step-start 10 \
        --profile-step-end 11 \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -ne 0 ]]; then
        echo "[WARN] ${tag} FAILED (exit ${exit_code}), continuing sweep..."
        echo "TUNE_STATUS: FAILED exit=${exit_code}" >> "$log_file"
    else
        echo "TUNE_STATUS: OK" >> "$log_file"
    fi

    sleep 5
}

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENABLE_COPY_ENGINE="${ENABLE_COPY_ENGINE:-0}"
case "$ENABLE_COPY_ENGINE" in
    1)
        # libpccl reads the PCCL_* names (NCCL_CTA_POLICY is not consumed).
        export ENABLE_COPY_ENGINE=1
        export PCCL_CTA_POLICY=2
        export PCCL_CE_ENABLE_AR=1
        export PCCL_CE_ENABLE_RS=1
        export CPATH="/usr/local/pccl/nccl_wrapper/include${CPATH:+:$CPATH}"
        export CPLUS_INCLUDE_PATH="/usr/local/pccl/nccl_wrapper/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
        export LIBRARY_PATH="/usr/local/pccl/nccl_wrapper/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export LD_LIBRARY_PATH="/usr/local/pccl/nccl_wrapper/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        ;;
    0)
        export ENABLE_COPY_ENGINE=0
        export PCCL_CTA_POLICY=0x1
        export PCCL_CE_ENABLE_AR=0
        export PCCL_CE_ENABLE_RS=0
        ;;
    *)
        echo "ENABLE_COPY_ENGINE must be 0 or 1" >&2
        exit 2
        ;;
esac
# Never write PCCL_MNNVL_ENABLE: the runtime enables MNNVL by default and the
# copy engine needs peer memory.
export NCCL_GRAPH_REGISTER=0
export PCCL_GRAPH_REGISTER=0
export MEGATRON_LOGGING_LEVEL=20
export FLA_CACHE_MODE="${FLA_CACHE_MODE:-default}"
export MEGATRON_SWIGLU_USE_CUDAC="${MEGATRON_SWIGLU_USE_CUDAC:-1}"
export MEGATRON_MOE_TOPK_RECONSTRUCT="${MEGATRON_MOE_TOPK_RECONSTRUCT:-triton}"
export NVTE_CPU_OFFLOAD_V1=1 ; export UB_SKIPMC=1 ;
export NVTE_USE_CUTLASS_GROUPED_GEMM=1;
# DeepEP-specific settings, only relevant for 35B-A3B MoE.
export DEEPEP_NUM_MAX_ICN_PEERS=16;
export TRITON_CACHE_DIR=/tmp/triton_cache_$(hostname)
export DEEP_EP_WARMUP_ENABLED=1;
export MEGATRON_VISION_FIXED_LENGTH_ATTENTION=1;
export MIMO_DEBUG="${MIMO_DEBUG:-1}";
export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1;
export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1;

BEST_LANG="${BEST_LANG:-language=tp=2,cp=1,ep=16,dp=8,etp=1,pp=1,rank_offset=0}"
BEST_IMG="${BEST_IMG:-images=tp=1,cp=1,dp=16,rank_offset=0}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-1920}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-1080}"
NUM_IMAGES="${NUM_IMAGES:-2}"
SEQ_LEN="${SEQ_LEN:-4096}"
MBS="${MBS:-1}"
GA="${GA:-8}"
ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-1}"

CUDA_GRAPH_ARGS=()
if [[ "$ENABLE_CUDA_GRAPH" == "1" ]]; then
    CUDA_GRAPH_ARGS=(--enable-cuda-graph-te)
fi

# EP overlap is a MIMO schedule-plan path: pure text and language TP=CP=PP=1.
# Image-heavy and CP>1 rows must leave it off (the entrypoint rejects them).
ENABLE_EP_OVERLAP="${ENABLE_EP_OVERLAP:-0}"
DELAY_WGRAD="${DELAY_WGRAD:-0}"
OVERLAP_ARGS=()
if [[ "$ENABLE_EP_OVERLAP" == "1" ]]; then
    OVERLAP_ARGS+=(--enable-ep-overlap)
    if [[ "$DELAY_WGRAD" == "1" ]]; then
        OVERLAP_ARGS+=(--delay-wgrad)
    fi
fi
MOE_DEEPEP_NUM_SMS="${MOE_DEEPEP_NUM_SMS:-20}"

run_exp "35ba3b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga${TAG_SUFFIX:-}" \
    "$BEST_LANG" "$BEST_IMG" \
    --moe-dispatcher-type flex --moe-dispatcher-backend deepep \
    --moe-deepep-num-sms "$MOE_DEEPEP_NUM_SMS" \
    --linear-cp-mode chunkwise --gdn-fused-pre-gdr \
    --enable-grad-overlap --mbs $MBS --ga $GA \
    --seq-length $SEQ_LEN --num-images $NUM_IMAGES --image-size-w $IMAGE_SIZE_W --image-size-h $IMAGE_SIZE_H \
    --enable-sp --enable-vision-cp --moe-force-load-balancing "${CUDA_GRAPH_ARGS[@]}" "${OVERLAP_ARGS[@]}" --enable-deepep-autotune --enable-vision-activation-offload --vision-offload-num-layers 20 --cross-entropy-fusion-impl te ${EXTRA_ARGS:-}
