#!/usr/bin/env bash
#
# Qwen3.5-122B-A10B FSDP tuning via the non-MIMO VLM entry
# (train_qwen35_vl_vlm.py), 4 nodes = world 32.
#
# The MIMO entry rejects FSDP outright
# ("FSDP is not yet supported for MegatronMIMO models"), so this script uses
# the plain VLM entry, which already ships Megatron-FSDP support and now maps
# the 122b_a10b recipe.
#
# Layout: tp=1, pp=1, cp=1, ep=16 -> dp=32. FSDP shards params/grads/optimizer
# states across those 32 ranks, so the ~37 GiB/rank of fp32 master weights +
# Adam moments that forced --optimizer-cpu-offload under DDP should no longer
# be needed. gbs = mbs(1) * dp(32) * ga(2) = 64, same as the DDP baselines.
#
# Usage (one invocation per node, same MASTER_ADDR/MASTER_PORT everywhere):
#   NNODES=4 RANK=<0..3> MASTER_ADDR=<node0-ip> bash tune_122b_fsdp.sh
#
# Results land under $OUTPUT_DIR/<tag>/train_node${RANK}.log.

set -uo pipefail

# ─── User config ────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-./mfu_tuning_122b_fsdp}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-4}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29624}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-122B-A10B}"

SEQ_LEN="${SEQ_LEN:-8192}"
MBS="${MBS:-1}"
GA="${GA:-2}"
NUM_IMAGES="${NUM_IMAGES:-1}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-256}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-256}"

# tp=1/pp=1/cp=1 gives dp=32 on 4 nodes; ep=16 keeps the two-node expert group
# layout used by the validated DDP baselines.
TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
EP="${EP:-16}"
ETP="${ETP:-1}"

SCRIPT="train_qwen35_vl_vlm.py"
TORCHRUN="python -m torch.distributed.run"
mkdir -p "$OUTPUT_DIR"

# ─── Helper ─────────────────────────────────────────────────────────────────
# run_exp <tag> [extra_args...]
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

    set +e
    $TORCHRUN \
        --nproc_per_node="$NPROC" \
        --nnodes="$NNODES" \
        --rdzv-backend=static \
        --node-rank="$RANK" \
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --max-restarts=0 \
        "$SCRIPT" \
        --model-size 122b_a10b \
        --hf-processor-path "$HF_MODEL" \
        --tp "$TP" --pp "$PP" --cp "$CP" --ep "$EP" --etp "$ETP" \
        --mbs "$MBS" --ga "$GA" --seq-length "$SEQ_LEN" \
        --num-images "$NUM_IMAGES" \
        --image-size-w "$IMAGE_SIZE_W" --image-size-h "$IMAGE_SIZE_H" \
        --output-dir "${log_dir}/tensorboard_node${RANK}" \
        --train-iters "$TRAIN_ITERS" \
        --log-interval "$LOG_INTERVAL" \
        --enable-profiler \
        --enable-memory-snapshot \
        --memory-snapshot-path "${log_dir}/memory_snapshot_node${RANK}.pickle" \
        --profile-step-start "${PROFILE_START:-10}" \
        --profile-step-end "${PROFILE_END:-11}" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -ne 0 ]]; then
        echo "[WARN] ${tag} FAILED (exit ${exit_code})"
        echo "TUNE_STATUS: FAILED exit=${exit_code}" >> "$log_file"
    else
        echo "TUNE_STATUS: OK" >> "$log_file"
    fi

    sleep 5
}

# ─── Environment (mirrors tune_122b.sh) ─────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENABLE_COPY_ENGINE="${ENABLE_COPY_ENGINE:-0}"
case "$ENABLE_COPY_ENGINE" in
    1)
        export NCCL_CTA_POLICY=2
        export PCCL_CE_ENABLE_RS=1
        ;;
    0)
        export NCCL_CTA_POLICY=0x1
        export PCCL_CE_ENABLE_RS=0
        ;;
    *)
        echo "ENABLE_COPY_ENGINE must be 0 or 1" >&2
        exit 2
        ;;
esac
export ENABLE_COPY_ENGINE
export NCCL_GRAPH_REGISTER=0
export MEGATRON_LOGGING_LEVEL=20
export NVTE_CPU_OFFLOAD_V1=1
export UB_SKIPMC=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=1
# 4 nodes: 16-rank EP group spans 2 nodes; 8 keeps the internode split normal
# (16 forced every rank onto ICN and trapped in cached_notify_combine).
export DEEPEP_NUM_MAX_ICN_PEERS="${DEEPEP_NUM_MAX_ICN_PEERS:-8}"
export TRITON_CACHE_DIR=/tmp/triton_cache_$(hostname)
export DEEP_EP_WARMUP_ENABLED=1
export MEGATRON_VISION_FIXED_LENGTH_ATTENTION=1
export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1
export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1
export PCCL_MNNVL_ENABLE=0
export HF_HOME="${HF_HOME:-/path/to/models}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1

# ─── Best-practice feature toggles ──────────────────────────────────────────
ENABLE_SP="${ENABLE_SP:-1}"
ENABLE_GRAD_OVERLAP="${ENABLE_GRAD_OVERLAP:-1}"
ENABLE_DEEPEP_AUTOTUNE="${ENABLE_DEEPEP_AUTOTUNE:-1}"
MOE_DISPATCHER_TYPE="${MOE_DISPATCHER_TYPE:-flex}"
MOE_DISPATCHER_BACKEND="${MOE_DISPATCHER_BACKEND:-deepep}"
ENABLE_VISION_OFFLOAD="${ENABLE_VISION_OFFLOAD:-1}"
# Recompute policy: full (default, +33% fwd re-run) | selective | none.
# selective frees ~9 GiB/mb of the ~24 GiB activation set at ~0.1 s cost; see
# the recompute trade-off report for per-module numbers.
RECOMPUTE_MODE="${RECOMPUTE_MODE:-full}"
RECOMPUTE_MODULES="${RECOMPUTE_MODULES:-moe_act,layernorm,shared_experts}"
# TE-scoped CUDA graph + fine-grained activation offload (both default off:
# graph x FSDP x hybrid-GDN is an untested combination on PPU).
ENABLE_TE_GRAPH="${ENABLE_TE_GRAPH:-0}"
CUDA_GRAPH_SCOPE="${CUDA_GRAPH_SCOPE:-}"
ENABLE_FINE_OFFLOAD="${ENABLE_FINE_OFFLOAD:-0}"
OFFLOAD_MODULES="${OFFLOAD_MODULES:-core_attn,qkv_linear}"

FEATURE_ARGS=()
[[ "$ENABLE_SP" == "1" ]] && FEATURE_ARGS+=(--enable-sp)
if [[ "$ENABLE_VISION_OFFLOAD" == "1" ]]; then
    FEATURE_ARGS+=(--enable-vision-activation-offload)
    [[ -n "${VISION_OFFLOAD_NUM_LAYERS:-}" ]] && \
        FEATURE_ARGS+=(--vision-activation-offload-num-layers "$VISION_OFFLOAD_NUM_LAYERS")
fi

DISPATCH_ARGS=(--moe-dispatcher-type "$MOE_DISPATCHER_TYPE")
[[ "$MOE_DISPATCHER_TYPE" == "flex" ]] && \
    DISPATCH_ARGS+=(--moe-dispatcher-backend "$MOE_DISPATCHER_BACKEND")

# ─── Recompute / graph / offload knobs ───────────────────────────────────────
RECOMPUTE_ARGS=()
case "$RECOMPUTE_MODE" in
    full)
        RECOMPUTE_ARGS+=(--enable-full-recompute)
        [[ "$ENABLE_TE_GRAPH" == "1" ]] && { echo "RECOMPUTE_MODE=full conflicts with ENABLE_TE_GRAPH=1 (mcore assert: full recompute needs full-iteration graph)" >&2; exit 2; }
        ;;
    selective)
        RECOMPUTE_ARGS+=(--enable-recompute --recompute-modules "$RECOMPUTE_MODULES")
        ;;
    none)
        # 122b_a10b recipe hard-enables full recompute; absence of flags is NOT off.
        RECOMPUTE_ARGS+=(--disable-recompute)
        ;;
    *)
        echo "RECOMPUTE_MODE must be full|selective|none" >&2
        exit 2
        ;;
esac
GRAPH_ARGS=()
if [[ "$ENABLE_TE_GRAPH" == "1" ]]; then
    GRAPH_ARGS+=(--enable-cuda-graph-te --cuda-graph-warmup "${CUDA_GRAPH_WARMUP:-3}")
    [[ -n "$CUDA_GRAPH_SCOPE" ]] && GRAPH_ARGS+=(--cuda-graph-scope "$CUDA_GRAPH_SCOPE")
fi
if [[ "$ENABLE_FINE_OFFLOAD" == "1" ]]; then
    [[ "$ENABLE_TE_GRAPH" != "1" ]] && { echo "ENABLE_FINE_OFFLOAD=1 requires ENABLE_TE_GRAPH=1" >&2; exit 2; }
    GRAPH_ARGS+=(--enable-fine-grained-offload --offload-modules "$OFFLOAD_MODULES")
fi

# ─── The run ────────────────────────────────────────────────────────────────
run_exp "122ba10b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga_${NNODES}n_fsdp_dp$((NPROC * NNODES / (TP * PP * CP)))${TAG_SUFFIX:-}" \
    "${DISPATCH_ARGS[@]}" \
    $([[ "$ENABLE_GRAD_OVERLAP" == "1" ]] && echo --enable-grad-overlap) \
    --enable-fsdp \
    "${RECOMPUTE_ARGS[@]}" "${GRAPH_ARGS[@]}" \
    $([[ "$ENABLE_DEEPEP_AUTOTUNE" == "1" && "$MOE_DISPATCHER_TYPE" == "flex" ]] && echo --enable-deepep-autotune) \
    --moe-force-load-balancing --cross-entropy-fusion-impl te \
    "${FEATURE_ARGS[@]}" ${EXTRA_ARGS:-}
