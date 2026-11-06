#!/usr/bin/env bash
#
# MFU tuning for Qwen3.5-VL 122B-A10B (MoE) — FSDP on the VLM entry
#
# Usage:
#   bash tune_122b_fsdp.sh
#
# The MIMO entry rejects FSDP outright ("FSDP is not yet supported for
# MegatronMIMO models"), so this script uses the plain VLM entry, which maps
# the 122b_a10b recipe.
#
# Layout: tp=1, pp=1, cp=1, ep=16 -> dp=64 on 8 nodes. FSDP shards params,
# grads and optimizer state across those 64 ranks, so the ~37 GiB/rank of fp32
# master weights + Adam moments that forced --optimizer-cpu-offload under DDP
# is not needed. gbs = mbs(1) * dp(64) * ga(2) = 128.

OUTPUT_DIR_DEFAULT="./mfu_tuning_122b_fsdp"
NNODES_DEFAULT=8
MASTER_PORT_DEFAULT=29624
HF_MODEL_DEFAULT="Qwen/Qwen3.5-122B-A10B"
SCRIPT="${SCRIPT:-train_qwen35_vl_vlm.py}"

source "$(dirname "${BASH_SOURCE[0]}")/tune_common.sh"
apply_common_env

# DeepEP-specific settings. A 16-rank EP group spans 2 nodes; 8 keeps the
# internode split normal (16 forced every rank onto ICN and trapped in
# cached_notify_combine).
export DEEPEP_NUM_MAX_ICN_PEERS="${DEEPEP_NUM_MAX_ICN_PEERS:-8}"
export DEEP_EP_WARMUP_ENABLED=1
export MEGATRON_VISION_FIXED_LENGTH_ATTENTION=1
export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
# Never write PCCL_MNNVL_ENABLE: the runtime enables MNNVL by default and the
# all-to-all path needs peer memory.

# ─── Best configuration ─────────────────────────────────────────────────────
# tp=1/pp=1/cp=1 gives dp=64 on 8 nodes; ep=16 keeps the two-node expert group
# layout used by the validated DDP baselines.
TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
EP="${EP:-16}"
ETP="${ETP:-1}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-256}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-256}"
NUM_IMAGES="${NUM_IMAGES:-1}"
SEQ_LEN="${SEQ_LEN:-8192}"
MBS="${MBS:-1}"
GA="${GA:-2}"

ENABLE_SP="${ENABLE_SP:-1}"
ENABLE_GRAD_OVERLAP="${ENABLE_GRAD_OVERLAP:-1}"
ENABLE_DEEPEP_AUTOTUNE="${ENABLE_DEEPEP_AUTOTUNE:-1}"
MOE_DISPATCHER_TYPE="${MOE_DISPATCHER_TYPE:-flex}"
MOE_DISPATCHER_BACKEND="${MOE_DISPATCHER_BACKEND:-deepep}"
ENABLE_VISION_OFFLOAD="${ENABLE_VISION_OFFLOAD:-1}"
# Recompute policy: full (default, +33% fwd re-run) | selective | none.
# selective frees ~9 GiB/mb of the ~24 GiB activation set at ~0.1 s cost.
RECOMPUTE_MODE="${RECOMPUTE_MODE:-full}"
RECOMPUTE_MODULES="${RECOMPUTE_MODULES:-moe_act,layernorm,shared_experts}"
# TE-scoped CUDA graph + fine-grained activation offload (both default off:
# graph x FSDP x hybrid-GDN is an untested combination on PPU).
ENABLE_TE_GRAPH="${ENABLE_TE_GRAPH:-0}"
CUDA_GRAPH_SCOPE="${CUDA_GRAPH_SCOPE:-}"
ENABLE_FINE_OFFLOAD="${ENABLE_FINE_OFFLOAD:-0}"
OFFLOAD_MODULES="${OFFLOAD_MODULES:-core_attn,qkv_linear}"

FEATURE_ARGS=""
[[ "$ENABLE_SP" == "1" ]] && FEATURE_ARGS="--enable-sp"
if [[ "$ENABLE_VISION_OFFLOAD" == "1" ]]; then
    FEATURE_ARGS+=" --enable-vision-activation-offload"
    [[ -n "${VISION_OFFLOAD_NUM_LAYERS:-}" ]] && \
        FEATURE_ARGS+=" --vision-activation-offload-num-layers $VISION_OFFLOAD_NUM_LAYERS"
fi

DISPATCH_ARGS="--moe-dispatcher-type $MOE_DISPATCHER_TYPE"
[[ "$MOE_DISPATCHER_TYPE" == "flex" ]] && \
    DISPATCH_ARGS+=" --moe-dispatcher-backend $MOE_DISPATCHER_BACKEND"

RECOMPUTE_ARGS=""
case "$RECOMPUTE_MODE" in
    full)
        RECOMPUTE_ARGS="--enable-full-recompute"
        [[ "$ENABLE_TE_GRAPH" == "1" ]] && { echo "RECOMPUTE_MODE=full conflicts with ENABLE_TE_GRAPH=1 (mcore assert: full recompute needs full-iteration graph)" >&2; exit 2; }
        ;;
    selective)
        RECOMPUTE_ARGS="--enable-recompute --recompute-modules $RECOMPUTE_MODULES"
        ;;
    none)
        # The 122b_a10b recipe hard-enables full recompute; absence of flags is NOT off.
        RECOMPUTE_ARGS="--disable-recompute"
        ;;
    *)
        echo "RECOMPUTE_MODE must be full|selective|none" >&2
        exit 2
        ;;
esac

GRAPH_ARGS=""
if [[ "$ENABLE_TE_GRAPH" == "1" ]]; then
    GRAPH_ARGS="--enable-cuda-graph-te --cuda-graph-warmup ${CUDA_GRAPH_WARMUP:-3}"
    [[ -n "$CUDA_GRAPH_SCOPE" ]] && GRAPH_ARGS+=" --cuda-graph-scope $CUDA_GRAPH_SCOPE"
fi
if [[ "$ENABLE_FINE_OFFLOAD" == "1" ]]; then
    [[ "$ENABLE_TE_GRAPH" != "1" ]] && { echo "ENABLE_FINE_OFFLOAD=1 requires ENABLE_TE_GRAPH=1" >&2; exit 2; }
    GRAPH_ARGS+=" --enable-fine-grained-offload --offload-modules $OFFLOAD_MODULES"
fi

OVERLAP_ARGS=""
[[ "$ENABLE_GRAD_OVERLAP" == "1" ]] && OVERLAP_ARGS="--enable-grad-overlap"
AUTOTUNE_ARGS=""
[[ "$ENABLE_DEEPEP_AUTOTUNE" == "1" && "$MOE_DISPATCHER_TYPE" == "flex" ]] && \
    AUTOTUNE_ARGS="--enable-deepep-autotune"

run_exp "122ba10b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga_${NNODES}n_fsdp_dp$((NPROC * NNODES / (TP * PP * CP)))${TAG_SUFFIX:-}" \
    --model-size 122b_a10b \
    --hf-processor-path "$HF_MODEL" \
    --tp "$TP" --pp "$PP" --cp "$CP" --ep "$EP" --etp "$ETP" \
    --mbs "$MBS" --ga "$GA" \
    --seq-length "$SEQ_LEN" --num-images "$NUM_IMAGES" \
    --image-size-w "$IMAGE_SIZE_W" --image-size-h "$IMAGE_SIZE_H" \
    --enable-fsdp \
    ${DISPATCH_ARGS} ${OVERLAP_ARGS} \
    ${RECOMPUTE_ARGS} ${GRAPH_ARGS} ${AUTOTUNE_ARGS} \
    --moe-force-load-balancing --cross-entropy-fusion-impl te \
    ${FEATURE_ARGS} ${EXTRA_ARGS:-}
