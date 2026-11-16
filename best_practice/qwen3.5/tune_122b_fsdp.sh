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

# DeepEP-specific settings. DEEPEP_NUM_MAX_ICN_PEERS follows EP below.
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
# DeepEP only takes the fabric path when this equals the EP group size and the
# group spans more ranks than one node: deep_ep then treats the whole group as
# one domain (num_rdma_ranks=1). Any other value falls back to the internode
# RDMA path -- with EP=16 across 2 nodes that measured dispatch 14.8 / combine
# 6.2 GB/s, against 174-243 GB/s on the fabric path (35B, same 2-node group).
# CP>1 stays on the RDMA path: the fabric path completed auto-tune and then
# every rank timed out on the first real dispatch of 122B cp2/16k on 8 nodes
# (RuntimeError: DeepEP error: CPU recv timeout in intranode_dispatch), while
# the same binary at cp1 with the identical dispatch shape passed. The fabric
# peer mapping does not cover the CP rank layout yet.
if [[ "$CP" == "1" ]]; then
    DEEPEP_NUM_MAX_ICN_PEERS="${DEEPEP_NUM_MAX_ICN_PEERS:-$EP}"
else
    DEEPEP_NUM_MAX_ICN_PEERS="${DEEPEP_NUM_MAX_ICN_PEERS:-1}"
fi
export DEEPEP_NUM_MAX_ICN_PEERS
IMAGE_SIZE_W="${IMAGE_SIZE_W:-256}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-256}"
NUM_IMAGES="${NUM_IMAGES:-1}"
SEQ_LEN="${SEQ_LEN:-8192}"
MBS="${MBS:-1}"
GA="${GA:-2}"

ENABLE_SP="${ENABLE_SP:-1}"
# GDN linear-attention CP path. Megatron's default "chunkwise" (FLA cp_context plus
# a GDN-internal CP all-to-all) deadlocks at the first GDN layer on 122B cp2/16k: two
# ranks sit in build_cp_context (blocked on a D2H .item()) while the rest wait in the
# MoE allgather, and the CP all-to-all watchdog fires on a 8192x3072 qkvzba transfer.
# "legacy" is the recipe label for the validated zigzag + all-to-all Ulysses path.
if [[ "$CP" == "1" ]]; then
    LINEAR_CP_MODE="${LINEAR_CP_MODE:-chunkwise}"
else
    LINEAR_CP_MODE="${LINEAR_CP_MODE:-legacy}"
fi
# FSDP grad-reduce/param-gather overlap stalls at CP>1: the cp2/16k run hangs
# in the first iteration with the watchdog on a coalesced DP_WITH_CP collective
# (1548288 in / 99090432 out, 64 buckets) that never completes. CP cells only
# need to finish, so the overlap stays off there.
if [[ "$CP" == "1" ]]; then
    ENABLE_GRAD_OVERLAP="${ENABLE_GRAD_OVERLAP:-1}"
else
    ENABLE_GRAD_OVERLAP="${ENABLE_GRAD_OVERLAP:-0}"
fi
# Auto-tune benchmarks 16 arms on the live buffer inside the first dispatch.
# Keep it off for CP>1 until the fabric mapping is fixed: the cp2 failure above
# happened on the first dispatch right after auto-tune and cannot be separated
# from it, and the RDMA path needs no per-call chunk search.
if [[ "$CP" == "1" ]]; then
    ENABLE_DEEPEP_AUTOTUNE="${ENABLE_DEEPEP_AUTOTUNE:-1}"
else
    ENABLE_DEEPEP_AUTOTUNE="${ENABLE_DEEPEP_AUTOTUNE:-0}"
fi
# CP>1 cannot use the DeepEP flex dispatcher yet: its static NVL buffer is
# sized from deep_ep's Config hint, and 122B cp2/16k overflows it -- metadata
# assert in internode_aot.cu:108 on the RDMA path, silent CPU recv timeout on
# the fabric path -- while cp1 at the same dispatch shape passes and the same
# cell passes on 2-node models. The NCCL all-to-all dispatcher avoids that
# buffer but the same cp2/16k cell then stalls on its first dispatch across the
# 16-rank EP group. The allgather dispatcher only needs allgather/reduce-scatter
# on the expert group; it costs extra activation memory (full token set on every
# EP rank) but the CP cells only have to complete, not win on throughput.
if [[ "$CP" == "1" ]]; then
    MOE_DISPATCHER_TYPE="${MOE_DISPATCHER_TYPE:-flex}"
else
    MOE_DISPATCHER_TYPE="${MOE_DISPATCHER_TYPE:-allgather}"
fi
MOE_DISPATCHER_BACKEND="${MOE_DISPATCHER_BACKEND:-deepep}"
# Vision activation offload matters once the images are large (1080p puts
# thousands of ViT tokens through 27 layers). The switch must be applied to
# the vision config before the model is built; images older than the fix in
# dockers/patches/qwen35.patch crash with KeyError: 0 on the first group
# commit whenever num_images>0, so keep IMAGE on a patched image.
ENABLE_VISION_OFFLOAD="${ENABLE_VISION_OFFLOAD:-1}"
# Recompute policy: none | selective | full. The default follows SEQ_LEN
# because the memory ceiling does: the CI 8k cell runs with full recompute
# (peak 66.6 GiB/rank), while the same cell without it OOMs in iteration 0's
# first forward (141.1 GiB allocated, 768 MiB request failed). A manual rerun
# that just calls this script must get the shape that fits, not the one that
# OOMs, so 8k and above defaults to full and below 8k keeps recompute off.
# FSDP already shards the fp32 master copies that made recompute necessary
# under DDP, so shorter sequences do not pay for it.
# selective frees ~9 GiB/mb of the ~24 GiB activation set at ~0.1 s cost.
if [[ -n "${RECOMPUTE_MODE:-}" ]]; then
    RECOMPUTE_MODE="$RECOMPUTE_MODE"
elif (( SEQ_LEN >= 8192 )); then
    RECOMPUTE_MODE="full"
else
    RECOMPUTE_MODE="none"
fi
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
# Forced routing is a benchmark shortcut for the CP=1 cells. Under CP>1 it makes
# the router skip real routing while the GDN context-parallel path builds its
# cu_seqlens from the same batch, and cp2/16k deadlocks in the first iteration
# with ranks parked in GDN build_cp_context and MoEAllGatherTokenDispatcher.
FORCE_LB_ARGS="--moe-force-load-balancing"
[[ "$CP" != "1" ]] && FORCE_LB_ARGS=""

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
    ${FORCE_LB_ARGS} --cross-entropy-fusion-impl te \
    --linear-cp-mode "$LINEAR_CP_MODE" \
    ${FEATURE_ARGS} ${EXTRA_ARGS:-}
