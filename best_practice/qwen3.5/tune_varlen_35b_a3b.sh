#!/usr/bin/env bash
#
# Variable-length SFT tuning for Qwen3.5-VL 35B-A3B (MoE) — LLaVA-NeXT-Data
# In-batch packing / THD + vlm_step; GBS = MBS * DP * GA.
# Usage: ENABLE_ASYS=1 PROFILE_RANGE=20,20 bash tune_varlen_35b_a3b.sh
# Logs: $LOG_DIR/<tag>/train_node<RANK>.log; TensorBoard: $OUTPUT_DIR/<tag>/.

set -uo pipefail

SCRIPT_DIR_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPU_RECIPES_ROOT="${PPU_RECIPES_ROOT:-$(cd "${SCRIPT_DIR_LOCAL}/../../.." && pwd)}"

# ─── User config ────────────────────────────────────────────────────────────
MODEL_SIZE="35b_a3b"
OUTPUT_DIR="${OUTPUT_DIR:-${PPU_RECIPES_ROOT}/output/llava_next_35b_a3b}"
LOG_DIR="${LOG_DIR:-${PPU_RECIPES_ROOT}/logs}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-2}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29501}"
TRAIN_ITERS="${TRAIN_ITERS:-24}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

DATA_DIR="${DATA_DIR:-/cpfs/datasets/llava-next-data/json}"
SCHEMA_ADAPTER="${SCHEMA_ADAPTER:-none}"
SEQ_LEN="${SEQ_LEN:-8192}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEFAULT_CHECKPOINT_NAME="Qwen3.5-35B-A3B"
if [[ "${ENABLE_FSDP:-1}" == "1" ]]; then
    DEFAULT_CHECKPOINT_NAME="Qwen3.5-35B-A3B-fsdp_dtensor_ep16"
fi
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${PPU_RECIPES_ROOT}/../output/megatron_checkpoints/${DEFAULT_CHECKPOINT_NAME}}"
# Real MBS12 packs may exceed 8192 tokens; align sequences without enforcing a fixed pack width.
PAD_TO_MAX_LENGTH="${PAD_TO_MAX_LENGTH:-0}"
TAG_SUFFIX="${TAG_SUFFIX:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"

RECOMPUTE_MODE="${RECOMPUTE_MODE:-full}"

TP="${TP:-4}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-12}"
GA="${GA:-2}"

ENABLE_FSDP="${ENABLE_FSDP:-1}"
FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-optim_grads}"
FSDP_ARGS=()
case "$ENABLE_FSDP" in
    1) FSDP_ARGS=(--enable-fsdp --fsdp-sharding-strategy "$FSDP_SHARDING_STRATEGY") ;;
    0) ;;
    *) echo "ENABLE_FSDP must be 0 or 1" >&2; exit 2 ;;
esac
TE_ARGS="--cross-entropy-fusion-impl te"
EXTRA_ARGS="${EXTRA_ARGS:-$TE_ARGS}"

ENABLE_ASYS="${ENABLE_ASYS:-0}"
PROFILE_RANGE="${PROFILE_RANGE:-20,20}"

SCRIPT="train_qwen35_varlen.py"
TORCHRUN="python -m torch.distributed.run"

EP="${EP:-16}"
MOE_FORCE_LB="${MOE_FORCE_LB:-0}"
CONV_MODE="${CONV_MODE:-pip}"

# This script runs SFT only; missing pretrained weights must not silently trigger random initialization.
CHECKPOINT_OK=0
if [[ -d "$PRETRAINED_CHECKPOINT" ]]; then
    if [[ -f "$PRETRAINED_CHECKPOINT/latest_checkpointed_iteration.txt" ]]; then
        CHECKPOINT_OK=1
    elif [[ -f "$PRETRAINED_CHECKPOINT/config.json" && -f "$PRETRAINED_CHECKPOINT/model.safetensors.index.json" ]]; then
        # Bridge can initialize from local HF weights and save a separate Megatron checkpoint.
        CHECKPOINT_OK=1
    fi
fi
if [[ "$CHECKPOINT_OK" == "0" ]]; then
    echo "SFT requires a valid PRETRAINED_CHECKPOINT: $PRETRAINED_CHECKPOINT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ─── Helper ─────────────────────────────────────────────────────────────────
# run_exp <tag> <tp> <pp> <cp> <ep> <mbs> <ga> [extra_args...]
run_exp() {
    local tag="$1"; shift
    local tp="$1"; shift
    local pp="$1"; shift
    local cp="$1"; shift
    local ep="$1"; shift
    local mbs="$1"; shift
    local ga="$1"; shift
    local extra_args=("$@")

    local world=$((NPROC * NNODES))
    if (( world % (tp * pp * cp) != 0 )); then
        echo "WORLD_SIZE=$world must be divisible by tp*pp*cp=$((tp * pp * cp))" >&2
        exit 2
    fi

    # With ETP=1, expert rank topology is independent of dense TP; do not multiply TP into the EP constraint.
    if (( world % (ep * pp) != 0 )); then
        echo "WORLD_SIZE=$world must be divisible by ETP*EP*PP=$((ep * pp)) with ETP=1" >&2
        exit 2
    fi
    local dp=$((world / (tp * pp * cp)))
    local gbs=$((mbs * dp * ga))
    local log_dir="${LOG_DIR}/${tag}"
    mkdir -p "$log_dir"
    local log_file="${log_dir}/train_node${RANK}.log"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  EXP : ${tag}"
    echo "  MODEL: qwen3.5-vl-${MODEL_SIZE}  DATA: ${DATA_DIR}"
    echo "  SEQ=${SEQ_LEN}  MBS=${mbs}  GA=${ga}  GBS=${gbs}  DP=${dp}  TP=${tp}  PP=${pp}  CP=${cp}  EP=${ep}"
    echo "  EXTRA: ${extra_args[*]:-}"
    echo "  LOG : ${log_file}"
    echo "  OUT : ${OUTPUT_DIR}/${tag}"
    echo "════════════════════════════════════════════════════════"

    local -a recompute_args=()
    case "$RECOMPUTE_MODE" in
        full) recompute_args+=(--enable-full-recompute) ;;
        selective) recompute_args+=(--enable-recompute) ;;
        none) ;;
        *) echo "RECOMPUTE_MODE must be full|selective|none" >&2; exit 2 ;;
    esac

    local -a moe_lb_args=()
    if [[ "$MOE_FORCE_LB" == "1" ]]; then
        moe_lb_args+=(--moe-force-load-balancing)
    fi

    local -a sp_args=()
    if (( tp > 1 )); then
        sp_args+=(--enable-sp)
    fi

    local -a entry_args=("$SCRIPT")
    if [[ "$ENABLE_ASYS" != "1" && -n "${CAPTURE_WORKER_WRAPPER:-}" ]]; then
        entry_args=("$CAPTURE_WORKER_WRAPPER" --worker-script "$SCRIPT")
    fi
    local -a cmd=(
        $TORCHRUN
        --nproc_per_node="$NPROC"
        --nnodes="$NNODES"
        --rdzv-backend=static
        --node-rank="$RANK"
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}"
        --max-restarts=0
        "${entry_args[@]}"
        --model-size "$MODEL_SIZE"
        --data "$DATA_DIR"
        --schema-adapter "$SCHEMA_ADAPTER"
        --seq-length "$SEQ_LEN"
        --mbs "$mbs"
        --ga "$ga"
        --train-iters "$TRAIN_ITERS"
        --tp "$tp" --pp "$pp" --cp "$cp" --ep "$ep" --etp 1
        --log-interval "$LOG_INTERVAL"
        --num-workers "$NUM_WORKERS"
        --output-dir "${OUTPUT_DIR}/${tag}"
        --save-interval "$SAVE_INTERVAL"
        "${recompute_args[@]}"
        "${sp_args[@]}"
        --moe-dispatcher-type flex
        --moe-dispatcher-backend deepep
        "${moe_lb_args[@]}"
        "${FSDP_ARGS[@]}"
        "${extra_args[@]}"
    )
    if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
        cmd+=(--pretrained-checkpoint "$PRETRAINED_CHECKPOINT")
    fi

    set +e
    if [[ "$ENABLE_ASYS" == "1" ]]; then
        MODEL_PROFILE_RANGE="$PROFILE_RANGE" \
            prof.sh "$log_dir/${tag}_$(hostname)" "${cmd[@]}" 2>&1 | tee "$log_file"
    else
        "${cmd[@]}" 2>&1 | tee "$log_file"
    fi
    exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -ne 0 ]]; then
        echo "[WARN] ${tag} FAILED (exit ${exit_code}), continuing sweep..."
        echo "TUNE_STATUS: FAILED exit=${exit_code}" >> "$log_file"
    else
        echo "TUNE_STATUS: OK" >> "$log_file"
    fi

    sleep 5
    return "$exit_code"
}

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
export NVTE_USE_CUTLASS_GROUPED_GEMM_RTC=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PPU_RECIPES_ROOT%/*}/triton_cache}"
export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT="${PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT:-1}"
export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1
# PPU's Inductor backend stalls compiling the first backward graph, so the
# variable-length path runs eager for now. Re-enable when tuning throughput.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export HF_HOME="${HF_HOME:-/cpfs/models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

# cuDNN normalization graphs fail for these shapes; use native TE kernels.
export NVTE_NORM_FWD_USE_CUDNN="${NVTE_NORM_FWD_USE_CUDNN:-0}"
export NVTE_NORM_BWD_USE_CUDNN="${NVTE_NORM_BWD_USE_CUDNN:-0}"
export DEEPEP_NUM_MAX_ICN_PEERS=16
export MEGATRON_VISION_FIXED_LENGTH_ATTENTION="${MEGATRON_VISION_FIXED_LENGTH_ATTENTION:-1}"
export DEEP_EP_WARMUP_ENABLED=1

CONV_ARGS=()
case "$CONV_MODE" in
    pip) ;;
    triton) CONV_ARGS=(--disable-causal-conv1d) ;;
    *) echo "CONV_MODE must be pip or triton" >&2; exit 2 ;;
esac

MODE="sft"
PAD_ARGS=()
case "$PAD_TO_MAX_LENGTH" in
    0) PAD_ARGS=(--no-pad-to-max-length); PAD_TAG="nopad" ;;
    1) PAD_TAG="padmax" ;;
    *) echo "PAD_TO_MAX_LENGTH must be 0 or 1" >&2; exit 2 ;;
esac
DIST_TAG="ddp"
if [[ "$ENABLE_FSDP" == "1" ]]; then
    DIST_TAG="fsdp_${FSDP_SHARDING_STRATEGY}"
fi

run_exp "llava_next_${MODEL_SIZE}_${MODE}_${SEQ_LEN}seq_${MBS}mbs_${GA}ga_tp${TP}_ep${EP}_${DIST_TAG}_${PAD_TAG}_${CONV_MODE}${TAG_SUFFIX}" \
    "$TP" "$PP" "$CP" "$EP" "$MBS" "$GA" \
    --enable-grad-overlap "${PAD_ARGS[@]}" "${CONV_ARGS[@]}" ${EXTRA_ARGS:-}
