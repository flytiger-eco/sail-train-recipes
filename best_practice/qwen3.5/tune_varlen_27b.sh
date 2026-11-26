#!/usr/bin/env bash
#
# Variable-length SFT tuning for Qwen3.5-VL 27B (dense) — LLaVA-NeXT-Data
# In-batch packing / THD + vlm_step; GBS = MBS * DP * GA.
# Run on both nodes with the same MASTER_ADDR and RANK=0 or RANK=1.
# For an asys capture, also set ENABLE_ASYS=1 and PROFILE_RANGE=20,20.
# Logs: $LOG_DIR/<tag>/train_node<RANK>.log; TensorBoard: $OUTPUT_DIR/<tag>/.

set -uo pipefail

SCRIPT_DIR_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPU_RECIPES_ROOT="${PPU_RECIPES_ROOT:-$(cd "${SCRIPT_DIR_LOCAL}/../../.." && pwd)}"

# ─── User config ────────────────────────────────────────────────────────────
MODEL_SIZE="27b"
OUTPUT_DIR="${OUTPUT_DIR:-${PPU_RECIPES_ROOT}/output/llava_next_27b}"
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
NUM_WORKERS="${NUM_WORKERS:-8}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${PPU_RECIPES_ROOT}/output/megatron_checkpoints/Qwen3.5-27B-fsdp-dtensor}"
TAG_SUFFIX="${TAG_SUFFIX:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
# At TP2/DP8 and MBS12, checkpoint the first 12 decoder layers on each stage.
RECOMPUTE="${RECOMPUTE:-block}"
BLOCK_RECOMPUTE_LAYERS="${BLOCK_RECOMPUTE_LAYERS:-12}"

TP="${TP:-2}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-12}"
GA="${GA:-8}"

# FSDP ZeRO-2 + grad/param overlap are the defaults for this sweep; append any
# extra entrypoint flags through EXTRA_ARGS.
FSDP_ARGS="--enable-fsdp --fsdp-sharding-strategy optim_grads --enable-grad-overlap"
TE_ARGS="--cross-entropy-fusion-impl te"
EXTRA_ARGS="${EXTRA_ARGS:-$FSDP_ARGS $TE_ARGS}"

# ENABLE_ASYS=1 wraps torchrun with prof.sh and sets MODEL_PROFILE_RANGE, which
# model_prof turns into cudaProfilerStart/Stop around those training steps.
# Default 20,20 captures a single steady-state step (step 20).
ENABLE_ASYS="${ENABLE_ASYS:-0}"
PROFILE_RANGE="${PROFILE_RANGE:-20,20}"

SCRIPT="${SCRIPT_DIR_LOCAL}/train_qwen35_varlen.py"
TORCHRUN="python -m torch.distributed.run"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ─── Helper ─────────────────────────────────────────────────────────────────
# run_exp <tag> <tp> <pp> <cp> <mbs> <ga> [extra_args...]
run_exp() {
    local tag="$1"; shift
    local tp="$1"; shift
    local pp="$1"; shift
    local cp="$1"; shift
    local mbs="$1"; shift
    local ga="$1"; shift
    local extra_args=("$@")

    local world=$((NPROC * NNODES))
    if (( world % (tp * pp * cp) != 0 )); then
        echo "WORLD_SIZE=$world must be divisible by tp*pp*cp=$((tp * pp * cp))" >&2
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
    echo "  SEQ=${SEQ_LEN}  MBS=${mbs}  GA=${ga}  GBS=${gbs}  DP=${dp}  TP=${tp}  PP=${pp}  CP=${cp}"
    echo "  EXTRA: ${extra_args[*]:-}"
    echo "  LOG : ${log_file}"
    echo "  OUT : ${OUTPUT_DIR}/${tag}"
    echo "════════════════════════════════════════════════════════"

    local -a cmd=(
        $TORCHRUN
        --nproc_per_node="$NPROC"
        --nnodes="$NNODES"
        --rdzv-backend=static
        --node-rank="$RANK"
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}"
        --max-restarts=0
        "$SCRIPT"
        --model-size "$MODEL_SIZE"
        --data "$DATA_DIR"
        --schema-adapter "$SCHEMA_ADAPTER"
        --seq-length "$SEQ_LEN"
        --mbs "$mbs"
        --ga "$ga"
        --train-iters "$TRAIN_ITERS"
        --tp "$tp" --pp "$pp" --cp "$cp"
        --log-interval "$LOG_INTERVAL"
        --num-workers "$NUM_WORKERS"
        --output-dir "${OUTPUT_DIR}/${tag}"
        --save-interval "$SAVE_INTERVAL"
        "${extra_args[@]}"
    )
    if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
        cmd+=(--pretrained-checkpoint "$PRETRAINED_CHECKPOINT")
    fi
    case "$RECOMPUTE" in
        block)
            cmd+=(--block-recompute-layers "$BLOCK_RECOMPUTE_LAYERS")
            ;;
        0|none)
            cmd+=(--disable-recompute)
            ;;
        1|selective)
            cmd+=(--enable-recompute)
            ;;
        full)
            cmd+=(--enable-full-recompute)
            ;;
        *)
            echo "RECOMPUTE must be one of block/0/none/1/selective/full" >&2
            exit 2
            ;;
    esac

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
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"
export PCCL_MNNVL_ENABLE="${PCCL_MNNVL_ENABLE:-0}"
export NCCL_GRAPH_REGISTER=0
export MEGATRON_LOGGING_LEVEL=20
export NVTE_CPU_OFFLOAD_V1=1
export UB_SKIPMC=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PPU_RECIPES_ROOT}/triton_cache}"
export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT="${PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT:-1}"
export PYTORCH_ENABLE_BROADCAST_SPECIALIZATION=1
# PPU's Inductor backend stalls compiling the first backward graph, so the
# variable-length path runs eager for now. Re-enable when tuning throughput.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
# Model snapshots live on CPFS. Bridge lock files and the dataset cache need a writable HF_HOME.
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-/cpfs/models}/hub}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

MODE="pretrain"
if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
    MODE="sft"
fi

run_exp "llava_next_${MODEL_SIZE}_${MODE}_${SEQ_LEN}seq_${MBS}mbs_${GA}ga_tp${TP}_fsdpz2${TAG_SUFFIX:-}" \
    "$TP" "$PP" "$CP" "$MBS" "$GA" ${EXTRA_ARGS:-}
