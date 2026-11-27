#!/usr/bin/env bash
#
# Shared launcher for the Megatron-LM-for-sail Qwen3.5-VL recipes.
#
# Drives Megatron-LM-for-sail's own multimodal entry point
# (examples/multimodal_dev/pretrain_multimodal.py); nothing on this path
# imports megatron.bridge. The model description lives in
# qwen_model_template.sh, the parallel layout in parallel_template.sh and the
# training loop in training_template.sh; this file keeps the environment,
# validation wiring, command assembly and artifact handling.
#
# Sourced by the tune_*.sh entry points one level up. Those files only declare
# the model variant, the process layout, the data shape and the run_megatron
# call.
#
# Environment (all overridable):
#   MEGATRON_LM_PATH    Megatron-LM-for-sail checkout; defaults to the path
#                       dockers/Dockerfile.Qwen35 installs
#   MODEL_VARIANT       2b | 4b | 9b | 27b | 35b_a3b | 122b_a10b (or proxy)
#   HF_MODEL            HF checkpoint path/id: tokenizer, processor and (with
#                       cord_v2) image processor source
#   TP / EP / PP / CP   parallelism sizes; EP must divide DP
#   MBS / GA            micro-batch size and gradient accumulation;
#                       global batch = MBS * GA * DP with DP = world/(TP*PP*CP)
#   SEQ_LEN             packed sequence length (--total-seq-length)
#   NUM_IMAGES          mock images per sample; 0 emits a text-only sample
#   IMAGE_SIZE          square mock image edge (wrappers default to 1088 high x
#                       1920 wide); must be a multiple of patch 16 x merge 2
#   IMAGE_SIZE_W/H      per-axis override of IMAGE_SIZE
#   IMAGE_SEQ_LENGTH    vision-token budget; must cover every merged patch token
#   ENABLE_VISION_CP    1 = split the batch images across CP ranks
#   USE_PACKED_SEQUENCE 1 = THD packing
#   LINEAR_CP_MODE      GDN context-parallel strategy (chunkwise | headwise)
#   CUDA_GRAPH_MODULES  comma list for TE CUDA graphs; empty leaves them off
#   USE_FSDP            1 = Megatron-FSDP. FSDP and PP are mutually exclusive on
#                       the standard path: an explicit USE_FSDP=1 with PP>1 is
#                       rejected, and the wrapper default (USE_FSDP_DEFAULT) is
#                       downgraded to 0.
#   USE_FSDP_DEFAULT    the wrapper's default when USE_FSDP is unset
#   DISPATCHER          alltoall | flex (MoE only)
#   FORCE_LOAD_BALANCING 1 = uniform router decisions (perf/mock data only)
#   MTP_NUM_LAYERS      MTP prediction depths
#   RECOMPUTE           1 = full uniform decoder recompute
#   CKPT_LOAD / CKPT_RESUME / CKPT_FORMAT / CKPT_OVERRIDE_SCHEDULER
#                       checkpoint loading. Bridge checkpoints are a GPT
#                       layout and do not load into this HybridModel; see
#                       README before pointing this at one.
#   TUNE_DRY_RUN        1 = print the torchrun command line and exit
#   PROFILE             off | torch | asys; one-based iteration window
#                       (PROFILE_STEPS, PROFILE_STEP_START/END)

set -uo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"
NPROC="${NPROC_PER_NODE:-${NPROC:-8}}"
NNODES="${NNODES:-$NNODES_DEFAULT}"
RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-$MASTER_PORT_DEFAULT}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
HF_MODEL="${HF_MODEL:-$HF_MODEL_DEFAULT}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/workspace/Megatron-LM-for-sail}"
ENTRYPOINT="${ENTRYPOINT:-${MEGATRON_LM_PATH}/examples/multimodal_dev/pretrain_multimodal.py}"

# ─── Profiling window (one-based, inclusive) ────────────────────────────────
ENABLE_PROFILER="${ENABLE_PROFILER:-0}"
CAPTURE_ASYS="${CAPTURE_ASYS:-0}"
PROFILE="${PROFILE:-}"
case "$(printf '%s' "$PROFILE" | tr '[:upper:]' '[:lower:]')" in
    "") ;;
    off|none|0) ENABLE_PROFILER=0; CAPTURE_ASYS=0 ;;
    torch|pytorch|trace|1|on|true|yes) ENABLE_PROFILER=1; CAPTURE_ASYS=0 ;;
    asys|model_prof|capture) ENABLE_PROFILER=0; CAPTURE_ASYS=1 ;;
    *) echo "PROFILE must be off, torch or asys (got '${PROFILE}')" >&2; exit 2 ;;
esac
case "$(printf '%s' "$CAPTURE_ASYS" | tr '[:upper:]' '[:lower:]')" in
    1|true|on|yes) CAPTURE_ASYS=1 ;;
    0|false|off|no|"") CAPTURE_ASYS=0 ;;
    *) echo "CAPTURE_ASYS must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$CAPTURE_ASYS" == "1" && "$ENABLE_PROFILER" == "1" ]]; then
    echo "CAPTURE_ASYS=1: disabling the torch profiler so the capture is clean" >&2
    ENABLE_PROFILER=0
fi
PROFILE_MODE="off"
if [[ "$CAPTURE_ASYS" == "1" ]]; then
    PROFILE_MODE=asys
elif [[ "$ENABLE_PROFILER" == "1" ]]; then
    PROFILE_MODE=torch
fi

# Keep the default eager window; torch profiling may also sample steady-state
# graph replay when an explicit later window is requested.
CUDA_GRAPH_WARMUP_STEPS="${CUDA_GRAPH_WARMUP_STEPS:-5}"
PROFILE_STEP_START="${PROFILE_STEP_START:-}"
PROFILE_STEP_END="${PROFILE_STEP_END:-}"
if [[ -n "${PROFILE_STEPS:-}" ]]; then
    case "$PROFILE_STEPS" in
        *-*) step_sep='-'; step_start="${PROFILE_STEPS%%-*}"; step_rest="${PROFILE_STEPS#*-}" ;;
        *,*) step_sep=','; step_start="${PROFILE_STEPS%%,*}"; step_rest="${PROFILE_STEPS#*,}" ;;
        *) echo "PROFILE_STEPS must be <start>-<end> (got '${PROFILE_STEPS}')" >&2; exit 2 ;;
    esac
    case "$step_rest" in
        *"$step_sep"*) echo "PROFILE_STEPS must be <start>-<end> (got '${PROFILE_STEPS}')" >&2; exit 2 ;;
    esac
    case "$step_start" in
        ""|*[!0-9]*) echo "PROFILE_STEPS must be numeric (got '${PROFILE_STEPS}')" >&2; exit 2 ;;
    esac
    case "$step_rest" in
        ""|*[!0-9]*) echo "PROFILE_STEPS must be numeric (got '${PROFILE_STEPS}')" >&2; exit 2 ;;
    esac
    PROFILE_STEP_START="${PROFILE_STEP_START:-$step_start}"
    PROFILE_STEP_END="${PROFILE_STEP_END:-$step_rest}"
fi
PROFILE_STEP_START="${PROFILE_STEP_START:-$CUDA_GRAPH_WARMUP_STEPS}"
PROFILE_STEP_END="${PROFILE_STEP_END:-$CUDA_GRAPH_WARMUP_STEPS}"
case "${PROFILE_STEP_START}${PROFILE_STEP_END}" in
    ""|*[!0-9]*) echo "PROFILE_STEP_START/END must be numeric" >&2; exit 2 ;;
esac
if (( PROFILE_STEP_START < 1 || PROFILE_STEP_END < PROFILE_STEP_START )); then
    echo "invalid profiling window ${PROFILE_STEP_START}-${PROFILE_STEP_END}" >&2; exit 2
fi
if [[ "$PROFILE_MODE" == "asys" ]] && (( PROFILE_STEP_END > CUDA_GRAPH_WARMUP_STEPS )); then
    echo "profiling window ${PROFILE_STEP_START}-${PROFILE_STEP_END} exceeds the ${CUDA_GRAPH_WARMUP_STEPS} eager warmup steps" >&2
    exit 2
fi
if [[ "$PROFILE_MODE" == "torch" ]] && (( PROFILE_STEP_END > TRAIN_ITERS )); then
    echo "profiling window ends after TRAIN_ITERS=${TRAIN_ITERS}" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

# ─── Model / data knobs ─────────────────────────────────────────────────────
MODEL_VARIANT="${MODEL_VARIANT:?MODEL_VARIANT must be set by the tune script}"
DATASET_PROVIDER="${DATASET_PROVIDER:-mock}"
NUM_IMAGES="${NUM_IMAGES:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-}"
IMAGE_SEQ_LENGTH="${IMAGE_SEQ_LENGTH:-256}"
SEQ_LEN="${SEQ_LEN:-4096}"
ENABLE_VISION_CP="${ENABLE_VISION_CP:-0}"
DISABLE_VISION_ENCODER_OVERRIDE="${DISABLE_VISION_ENCODER:-0}"
USE_PACKED_SEQUENCE="${USE_PACKED_SEQUENCE:-0}"
DISPATCHER="${DISPATCHER:-alltoall}"
# GDN context-parallel strategy; inert at CP=1. Chunkwise is the default;
# 122B-A10B overrides it to headwise at CP>1 (its chunkwise path deadlocks at
# the first GDN layer).
LINEAR_CP_MODE="${LINEAR_CP_MODE:-chunkwise}"
CP_PARTITION_MODE="${CP_PARTITION_MODE:-}"
CUDA_GRAPH_MODULES="${CUDA_GRAPH_MODULES:-}"
if [[ "${ENABLE_CUDA_GRAPH:-1}" == "0" ]]; then
    CUDA_GRAPH_MODULES=""
fi
TP="${TP:-1}"
EP="${EP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
ETP="${ETP:-1}"
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"
FORCE_LOAD_BALANCING="${FORCE_LOAD_BALANCING:-0}"
RECOMPUTE="${RECOMPUTE:-0}"
RECOMPUTE_MODE="${RECOMPUTE_MODE:-}"
RECOMPUTE_VISION="${RECOMPUTE_VISION:-0}"
GA="${GA:-1}"
CKPT_LOAD="${CKPT_LOAD:-}"
CKPT_FORMAT="${CKPT_FORMAT:-}"
CKPT_RESUME="${CKPT_RESUME:-0}"
CKPT_OVERRIDE_SCHEDULER="${CKPT_OVERRIDE_SCHEDULER:-0}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
# Wrappers declare their default in USE_FSDP_DEFAULT instead of assigning
# USE_FSDP, so the FSDP/PP guard can tell a caller's explicit USE_FSDP=1 from a
# wrapper default.
USE_FSDP_WAS_SET=${USE_FSDP:+x}
USE_FSDP=${USE_FSDP:-${USE_FSDP_DEFAULT:-0}}
MBS="${MBS:-1}"

# ─── Module templates ───────────────────────────────────────────────────────
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/qwen_model_template.sh"
source "${_HERE}/parallel_template.sh"
source "${_HERE}/training_template.sh"

qwen_model_resolve
# Restore the measured Bridge defaults for dense models. MoE defaults are
# unchanged until their separate performance validation is complete.
if (( NUM_EXPERTS == 0 )); then
    export MEGATRON_SWIGLU_USE_CUDAC="${MEGATRON_SWIGLU_USE_CUDAC:-1}"
    export MEGATRON_VISION_FUSION="${MEGATRON_VISION_FUSION:-1}"
    export MEGATRON_VISION_FIXED_LENGTH_ATTENTION="${MEGATRON_VISION_FIXED_LENGTH_ATTENTION:-1}"
    export MEGATRON_SKIP_SINGLETON_PIXEL_CAT="${MEGATRON_SKIP_SINGLETON_PIXEL_CAT:-1}"
    export MEGATRON_VISION_ROPE_TILED="${MEGATRON_VISION_ROPE_TILED:-1}"
    CP_COMM_TYPE="${CP_COMM_TYPE:-p2p}"
    ENABLE_GRAD_OVERLAP="${ENABLE_GRAD_OVERLAP:-1}"
    ENABLE_GDN_FUSED_PRE_GDR="${ENABLE_GDN_FUSED_PRE_GDR:-1}"
    CROSS_ENTROPY_FUSION_IMPL="${CROSS_ENTROPY_FUSION_IMPL:-fla}"
    # Bridge's scheduler overwrote the optimizer's 0.1 with constant 0.033.
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.033}"
    LR="${LR:-5e-6}"; MIN_LR="${MIN_LR:-5e-7}"
    LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-200}"
    LR_DECAY_ITERS="${LR_DECAY_ITERS:-300000}"
    MANUAL_GC_INTERVAL="${MANUAL_GC_INTERVAL:-100}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-0}"; EVAL_ITERS="${EVAL_ITERS:-0}"
    LOG_PARAMS_NORM="${LOG_PARAMS_NORM:-0}"
else
    CP_COMM_TYPE="${CP_COMM_TYPE:-a2a}"
    ENABLE_GRAD_OVERLAP="${ENABLE_GRAD_OVERLAP:-0}"
    ENABLE_GDN_FUSED_PRE_GDR="${ENABLE_GDN_FUSED_PRE_GDR:-0}"
    CROSS_ENTROPY_FUSION_IMPL="${CROSS_ENTROPY_FUSION_IMPL:-none}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
    LR="${LR:-1.2e-4}"; MIN_LR="${MIN_LR:-1.2e-5}"
    LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-100}"
    LR_DECAY_ITERS="${LR_DECAY_ITERS:-2000}"
    MANUAL_GC_INTERVAL="${MANUAL_GC_INTERVAL:-50}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-500}"; EVAL_ITERS="${EVAL_ITERS:-10}"
    LOG_PARAMS_NORM="${LOG_PARAMS_NORM:-1}"
fi
parallel_validate || exit $?
qwen_validate_data_geometry || exit $?

# ─── Environment ────────────────────────────────────────────────────────────
# The dense SwiGLU kernel is included by Dockerfile.Qwen35. The MoE-only
# MEGATRON_MOE_TOPK_RECONSTRUCT port is outside this dense parity change.
COPY_ENGINE_WRAPPER="${COPY_ENGINE_WRAPPER:-0}"

apply_megatron_env() {
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export FLA_CACHE_MODE="${FLA_CACHE_MODE:-default}"
    ENABLE_COPY_ENGINE="${ENABLE_COPY_ENGINE:-0}"
    case "$ENABLE_COPY_ENGINE" in
        1)
            export ENABLE_COPY_ENGINE=1
            export PCCL_CTA_POLICY=2
            export PCCL_CE_ENABLE_RS=1
            if [[ "$COPY_ENGINE_WRAPPER" == "1" ]]; then
                export PCCL_CE_ENABLE_AR=1
                export CPATH="/usr/local/pccl/nccl_wrapper/include${CPATH:+:$CPATH}"
                export CPLUS_INCLUDE_PATH="/usr/local/pccl/nccl_wrapper/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
                export LIBRARY_PATH="/usr/local/pccl/nccl_wrapper/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
                export LD_LIBRARY_PATH="/usr/local/pccl/nccl_wrapper/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            else
                export NCCL_CTA_POLICY=2
            fi
            ;;
        0)
            export ENABLE_COPY_ENGINE=0
            export PCCL_CTA_POLICY=0x1
            export PCCL_CE_ENABLE_RS=0
            if [[ "$COPY_ENGINE_WRAPPER" == "1" ]]; then
                export PCCL_CE_ENABLE_AR=0
            else
                export NCCL_CTA_POLICY=0x1
            fi
            ;;
        *)
            echo "ENABLE_COPY_ENGINE must be 0 or 1" >&2
            exit 2
            ;;
    esac

    export NCCL_GRAPH_REGISTER=0
    export PCCL_GRAPH_REGISTER=0
    export PCCL_RAS_ENABLE="${PCCL_RAS_ENABLE:-0}"
    export TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL="${TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL:-0}"
    export MEGATRON_LOGGING_LEVEL=20
    export NVTE_CPU_OFFLOAD_V1=1
    export UB_SKIPMC=1
    export NVTE_USE_CUTLASS_GROUPED_GEMM=1
    export NVTE_GG_ALLOW_FP32_ACCUM="${NVTE_GG_ALLOW_FP32_ACCUM:-1}"
    export NVTE_CUTLASS3_INCLUDE_DIR="${NVTE_CUTLASS3_INCLUDE_DIR:-/usr/local/lib/python3.12/site-packages/include/cutlass3}"
    export TRITON_CACHE_DIR=/tmp/triton_cache_$(hostname)
    export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1
    # The fused-attention kernels run through TE on this path.
    export NVTE_FUSED_ATTN=1
    # Bridge left this unset. On the dense PPU TP2/CP1 path, a single connection
    # can trade faster collectives for slower overlapping backward GEMMs.
    # Preserve explicit overrides and the native policy on other topologies.
    local default_connections=1
    if (( NUM_EXPERTS == 0 && TP == 2 && CP == 1 && PP == 1 && USE_FSDP == 0 )); then
        default_connections=8
        export MEGATRON_PPU_TP_CONNECTIONS8=1
    fi
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-$default_connections}"
}

# ─── Command assembly ───────────────────────────────────────────────────────
build_megatron_args() {
    qwen_model_args || exit $?
    parallel_args
    training_args || exit $?
    MEGATRON_ARGS=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "${TOKENIZER_MODEL:-$HF_MODEL}"
        "${TRAINING_ARGS[@]}"
        "${PARALLEL_ARGS[@]}"
        "${LOGGING_ARGS[@]}"
        "${QWEN_MODEL_ARGS[@]}"
        "${RECOMPUTE_ARGS[@]}"
        "${FSDP_ARGS[@]}"
        "${CKPT_ARGS[@]}"
    )
}

# run_megatron <tag> [extra args...]
run_megatron() {
    local tag="$1"; shift
    RUN_TAG="$tag"
    local run_args=("$@")

    local log_dir="${OUTPUT_DIR}/${tag}"
    mkdir -p "$log_dir"
    local log_file="${log_dir}/train_node${RANK}.log"
    local default_profile_dir="${log_dir}/profile"
    [[ "$PROFILE_MODE" == "torch" ]] && default_profile_dir="${log_dir}/torch_profile"
    local profile_dir="${PROFILE_DIR:-$default_profile_dir}"
    if [[ "$PROFILE_MODE" != "off" ]]; then
        mkdir -p "$profile_dir"
    fi

    build_megatron_args

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  EXP : ${tag}"
    echo "  RUN : Megatron-LM-for-sail entry point"
    echo "  LOG : ${log_file}"
    [[ "$PROFILE_MODE" != "off" ]] && echo "  PROF: ${PROFILE_MODE} steps ${PROFILE_STEP_START}-${PROFILE_STEP_END} -> ${profile_dir}"
    echo "════════════════════════════════════════════════════════"

    local -a profile_args=()
    if [[ "$PROFILE_MODE" == "torch" ]]; then
        # The pinned MCore CLI converter requires --profile as its master
        # switch; --use-pytorch-profiler selects torch instead of nsys.
        # The launcher window is one-based and inclusive.
        profile_args=(
            --profile
            --use-pytorch-profiler
            --pytorch-profiler-collect-shapes
            --profile-step-start $(( PROFILE_STEP_START - 1 ))
            --profile-step-end "$PROFILE_STEP_END"
            --profile-ranks "${PROFILE_RANKS:-0}"
        )
    fi

    local -a torchrun_cmd=(
        python -m torch.distributed.run
        --nproc_per_node="$NPROC"
        --nnodes="$NNODES"
        --rdzv-backend=static
        --node-rank="$RANK"
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}"
        --max-restarts=0
    )
    local -a launch_cmd=("${torchrun_cmd[@]}")
    if [[ "$PROFILE_MODE" == "asys" ]]; then
        local report_prefix="${CAPTURE_REPORT_PREFIX:-${profile_dir}/capture_node${RANK}}"
        launch_cmd=(bash "${_HERE}/capture_report.sh" "$report_prefix" "${torchrun_cmd[@]}")
        export MODEL_PROFILE_RANGE="${MODEL_PROFILE_RANGE:-$(( PROFILE_STEP_START - 1 )),$(( PROFILE_STEP_END - 1 ))}"
        echo "  capture: ${report_prefix}.asysrep"
    fi

    local -a run_cmd=("${launch_cmd[@]}"
        "$ENTRYPOINT"
        --train-iters "$TRAIN_ITERS"
        --log-interval "$LOG_INTERVAL"
        --tensorboard-dir "${log_dir}/tensorboard"
        "${profile_args[@]}"
        "${MEGATRON_ARGS[@]}"
        "${run_args[@]}")

    local meta_dir="${log_dir}/meta"
    mkdir -p "$meta_dir"
    {
        echo "tag=${tag}"
        echo "path=megatron"
        echo "profile=${PROFILE_MODE}"
        echo "profile_steps=${PROFILE_STEP_START}-${PROFILE_STEP_END}"
        echo "profile_dir=${profile_dir}"
        echo "output_dir=${OUTPUT_DIR}"
        echo "log_file=${log_file}"
        echo "train_iters=${TRAIN_ITERS}"
        echo "log_interval=${LOG_INTERVAL}"
        echo "nproc=${NPROC}"
        echo "nnodes=${NNODES}"
        echo "node_rank=${RANK}"
        echo "master_addr=${MASTER_ADDR}"
        echo "master_port=${MASTER_PORT}"
        echo "entrypoint=${ENTRYPOINT}"
        echo "megatron_lm_path=${MEGATRON_LM_PATH}"
        echo "hf_model=${HF_MODEL}"
        echo "model_variant=${MODEL_VARIANT}"
        echo "tp=${TP} ep=${EP} pp=${PP} cp=${CP} etp=${ETP} dp=${DP}"
        echo "mbs=${MBS} ga=${GA} gbs=${GBS}"
        echo "fsdp=${USE_FSDP}"
        echo "cp_comm_type=${CP_COMM_TYPE}"
        echo "grad_overlap=${ENABLE_GRAD_OVERLAP}"
        echo "gdn_fused_pre_gdr=${ENABLE_GDN_FUSED_PRE_GDR}"
        echo "mtp_num_layers=${MTP_NUM_LAYERS}"
        echo "cuda_graph_modules=${CUDA_GRAPH_MODULES}"
        echo "cross_entropy_fusion_impl=${CROSS_ENTROPY_FUSION_IMPL}"
        echo "weight_decay=${WEIGHT_DECAY}"
        echo "dense_swiglu_cudac=${MEGATRON_SWIGLU_USE_CUDAC:-0}"
        echo "vision_fusion=${MEGATRON_VISION_FUSION:-0}"
        echo "vision_rope_tiled=${MEGATRON_VISION_ROPE_TILED:-0}"
        echo "skip_singleton_pixel_cat=${MEGATRON_SKIP_SINGLETON_PIXEL_CAT:-0}"
        echo "cuda_device_max_connections=${CUDA_DEVICE_MAX_CONNECTIONS:-unset}"
        echo "ppu_tp_connections8=${MEGATRON_PPU_TP_CONNECTIONS8:-0}"
        echo "vision_fixed_length_attention=${MEGATRON_VISION_FIXED_LENGTH_ATTENTION:-0}"
        echo "cuda_graph_warmup_steps=${CUDA_GRAPH_WARMUP_STEPS}"
        echo "started_at=$(date -u '+%F %T') UTC"
    } > "${meta_dir}/run.env"
    printf '%q ' "${run_cmd[@]}" > "${meta_dir}/cmd.txt"
    echo >> "${meta_dir}/cmd.txt"

    if [[ "${TUNE_DRY_RUN:-0}" == "1" ]]; then
        echo "TUNE_DRY_RUN: profile=${PROFILE_MODE} steps=${PROFILE_STEP_START}-${PROFILE_STEP_END} dir=${profile_dir} range=${MODEL_PROFILE_RANGE:-none} capture=${CAPTURE_ASYS}"
        printf 'TUNE_DRY_RUN_CMD:'
        printf ' %q' "${run_cmd[@]}"
        echo
        return 0
    fi

    set +e
    {
        echo "CI_PATH: megatron"
        echo "CI_PROFILE: ${PROFILE_MODE}"
        echo "CI_PROFILE_STEPS: ${PROFILE_STEP_START}-${PROFILE_STEP_END}"
        echo "CI_PROFILER: ${ENABLE_PROFILER}"
        echo "CI_ASYS: ${CAPTURE_ASYS}"
        [[ "$PROFILE_MODE" != "off" ]] && echo "CI_PROFILE_DIR: ${profile_dir}"
        "${run_cmd[@]}"
    } 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -ne 0 ]]; then
        echo "[WARN] ${tag} FAILED (exit ${exit_code})" >&2
        echo "TUNE_STATUS: FAILED exit=${exit_code}" >> "$log_file"
    else
        echo "TUNE_STATUS: OK" >> "$log_file"
    fi
    return "$exit_code"
}
