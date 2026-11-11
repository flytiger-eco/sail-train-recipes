#!/usr/bin/env bash
#
# Shared launcher for the Qwen3.5-VL tuning entrypoints.
#
# Sourced by tune_2b.sh / tune_4b.sh / tune_9b.sh / tune_27b.sh /
# tune_35b_a3b.sh. Each of those files only declares its model (default
# checkpoint, process layout, best component specs) and the run_exp call; the
# process launch, environment and exit-status handling live here.

set -uo pipefail

# ─── User config ────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"
NPROC="${NPROC_PER_NODE:-${NPROC:-8}}"
NNODES="${NNODES:-$NNODES_DEFAULT}"
RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-$MASTER_PORT_DEFAULT}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
HF_MODEL="${HF_MODEL:-$HF_MODEL_DEFAULT}"
# Profiling is one switch with three modes:
#   off   (default) no trace; local runs are unaffected
#   torch           torch profiler trace + memory snapshot
#   asys            model_prof/asys capture (record-only; torch profiler off)
# PROFILE wins over the legacy ENABLE_PROFILER/CAPTURE_ASYS pair so CI, the
# nightly harness and a local run can all use the same name.
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
    echo "CAPTURE_ASYS=1: disabling the torch profiler (trace and snapshot) so the capture is clean" >&2
    ENABLE_PROFILER=0
fi

PROFILE_MODE="off"
if [[ "$CAPTURE_ASYS" == "1" ]]; then
    PROFILE_MODE=asys
elif [[ "$ENABLE_PROFILER" == "1" ]]; then
    PROFILE_MODE=torch
fi

# Traces are captured during the CUDA-graph warmup: the first
# CUDA_GRAPH_WARMUP_STEPS iterations run eagerly, so the trace keeps real kernel
# launches and never covers graph capture or replay. Both profilers use this one
# window, and the capture step that follows it needs no special case.
CUDA_GRAPH_WARMUP_STEPS="${CUDA_GRAPH_WARMUP_STEPS:-3}"

# One window definition for both tools (1-based, closed interval). Explicit
# PROFILE_STEP_START/END win over PROFILE_STEPS.
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
PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
PROFILE_STEP_END="${PROFILE_STEP_END:-$CUDA_GRAPH_WARMUP_STEPS}"
case "${PROFILE_STEP_START}${PROFILE_STEP_END}" in
    ""|*[!0-9]*) echo "PROFILE_STEP_START/END must be numeric" >&2; exit 2 ;;
esac
if (( PROFILE_STEP_START < 1 || PROFILE_STEP_END < PROFILE_STEP_START )); then
    echo "invalid profiling window ${PROFILE_STEP_START}-${PROFILE_STEP_END}" >&2; exit 2
fi

# The MIMO entrypoint is the default; tune_27b.sh overrides this.
SCRIPT="${SCRIPT:-train_qwen35_vl_mimo.py}"
TORCHRUN="python -m torch.distributed.run"

mkdir -p "$OUTPUT_DIR"

# ─── Helper ─────────────────────────────────────────────────────────────────
# run_exp <tag> <entrypoint args...>
#
# The entrypoint args are everything that follows the script name, except the
# output-dir/train-iters/log-interval triple and the profiler flags, which are
# derived from the environment here.
run_exp() {
    local tag="$1"; shift
    local run_args=("$@")

    local log_dir="${OUTPUT_DIR}/${tag}"
    mkdir -p "$log_dir"
    local log_file="${log_dir}/train_node${RANK}.log"
    local profile_dir="${PROFILE_DIR:-${log_dir}/profile}"
    if [[ "$PROFILE_MODE" != "off" ]]; then
        mkdir -p "$profile_dir"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  EXP : ${tag}"
    if [[ -n "${BEST_LANG:-}" ]]; then echo "  LANG: ${BEST_LANG}"; fi
    if [[ -n "${BEST_IMG:-}" ]]; then echo "  IMG : ${BEST_IMG}"; fi
    echo "  LOG : ${log_file}"
    if [[ "$PROFILE_MODE" != "off" ]]; then
        echo "  PROF: ${PROFILE_MODE} steps ${PROFILE_STEP_START}-${PROFILE_STEP_END} -> ${profile_dir}"
    fi
    echo "════════════════════════════════════════════════════════"

    local profile_args=""
    if [[ "$PROFILE_MODE" == "torch" ]]; then
        profile_args="--enable-profiler --enable-memory-snapshot "
        profile_args+="--memory-snapshot-path ${profile_dir}/memory_snapshot_node${RANK}.pickle "
        profile_args+="--profile-step-start ${PROFILE_STEP_START} --profile-step-end ${PROFILE_STEP_END}"
    fi

    local torchrun_cmd=($TORCHRUN)
    local launch_cmd=("${torchrun_cmd[@]}")
    if [[ "$PROFILE_MODE" == "asys" ]]; then
        local helper_dir="${BASH_SOURCE[0]%/*}"
        [[ "$helper_dir" == "${BASH_SOURCE[0]}" ]] && helper_dir="."
        local report_prefix="${CAPTURE_REPORT_PREFIX:-${profile_dir}/capture_node${RANK}}"
        launch_cmd=("bash" "${helper_dir}/capture_report.sh" "$report_prefix" "${torchrun_cmd[@]}")
        # One window definition for both capture paths.
        export MODEL_PROFILE_RANGE="${MODEL_PROFILE_RANGE:-${PROFILE_STEP_START},${PROFILE_STEP_END}}"
        echo "  capture: ${report_prefix}.asysrep"
    fi

    local run_cmd=("${launch_cmd[@]}"
        --nproc_per_node="$NPROC"
        --nnodes="$NNODES"
        --rdzv-backend=static
        --node-rank="$RANK"
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}"
        --max-restarts=0
        "$SCRIPT"
        --output-dir "${log_dir}/tensorboard_node${RANK}"
        --train-iters "$TRAIN_ITERS"
        --log-interval "$LOG_INTERVAL"
        ${profile_args}
        "${run_args[@]}")

    # Structured artifacts: the training log and tensorboard directory stay at
    # the tag root (collectors read them there); profile output and a small
    # manifest live in subdirectories so a run directory stays readable.
    local meta_dir="${log_dir}/meta"
    mkdir -p "$meta_dir"
    # The model structure is an artifact, not log output: the entrypoint writes
    # it here and the FSDP parameter-group block stays out of the training log.
    local model_structure="${meta_dir}/model_structure.json"
    export MODEL_STRUCTURE_JSON="$model_structure"
    {
        echo "tag=${tag}"
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
        echo "script=${SCRIPT}"
        echo "hf_model=${HF_MODEL}"
        echo "cuda_graph_warmup_steps=${CUDA_GRAPH_WARMUP_STEPS}"
        echo "model_structure=${model_structure}"
        echo "started_at=$(date -u '+%F %T') UTC"
    } > "${meta_dir}/run.env"
    printf '%q ' "${run_cmd[@]}" > "${meta_dir}/cmd.txt"
    echo >> "${meta_dir}/cmd.txt"

    if [[ "${TUNE_DRY_RUN:-0}" == "1" ]]; then
        echo "TUNE_DRY_RUN: profile=${PROFILE_MODE} steps=${PROFILE_STEP_START}-${PROFILE_STEP_END} dir=${profile_dir} range=${MODEL_PROFILE_RANGE:-none} capture=${CAPTURE_ASYS} structure=${model_structure}"
        printf 'TUNE_DRY_RUN_CMD:'
        printf ' %q' "${run_cmd[@]}"
        echo
        return 0
    fi

    set +e
    {
        # Machine-readable markers: the console derives the steady-state window
        # and the record-only flag from the run's own log, not from this file.
        echo "CI_PROFILE: ${PROFILE_MODE}"
        echo "CI_PROFILE_STEPS: ${PROFILE_STEP_START}-${PROFILE_STEP_END}"
        echo "CI_PROFILER: ${ENABLE_PROFILER}"
        echo "CI_ASYS: ${CAPTURE_ASYS}"
        if [[ "$PROFILE_MODE" != "off" ]]; then
            echo "CI_PROFILE_DIR: ${profile_dir}"
        fi
        "${run_cmd[@]}"
    } 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -ne 0 ]]; then
        echo "[WARN] ${tag} FAILED (exit ${exit_code})" >&2
        echo "TUNE_STATUS: FAILED exit=${exit_code}" >> "$log_file"
    else
        echo "TUNE_STATUS: OK" >> "$log_file"
    fi

    # The run's exit status is the script's exit status, for a local user and a
    # CI cell alike: one run_exp call per entrypoint.
    return "$exit_code"
}

# ─── Environment ────────────────────────────────────────────────────────────
# ENABLE_COPY_ENGINE=1 routes collectives through the copy engine. The MoE image
# (35B-A3B) additionally ships the CE headers, which dense images do not.
COPY_ENGINE_WRAPPER="${COPY_ENGINE_WRAPPER:-0}"

apply_common_env() {
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
    export MEGATRON_LOGGING_LEVEL=20
    export MEGATRON_SWIGLU_USE_CUDAC="${MEGATRON_SWIGLU_USE_CUDAC:-1}"
    export MEGATRON_MOE_TOPK_RECONSTRUCT="${MEGATRON_MOE_TOPK_RECONSTRUCT:-triton}"
    export NVTE_CPU_OFFLOAD_V1=1
    export UB_SKIPMC=1
    export NVTE_USE_CUTLASS_GROUPED_GEMM=1
    export TRITON_CACHE_DIR=/tmp/triton_cache_$(hostname)
    export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1
}

# cuda_graph_args <scope>
# Sets CUDA_GRAPH_ARGS to the CUDA Graph flags, or to the empty string. Kept as
# a string so an empty value expands to nothing under set -u on bash < 4.4.
# The warmup steps are passed explicitly so the entrypoint and the profiling
# window cannot drift apart.
cuda_graph_args() {
    CUDA_GRAPH_ARGS=""
    [[ "${ENABLE_CUDA_GRAPH:-1}" == "1" ]] || return 0
    CUDA_GRAPH_ARGS="--enable-cuda-graph-te --cuda-graph-warmup ${CUDA_GRAPH_WARMUP_STEPS}"
    [[ -n "${1:-}" ]] && CUDA_GRAPH_ARGS+=" --cuda-graph-scope ${1}"
    return 0
}
