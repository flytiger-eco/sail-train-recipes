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
# Off by default for local runs: profiling writes a trace and a memory snapshot
# and perturbs the profiled steps. CI sets ENABLE_PROFILER=1 because those files
# are its artifacts; the console excludes the captured steps from steady state.
ENABLE_PROFILER="${ENABLE_PROFILER:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-10}"
PROFILE_STEP_END="${PROFILE_STEP_END:-11}"
# CAPTURE_ASYS=1 runs the same entrypoint under model_prof's prof.sh and saves a
# captured report. It is record-only: the torch profiler must stay off so the
# capture is not perturbed, and the console does not publish its metrics.
CAPTURE_ASYS="${CAPTURE_ASYS:-0}"
case "$(printf '%s' "$CAPTURE_ASYS" | tr '[:upper:]' '[:lower:]')" in
    1|true|on|yes) CAPTURE_ASYS=1 ;;
    0|false|off|no|"") CAPTURE_ASYS=0 ;;
    *) echo "CAPTURE_ASYS must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$CAPTURE_ASYS" == "1" && "$ENABLE_PROFILER" == "1" ]]; then
    echo "CAPTURE_ASYS=1: disabling the torch profiler (trace and snapshot) so the capture is clean" >&2
    ENABLE_PROFILER=0
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

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  EXP : ${tag}"
    if [[ -n "${BEST_LANG:-}" ]]; then echo "  LANG: ${BEST_LANG}"; fi
    if [[ -n "${BEST_IMG:-}" ]]; then echo "  IMG : ${BEST_IMG}"; fi
    echo "  LOG : ${log_file}"
    echo "════════════════════════════════════════════════════════"

    local profile_args=""
    if [[ "$ENABLE_PROFILER" == "1" ]]; then
        profile_args="--enable-profiler --enable-memory-snapshot "
        profile_args+="--memory-snapshot-path ${log_dir}/memory_snapshot_node${RANK}.pickle "
        profile_args+="--profile-step-start ${PROFILE_STEP_START} --profile-step-end ${PROFILE_STEP_END}"
    fi

    local torchrun_cmd=($TORCHRUN)
    local launch_cmd=("${torchrun_cmd[@]}")
    if [[ "$CAPTURE_ASYS" == "1" ]]; then
        local helper_dir="${BASH_SOURCE[0]%/*}"
        [[ "$helper_dir" == "${BASH_SOURCE[0]}" ]] && helper_dir="."
        local report_prefix="${CAPTURE_REPORT_PREFIX:-${log_dir}/capture_node${RANK}}"
        launch_cmd=("bash" "${helper_dir}/capture_report.sh" "$report_prefix" "${torchrun_cmd[@]}")
        # One window definition for both capture paths.
        export MODEL_PROFILE_RANGE="${MODEL_PROFILE_RANGE:-${PROFILE_STEP_START},${PROFILE_STEP_END}}"
        echo "  capture: ${report_prefix}.asysrep"
    fi

    set +e
    {
        # Machine-readable markers: the console derives the steady-state window
        # and the record-only flag from the run's own log, not from this file.
        echo "CI_PROFILE_STEPS: ${PROFILE_STEP_START}-${PROFILE_STEP_END}"
        echo "CI_PROFILER: ${ENABLE_PROFILER}"
        echo "CI_ASYS: ${CAPTURE_ASYS}"
        "${launch_cmd[@]}" \
            --nproc_per_node="$NPROC" \
            --nnodes="$NNODES" \
            --rdzv-backend=static \
            --node-rank="$RANK" \
            --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
            --max-restarts=0 \
            "$SCRIPT" \
            --output-dir "${log_dir}/tensorboard_node${RANK}" \
            --train-iters "$TRAIN_ITERS" \
            --log-interval "$LOG_INTERVAL" \
            ${profile_args} \
            "${run_args[@]}"
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
cuda_graph_args() {
    CUDA_GRAPH_ARGS=""
    [[ "${ENABLE_CUDA_GRAPH:-1}" == "1" ]] || return 0
    CUDA_GRAPH_ARGS="--enable-cuda-graph-te"
    [[ -n "${1:-}" ]] && CUDA_GRAPH_ARGS+=" --cuda-graph-scope ${1}"
    return 0
}
