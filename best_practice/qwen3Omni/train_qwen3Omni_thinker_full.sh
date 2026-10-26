#!/usr/bin/env bash
#
# Qwen3-Omni-30B-A3B-Instruct  thinker-only training script
#
# Based on Megatron-Bridge run_recipe.py entry point, supports single/multi-node.
# Usage:
#   Single-node:
#     HF_MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
#     TRAIN_JSONL=/path/to/train.jsonl \
#     bash train_qwen3Omni_thinker_full.sh
#
#   Dual-node (node 0 @ <MASTER_NODE_IP>):
#     NNODES=2 NODE_RANK=0 MASTER_ADDR=<MASTER_NODE_IP> MASTER_PORT=29513 \
#     HF_MODEL_PATH=... TRAIN_JSONL=... \
#     bash train_qwen3Omni_thinker_full.sh
#
#   Dual-node (node 1):
#     NNODES=2 NODE_RANK=1 MASTER_ADDR=<MASTER_NODE_IP> MASTER_PORT=29513 \
#     HF_MODEL_PATH=... TRAIN_JSONL=... \
#     bash train_qwen3Omni_thinker_full.sh

set -euo pipefail

# ─── Locate Megatron-Bridge root directory ────────────────────────────────────
# Inside container: /workspace/Megatron-Bridge (Dockerfile WORKDIR)
# Outside container: specified via MBRIDGE_ROOT env var
MBRIDGE_ROOT="${MBRIDGE_ROOT:-/workspace/Megatron-Bridge}"
if [[ ! -d "${MBRIDGE_ROOT}/src" ]]; then
  echo "[error] MBRIDGE_ROOT=${MBRIDGE_ROOT} invalid, src/ directory not found" >&2
  echo "       Please set MBRIDGE_ROOT pointing to Megatron-Bridge root directory" >&2
  exit 1
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="${MBRIDGE_ROOT}/src:${MBRIDGE_ROOT}/3rdparty/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

# ─── Cache / temp directories ─────────────────────────────────────────────────
V="${CACHE_ROOT:-/root/omni}"
export HF_HOME="${HF_HOME:-${V}/cache/hf}"
export TMPDIR="${TMPDIR:-${V}/tmp}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${V}/cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${V}/cache/triton}"
export TORCH_HOME="${TORCH_HOME:-${V}/cache/torch}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${V}/cache/numba}"
mkdir -p "${V}/tmp" "${V}/cache"/{hf,triton,torch,numba,torch_extensions}

WORKSPACE="${WORKSPACE:-${MBRIDGE_ROOT}/.cache/qwen3_omni_train}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE}/results}"
LOG_DIR="${LOG_DIR:-${WORKSPACE}/logs}"
mkdir -p "${WORKSPACE}" "${RESULTS_DIR}" "${LOG_DIR}"

# ─── Model & Data ─────────────────────────────────────────────────────────────
HF_MODEL_PATH="${HF_MODEL_PATH:-}"
TRAIN_JSONL="${TRAIN_JSONL:-}"
VALID_JSONL="${VALID_JSONL:-}"
TEST_JSONL="${TEST_JSONL:-}"

if [[ -z "${HF_MODEL_PATH}" ]]; then
  echo "[error] HF_MODEL_PATH must be set (local HF checkpoint path)" >&2
  exit 1
fi
if [[ -z "${TRAIN_JSONL}" ]]; then
  echo "[error] TRAIN_JSONL must be set (training data jsonl path)" >&2
  exit 1
fi

# ─── Parallelism & Training hyperparameters ──────────────────────────────────
NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

RECIPE="${RECIPE:-qwen3_omni_30b_a3b_sft_hf_json_config}"
STEP_FUNC="${STEP_FUNC:-qwen3_omni_step}"

SEQ_LENGTH="${SEQ_LENGTH:-8192}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
EVAL_ITERS="${EVAL_ITERS:-0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"

# ─── Parallelism ──────────────────────────────────────────────────────────────
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-False}"

# ─── MoE Token Dispatcher / DeepEP ──────────────────────────────────────────
# Default flex + deepep: MoE token dispatch via DeepEP all-to-all (deep_ep pre-installed, takes effect when EP>1)
# Fallback path: MOE_TOKEN_DISPATCHER_TYPE=alltoall (or allgather)
# Note: current image does not support model.moe_deepep_autotune override (qwen35.patch not included in this image)
MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-flex}"
MOE_FLEX_DISPATCHER_BACKEND="${MOE_FLEX_DISPATCHER_BACKEND:-deepep}"
MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-True}"
MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-True}"
MOE_ROUTER_FUSION="${MOE_ROUTER_FUSION:-True}"
MOE_DEEPEP_NUM_SMS="${MOE_DEEPEP_NUM_SMS:-28}"

# ─── Precision & Memory optimization ──────────────────────────────────────────
BIAS_ACTIVATION_FUSION="${BIAS_ACTIVATION_FUSION:-True}"
VIT_GRADIENT_CHECKPOINTING="${VIT_GRADIENT_CHECKPOINTING:-False}"
FREEZE_LANGUAGE_MODEL="${FREEZE_LANGUAGE_MODEL:-False}"
FREEZE_VISION_MODEL="${FREEZE_VISION_MODEL:-True}"
FREEZE_AUDIO_MODEL="${FREEZE_AUDIO_MODEL:-True}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-False}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.0}"
OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-False}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-}"

# ─── Recompute ───────────────────────────────────────────────────────────────
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-block}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-26}"
RECOMPUTE_MODULES="${RECOMPUTE_MODULES:-null}"

# ─── Communication overlap ────────────────────────────────────────────────────
OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-True}"
OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-True}"
ALIGN_PARAM_GATHER="${ALIGN_PARAM_GATHER:-}"

# ─── Attention ────────────────────────────────────────────────────────────────
MULTIMODAL_ATTN_IMPL="${MULTIMODAL_ATTN_IMPL:-flash_attention_3}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"

# ─── Data loading ─────────────────────────────────────────────────────────────
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-8}"
DATASET_PERSISTENT_WORKERS="${DATASET_PERSISTENT_WORKERS:-False}"

# ─── Profiling ────────────────────────────────────────────────────────────────
USE_PYTORCH_PROFILER="${USE_PYTORCH_PROFILER:-}"
PROFILE_STEP_START="${PROFILE_STEP_START:-}"
PROFILE_STEP_END="${PROFILE_STEP_END:-}"
PROFILE_RANKS="${PROFILE_RANKS:-}"

# ─── Logging ──────────────────────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-qwen3_omni_sail_train}"
PERF_TAG="${PERF_TAG:-full}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_NAME}_${PERF_TAG}_node${NODE_RANK}.log}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RESULTS_DIR}/${RUN_NAME}/tb_logs}"

# ─── PPU / NCCL optimization ─────────────────────────────────────────────────
export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT="${PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT:-True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# ─── Video/image token limit (optional) ───────────────────────────────────────
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-}"
export MAX_PIXELS="${MAX_PIXELS:-}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-}"

# ─── Thinker-only mirror (disable audio output) ──────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
THINKER_ONLY_MIRROR_DIR="${THINKER_ONLY_MIRROR_DIR:-${WORKSPACE}/hf_thinker_only}"
mkdir -p "${THINKER_ONLY_MIRROR_DIR}"

prepare_thinker_only_hf_path() {
    local src_path=$1
    local mirror_root=$2
    local python_bin=$3

    local config_path="${src_path}/config.json"
    if [[ ! -f "${config_path}" ]]; then
        echo "${src_path}"
        return 0
    fi

    local mirror_path
    mirror_path=$("${python_bin}" - "${src_path}" "${mirror_root}" <<'PY'
import json, os, sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
mirror_root = Path(sys.argv[2]).resolve()
config_path = src / "config.json"

with config_path.open() as f:
    config = json.load(f)

if not config.get("enable_audio_output", False):
    print(str(src))
    raise SystemExit(0)

mirror = mirror_root / f"{src.name}-thinker-only"
mirror.mkdir(parents=True, exist_ok=True)

for child in src.iterdir():
    dst = mirror / child.name
    if child.name == "config.json":
        continue
    if dst.exists() or dst.is_symlink():
        continue
    try:
        os.symlink(child, dst, target_is_directory=child.is_dir())
    except FileExistsError:
        pass

patched = dict(config)
patched["enable_audio_output"] = False
patched["talker_config"] = None
patched["code2wav_config"] = None

tmp = mirror / f"config.json.tmp.{os.getpid()}"
with tmp.open("w") as f:
    json.dump(patched, f, indent=2, ensure_ascii=False)
    f.write("\n")
tmp.replace(mirror / "config.json")

print(str(mirror))
PY
)
    echo "${mirror_path}"
}

EFFECTIVE_HF_MODEL_PATH=$(prepare_thinker_only_hf_path "${HF_MODEL_PATH}" "${THINKER_ONLY_MIRROR_DIR}" "${PYTHON_BIN}")

# Offline mode: prevent recipe from accessing Hub during build
export QWEN3_OMNI_HF_PATH="${EFFECTIVE_HF_MODEL_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ─── Resolve MASTER_ADDR (hostname -> IPv4) ───────────────────────────────────
if [[ "${MASTER_ADDR}" == *,* ]]; then
    MASTER_ADDR=${MASTER_ADDR%%,*}
fi

resolve_master_addr_ipv4() {
    local addr=$1
    "${PYTHON_BIN}" - "$addr" <<'PYEOF'
import re, socket, sys
addr = sys.argv[1]
match = re.match(r"^(\d+)-(\d+)-(\d+)-(\d+)(?:\.|$)", addr)
if match:
    print(".".join(match.groups()))
    raise SystemExit(0)
try:
    infos = socket.getaddrinfo(addr, None, socket.AF_INET, socket.SOCK_STREAM)
except socket.gaierror:
    print(addr); raise SystemExit(0)
seen = set()
for info in infos:
    ip = info[4][0]
    if ip not in seen:
        print(ip); break
    seen.add(ip)
else:
    print(addr)
PYEOF
}
MASTER_ADDR=$(resolve_master_addr_ipv4 "${MASTER_ADDR}")
export MASTER_ADDR MASTER_PORT

# ─── Build torchrun command ───────────────────────────────────────────────────
CMD=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nproc_per_node="${NUM_GPUS}"
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    "${MBRIDGE_ROOT}/scripts/training/run_recipe.py"
    --recipe "${RECIPE}"
    --step_func "${STEP_FUNC}"
    checkpoint.pretrained_checkpoint="${EFFECTIVE_HF_MODEL_PATH}"
    checkpoint.save="${RESULTS_DIR}/${RUN_NAME}"
    checkpoint.save_interval="${SAVE_INTERVAL}"
    model.seq_length="${SEQ_LENGTH}"
    model.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE}"
    model.pipeline_model_parallel_size="${PIPELINE_PARALLEL_SIZE}"
    model.context_parallel_size="${CONTEXT_PARALLEL_SIZE}"
    model.expert_model_parallel_size="${EXPERT_MODEL_PARALLEL_SIZE}"
    model.expert_tensor_parallel_size="${EXPERT_TENSOR_PARALLEL_SIZE}"
    model.sequence_parallel="${SEQUENCE_PARALLEL}"
    model.freeze_language_model="${FREEZE_LANGUAGE_MODEL}"
    model.freeze_vision_model="${FREEZE_VISION_MODEL}"
    model.freeze_audio_model="${FREEZE_AUDIO_MODEL}"
    model.vit_gradient_checkpointing="${VIT_GRADIENT_CHECKPOINTING}"
    model.multimodal_attn_impl="${MULTIMODAL_ATTN_IMPL}"
    model.attention_backend="${ATTENTION_BACKEND}"
    model.flash_attention_version=3
    model.attention_dropout=0.0
    model.moe_token_dispatcher_type="${MOE_TOKEN_DISPATCHER_TYPE}"
    model.moe_flex_dispatcher_backend="${MOE_FLEX_DISPATCHER_BACKEND}"
    model.moe_grouped_gemm="${MOE_GROUPED_GEMM}"
    model.moe_permute_fusion="${MOE_PERMUTE_FUSION}"
    model.moe_router_fusion="${MOE_ROUTER_FUSION}"
    model.bias_activation_fusion="${BIAS_ACTIVATION_FUSION}"
    train.train_iters="${TRAIN_ITERS}"
    train.global_batch_size="${GLOBAL_BATCH_SIZE}"
    train.micro_batch_size="${MICRO_BATCH_SIZE}"
    validation.eval_iters="${EVAL_ITERS}"
    validation.eval_interval="${EVAL_INTERVAL}"
    logger.log_interval="${LOG_INTERVAL}"
    logger.log_throughput=True
    logger.log_throughput_to_tensorboard=True
    logger.tensorboard_dir="${TENSORBOARD_DIR}"
    logger.log_timers_to_tensorboard=False
    logger.wandb_exp_name="${RUN_NAME}"
    dataset.seq_length="${SEQ_LENGTH}"
    dataset.hf_processor_path="${EFFECTIVE_HF_MODEL_PATH}"
    "dataset.source.load_kwargs={data_files:{train:${TRAIN_JSONL}}}"
    dataset.num_workers="${DATASET_NUM_WORKERS}"
    dataset.persistent_workers="${DATASET_PERSISTENT_WORKERS}"
    dataset.enable_in_batch_packing=False
)

# Validate data
if [[ -n "${VALID_JSONL}" ]]; then
    CMD+=("dataset.validation_source.load_kwargs={data_files:{validation:${VALID_JSONL}}}")
else
    CMD+=(dataset.do_validation=False dataset.validation_source=null)
fi
if [[ -n "${TEST_JSONL}" ]]; then
    CMD+=("dataset.test_source.load_kwargs={data_files:{test:${TEST_JSONL}}}")
else
    CMD+=(dataset.do_test=False dataset.test_source=null)
fi

# Recompute
[[ -n "${RECOMPUTE_GRANULARITY}" ]] && CMD+=(model.recompute_granularity="${RECOMPUTE_GRANULARITY}")
[[ -n "${RECOMPUTE_METHOD}" ]]       && CMD+=(model.recompute_method="${RECOMPUTE_METHOD}")
[[ -n "${RECOMPUTE_NUM_LAYERS}" ]]   && CMD+=(model.recompute_num_layers="${RECOMPUTE_NUM_LAYERS}")
[[ -n "${RECOMPUTE_MODULES}" ]]      && CMD+=(model.recompute_modules="${RECOMPUTE_MODULES}")

# Optimizer
[[ -n "${OPTIMIZER_CPU_OFFLOAD}" ]]          && CMD+=(optimizer.optimizer_cpu_offload="${OPTIMIZER_CPU_OFFLOAD}")
[[ -n "${OPTIMIZER_OFFLOAD_FRACTION}" ]]     && CMD+=(optimizer.optimizer_offload_fraction="${OPTIMIZER_OFFLOAD_FRACTION}")
[[ -n "${OVERLAP_CPU_OPTIMIZER_D2H_H2D}" ]]  && CMD+=(optimizer.overlap_cpu_optimizer_d2h_h2d="${OVERLAP_CPU_OPTIMIZER_D2H_H2D}")
[[ -n "${USE_PRECISION_AWARE_OPTIMIZER}" ]]  && CMD+=(optimizer.use_precision_aware_optimizer="${USE_PRECISION_AWARE_OPTIMIZER}")

# DDP overlap
[[ -n "${OVERLAP_GRAD_REDUCE}" ]]   && CMD+=(ddp.overlap_grad_reduce="${OVERLAP_GRAD_REDUCE}")
[[ -n "${OVERLAP_PARAM_GATHER}" ]]  && CMD+=(ddp.overlap_param_gather="${OVERLAP_PARAM_GATHER}")
[[ -n "${ALIGN_PARAM_GATHER}" ]]    && CMD+=(ddp.align_param_gather="${ALIGN_PARAM_GATHER}")

# DeepEP tuning (optional): SM count for communication kernel
[[ -n "${MOE_DEEPEP_NUM_SMS}" ]] && CMD+=(model.moe_deepep_num_sms="${MOE_DEEPEP_NUM_SMS}")

# Profiling
[[ -n "${USE_PYTORCH_PROFILER}" ]]  && CMD+=(profiling.use_pytorch_profiler="${USE_PYTORCH_PROFILER}")
[[ -n "${PROFILE_STEP_START}" ]]    && CMD+=(profiling.profile_step_start="${PROFILE_STEP_START}")
[[ -n "${PROFILE_STEP_END}" ]]      && CMD+=(profiling.profile_step_end="${PROFILE_STEP_END}")
[[ -n "${PROFILE_RANKS}" ]]         && CMD+=(profiling.profile_ranks="${PROFILE_RANKS}")

# ─── Print config ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Qwen3-Omni Thinker Training (Megatron-Bridge)"
echo "════════════════════════════════════════════════════════"
echo "  MBRIDGE_ROOT       = ${MBRIDGE_ROOT}"
echo "  HF_MODEL_PATH      = ${HF_MODEL_PATH}"
echo "  EFFECTIVE_HF_PATH  = ${EFFECTIVE_HF_MODEL_PATH}"
echo "  TRAIN_JSONL        = ${TRAIN_JSONL}"
echo "  NUM_GPUS           = ${NUM_GPUS}"
echo "  NNODES             = ${NNODES}"
echo "  NODE_RANK          = ${NODE_RANK}"
echo "  MASTER_ADDR        = ${MASTER_ADDR}"
echo "  MASTER_PORT        = ${MASTER_PORT}"
echo "  SEQ_LENGTH         = ${SEQ_LENGTH}"
echo "  TRAIN_ITERS        = ${TRAIN_ITERS}"
echo "  GBS                = ${GLOBAL_BATCH_SIZE}"
echo "  MBS                = ${MICRO_BATCH_SIZE}"
echo "  TP/PP/CP           = ${TENSOR_PARALLEL_SIZE}/${PIPELINE_PARALLEL_SIZE}/${CONTEXT_PARALLEL_SIZE}"
echo "  EP/ETP             = ${EXPERT_MODEL_PARALLEL_SIZE}/${EXPERT_TENSOR_PARALLEL_SIZE}"
echo "  MoE DISPATCH       = ${MOE_TOKEN_DISPATCHER_TYPE}/${MOE_FLEX_DISPATCHER_BACKEND}"
echo "  MOE_ROUTER_FUSION  = ${MOE_ROUTER_FUSION}"
echo "  MOE_DEEPEP_NUM_SMS = ${MOE_DEEPEP_NUM_SMS}"
echo "  SP                 = ${SEQUENCE_PARALLEL}"
echo "  RECOMPUTE          = ${RECOMPUTE_GRANULARITY}/${RECOMPUTE_METHOD}/layers=${RECOMPUTE_NUM_LAYERS}/mods=${RECOMPUTE_MODULES}"
echo "  VIT_CKPT           = ${VIT_GRADIENT_CHECKPOINTING}"
echo "  BIAS_ACTIVATION_FUSION = ${BIAS_ACTIVATION_FUSION}"
echo "  PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT = ${PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT}"
echo "  OPTIMIZER_OFFLOAD  = ${OPTIMIZER_CPU_OFFLOAD}"
echo "  FREEZE(L/V/A)      = ${FREEZE_LANGUAGE_MODEL}/${FREEZE_VISION_MODEL}/${FREEZE_AUDIO_MODEL}"
echo "  LOG_PATH           = ${LOG_PATH}"
echo "  TENSORBOARD_DIR    = ${TENSORBOARD_DIR}"
echo "════════════════════════════════════════════════════════"
echo ""

# ─── Execute ──────────────────────────────────────────────────────────────────
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] %q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

cd "${MBRIDGE_ROOT}"
SECONDS=0
set -o pipefail
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
WALL_TIME=${SECONDS}

# ─── Performance summary ─────────────────────────────────────────────────────
python - "${LOG_PATH}" "${GLOBAL_BATCH_SIZE}" "${SEQ_LENGTH}" "${WALL_TIME}" <<'PY'
import re, sys
from pathlib import Path

log_path = Path(sys.argv[1])
global_batch = int(sys.argv[2])
seq_length = int(sys.argv[3])
wall_time = int(sys.argv[4])

elapsed, throughput = [], []
for line in log_path.read_text().splitlines():
    if m := re.search(r"elapsed time per iteration \(ms\):\s+([\d.]+)", line):
        elapsed.append(float(m.group(1)))
    if m := re.search(r"throughput per GPU \(TFLOP/s/GPU\):\s+([\d.]+)", line):
        throughput.append(float(m.group(1)))

print("[perf] wall_time_s=", wall_time, sep="")
if elapsed:
    last_ms = elapsed[-1]
    avg_ms = sum(elapsed) / len(elapsed)
    nominal_tokens_per_sec = global_batch * seq_length / (avg_ms / 1000.0)
    print(f"[perf] last_iter_ms={last_ms:.2f}")
    print(f"[perf] avg_iter_ms={avg_ms:.2f}")
    print(f"[perf] nominal_tokens_per_sec={nominal_tokens_per_sec:.2f}")
else:
    print("[perf] no iteration timing found in log")

if throughput:
    print(f"[perf] last_tflops_per_gpu={throughput[-1]:.2f}")
else:
    print("[perf] no throughput line found in log")
PY
