#!/bin/bash
# ============================================================================
# Qwen3.5-35B-A3B GoLongRL GRPO training / tuning entry — single-node 8x M890
# (TP x CP x EP8, colocate rollout). Derived from the slime 0.4.0 GoLongRL recipe.
#
# The per-band defaults below are the validated best-performance configurations
# measured on 8x M890 (see README "Performance Results"). Override any knob via
# environment variable for controlled comparisons.
#
# Band -> parallelism -> weight layout (conversion TP must match training TP):
#   32k  TP1 CP4 EP8 DP2  recompute off        expandable off  -> TP1 weights
#   64k  TP2 CP4 EP8 DP1  recompute off        expandable off  -> TP2 weights
#   128k TP2 CP4 EP8 DP1  recompute block(20)  expandable on   -> TP2 weights
#   256k TP1 CP8 EP8 DP1  recompute full-unif  expandable on   -> TP1 weights
# GDN_TORCH_DIST defaults to the _tp${TP} directory resolved from the effective TP
# (after band defaults + TP_OVERRIDE); an explicit GDN_TORCH_DIST always wins. Prepare
# each layout with prepare_weights.sh CONVERT_TP=<TP> (TP1 for 32k/256k, TP2 for 64k/128k).
#
# Usage (inside container):
#   BAND=32k  bash tune_slime_golongrl.sh                 # 32k recommended config
#   BAND=128k bash tune_slime_golongrl.sh                 # 128k block-recompute + expandable
#   BAND=64k RECOMPUTE=on bash tune_slime_golongrl.sh     # override recompute
#   EXP_TAG=ablation NUM_ROLLOUT=10 bash tune_slime_golongrl.sh
#
# Main knobs (env): BAND / RECOMPUTE(off|on|selective|block) / OFFLOAD(on|off) /
#   NUM_ROLLOUT / SAVE_INTERVAL / EXP_TAG / TP_OVERRIDE / CP_OVERRIDE / MTPG_OVERRIDE /
#   RECOMPUTE_OVERRIDE / RECOMPUTE_NUM_LAYERS / SLIME_ENABLE_EXPANDABLE_SEGMENTS /
#   PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT(0|1, default 1) / MOE_GROUPED_GEMM(0|1) /
#   EXTRA_TRAIN_ENV_JSON (JSON object of extra env_vars).
#
# The 32k switch ablation (9 arms, warm-step mean, ±2% noise floor) found only
# PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT beneficial (-3.0% train time / +3.1% TFLOP/s),
# so it defaults to 1 here. Disabling torch.compile was harmful (+13.9%) and grouped
# GEMM slower (+8.1%) on this uneven-m dynamic batch; both stay off by default.
# Validation runs default to 3 rollouts and no checkpoint save (SAVE_INTERVAL=999)
# with an isolated ckpt dir to avoid resume pollution.
# ============================================================================
set -ex
# Shared config — env-overridable; defaults are neutral placeholders (set to your
# storage layout). GCS_PORT/DASH_PORT avoid the ray worker range 10002-19999;
# RAY_TMP_DIR kept short (AF_UNIX 107-byte limit). Conversion TP must match training TP.
NAS_ROOT=${NAS_ROOT:-/workspace/golongrl}
MODEL_ROOT=${MODEL_ROOT:-/workspace/models}
HF_CKPT=${HF_CKPT:-${MODEL_ROOT}/Qwen3.5-35B-A3B}
GDN_TORCH_DIST=${GDN_TORCH_DIST:-}   # resolved from effective TP after the override block
SLIME_DATA_ROOT=${SLIME_DATA_ROOT:-${NAS_ROOT}/data}
DATA_SUFFIX=${DATA_SUFFIX:-}
CKPT_ROOT=${CKPT_ROOT:-${NAS_ROOT}/checkpoints}
NUM_GPUS=${NUM_GPUS:-8}
GCS_PORT=${GCS_PORT:-26580}
DASH_PORT=${DASH_PORT:-28366}
RAY_TMP_DIR=${RAY_TMP_DIR:-/tmp/ray_golongrl}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
SLIME_ROOT=${SLIME_ROOT:-$(pip show slime 2>/dev/null | awk '/Editable project location:/{print $NF}')}
MEGATRON_ROOT=${MEGATRON_ROOT:-$(pip show megatron-core 2>/dev/null | awk '/Editable project location:/{print $NF}')}
export NAS_ROOT MODEL_ROOT HF_CKPT GDN_TORCH_DIST SLIME_DATA_ROOT DATA_SUFFIX \
       CKPT_ROOT NUM_GPUS GCS_PORT DASH_PORT RAY_TMP_DIR MASTER_ADDR SLIME_ROOT MEGATRON_ROOT

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTHONBUFFERED=16
export RAY_DEDUP_LOGS=0

BAND=${BAND:-32k}
# Capture caller plain RECOMPUTE before band defaults overwrite it; final priority is
# RECOMPUTE_OVERRIDE > caller RECOMPUTE > band default (see override block below).
RECOMPUTE_USER=${RECOMPUTE:-}
RECOMPUTE=${RECOMPUTE:-off}
OFFLOAD=${OFFLOAD:-on}
NUM_ROLLOUT=${NUM_ROLLOUT:-3}
SAVE_INTERVAL=${SAVE_INTERVAL:-999}
DP_ATTN=${DP_ATTN:-off}
RADIX=${RADIX:-on}
EXP_TAG=${EXP_TAG:-run}
EP_SIZE=${EP_SIZE:-${NUM_GPUS}}
# expandable_segments: historically triggered an alixpu kernel NULL deref on train
# actor NCCL process-group reload (whole-machine reboot), and later a pidfd_getfd
# failure at the first colocate weight sync. Both are resolved by the
# expandable IPC export-buffer fix in qwen35-golongrl-slime.patch (baked into the
# Dockerfile); with it applied, expandable=1 is safe and is the default for 128k/256k.
EXPANDABLE=${SLIME_ENABLE_EXPANDABLE_SEGMENTS:-0}
# TE fused cross-entropy (default 0): when 1, replaces the hand-rolled
# vocab_parallel_softmax with transformer_engine.parallel_cross_entropy; backward
# recomputes softmax without materializing the full [T,V/TP] fp32 buffer, lowering
# the log-prob memory peak. Only effective when entropy_coef=0 and no top-p keep-mask
# (TE cross-entropy change in qwen35-golongrl-slime.patch).
TE_CE=${SLIME_USE_TE_CROSS_ENTROPY:-0}

# ---------- pre-flight (shared-host safety: refuse to start unless NUM_GPUS idle cards + ports free) ----------
PPU_SMI_OUT=$(ppu-smi) || { echo "ERROR: ppu-smi failed; cannot verify GPU state" >&2; exit 1; }
N_CARDS=$(printf '%s\n' "${PPU_SMI_OUT}" | grep -oE '[0-9]+MiB / [0-9]+MiB' | wc -l)
if [ "${N_CARDS}" -ne "${NUM_GPUS}" ]; then
    echo "ERROR: expected ${NUM_GPUS} cards, ppu-smi reports ${N_CARDS}; refuse to start" >&2
    exit 1
fi
printf '%s\n' "${PPU_SMI_OUT}" | grep -oE '[0-9]+MiB / [0-9]+MiB' | awk -F'MiB' '{if ($1+0 > 1000) {print "BUSY card mem: " $0; exit 1}}'
for p in ${GCS_PORT} ${DASH_PORT}; do
    if ! python3 -c "import socket;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('0.0.0.0',${p}));s.close()" 2>/dev/null; then
        echo "ERROR: port ${p} already in use, refuse to start" >&2
        exit 1
    fi
done

source "${SLIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"

# ---------- per-band recommended configuration ----------
# ROLLOUT_MEM_UTIL: sglang engine mem_fraction_static.
case "${BAND}" in
  32k)  TP=1; CP=4; RESP=4096; RB=16; GBS=128; MTPG=8192;  ROLLOUT_MEM_UTIL=0.7;  RECOMPUTE=off;   EXPANDABLE=0 ;;
  64k)  TP=2; CP=4; RESP=4096; RB=16; GBS=128; MTPG=8192;  ROLLOUT_MEM_UTIL=0.7;  RECOMPUTE=off;   EXPANDABLE=0 ;;
  128k) TP=2; CP=4; RESP=8192; RB=16; GBS=128; MTPG=8192;  ROLLOUT_MEM_UTIL=0.7;  RECOMPUTE=block; EXPANDABLE=1; RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-20} ;;
  256k) TP=1; CP=8; RESP=8192; RB=16; GBS=128; MTPG=16384; ROLLOUT_MEM_UTIL=0.55; RECOMPUTE=on;    EXPANDABLE=1 ;;
  *) echo "unknown BAND=${BAND}" >&2; exit 1 ;;
esac
# honor caller-provided expandable override even after band defaults
EXPANDABLE=${SLIME_ENABLE_EXPANDABLE_SEGMENTS:-${EXPANDABLE}}

# TP*CP must divide total cards (DP=NUM_GPUS/(TP*CP)); EP must not exceed total cards.
if (( NUM_GPUS % (TP * CP) != 0 )); then
   echo "ERROR: TP${TP}*CP${CP} does not divide NUM_GPUS=${NUM_GPUS}" >&2; exit 1
fi
if (( EP_SIZE > NUM_GPUS )); then
   echo "ERROR: EP_SIZE=${EP_SIZE} > NUM_GPUS=${NUM_GPUS}" >&2; exit 1
fi

# Optional overrides (controlled comparisons / batch analysis).
RB=${RB_OVERRIDE:-${RB}}
GBS=${GBS_OVERRIDE:-${GBS}}
MTPG=${MTPG_OVERRIDE:-${MTPG}}
TP=${TP_OVERRIDE:-${TP}}
CP=${CP_OVERRIDE:-${CP}}
RECOMPUTE=${RECOMPUTE_OVERRIDE:-${RECOMPUTE_USER:-${RECOMPUTE}}}

# Resolve the GDN checkpoint default from the effective TP (after band defaults +
# TP_OVERRIDE); an explicit caller GDN_TORCH_DIST always wins. Conversion TP must
# match training TP: TP1 bands (32k/256k) -> _tp1, TP2 bands (64k/128k) -> _tp2.
GDN_TORCH_DIST=${GDN_TORCH_DIST:-${MODEL_ROOT}/Qwen3.5-35B-A3B_slime_torch_dist_gdn_tp${TP}}
export GDN_TORCH_DIST
if [ ! -f "${GDN_TORCH_DIST}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: ${GDN_TORCH_DIST} missing latest_checkpointed_iteration.txt (run prepare_weights.sh with CONVERT_TP=${TP} first)" >&2
    exit 1
fi
# Optional log-probs-chunk-size (chunk CE along token dim to lower forward peak; empty=no chunking).
LOG_PROBS_CHUNK=${LOG_PROBS_CHUNK:-}
# Optional log-probs-max-tokens-per-gpu (ref/old_log_prob forward packing cap; empty=follow MTPG).
LOG_PROBS_MTPG=${LOG_PROBS_MTPG:-}

GOLONGRL_DATA=${SLIME_DATA_ROOT}/band${BAND}${DATA_SUFFIX}
CKPT_DIR=${CKPT_ROOT}/golongrl_${BAND}_${EXP_TAG}${DATA_SUFFIX}
CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${GDN_TORCH_DIST}/"
   --load "${CKPT_DIR}/"
   --save "${CKPT_DIR}/"
   --save-interval ${SAVE_INTERVAL}
   --no-save-optim
)

ROLLOUT_ARGS=(
   --prompt-data "${GOLONGRL_DATA}/train.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --custom-rm-path slime_plugins.rm.golongrl.golongrl_rm
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${RB}
   --n-samples-per-prompt 8
   --rollout-max-response-len ${RESP}
   --rollout-temperature 1

   --global-batch-size ${GBS}
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CP}
   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MTPG}
)
if [ -n "${LOG_PROBS_CHUNK}" ]; then
   PERF_ARGS+=(--log-probs-chunk-size ${LOG_PROBS_CHUNK})
fi
if [ -n "${LOG_PROBS_MTPG}" ]; then
   PERF_ARGS+=(--log-probs-max-tokens-per-gpu ${LOG_PROBS_MTPG})
fi
if [ "${RECOMPUTE}" = "on" ]; then
   PERF_ARGS+=(
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
   )
elif [ "${RECOMPUTE}" = "selective" ]; then
   PERF_ARGS+=(
      --recompute-granularity selective
      --recompute-modules ${RECOMPUTE_MODULES:-layernorm}
   )
elif [ "${RECOMPUTE}" = "block" ]; then
   # Block recompute: only recompute the first RECOMPUTE_NUM_LAYERS layers; the rest
   # are not recomputed (trade memory for speed).
   PERF_ARGS+=(
      --recompute-granularity full
      --recompute-method block
      --recompute-num-layers ${RECOMPUTE_NUM_LAYERS:-20}
   )
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-precision-aware-optimizer
)
if [ "${OFFLOAD}" = "on" ]; then
   OPTIMIZER_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
   )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static ${ROLLOUT_MEM_UTIL}
   --sglang-ep-size 8
   --sglang-moe-dense-tp-size 1

   --sglang-max-mamba-cache-size 512

   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 32 256)

   --sglang-server-concurrency 1024
   --sglang-max-running-requests 512

   --sglang-chunked-prefill-size 8192
   --sglang-page-size 64
   --sglang-disable-shared-experts-fusion
   --sglang-attention-backend fa3
   --sglang-disable-piecewise-cuda-graph
   --sglang-stream-interval 128

   --sglang-watchdog-timeout 3600
)
# DP attention (default off: single TP8-EP8 engine; n samples share prefix via radix).
if [ "${DP_ATTN}" = "on" ]; then
   SGLANG_ARGS+=(--sglang-enable-dp-attention --sglang-dp-size 8 --sglang-enable-dp-lm-head)
else
   SGLANG_ARGS+=(--sglang-dp-size 1)
fi
# radix cache (default on: prefix caching, GDN supported via MambaRadixCache).
if [ "${RADIX}" = "off" ]; then
   SGLANG_ARGS+=(--sglang-disable-radix-cache)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   --moe-token-dispatcher-type flex
   --moe-enable-deepep

   --qwen-gdn-impl native
   --linear-cp-mode "${LINEAR_CP_MODE:-chunkwise}"
)
# MoE grouped GEMM (default 0): route expert FFN through TEGroupedMLP/GroupedLinear so
# NVTE_USE_CUTLASS_GROUPED_GEMM has an effect. Pair with that env via EXTRA_TRAIN_ENV_JSON.
if [ "${MOE_GROUPED_GEMM:-0}" = "1" ]; then
   MISC_ARGS+=(--moe-grouped-gemm)
fi

# Optional torch profiler (default empty=disabled). slime TrainProfiler advances the
# schedule by rollout_id: start=2/end=3 -> RECORD_AND_SAVE on step 3.
PROFILER_ARGS=()
if [ "${USE_PYTORCH_PROFILER:-0}" = "1" ]; then
   PROFILER_ARGS+=(
      --use-pytorch-profiler
      --profile-step-start ${PROFILE_STEP_START:-2}
      --profile-step-end ${PROFILE_STEP_END:-3}
      --tensorboard-dir "${PROFILE_TB_DIR}"
   )
fi

export no_proxy="127.0.0.1,${MASTER_ADDR}"
mkdir -p "${RAY_TMP_DIR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --port ${GCS_PORT} --disable-usage-stats --temp-dir="${RAY_TMP_DIR}" \
   --dashboard-host=0.0.0.0 --dashboard-port=${DASH_PORT}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"no_proxy\": \"${no_proxy}\",
    \"CUDA_LAUNCH_BLOCKING\": \"0\",
    \"HGGC_LOGGER_ENABLE_MASK\": \"0x00\",
    \"HGGCRT_LOGGER_ENABLE_MASK\": \"0x00\",
    \"NCCL_IB_DISABLE\": \"1\",
    \"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK\": \"512\",
    \"DEEPEP_TIMEOUT_SECONDS\": \"3600\",
    \"QWEN_GDN_IMPL\": \"native\",
    \"SLIME_ENABLE_EXPANDABLE_SEGMENTS\": \"${EXPANDABLE}\",
    \"SLIME_USE_TE_CROSS_ENTROPY\": \"${TE_CE}\",
    \"PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT\": \"${PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT:-1}\"
  }
}"
# Conditional NVTE env passthrough (default empty=not injected).
if [ -n "${NVTE_USE_CUTLASS_GROUPED_GEMM:-}" ]; then
   RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON%%\}*}"
   RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}, \"NVTE_USE_CUTLASS_GROUPED_GEMM\": \"${NVTE_USE_CUTLASS_GROUPED_GEMM}\"
  }
}"
fi
# Profiler rank gating / with_stack passthrough (default empty=not injected).
if [ -n "${SLIME_PROFILER_RANKS:-}" ]; then
   RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON%%\}*}"
   RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}, \"SLIME_PROFILER_RANKS\": \"${SLIME_PROFILER_RANKS}\"
  }
}"
fi
if [ -n "${SLIME_PROFILER_WITH_STACK:-}" ]; then
   RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON%%\}*}"
   RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}, \"SLIME_PROFILER_WITH_STACK\": \"${SLIME_PROFILER_WITH_STACK}\"
  }
}"
fi
# Generic extra env_vars merge (default empty=none). EXTRA_TRAIN_ENV_JSON is a JSON
# object, e.g. '{"PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT":"1"}', merged into env_vars so
# ablation knobs reach the Ray train/rollout actors. Used by the 32k ablation sweep.
if [ -n "${EXTRA_TRAIN_ENV_JSON:-}" ]; then
   RUNTIME_ENV_JSON=$(python3 -c 'import json,sys; b=json.loads(sys.argv[1]); b["env_vars"].update(json.loads(sys.argv[2])); print(json.dumps(b))' "${RUNTIME_ENV_JSON}" "${EXTRA_TRAIN_ENV_JSON}")
fi

set +e
ray job submit --address="http://127.0.0.1:${DASH_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${SLIME_ROOT}/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${PROFILER_ARGS[@]}
rc=$?
set -e
if [ ${rc} -eq 0 ]; then
   echo "TUNE_STATUS: OK (BAND=${BAND} EXP_TAG=${EXP_TAG})"
else
   echo "TUNE_STATUS: FAILED (BAND=${BAND} EXP_TAG=${EXP_TAG} rc=${rc})"
fi
exit ${rc}