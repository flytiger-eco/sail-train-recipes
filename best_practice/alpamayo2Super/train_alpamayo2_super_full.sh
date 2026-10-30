#!/usr/bin/env bash
#
# Alpamayo 2 Super (34B VLA) - Stage-1 VLM supervised fine-tuning, best-practice launcher.
#
# Trains the release-native nvidia/Alpamayo2-Super checkpoint on the
# nvidia/PhysicalAI-Autonomous-Vehicles (PAI) dataset with Hugging Face Trainer + DeepSpeed
# ZeRO-3, measured on Zhenwu M890 (144 GiB/card). Supports single-node (8 GPU) and
# dual-node (16 GPU) with one code path.
#
# Defaults reproduce the measured best configuration:
#   8 GPU  (1 node) : ZeRO-3 + selective recompute K48 + 4 comm channels -> ~6.58 s/step
#   16 GPU (2 nodes): ZeRO-3 + selective recompute K0  + 4 comm channels -> ~5.72 s/step
# Every optimization switch (FA3-varlen vision, TE fused RMSNorm, Triton RoPE, Liger SwiGLU,
# selective checkpoint) is ON by default. They are runtime monkey-patches registered through
# hydra callbacks, so the upstream sources are never edited in-tree (see dockers/patches/).
#
# Usage:
#   Single-node (8 GPU):
#     MODEL_PATH=/path/to/Alpamayo2-Super \
#     DATA_PATH=/path/to/PhysicalAI-Autonomous-Vehicles \
#     bash train_alpamayo2_super_full.sh
#
#   Dual-node (16 GPU) - node 0 (master); OUTPUT_DIR must be a path shared by both nodes:
#     NNODES=2 NODE_RANK=0 MASTER_ADDR=<MASTER_NODE_IP> MASTER_PORT=29517 \
#     NCCL_SOCKET_IFNAME=<bootstrap-nic> NCCL_IB_HCA=<rdma-hca-list> \
#     OUTPUT_DIR=/shared/outputs/alpamayo2_stage1 \
#     MODEL_PATH=... DATA_PATH=... \
#     bash train_alpamayo2_super_full.sh
#
#   Dual-node (16 GPU) - node 1 (same OUTPUT_DIR / MASTER_ADDR, NODE_RANK=1):
#     NNODES=2 NODE_RANK=1 MASTER_ADDR=<MASTER_NODE_IP> MASTER_PORT=29517 \
#     NCCL_SOCKET_IFNAME=<bootstrap-nic> NCCL_IB_HCA=<rdma-hca-list> \
#     OUTPUT_DIR=/shared/outputs/alpamayo2_stage1 \
#     MODEL_PATH=... DATA_PATH=... \
#     bash train_alpamayo2_super_full.sh
#
#   Preview the fully-resolved launch command without running it:
#     DRY_RUN=1 MODEL_PATH=... DATA_PATH=... bash train_alpamayo2_super_full.sh

set -euo pipefail

die() { echo "[error] $*" >&2; exit 1; }

# ─── Interpreter & installed recipe package ───────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "PYTHON_BIN=$PYTHON_BIN not found"
"$PYTHON_BIN" -c "import alpamayo2_sft" >/dev/null 2>&1 \
  || die "alpamayo2_sft is not importable; build dockers/Dockerfile.Alpamayo2Super (or install the recipe package) first"

# Absolute path of the installed alpamayo2_sft package (used to resolve the DeepSpeed config).
PKG_DIR="$("$PYTHON_BIN" -c 'import os, alpamayo2_sft; print(os.path.dirname(alpamayo2_sft.__file__))')"

# ─── Cache / temp directories ─────────────────────────────────────────────────
V="${CACHE_ROOT:-${HOME:-/root}/alpamayo2}"
export HF_HOME="${HF_HOME:-${V}/cache/hf}"
export TMPDIR="${TMPDIR:-${V}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${V}/cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${V}/cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${V}/cache/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${V}/cache/cuda}"
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${V}/cache/torch_kernels}"
mkdir -p "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
         "$PYTORCH_KERNEL_CACHE_PATH" "$TORCH_EXTENSIONS_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# ─── Model & Data (required) ──────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-}"
DATA_PATH="${DATA_PATH:-}"
[[ -n "$MODEL_PATH" ]] || die "MODEL_PATH must be set (local snapshot of nvidia/Alpamayo2-Super)"
[[ -n "$DATA_PATH"  ]] || die "DATA_PATH must be set (local nvidia/PhysicalAI-Autonomous-Vehicles dir)"
[[ -f "$MODEL_PATH/config.json" ]] || die "MODEL_PATH=$MODEL_PATH has no config.json (expected a local HF snapshot)"
[[ -d "$DATA_PATH" ]] || die "DATA_PATH=$DATA_PATH is not a directory"

# ─── Topology (one code path for single- and dual-node) ───────────────────────
NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29517}"

# ─── Data / schedule hyperparameters (match the measured best run) ─────────────
CONFIG_NAME="${CONFIG_NAME:-sft_stage1}"
TRAIN_CHUNKS="${TRAIN_CHUNKS:-0-99}"    # PAI chunks 0-98 for training (half-open range)
VAL_CHUNKS="${VAL_CHUNKS:-99-100}"      # held-out chunk 99 for validation
NUM_FRAMES="${NUM_FRAMES:-4}"           # frames per camera (the 7 cameras are baked into the config)
MAX_LENGTH="${MAX_LENGTH:-6144}"        # hard token cap per sample
MAX_STEPS="${MAX_STEPS:-1000}"          # LR-schedule horizon
WARMUP_STEPS="${WARMUP_STEPS:-50}"
STOP_AFTER="${STOP_AFTER:-30}"          # full_monitor early-stop step (0 = run to MAX_STEPS)
MBS="${MBS:-1}"                         # per-device train batch size
GRAD_ACCUM="${GRAD_ACCUM:-1}"           # gradient accumulation steps
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
REPORT_TO="${REPORT_TO:-none}"
RESUME_FROM="${RESUME_FROM:-null}"

# ─── DeepSpeed ZeRO-3 config (resolved from the installed package) ─────────────
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero3.json}"
DS_CFG="$PKG_DIR/configs/deepspeed/$DEEPSPEED_CONFIG"
[[ -f "$DS_CFG" ]] || die "DeepSpeed config not found: $DS_CFG"

# ─── Optimization switches (all ON by default = measured best config) ─────────
ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
case "$ATTN_IMPL" in
  sdpa|eager|flash_attention_2|flash_attention_3) ;;
  *) die "unsupported ATTN_IMPL: $ATTN_IMPL" ;;
esac

VISION_FA3_VARLEN="${VISION_FA3_VARLEN:-1}"            # vision tower concurrent varlen under FA3
TE_RMSNORM="${TE_RMSNORM:-1}"                          # TransformerEngine fused RMSNorm (all norm sites)
TRITON_ROPE="${TRITON_ROPE:-1}"                        # 0/1/text/vision fused rotary embedding
TRITON_ROPE_BACKEND="${TRITON_ROPE_BACKEND:-triton}"   # triton/flash/liger
LIGER_SWIGLU="${LIGER_SWIGLU:-1}"                      # Liger fused SwiGLU on the 64 text MLPs
PAD_TO_FIXED_LENGTH="${PAD_TO_FIXED_LENGTH:-false}"    # false = dynamic pad to batch-longest
GRAD_CKPT="${GRAD_CKPT:-1}"                            # trainer-level checkpointing (vision tower)
SUPERVISED_LM_HEAD="${SUPERVISED_LM_HEAD:-0}"          # optional compact stage-1 logits projection

for kv in "VISION_FA3_VARLEN=$VISION_FA3_VARLEN" "TE_RMSNORM=$TE_RMSNORM" \
          "LIGER_SWIGLU=$LIGER_SWIGLU" "GRAD_CKPT=$GRAD_CKPT" "SUPERVISED_LM_HEAD=$SUPERVISED_LM_HEAD"; do
  [[ "${kv#*=}" =~ ^[01]$ ]] || die "${kv%%=*} must be 0 or 1, got: ${kv#*=}"
done
case "$TRITON_ROPE" in
  0|1|text|vision) ;; *) die "TRITON_ROPE must be 0/1/text/vision, got: $TRITON_ROPE" ;;
esac
case "$TRITON_ROPE_BACKEND" in
  triton|flash|liger) ;; *) die "TRITON_ROPE_BACKEND must be triton/flash/liger, got: $TRITON_ROPE_BACKEND" ;;
esac
case "$PAD_TO_FIXED_LENGTH" in
  true|false) ;; *) die "PAD_TO_FIXED_LENGTH must be true/false, got: $PAD_TO_FIXED_LENGTH" ;;
esac

# Selective recompute of the 64-layer text backbone: keep K layers recomputed, hold 64-K as
# activations. Memory headroom differs by topology, so the best K does too:
#   1 node / 8 GPU   -> K48 (ZeRO-3 shards over 8 ranks; peak_alloc ~94.7 GiB, ~6.58 s/step)
#   2 nodes / 16 GPU -> K0  (16-way sharding frees room to skip text recompute entirely;
#                            peak_reserved ~125.6 GiB, ~5.72 s/step)
# Both fit the 144 GiB card. Set CKPT_LAYERS explicitly to pin a value.
if [[ -z "${CKPT_LAYERS:-}" ]]; then
  if (( NNODES >= 2 )); then CKPT_LAYERS=0; else CKPT_LAYERS=48; fi
fi
[[ "$CKPT_LAYERS" =~ ^(0|[1-9][0-9]*)$ ]] || die "CKPT_LAYERS must be a non-negative integer, got: $CKPT_LAYERS"
(( CKPT_LAYERS <= 64 )) || die "CKPT_LAYERS must be <= 64 (text backbone has 64 layers), got: $CKPT_LAYERS"
[[ "$GRAD_CKPT" == "1" ]] || die "selective recompute (CKPT_LAYERS) requires GRAD_CKPT=1"

# ─── Communication / memory tuning (match the measured best run) ───────────────
# Capping PCCL/NCCL channels at 4 reduces ZeRO-3 param-AllGather vs GEMM contention on the
# compute units; 4 was the saturation point for both the 8-GPU and 16-GPU ZeRO-3 best configs.
export PCCL_MAX_NCHANNELS="${PCCL_MAX_NCHANNELS:-4}"
export PCCL_MAX_P2P_NCHANNELS="${PCCL_MAX_P2P_NCHANNELS:-4}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
export NCCL_MAX_P2P_NCHANNELS="${NCCL_MAX_P2P_NCHANNELS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_ALLOC_CONF  # newer alias can conflict with PYTORCH_CUDA_ALLOC_CONF; benchmark ran without it
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"
# Multi-node fabric (hardware/topology specific - set per your cluster; empty = NCCL auto-detect):
#   NCCL_SOCKET_IFNAME : bootstrap/OOB NIC (rendezvous + PCCL bootstrap)
#   NCCL_IB_HCA        : RDMA HCAs for the data plane, e.g. mlx5_bond_0,mlx5_bond_1,...
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
if [[ -n "${NCCL_IB_HCA:-}" ]]; then export NCCL_IB_HCA; fi

# ─── Output directories ───────────────────────────────────────────────────────
# ZeRO-3 reassembles a full checkpoint only if every rank writes the SAME trainer output_dir,
# so paths.output_dir is shared across nodes. Per-node logs / hydra / torchrun stay separate so
# two nodes never clobber each other.
OUTPUT_DIR="${OUTPUT_DIR:-${V}/outputs/alpamayo2_super_stage1}"
TRAINER_OUT="$OUTPUT_DIR"
if (( NNODES >= 2 )); then
  RUN_DIR="$OUTPUT_DIR/node${NODE_RANK}"
else
  RUN_DIR="$OUTPUT_DIR"
fi
LOG_PATH="${LOG_PATH:-$RUN_DIR/train.log}"
mkdir -p "$RUN_DIR" "$RUN_DIR/torchrun"

# ─── Checkpointing (off by default: benchmark-friendly, avoids ~445 GB writes) ──
SAVE_CKPT="${SAVE_CKPT:-0}"
SAVE_STEPS="${SAVE_STEPS:-500}"
case "$SAVE_CKPT" in 0|1) ;; *) die "SAVE_CKPT must be 0 or 1, got: $SAVE_CKPT" ;; esac
if [[ "$SAVE_CKPT" == "1" ]]; then
  SAVE_LINE=( ++trainer.save_strategy=steps "trainer.save_steps=$SAVE_STEPS" trainer.save_total_limit=2
              ++trainer.save_only_model=false ++trainer.eval_strategy=no )
  SAVE_AT_STOP=true
else
  SAVE_LINE=( ++trainer.save_strategy=no ++trainer.save_only_model=false ++trainer.eval_strategy=no )
  SAVE_AT_STOP=false
fi

# ─── Optional torch.profiler capture (off by default) ─────────────────────────
PROF_ENABLE="${PROF_ENABLE:-0}"
PROF_START="${PROF_START:-20}"
PROF_END="${PROF_END:-21}"
PROF_RANKS="${PROF_RANKS:-[0]}"
PROF_WITH_STACK="${PROF_WITH_STACK:-0}"
case "$PROF_ENABLE" in 0|1) ;; *) die "PROF_ENABLE must be 0 or 1, got: $PROF_ENABLE" ;; esac
case "$PROF_WITH_STACK" in 0|1) ;; *) die "PROF_WITH_STACK must be 0 or 1, got: $PROF_WITH_STACK" ;; esac

# ─── Resolve MASTER_ADDR (hostname -> IPv4) ───────────────────────────────────
if [[ "$MASTER_ADDR" == *,* ]]; then MASTER_ADDR="${MASTER_ADDR%%,*}"; fi
resolve_master_addr_ipv4() {
  "$PYTHON_BIN" - "$1" <<'PYEOF'
import re, socket, sys
addr = sys.argv[1]
m = re.match(r"^(\d+)-(\d+)-(\d+)-(\d+)(?:\.|$)", addr)
if m:
    print(".".join(m.groups())); raise SystemExit(0)
try:
    infos = socket.getaddrinfo(addr, None, socket.AF_INET, socket.SOCK_STREAM)
except socket.gaierror:
    print(addr); raise SystemExit(0)
for info in infos:
    print(info[4][0]); break
else:
    print(addr)
PYEOF
}
MASTER_ADDR="$(resolve_master_addr_ipv4 "$MASTER_ADDR")"
export MASTER_ADDR MASTER_PORT

# ─── Build hydra overrides ────────────────────────────────────────────────────
GRAD_CKPT_BOOL=true
[[ "$GRAD_CKPT" == "1" ]] || GRAD_CKPT_BOOL=false

OVERRIDES=(
  "model.pretrained_model_name_or_path=$MODEL_PATH"
  "data.train_dataset.local_dir=$DATA_PATH"
  "data.val_dataset.local_dir=$DATA_PATH"
  "data.train_dataset.chunk_ids=$TRAIN_CHUNKS"
  "data.val_dataset.chunk_ids=$VAL_CHUNKS"
  "num_frames=$NUM_FRAMES"
  "data.collator.max_length=$MAX_LENGTH"
  "data.collator.pad_to_fixed_length=$PAD_TO_FIXED_LENGTH"
  "trainer.max_steps=$MAX_STEPS"
  "trainer.warmup_steps=$WARMUP_STEPS"
  "trainer.per_device_train_batch_size=$MBS"
  "trainer.gradient_accumulation_steps=$GRAD_ACCUM"
  "trainer.deepspeed=$DS_CFG"
  "trainer.dataloader_num_workers=$DATALOADER_NUM_WORKERS"
  "++trainer.dataloader_prefetch_factor=$PREFETCH_FACTOR"
  "trainer.logging_steps=$LOGGING_STEPS"
  "++trainer.logging_nan_inf_filter=false"
  "++trainer.gradient_checkpointing=$GRAD_CKPT_BOOL"
  "trainer.report_to=$REPORT_TO"
  "trainer.resume_from_checkpoint=$RESUME_FROM"
  "${SAVE_LINE[@]}"
  "+model.attn_implementation=$ATTN_IMPL"
  "+callbacks.full_monitor._target_=alpamayo2_sft.stage1_full_monitor.FullRunMonitor"
  "+callbacks.full_monitor.stop_after=$STOP_AFTER"
  "+callbacks.full_monitor.save_at_stop=$SAVE_AT_STOP"
  "+callbacks.selective_ckpt._target_=alpamayo2_sft.stage1_selective_ckpt.SelectiveCheckpointCallback"
  "+callbacks.selective_ckpt.num_ckpt_layers=$CKPT_LAYERS"
  "paths.output_dir=$TRAINER_OUT"
  "hydra.run.dir=$RUN_DIR/hydra"
)

[[ "$VISION_FA3_VARLEN"  == 1 ]] && OVERRIDES+=( "+callbacks.vision_fa3_varlen._target_=alpamayo2_sft.stage1_vision_fa3_varlen.VisionFA3VarlenCallback" )
[[ "$TE_RMSNORM"         == 1 ]] && OVERRIDES+=( "+callbacks.te_rmsnorm._target_=alpamayo2_sft.stage1_te_rmsnorm.TERMSNormCallback" )
[[ "$LIGER_SWIGLU"       == 1 ]] && OVERRIDES+=( "+callbacks.liger_swiglu._target_=alpamayo2_sft.liger_patch.LigerSwiGLUCallback" )
[[ "$SUPERVISED_LM_HEAD" == 1 ]] && OVERRIDES+=( "+model.stage1_supervised_logits=true" )

if [[ "$TRITON_ROPE" != 0 ]]; then
  ROPE_TEXT=true; ROPE_VISION=true
  [[ "$TRITON_ROPE" == vision ]] && ROPE_TEXT=false
  [[ "$TRITON_ROPE" == text  ]] && ROPE_VISION=false
  OVERRIDES+=(
    "+callbacks.triton_rope._target_=alpamayo2_sft.stage1_triton_rope.TritonRoPECallback"
    "+callbacks.triton_rope.text=$ROPE_TEXT"
    "+callbacks.triton_rope.vision=$ROPE_VISION"
    "+callbacks.triton_rope.backend=$TRITON_ROPE_BACKEND"
  )
fi

if [[ "$PROF_ENABLE" == 1 ]]; then
  OVERRIDES+=(
    "+callbacks.profiler._target_=alpamayo2_sft.stage1_profiler.TorchProfilerCallback"
    "+callbacks.profiler.start_step=$PROF_START"
    "+callbacks.profiler.end_step=$PROF_END"
    "+callbacks.profiler.ranks=$PROF_RANKS"
  )
  [[ "$PROF_WITH_STACK" == 1 ]] && OVERRIDES+=( "+callbacks.profiler.with_stack=true" )
fi

# ─── Print config ─────────────────────────────────────────────────────────────
GLOBAL_BATCH=$(( MBS * GRAD_ACCUM * NUM_GPUS * NNODES ))
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Alpamayo 2 Super - Stage-1 VLM SFT (HF Trainer + DeepSpeed ZeRO-3)"
echo "════════════════════════════════════════════════════════"
echo "  MODEL_PATH         = $MODEL_PATH"
echo "  DATA_PATH          = $DATA_PATH"
echo "  CONFIG_NAME        = $CONFIG_NAME"
echo "  NUM_GPUS           = $NUM_GPUS"
echo "  NNODES / NODE_RANK = $NNODES / $NODE_RANK"
echo "  MASTER_ADDR:PORT   = $MASTER_ADDR:$MASTER_PORT"
echo "  GLOBAL_BATCH       = $GLOBAL_BATCH  (MBS=$MBS x GA=$GRAD_ACCUM x $NUM_GPUS x $NNODES)"
echo "  DEEPSPEED          = $DEEPSPEED_CONFIG"
echo "  MAX_STEPS / STOP   = $MAX_STEPS / stop_after=$STOP_AFTER"
echo "  WARMUP_STEPS       = $WARMUP_STEPS"
echo "  NUM_FRAMES/MAXLEN  = $NUM_FRAMES / $MAX_LENGTH  (pad_to_fixed=$PAD_TO_FIXED_LENGTH)"
echo "  TRAIN/VAL CHUNKS   = $TRAIN_CHUNKS / $VAL_CHUNKS"
echo "  ATTN_IMPL          = $ATTN_IMPL"
echo "  selective recompute= CKPT_LAYERS=$CKPT_LAYERS (of 64 text layers), GRAD_CKPT=$GRAD_CKPT"
echo "  fusion switches    = vision_fa3_varlen=$VISION_FA3_VARLEN te_rmsnorm=$TE_RMSNORM triton_rope=$TRITON_ROPE($TRITON_ROPE_BACKEND) liger_swiglu=$LIGER_SWIGLU"
echo "  comm channels      = PCCL/NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS"
echo "  alloc conf         = $PYTORCH_CUDA_ALLOC_CONF"
echo "  SAVE_CKPT          = $SAVE_CKPT (save_steps=$SAVE_STEPS)"
echo "  PROF_ENABLE        = $PROF_ENABLE"
echo "  OUTPUT_DIR         = $OUTPUT_DIR"
echo "  RUN_DIR (this node)= $RUN_DIR"
echo "  LOG_PATH           = $LOG_PATH"
echo "════════════════════════════════════════════════════════"
echo ""

# ─── Build torchrun command ───────────────────────────────────────────────────
CMD=(
  "$PYTHON_BIN" -m torch.distributed.run
  --nnodes="$NNODES"
  --nproc_per_node="$NUM_GPUS"
  --node_rank="$NODE_RANK"
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
  --log-dir "$RUN_DIR/torchrun" --redirects 3 --tee 3
  -m alpamayo2_sft.train_hf
  --config-path pkg://alpamayo2_sft/configs
  --config-name "$CONFIG_NAME"
  "${OVERRIDES[@]}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run] %q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

# ─── Execute ──────────────────────────────────────────────────────────────────
SECONDS=0
set +e
"${CMD[@]}" 2>&1 | tee "$LOG_PATH"
RC=${PIPESTATUS[0]}
set -e
WALL_TIME=$SECONDS
echo "[exit] torchrun rc=$RC wall=${WALL_TIME}s log=$LOG_PATH"

# ─── Performance summary (steady state from the full_monitor metrics) ─────────
"$PYTHON_BIN" - "$TRAINER_OUT" "$GLOBAL_BATCH" "$WALL_TIME" "$RC" <<'PY' || true
import json, statistics, sys
from pathlib import Path

trainer_out = Path(sys.argv[1]); global_batch = int(sys.argv[2])
wall = int(sys.argv[3]); rc = sys.argv[4]
print(f"[perf] wall_time_s={wall} exit={rc} global_batch={global_batch}")
m = trainer_out / "metrics_rank0.jsonl"
if not m.exists():
    print(f"[perf] no metrics at {m} (full_monitor writes metrics_rank*.jsonl to paths.output_dir)")
    raise SystemExit(0)

steps, peaks, losses = [], [], []
for line in m.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        ev = json.loads(line)
    except Exception:
        continue
    if ev.get("event") == "step":
        gs = ev.get("global_step", 0)
        if "step_window_seconds" in ev:
            steps.append((gs, ev["step_window_seconds"]))
        if "peak_reserved_gib" in ev:
            peaks.append((gs, ev["peak_reserved_gib"]))
    elif ev.get("event") == "log":
        lm = ev.get("metrics") or {}
        if lm.get("loss") is not None:
            losses.append(lm["loss"])

steady = [t for gs, t in steps if gs >= 5]
if steady:
    med = statistics.median(steady)
    print(f"[perf] steady_step_median_s={med:.4f} (n={len(steady)}, global_step>=5)")
    print(f"[perf] throughput_samples_per_s={global_batch / med:.3f}")
p2 = [p for gs, p in peaks if gs >= 2]
if p2:
    print(f"[perf] peak_reserved_gib={max(p2):.3f}")
if losses:
    print(f"[perf] last_loss={losses[-1]:.4f}")
PY

exit "$RC"
