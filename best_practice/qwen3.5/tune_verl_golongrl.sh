#!/usr/bin/env bash
# ============================================================================
# Qwen3.5-35B-A3B GoLongRL GRPO tuning entry — verl 0.9.1 on single-node
# 8x M890 144GiB (Megatron actor + colocate sglang rollout, chunkwise GDN CP).
#
# Self-contained: band defaults, preflight, hydra launch and the TUNE_STATUS
# marker all live in this file. Run inside the image built from
# dockers/Dockerfile.Qwen35VerlGoLongRL (patch chain 01-09 applied).
# verl loads HF weights directly via Megatron-Bridge (no offline TP/PP
# conversion); main_ppo starts its own ray head (no ray start / job submit).
#
# Band defaults (validated on 8x M890; see README "Performance Results"):
#   32k  TP1 CP4 EP8 DP2  recompute off          frac 1.0   <- tuned default
#        (chunkwise unlocks TP1; -20.9% update_actor / -13.9% warm step vs
#         TP2CP4 DP1; reserved 113.7GB. TP2CP4 DP1 fallback: TP_OVERRIDE=2)
#   64k  TP2 CP4 EP8 DP1  recompute off          frac 1.0
#   128k TP2 CP4 EP8 DP1  recompute block(20)    frac 1.0
#   256k TP1 CP8 EP8 DP1  recompute full-uniform frac 0.8   util 0.55
# Rollout is always TP8 EP8. expandable_segments on for all bands.
#
# Usage (inside container):
#   BAND=32k  bash tune_verl_golongrl.sh
#   BAND=128k RECOMPUTE_OVERRIDE=off bash tune_verl_golongrl.sh   # memory probe
#   BAND=32k TP_OVERRIDE=2 CP_OVERRIDE=4 EXP_TAG=dp1 bash tune_verl_golongrl.sh
#
# Main knobs (env): BAND / EXP_TAG / TOTAL_STEPS / TP_OVERRIDE / CP_OVERRIDE /
#   EP_OVERRIDE / RECOMPUTE(off|on|block|selective) / RECOMPUTE_OVERRIDE /
#   RECOMPUTE_NUM_LAYERS / OFFLOAD_FRAC / EXPANDABLE(on|off) / LINEAR_CP_MODE /
#   ROLLOUT_MEM_UTIL / TRAIN_BATCH_SIZE / ROLLOUT_N / DATA_DIR / HF_MODEL_PATH /
#   EXTRA_ARGS (extra hydra overrides, last-word wins).
# RECOMPUTE priority: RECOMPUTE_OVERRIDE > caller RECOMPUTE > band default.
#
# Tuning method: paired warm-step comparison against a same-session baseline;
# exclude cold-JIT step1; noise floor ±2%. Shared-host red line: preflight
# refuses unless all 8 cards are idle; never ray stop (--pid=host containers).
# ============================================================================
set -euo pipefail

# ----------------------------- config (inline) -----------------------------
NAS_ROOT=${NAS_ROOT:-/workspace/golongrl}
export HF_MODEL_PATH=${HF_MODEL_PATH:-${NAS_ROOT}/models/Qwen3.5-35B-A3B}
export DATA_DIR=${DATA_DIR:-${NAS_ROOT}/datas}
export VERL_ROOT=${VERL_ROOT:-/opt/verl}        # image editable tree, patches applied
LOG_DIR=${LOG_DIR:-${NAS_ROOT}/logs}
BAND=${BAND:-32k}
EXP_TAG=${EXP_TAG:-run}
export N_GPUS=${N_GPUS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOTAL_STEPS=${TOTAL_STEPS:-5}
export SAVE_FREQ=${SAVE_FREQ:--1}
export TEST_FREQ=${TEST_FREQ:--1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export ROLLOUT_N=${ROLLOUT_N:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export LR=${LR:-1e-6}
# Capture caller RECOMPUTE before band defaults (3-level priority below).
RECOMPUTE_USER=${RECOMPUTE:-}

mkdir -p "${LOG_DIR}"

# ----------------------------- band defaults -------------------------------
case "${BAND}" in
  32k)  export MAX_PROMPT_LENGTH=28600;  export MAX_RESPONSE_LENGTH=4096
        # Tuned: chunkwise unlocks TP1 -> TP1CP4 DP2 beats TP2CP4 DP1 by
        # -20.9% update_actor / -13.9% warm step; reserved 113.7GB (TP1
        # disables SP, +44GB activations) still fits with ~30GB headroom.
        TP_D=1; CP_D=4; EP_D=8; FRAC_D=1.0; REC_D=off ;;
  64k)  export MAX_PROMPT_LENGTH=61380;  export MAX_RESPONSE_LENGTH=4096
        # Per-card 16k with CP4; TP2+SP splits projection and LN activations.
        # TP1CP8 matched-step shows no speed gain and less headroom (slime);
        # TP1CP4 DP2 is a structural OOM (~140GiB allocated, slime probe).
        TP_D=2; CP_D=4; EP_D=8; FRAC_D=1.0; REC_D=off ;;
  128k) export MAX_PROMPT_LENGTH=122800; export MAX_RESPONSE_LENGTH=8192
        # Per-card 32k: recompute off extrapolates past 144GiB (structural);
        # block(20) recomputes only the first 20 layers.
        TP_D=2; CP_D=4; EP_D=8; FRAC_D=1.0; REC_D=block
        export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-20} ;;
  256k) export MAX_PROMPT_LENGTH=253900; export MAX_RESPONSE_LENGTH=8192
        # Per-card 32k at the CP8 ceiling; full-uniform recompute + frac 0.8
        # (host-RAM guard) + engine util 0.55. Reserved ~135GB: no
        # memory-adding knob without a headroom audit.
        TP_D=1; CP_D=8; EP_D=8; FRAC_D=0.8; REC_D=on
        export ROLLOUT_MEM_UTIL=${ROLLOUT_MEM_UTIL:-0.55} ;;
  *) echo "ERROR: BAND=${BAND} (expect 32k/64k/128k/256k)" >&2; exit 1 ;;
esac

# ----------------------------- data ----------------------------------------
export TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/golongrl-verl-bands/band${BAND}/train.parquet}
export VAL_FILE=${VAL_FILE:-${DATA_DIR}/golongrl-verl-bands/band${BAND}/val.parquet}
[[ -f "${TRAIN_FILE}" ]] || { echo "ERROR: missing ${TRAIN_FILE} (run golongrl_prepare_data_verl.sh)" >&2; exit 1; }

# ----------------------------- parallelism ---------------------------------
export ACTOR_TP=${TP_OVERRIDE:-${ACTOR_TP:-${TP_D}}}
export ACTOR_CP=${CP_OVERRIDE:-${ACTOR_CP:-${CP_D}}}
export ACTOR_EP=${EP_OVERRIDE:-${ACTOR_EP:-${EP_D}}}
export ACTOR_PP=1
export ACTOR_ETP=1
export REF_TP=${REF_TP:-${ACTOR_TP}}
export REF_CP=${REF_CP:-${ACTOR_CP}}
export REF_EP=${REF_EP:-${ACTOR_EP}}
export REF_ETP=1
# PP1 fixed; TP*CP must divide N_GPUS (DP = N_GPUS/(TP*CP)); EP <= N_GPUS.
if (( N_GPUS % (ACTOR_TP * ACTOR_CP) != 0 )); then
    echo "ERROR: TP${ACTOR_TP}*CP${ACTOR_CP} does not divide N_GPUS=${N_GPUS}" >&2; exit 1
fi
if (( ACTOR_EP > N_GPUS )); then
    echo "ERROR: ACTOR_EP=${ACTOR_EP} > N_GPUS=${N_GPUS}" >&2; exit 1
fi
OFFLOAD_FRAC=${OFFLOAD_FRAC:-${FRAC_D}}
export RECOMPUTE=${RECOMPUTE_OVERRIDE:-${RECOMPUTE_USER:-${REC_D}}}
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-20}
export RECOMPUTE_MODULES=${RECOMPUTE_MODULES:-layernorm}
# GDN CP mode: chunkwise default (m890); headwise falls back to native 0.18.0.
# Outer data plane stays zigzag in BOTH modes (bridge VLM packs with
# align=tp*cp*2); chunkwise converts zigzag->contiguous inside GDN layers.
export LINEAR_CP_MODE=${LINEAR_CP_MODE:-chunkwise}
export CP_PARTITION_MODE=${CP_PARTITION_MODE:-zigzag}
# CP>1 requires THD/packed sequences (qwen35 provider forces
# calculate_per_token_loss=True; BSHD would explode grad_norm via MoE aux loss).
export USE_REMOVE_PADDING=True

# ----------------------------- preflight -----------------------------------
PPU_SMI_OUT=$(ppu-smi) || { echo "ERROR: ppu-smi failed; cannot verify GPU state" >&2; exit 1; }
N_CARDS=$(printf '%s\n' "${PPU_SMI_OUT}" | grep -oE '[0-9]+MiB / [0-9]+MiB' | wc -l)
if [ "${N_CARDS}" -ne "${N_GPUS}" ]; then
    echo "ERROR: expected ${N_GPUS} cards, ppu-smi reports ${N_CARDS}; refuse to start" >&2
    exit 1
fi
printf '%s\n' "${PPU_SMI_OUT}" | grep -oE '[0-9]+MiB / [0-9]+MiB' \
    | awk -F'MiB' '{if ($1+0 > 1000) {print "ERROR: busy card mem: " $0; exit 1}}' \
    || { echo "ERROR: cards not idle; refuse to start" >&2; exit 1; }

# ----------------------------- runtime env ---------------------------------
export ROLLOUT_TP=8
export ROLLOUT_EP=8
export ROLLOUT_MEM_UTIL=${ROLLOUT_MEM_UTIL:-0.7}
export ROLLOUT_MAX_SEQS=${ROLLOUT_MAX_SEQS:-512}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export ALL_OFFLOAD=True
export PROJECT_NAME=${PROJECT_NAME:-verl_golongrl_qwen3_5_35b}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-golongrl_${BAND}_${EXP_TAG}_tp${ACTOR_TP}cp${ACTOR_CP}ep${ACTOR_EP}_${LINEAR_CP_MODE}}
export LOGGER=${LOGGER:-'["console"]'}
export VERL_SET_CPU_AFFINITY=${VERL_SET_CPU_AFFINITY:-0}
# expandable_segments on by default (validated with patch 04 TMS coexistence;
# EXPANDABLE=off switches both train and engine processes back for ablations).
if [[ "${EXPANDABLE:-on}" == "on" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
export RAY_memory_usage_threshold=0.99
BUCKET_MB=${BUCKET_MB:-256}
ATTN_BACKEND=${ATTN_BACKEND:-fa3}
# CPU-only TaskRunnerV1 driver MKL VML GNU-thread NULL-callback SIGSEGV guard:
# multi-thread OMP + ATen thread rebuild during advantage compute segfaults
# the driver. MKL_THREADING_LAYER=sequential cannot fix it (libtorch_cpu.so
# DT_NEEDED hard-links libmkl_gnu_thread); OMP_NUM_THREADS=1 is the only
# reliable fix (also baked as an image ENV). Cost: single-thread CPU optimizer.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-sequential}   # redundant fallback only
# PPU: decode cuda graph + memory saver segfaults (same finding as miles).
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=false
export SGLANG_SAIL_FLA_CUDA=0
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export VERL_USE_UV=0
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOG_FILE=${LOG_FILE:-${LOG_DIR}/tune_${BAND}_${EXP_TAG}_tp${ACTOR_TP}cp${ACTOR_CP}ep${ACTOR_EP}_${LINEAR_CP_MODE}.log}
export LOG_FILE

# ----------------------------- hydra args ----------------------------------
DATA=(
    data.train_files=${TRAIN_FILE}
    data.val_files=${VAL_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.truncation='error'
    data.filter_overlong_prompts=True
)
MODEL=(
    actor_rollout_ref.model.path=${HF_MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}
)
ACTOR=(
    actor_rollout_ref.actor.optim.lr=${LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    # Qwen3.5 MoE needs the Megatron-Bridge qwen35_vl_provider; the legacy
    # mbridge backend raises TypeError on Qwen3_5VLTransformerConfig.
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=${USE_REMOVE_PADDING}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    # m890 chunkwise CP wiring (patch 07 fields; ref inherits via oc.select).
    +actor_rollout_ref.actor.megatron.override_transformer_config.linear_cp_mode=${LINEAR_CP_MODE}
    +actor_rollout_ref.actor.megatron.override_transformer_config.cp_partition_mode=${CP_PARTITION_MODE}
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)
ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_MEM_UTIL}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_SEQS}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.calculate_log_probs=True
)
REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${REF_ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.ref.megatron.use_remove_padding=${USE_REMOVE_PADDING}
)
ALGORITHM=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.use_kl_in_reward=False
)
# Recompute keys injected only when RECOMPUTE!=off (off = pass nothing;
# megatron default None means disabled). block = checkpoint only the first
# RECOMPUTE_NUM_LAYERS layers; uniform always recomputes all layers.
case "${RECOMPUTE}" in
    on)
        REC_ARGS=(
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
        ) ;;
    block)
        REC_ARGS=(
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=${RECOMPUTE_NUM_LAYERS}
        ) ;;
    selective)
        REC_ARGS=(
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=selective
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_modules=[${RECOMPUTE_MODULES}]
        ) ;;
    off)
        REC_ARGS=() ;;
    *)
        echo "ERROR: RECOMPUTE=${RECOMPUTE} (expect off|on|block|selective)" >&2; exit 1 ;;
esac
TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=${LOGGER}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=False
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_STEPS}
)
EXTRA=(
    model_engine=megatron
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${BUCKET_MB}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=${ATTN_BACKEND}
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OFFLOAD_FRAC}
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_fusion=True
)

# ----------------------------- launch --------------------------------------
echo "[$(date '+%F %T')] tune_verl_golongrl: BAND=${BAND} EXP_TAG=${EXP_TAG} tp${ACTOR_TP}cp${ACTOR_CP}ep${ACTOR_EP} ${LINEAR_CP_MODE} RECOMPUTE=${RECOMPUTE} frac=${OFFLOAD_FRAC} LOG=${LOG_FILE}"
cd "${VERL_ROOT}"
set +e
python3 -u -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    ${REC_ARGS[@]+"${REC_ARGS[@]}"} \
    ${EXTRA_ARGS:-} \
    >> "${LOG_FILE}" 2>&1
rc=$?
set -e
if [ ${rc} -eq 0 ]; then
    echo "TUNE_STATUS: OK (BAND=${BAND} EXP_TAG=${EXP_TAG})"
else
    echo "TUNE_STATUS: FAILED (BAND=${BAND} EXP_TAG=${EXP_TAG} rc=${rc})"
fi
exit ${rc}