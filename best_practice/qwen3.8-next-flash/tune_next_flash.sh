#!/usr/bin/env bash
#
# Qwen3.8-Flash-Next (125B-A6B MoE) Megatron-SWIFT tuning entrypoint, 1 node.
#
# Usage:
#   HF_MODEL=/workspace/models/Qwen3.8-Flash-Next bash tune_next_flash.sh
#
# Results are written under $OUTPUT_DIR; the per-step line the README quotes is
# the `logging.jsonl` entry the CLI emits each step:
#   {'loss': ..., 'train_speed(s/it)': ..., 'memory(GiB)': ...}
#
# The stack is ms-swift -> mcore-bridge -> megatron-core, NOT Megatron-Bridge,
# so this entrypoint does not source tune_common.sh (that one launches
# MegatronMIMO through Megatron-Bridge and has no qwen4_exp model).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG_HINT="ppu-qwen38-next-flash:M890"

HF_MODEL="${HF_MODEL:?set HF_MODEL to the Qwen3.8-Flash-Next checkpoint directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${HERE}/output_next_flash}"
SEQ_LEN="${SEQ_LEN:-8192}"
MBS="${MBS:-1}"
GA="${GA:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
NPROC="${NPROC_PER_NODE:-8}"

# One node, 8 M890P. TP=2 keeps the 512-expert MoE at EP=4 inside the node;
# PP=2 with an even 24/24 split avoids the uneven first stage.
TP="${TP:-2}"
PP="${PP:-2}"
EP="${EP:-4}"
ETP="${ETP:-1}"

# LoRA r8 on the attention projections: the baseline is a fine-tuning run, and
# a full-parameter run of a 125B model needs far more than one node.
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
TARGET_MODULES="${TARGET_MODULES:-in_proj out_proj linear_proj linear_qkv}"

# The N-gram table (51B rows) is host-resident and receives no gradient; without
# this switch it is materialized on the PPU and the run does not fit.
PLE_CPU_OFFLOAD="${PLE_CPU_OFFLOAD:-1}"

# Merging the LoRA adapter back into a full safetensors copy re-writes the whole
# 331 GB checkpoint (about 3 minutes of wall time). Off by default here because
# this entry point is the throughput benchmark; set MERGE_LORA=true to get the
# merged export as well.
MERGE_LORA="${MERGE_LORA:-false}"

DATA="${DATA:-${OUTPUT_DIR}/bench_seq${SEQ_LEN}.jsonl}"
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${DATA}" ]]; then
    python "${HERE}/bench_data.py" \
        --model "${HF_MODEL}" --seq-len "${SEQ_LEN}" \
        --samples $(( TRAIN_ITERS * GA * MBS )) --out "${DATA}" || exit 1
fi

export PLE_CPU_OFFLOAD
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[tune_next_flash] image=${IMAGE_TAG_HINT} seq_len=${SEQ_LEN} mbs=${MBS} ga=${GA} iters=${TRAIN_ITERS} tp=${TP} pp=${PP} ep=${EP}"

megatron sft \
    --model "${HF_MODEL}" \
    --dataset "${DATA}" \
    --truncation_strategy left \
    --num_train_epochs 1 \
    --train_iters "${TRAIN_ITERS}" \
    --tuner_type lora --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" \
    --target_modules ${TARGET_MODULES} \
    --tensor_model_parallel_size "${TP}" \
    --expert_model_parallel_size "${EP}" \
    --expert_tensor_parallel_size "${ETP}" \
    --pipeline_model_parallel_size "${PP}" \
    --decoder_first_pipeline_num_layers $(( 48 / PP )) \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --padding_free true \
    --max_length "${SEQ_LEN}" \
    --micro_batch_size "${MBS}" --global_batch_size $(( MBS * GA )) \
    --lr 1e-4 --logging_steps 1 --save_steps 1000000 \
    --merge_lora "${MERGE_LORA}" \
    --output_dir "${OUTPUT_DIR}"
