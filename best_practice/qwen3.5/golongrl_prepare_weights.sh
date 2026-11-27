#!/bin/bash
# ============================================================================
# Convert HF weights -> native GDN torch_dist layout (run inside the container).
#
# Output layout: six-section rank-major in_proj + three-section conv1d pre-permute
# + 1p gamma minus 1.
# IMPORTANT: --tensor-model-parallel-size MUST match the training TP. The native
#   pre-permute (_permute_sections_rank_major) uses the conversion process's mpu TP
#   size; converting at TP1 and loading at TP2 misaligns GDN weights -> garbled
#   rollout, reward ~= 0.
#   64k/128k train at TP2 (default CONVERT_TP=2); 32k/256k train at TP1, so convert
#   separately with CONVERT_TP=1 SAVE_DIR=..._tp1 GDN_TORCH_DIST=..._tp1.
#
# Usage:
#   bash prepare_weights.sh
#   HF_CKPT=/path/to/HF SAVE_DIR=/path/to/torch_dist CONVERT_TP=1 bash prepare_weights.sh
# ============================================================================
set -ex
# Shared config — env-overridable; defaults are neutral placeholders (set to your
# storage layout). GCS_PORT/DASH_PORT avoid the ray worker range 10002-19999;
# RAY_TMP_DIR kept short (AF_UNIX 107-byte limit). Conversion TP must match training TP.
NAS_ROOT=${NAS_ROOT:-/workspace/golongrl}
MODEL_ROOT=${MODEL_ROOT:-/workspace/models}
HF_CKPT=${HF_CKPT:-${MODEL_ROOT}/Qwen3.5-35B-A3B}
GDN_TORCH_DIST=${GDN_TORCH_DIST:-${MODEL_ROOT}/Qwen3.5-35B-A3B_slime_torch_dist_gdn_tp2}
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

SAVE_DIR=${SAVE_DIR:-${GDN_TORCH_DIST}}
CONVERT_TP=${CONVERT_TP:-2}     # must match training TP (TP2 for 64k/128k)
CONVERT_PP=${CONVERT_PP:-4}
CONVERT_NPROC=${CONVERT_NPROC:-8}

if [ -e "${SAVE_DIR}" ]; then
    echo "ERROR: ${SAVE_DIR} already exists; refusing to overwrite. Move it aside to reconvert." >&2
    exit 1
fi
[ -d "${HF_CKPT}" ] || { echo "ERROR: HF checkpoint dir not found: ${HF_CKPT}" >&2; exit 1; }

source "${SLIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"   # load MODEL_ARGS
export QWEN_GDN_IMPL=native                              # native branch (megatron parser fallback)
export CUDA_DEVICE_MAX_CONNECTIONS=1                     # required by Megatron validate_args when TP>1
cd "${SLIME_ROOT}"

torchrun --nproc_per_node=${CONVERT_NPROC} tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${SAVE_DIR}" \
    --tensor-model-parallel-size ${CONVERT_TP} \
    --pipeline-model-parallel-size ${CONVERT_PP} \
    "${MODEL_ARGS[@]}"

# Verify output.
cat "${SAVE_DIR}/latest_checkpointed_iteration.txt"   # expect: release
ls "${SAVE_DIR}/release/" | head -5
echo "NATIVE_GDN_CONVERT_OK"