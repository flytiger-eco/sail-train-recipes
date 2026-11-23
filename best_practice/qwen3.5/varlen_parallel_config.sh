#!/usr/bin/env bash
# 四个dense tune脚本共用的配置，保留原入口与逻辑batch语义。
GLOBAL_BATCH_PACKING="${GLOBAL_BATCH_PACKING:-0}"
DYNAMIC_CP="${DYNAMIC_CP:-0}"
MIN_DYNAMIC_CP="${MIN_DYNAMIC_CP:-1}"
MAX_SEQLEN_PER_RANK="${MAX_SEQLEN_PER_RANK:-$((SEQ_LEN / CP))}"
FSDP="${FSDP:-1}"
FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-optim_grads}"
GRAD_OVERLAP="${GRAD_OVERLAP:-1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
CONFIG_CHECK="${CONFIG_CHECK:-0}"
PRINT_COMMAND_ONLY="${PRINT_COMMAND_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

for flag in GLOBAL_BATCH_PACKING DYNAMIC_CP FSDP GRAD_OVERLAP CONFIG_CHECK PRINT_COMMAND_ONLY; do
    if [[ "${!flag}" != 0 && "${!flag}" != 1 ]]; then
        echo "$flag必须为0或1" >&2
        exit 2
    fi
done
if [[ "$DYNAMIC_CP" == 1 ]]; then
    GLOBAL_BATCH_PACKING=1
fi
if [[ "$GLOBAL_BATCH_PACKING" == 1 && ( "$TP" != 1 || "$PP" != 1 ) ]]; then
    echo "图文global packing要求显式TP=1 PP=1，不自动修改模型拓扑。" >&2
    exit 2
fi
if [[ "$GLOBAL_BATCH_PACKING" == 1 && "$FSDP" == 1 ]]; then
    if [[ "$FSDP_SHARDING_STRATEGY" != optim_grads || "$GRAD_OVERLAP" != 1 ]]; then
        echo "global packing与FSDP当前要求ZeRO2 optim_grads及GRAD_OVERLAP=1。" >&2
        exit 2
    fi
fi
case "$FSDP_SHARDING_STRATEGY" in
    no_shard|optim|optim_grads|optim_grads_params) ;;
    *) echo "未知FSDP_SHARDING_STRATEGY：$FSDP_SHARDING_STRATEGY" >&2; exit 2 ;;
esac

VARLEN_PARALLEL_ARGS=()
[[ "$GLOBAL_BATCH_PACKING" == 1 ]] && VARLEN_PARALLEL_ARGS+=(--enable-global-batch-packing)
[[ "$DYNAMIC_CP" == 1 ]] && VARLEN_PARALLEL_ARGS+=(--dynamic-cp)
if [[ "$GLOBAL_BATCH_PACKING" == 1 ]]; then
    VARLEN_PARALLEL_ARGS+=(--min-dynamic-cp "$MIN_DYNAMIC_CP" --max-seqlen-per-rank "$MAX_SEQLEN_PER_RANK")
fi
[[ "$FSDP" == 1 ]] && VARLEN_PARALLEL_ARGS+=(--enable-fsdp --fsdp-sharding-strategy "$FSDP_SHARDING_STRATEGY")
[[ "$GRAD_OVERLAP" == 1 ]] && VARLEN_PARALLEL_ARGS+=(--enable-grad-overlap)
[[ -n "$RESUME_CHECKPOINT" ]] && VARLEN_PARALLEL_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
[[ "$CONFIG_CHECK" == 1 ]] && VARLEN_PARALLEL_ARGS+=(--config-check)
# 固定生效的源码，避免运行到镜像内旧版本；用户仍可显式覆盖路径。
MEGATRON_CORE_ROOT="${MEGATRON_CORE_ROOT:-${PPU_RECIPES_ROOT}/Megatron-LM-dcp-fsdp}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PPU_RECIPES_ROOT}/Megatron-Bridge}"
if [[ -d "$MEGATRON_CORE_ROOT/megatron/core" && -d "$MEGATRON_BRIDGE_ROOT/src/megatron/bridge" ]]; then
    export PYTHONPATH="$MEGATRON_BRIDGE_ROOT:$MEGATRON_CORE_ROOT:$MEGATRON_BRIDGE_ROOT/src:$PPU_RECIPES_ROOT/model_prof${PYTHONPATH:+:$PYTHONPATH}"
elif [[ "$GLOBAL_BATCH_PACKING" == 1 ]]; then
    echo "global packing需要显式有效的MEGATRON_CORE_ROOT与MEGATRON_BRIDGE_ROOT。" >&2
    exit 2
fi
if [[ "$FSDP" == 1 ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
    if [[ "$CUDA_DEVICE_MAX_CONNECTIONS" == 1 ]]; then
        echo "FSDP要求CUDA_DEVICE_MAX_CONNECTIONS>1。" >&2
        exit 2
    fi
else
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
fi
PARALLEL_TAG="gbp${GLOBAL_BATCH_PACKING}_dcp${DYNAMIC_CP}_fsdp${FSDP}"
[[ "$FSDP" == 1 ]] && PARALLEL_TAG+="_${FSDP_SHARDING_STRATEGY}"

validate_varlen_extra_args() {
    local arg
    for arg in "$@"; do
        case "$arg" in
            --enable-fsdp|--fsdp-sharding-strategy|--enable-grad-overlap|--dynamic-cp|--enable-global-batch-packing|--gbs|--tp|--pp|--cp|--mbs|--ga|--min-dynamic-cp|--max-seqlen-per-rank|--resume-checkpoint|--config-check|--enable-fsdp=*|--fsdp-sharding-strategy=*|--gbs=*|--tp=*|--pp=*|--cp=*|--mbs=*|--ga=*)
                echo "EXTRA_ARGS中的$arg与显式配置重复；请使用对应环境变量。" >&2
                return 2
                ;;
        esac
    done
}
