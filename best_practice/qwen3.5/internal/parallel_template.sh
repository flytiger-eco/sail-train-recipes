#!/usr/bin/env bash
#
# Parallel-layout template for the Megatron-LM-for-sail launchers.
#
# Owns the topology checks (world vs TP*PP*CP, EP vs DP, FSDP vs PP) and the
# parallel / Megatron-FSDP flags. Sourced by common.sh.

# parallel_validate: size the data-parallel axis, check the layout, apply the
# FSDP/PP rule.
parallel_validate() {
    WORLD=$(( NPROC * NNODES ))
    PAR=$(( TP * PP * CP ))
    if (( WORLD % PAR != 0 )); then
        echo "world size ${WORLD} (${NNODES} nodes x ${NPROC}) is not divisible by TP*PP*CP=${PAR}" >&2
        return 2
    fi
    DP=$(( WORLD / PAR ))
    if (( NUM_EXPERTS > 0 )); then
        if (( DP % EP != 0 )); then
            echo "expert parallel size EP=${EP} must divide data parallel size DP=${DP}" >&2
            echo "  (world/${PAR} = ${DP}; pick EP in {1,2,4,8,16,...} up to DP, or resize the layout)" >&2
            return 2
        fi
    elif (( EP > 1 )); then
        echo "MODEL_VARIANT=${MODEL_VARIANT} has NUM_EXPERTS=0 (dense) but EP=${EP}; set EP=1" >&2
        return 2
    fi
    # global batch = micro batch x accumulation x data parallel replicas
    GBS="${GBS:-$(( MBS * GA * DP ))}"

    # FSDP and PP are mutually exclusive on Megatron's standard path.
    # Downgrading an explicit USE_FSDP=1 would drop the FSDP flags, so the run
    # would no longer be the configuration the caller asked to validate; fail
    # instead. The implicit default is still downgraded so plain PP runs work
    # out of the box.
    if (( PP > 1 )) && (( USE_FSDP == 1 )); then
        if [[ -n "$USE_FSDP_WAS_SET" ]]; then
            echo "USE_FSDP=1 was explicitly requested with PP=${PP} > 1." >&2
            echo "FSDP and PP are mutually exclusive on the standard path; set USE_FSDP=0 to run with PP>1." >&2
            return 1
        fi
        echo "PP=${PP} > 1 -> defaulting USE_FSDP=0"
        USE_FSDP=0
    fi
    return 0
}

# parallel_args: build PARALLEL_ARGS and FSDP_ARGS.
parallel_args() {
    PARALLEL_ARGS=(
        --tensor-model-parallel-size "$TP"
        --pipeline-model-parallel-size "$PP"
        --expert-model-parallel-size "$EP"
        --expert-tensor-parallel-size "$ETP"
        --context-parallel-size "$CP"
        --cp-comm-type "$CP_COMM_TYPE"
        --use-distributed-optimizer
        --sequence-parallel
    )
    if (( ENABLE_GRAD_OVERLAP == 1 )); then
        PARALLEL_ARGS+=( --overlap-grad-reduce --overlap-param-gather )
    fi
    if [[ "${ENABLE_COPY_ENGINE:-0}" == "1" ]]; then
        PARALLEL_ARGS+=( --use-nccl-ub )
    fi
    FSDP_ARGS=()
    if (( USE_FSDP == 1 )); then
        # The legacy --grad-reduce-in-fp32 flag does not reach the FSDP
        # precision policy; FP32 accumulation is requested through the FSDP
        # dtype knob instead.
        FSDP_ARGS=(
            --use-megatron-fsdp
            --data-parallel-sharding-strategy optim_grads_params
            --init-model-with-meta-device
            --use-distributed-optimizer
            --ckpt-format fsdp_dtensor
            --megatron-fsdp-main-grads-dtype fp32
        )
        export CUDA_DEVICE_MAX_CONNECTIONS=8
    fi
}
