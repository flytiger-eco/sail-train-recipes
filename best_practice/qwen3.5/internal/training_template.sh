#!/usr/bin/env bash
#
# Training-loop template for the Megatron-LM-for-sail launchers.
#
# Owns optimizer/schedule/precision, logging, recompute and checkpoint flags
# plus the TE CUDA-graph switch. Sourced by common.sh.

training_args() {
    TRAINING_ARGS=(
        --micro-batch-size "$MBS"
        --global-batch-size "$GBS"
        --adam-beta1 0.9
        --adam-beta2 0.95
        --lr "$LR"
        --min-lr "$MIN_LR"
        --lr-decay-style cosine
        --lr-warmup-iters "$LR_WARMUP_ITERS"
        --lr-decay-iters "$LR_DECAY_ITERS"
        --weight-decay "$WEIGHT_DECAY"
        --clip-grad 1.0
        --bf16
        --use-mcore-models
        --transformer-impl transformer_engine
        --enable-experimental
        --manual-gc
        --manual-gc-interval "$MANUAL_GC_INTERVAL"
        --use-flash-attn
        --calculate-per-token-loss
    )
    if (( ENABLE_GDN_FUSED_PRE_GDR == 1 )); then
        TRAINING_ARGS+=( --gdn-pre-gated-delta-rule-fusion )
    fi
    # The pinned Megatron rejects TE CE on TE < 2.19. Dense recipes use the
    # tested FLA adapter; native remains available without bypassing that guard.
    local ce_impl="$CROSS_ENTROPY_FUSION_IMPL"
    if (( CP == 1 )) && [[ "$ce_impl" != "none" ]]; then
        TRAINING_ARGS+=( --cross-entropy-loss-fusion --cross-entropy-fusion-impl "$ce_impl" )
    fi
    if (( MTP_NUM_LAYERS > 0 )); then
        TRAINING_ARGS+=(
            --mtp-num-layers "$MTP_NUM_LAYERS"
            --mtp-loss-scaling-factor 0.1
        )
    fi
    if [[ -n "$CUDA_GRAPH_MODULES" ]]; then
        # Megatron asserts this pairing: TE CUDA graphs plus expandable
        # segments illegal-access unless the graph is kept out of the NCCL
        # registration path.
        export NCCL_GRAPH_REGISTER=0
        local -a graph_modules
        IFS=',' read -r -a graph_modules <<< "$CUDA_GRAPH_MODULES"
        TRAINING_ARGS+=(
            --cuda-graph-impl transformer_engine
            --cuda-graph-modules "${graph_modules[@]}"
            --cuda-graph-warmup-steps "$CUDA_GRAPH_WARMUP_STEPS"
        )
    fi

    LOGGING_ARGS=(
        --eval-interval "$EVAL_INTERVAL"
        --eval-iters "$EVAL_ITERS"
        --log-throughput
        --log-timers-to-tensorboard
    )
    (( LOG_PARAMS_NORM == 1 )) && LOGGING_ARGS+=( --log-params-norm )
    if (( SAVE_CHECKPOINT == 1 )); then
        LOGGING_ARGS+=(
            --save-interval "$SAVE_INTERVAL"
            --save "${OUTPUT_DIR}/${RUN_TAG:-run}"
        )
    fi

    RECOMPUTE_ARGS=()
    local recompute_mode="${RECOMPUTE_MODE:-}"
    [[ -n "$recompute_mode" ]] || { if (( RECOMPUTE == 1 )); then recompute_mode=full; else recompute_mode=none; fi; }
    if [[ "$recompute_mode" == "full" ]]; then
        RECOMPUTE_ARGS=(
            --recompute-granularity full
            --recompute-method uniform
            --recompute-num-layers 1
        )
    elif [[ "$recompute_mode" == "selective" ]]; then
        local -a modules
        IFS=',' read -r -a modules <<< "${RECOMPUTE_MODULES:-moe_act,layernorm,shared_experts}"
        RECOMPUTE_ARGS=( --recompute-granularity selective --recompute-modules "${modules[@]}" )
    elif [[ "$recompute_mode" != "none" ]]; then
        echo "RECOMPUTE_MODE must be none, selective or full" >&2
        return 2
    fi
    (( RECOMPUTE_VISION == 1 )) && RECOMPUTE_ARGS+=( --recompute-vision )

    CKPT_ARGS=()
    if [[ -n "$CKPT_LOAD" ]]; then
        CKPT_ARGS+=( --load "$CKPT_LOAD" )
        if (( CKPT_RESUME == 0 )); then
            CKPT_ARGS+=( --finetune --no-load-optim --no-load-rng )
        fi
        [[ -n "$CKPT_FORMAT" ]] && CKPT_ARGS+=( --ckpt-format "$CKPT_FORMAT" )
        (( CKPT_OVERRIDE_SCHEDULER == 1 )) && CKPT_ARGS+=( --override-opt-param-scheduler )
    fi
    return 0
}
