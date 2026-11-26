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
        --lr 1.2e-4
        --min-lr 1.2e-5
        --lr-decay-style cosine
        --lr-warmup-iters 100
        --lr-decay-iters 2000
        --weight-decay 0.1
        --clip-grad 1.0
        --bf16
        --use-mcore-models
        --transformer-impl transformer_engine
        --enable-experimental
        --manual-gc
        --manual-gc-interval 50
        --use-flash-attn
        --calculate-per-token-loss
    )
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
        --eval-interval 500
        --eval-iters 10
        --log-throughput
        --log-timers-to-tensorboard
        --log-params-norm
    )
    if (( SAVE_CHECKPOINT == 1 )); then
        LOGGING_ARGS+=(
            --save-interval "$SAVE_INTERVAL"
            --save "${OUTPUT_DIR}/${RUN_TAG:-run}"
        )
    fi

    RECOMPUTE_ARGS=()
    if (( RECOMPUTE == 1 )); then
        RECOMPUTE_ARGS=(
            --recompute-granularity full
            --recompute-method uniform
            --recompute-num-layers 1
        )
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
