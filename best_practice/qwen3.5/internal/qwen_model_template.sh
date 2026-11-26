#!/usr/bin/env bash
#
# Qwen3.5 architecture template for the Megatron-LM-for-sail launchers.
#
# Owns everything that describes the model: the per-variant size table, the
# mock-data geometry that follows from the vision tower, and the model-side
# CLI flags (multimodal/vision, language, hybrid layer pattern, MoE).
# common.sh only sources this file and calls the functions; it adds
# no model flags of its own.

# qwen_model_resolve: MODEL_VARIANT -> architecture sizes. These must match
# configuration.py, which is the single source of truth for the HF-parity
# tables; Megatron also needs them on the CLI for PP splits and parameter
# counting.
qwen_model_resolve() {
    case "$MODEL_VARIANT" in
        proxy)          NUM_LAYERS=${NUM_LAYERS:-4};  NUM_EXPERTS=${NUM_EXPERTS:-16}
                        HIDDEN_SIZE=4096; FFN_HIDDEN_SIZE=10240; NUM_ATTN_HEADS=32
                        NUM_QUERY_GROUPS=2; LINEAR_NUM_VALUE_HEADS=64
                        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-2} ;;
        0.8b)           NUM_LAYERS=${NUM_LAYERS:-24}; NUM_EXPERTS=${NUM_EXPERTS:-0}
                        HIDDEN_SIZE=1024; FFN_HIDDEN_SIZE=3584; NUM_ATTN_HEADS=8
                        NUM_QUERY_GROUPS=2; LINEAR_NUM_VALUE_HEADS=16
                        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-12} ;;
        2b)             NUM_LAYERS=${NUM_LAYERS:-24}; NUM_EXPERTS=${NUM_EXPERTS:-0}
                        HIDDEN_SIZE=2048; FFN_HIDDEN_SIZE=6144; NUM_ATTN_HEADS=8
                        NUM_QUERY_GROUPS=2; LINEAR_NUM_VALUE_HEADS=16
                        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-24} ;;
        4b)             NUM_LAYERS=${NUM_LAYERS:-32}; NUM_EXPERTS=${NUM_EXPERTS:-0}
                        HIDDEN_SIZE=2560; FFN_HIDDEN_SIZE=9216; NUM_ATTN_HEADS=16
                        NUM_QUERY_GROUPS=4; LINEAR_NUM_VALUE_HEADS=32
                        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-24} ;;
        9b)             NUM_LAYERS=${NUM_LAYERS:-32}; NUM_EXPERTS=${NUM_EXPERTS:-0}
                        HIDDEN_SIZE=4096; FFN_HIDDEN_SIZE=12288; NUM_ATTN_HEADS=16
                        NUM_QUERY_GROUPS=4; LINEAR_NUM_VALUE_HEADS=32
                        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27} ;;
        27b)            NUM_LAYERS=${NUM_LAYERS:-64}; NUM_EXPERTS=${NUM_EXPERTS:-0}
                        HIDDEN_SIZE=5120; FFN_HIDDEN_SIZE=17408; NUM_ATTN_HEADS=24
                        NUM_QUERY_GROUPS=4; LINEAR_NUM_VALUE_HEADS=48
                        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27} ;;
        35b_a3b|35b_a3b_light)
                        NUM_LAYERS=${NUM_LAYERS:-40}; NUM_EXPERTS=${NUM_EXPERTS:-256}
                        HIDDEN_SIZE=2048; FFN_HIDDEN_SIZE=4096; NUM_ATTN_HEADS=16
                        NUM_QUERY_GROUPS=2; LINEAR_NUM_VALUE_HEADS=32
                        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27} ;;
        122b_a10b)      NUM_LAYERS=${NUM_LAYERS:-48}; NUM_EXPERTS=${NUM_EXPERTS:-256}
                        HIDDEN_SIZE=3072; FFN_HIDDEN_SIZE=8192; NUM_ATTN_HEADS=32
                        NUM_QUERY_GROUPS=2; LINEAR_NUM_VALUE_HEADS=64
                        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27} ;;
        397b_a17b)      NUM_LAYERS=${NUM_LAYERS:-60}; NUM_EXPERTS=${NUM_EXPERTS:-512}
                        HIDDEN_SIZE=4096; FFN_HIDDEN_SIZE=10240; NUM_ATTN_HEADS=32
                        NUM_QUERY_GROUPS=2; LINEAR_NUM_VALUE_HEADS=64
                        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27} ;;
        *)
            : "${NUM_LAYERS:?NUM_LAYERS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
            : "${NUM_EXPERTS:?NUM_EXPERTS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
            : "${HIDDEN_SIZE:?HIDDEN_SIZE must be set for MODEL_VARIANT=$MODEL_VARIANT}"
            : "${FFN_HIDDEN_SIZE:?FFN_HIDDEN_SIZE must be set for MODEL_VARIANT=$MODEL_VARIANT}"
            : "${NUM_ATTN_HEADS:?NUM_ATTN_HEADS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
            : "${NUM_QUERY_GROUPS:?NUM_QUERY_GROUPS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
            : "${LINEAR_NUM_VALUE_HEADS:?LINEAR_NUM_VALUE_HEADS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
            VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27} ;;
    esac
    ATTN_CADENCE="${ATTN_CADENCE:-GGG*}"
}

# qwen_validate_data_geometry: the mock provider does not resize. grid_thw and
# the token span are built from the requested geometry, so every edge must be
# patch-aligned and the token budget must cover the merged patches of every
# image.
qwen_validate_data_geometry() {
    [[ "$DATASET_PROVIDER" == "mock" ]] || return 0
    local img_h="${IMAGE_SIZE_H:-$IMAGE_SIZE}"
    local img_w="${IMAGE_SIZE_W:-$IMAGE_SIZE}"
    if (( img_h % 16 != 0 || img_w % 16 != 0 )); then
        echo "mock image ${img_w}x${img_h} must be a multiple of the ViT patch size 16" >&2
        return 2
    fi
    if (( NUM_IMAGES > 0 )); then
        local merged_per_image=$(( 2 * (img_h / 16 / 2) * (img_w / 16 / 2) ))
        local merged=$(( NUM_IMAGES * merged_per_image ))
        if (( IMAGE_SEQ_LENGTH < merged )); then
            echo "IMAGE_SEQ_LENGTH=${IMAGE_SEQ_LENGTH} < ${merged} merged vision tokens" >&2
            echo "  (${NUM_IMAGES} x ${img_w}x${img_h}: pay attention to the ViT patch 16 / merge 2 geometry)" >&2
            return 2
        fi
        if (( SEQ_LEN < merged + NUM_IMAGES )); then
            echo "SEQ_LEN=${SEQ_LEN} cannot fit ${NUM_IMAGES} image(s) plus text" >&2
            return 2
        fi
    fi
    return 0
}

# qwen_model_args: append the model-side flags to QWEN_MODEL_ARGS.
qwen_model_args() {
    local -a multimodal_args=(
        --model-arch qwen35_vl
        --model-variant "$MODEL_VARIANT"
        --dataset-provider "$DATASET_PROVIDER"
        --hf-processor-path "${HF_PROCESSOR_PATH:-$HF_MODEL}"
        --use-vanilla-collate-fn
        --image-token-id 248056
        --image-size "$IMAGE_SIZE"
        --num-images "$NUM_IMAGES"
        --total-seq-length "$SEQ_LEN"
        --image-seq-length "$IMAGE_SEQ_LENGTH"
        --vision-num-layers "$VISION_NUM_LAYERS"
    )
    [[ -n "$IMAGE_SIZE_W" ]] && multimodal_args+=( --image-size-w "$IMAGE_SIZE_W" )
    [[ -n "$IMAGE_SIZE_H" ]] && multimodal_args+=( --image-size-h "$IMAGE_SIZE_H" )
    (( ENABLE_VISION_CP == 1 )) && multimodal_args+=( --enable-vision-cp )
    # A text-only run never calls the encoder; Megatron-FSDP hangs on an unused
    # unit because it never receives gradients, so omit it from the model.
    if (( DISABLE_VISION_ENCODER_OVERRIDE == 1 )) || { (( NUM_IMAGES == 0 )) && [[ "$DATASET_PROVIDER" == "mock" ]]; }; then
        multimodal_args+=( --disable-vision-encoder )
    fi
    (( USE_PACKED_SEQUENCE == 1 )) && multimodal_args+=( --use-packed-sequence )

    local -a language_args=(
        --hidden-size "$HIDDEN_SIZE"
        --ffn-hidden-size "$FFN_HIDDEN_SIZE"
        --num-attention-heads "$NUM_ATTN_HEADS"
        --group-query-attention
        --num-query-groups "$NUM_QUERY_GROUPS"
        --kv-channels 256
        --max-position-embeddings 262144
        --seq-length "$SEQ_LEN"
        --normalization RMSNorm
        --apply-layernorm-1p
        --norm-epsilon 1e-06
        --swiglu
        --disable-bias-linear
        # 3D MRoPE; the split must match MROPE_SECTION and sum to half the
        # rotary dimension (kv_channels * rotary_percent / 2 = 32).
        --position-embedding-type mrope
        --mrope-section 11 11 10
        --rotary-percent 0.25
        --rotary-base 10000000
        --rotary-seq-len-interpolation-factor 1
        --qk-layernorm
        --attention-output-gate
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --experimental-attention-variant gated_delta_net
        --linear-conv-kernel-dim 4
        --linear-key-head-dim 128
        --linear-value-head-dim 128
        --linear-num-key-heads 16
        --linear-num-value-heads "$LINEAR_NUM_VALUE_HEADS"
        --make-vocab-size-divisible-by 485
    )
    [[ -n "$LINEAR_CP_MODE" ]] && language_args+=( --linear-cp-mode "$LINEAR_CP_MODE" )
    [[ -n "$CP_PARTITION_MODE" ]] && language_args+=( --cp-partition-mode "$CP_PARTITION_MODE" )
    case "$MODEL_VARIANT" in
        0.8b|2b|4b) ;;
        *) language_args+=( --untie-embeddings-and-output-weights ) ;;
    esac

    local -a moe_args=()
    if (( NUM_EXPERTS > 0 )); then
        local moe_topk moe_ffn moe_shared
        case "$MODEL_VARIANT" in
            proxy)              moe_topk=2;  moe_ffn=1024; moe_shared=1024 ;;
            35b_a3b|35b_a3b_light) moe_topk=8;  moe_ffn=512;  moe_shared=512 ;;
            122b_a10b)          moe_topk=8;  moe_ffn=1024; moe_shared=1024 ;;
            397b_a17b)          moe_topk=10; moe_ffn=1024; moe_shared=1024 ;;
            *)
                : "${MOE_TOPK:?MOE_TOPK must be set for MoE MODEL_VARIANT=$MODEL_VARIANT}"
                : "${MOE_FFN_HIDDEN:?MOE_FFN_HIDDEN must be set for MoE MODEL_VARIANT=$MODEL_VARIANT}"
                : "${MOE_SHARED_HIDDEN:?MOE_SHARED_HIDDEN must be set for MoE MODEL_VARIANT=$MODEL_VARIANT}"
                moe_topk=$MOE_TOPK; moe_ffn=$MOE_FFN_HIDDEN; moe_shared=$MOE_SHARED_HIDDEN ;;
        esac
        moe_args=(
            --num-experts "$NUM_EXPERTS"
            --moe-ffn-hidden-size "$moe_ffn"
            --moe-shared-expert-intermediate-size "$moe_shared"
            --moe-shared-expert-gate
            --moe-router-load-balancing-type aux_loss
            --moe-router-topk "$moe_topk"
            --moe-grouped-gemm
            --moe-aux-loss-coeff 1e-3
            --moe-router-dtype fp32
            --moe-permute-fusion
            --moe-router-fusion
        )
        case "$DISPATCHER" in
            alltoall) moe_args+=( --moe-token-dispatcher-type alltoall ) ;;
            flex)     moe_args+=( --moe-token-dispatcher-type flex
                                  --moe-flex-dispatcher-backend "${MOE_DISPATCHER_BACKEND:-deepep}" ) ;;
            *) echo "Unsupported DISPATCHER=$DISPATCHER (expected alltoall or flex)" >&2; return 2 ;;
        esac
        # Perf / mock-data only: forces uniform routing and freezes
        # data-dependent routing, so never enable it for a real finetune.
        (( FORCE_LOAD_BALANCING == 1 )) && moe_args+=( --moe-router-force-load-balancing )
    fi

    # HybridModel orders each historical GPT block as two layers: the attention
    # layer (GatedDeltaNet 'G' or full attention '*') followed by the MLP layer
    # ('-' dense, 'E' MoE). ATTN_CADENCE is one period, repeated and truncated to
    # NUM_LAYERS; each MTP depth replicates the final block.
    local mlp_symbol="-"
    (( NUM_EXPERTS > 0 )) && mlp_symbol="E"
    local pattern="" block_idx attn_symbol
    for (( block_idx = 0; block_idx < NUM_LAYERS; block_idx++ )); do
        attn_symbol="${ATTN_CADENCE:block_idx % ${#ATTN_CADENCE}:1}"
        pattern+="${attn_symbol}${mlp_symbol}"
    done
    local last_symbol="${ATTN_CADENCE:(NUM_LAYERS - 1) % ${#ATTN_CADENCE}:1}"
    local mtp_depth
    for (( mtp_depth = 0; mtp_depth < MTP_NUM_LAYERS; mtp_depth++ )); do
        pattern+="/${last_symbol}${mlp_symbol}"
    done

    QWEN_MODEL_ARGS=(
        "${multimodal_args[@]}"
        "${language_args[@]}"
        --hybrid-layer-pattern "$pattern"
        "${moe_args[@]}"
    )
    return 0
}
