"""Qwen3.5-VL variable-length training entrypoint (2B/4B/9B/27B/35B-A3B).

Trains on pre-converted variable-length conversation JSONL: CORD-v2 through the
``cord_v2`` schema adapter, or typed conversations such as LLaVA-NeXT with no
adapter. In-batch packing turns each microbatch into a THD pack in the Qwen2.5/
Qwen3-VL collate; ``vlm_step`` consumes the resulting ``cu_seqlens``. The
Qwen3-VL-specific step (``qwen3_vl_step``) would pad the batch to a fixed
length instead.

Variable-length pretrain (random init)::

    torchrun --standalone --nproc_per_node=8 \\
        train_varlen_9b.py \\
        --model-size 9b --data /cpfs/datasets/llava-next-data/json \\
        --schema-adapter none --seq-length 8192 --mbs 8 --ga 1 \\
        --tp 2 --pp 1 --cp 1 --train-iters 20

SFT from a converted Megatron checkpoint::

    torchrun --standalone --nproc_per_node=8 \\
        train_varlen_9b.py \\
        --model-size 9b --data /cpfs/datasets/llava-next-data/json \\
        --schema-adapter none --seq-length 8192 --mbs 8 --ga 1 \\
        --tp 2 --pp 1 --cp 1 --train-iters 100 \\
        --pretrained-checkpoint /path/to/megatron_checkpoints/Qwen3.5-9B-tp2

MoE example (35B-A3B, 16 GPUs)::

    torchrun --nproc_per_node=8 --nnodes=2 \\
        train_varlen_9b.py \\
        --model-size 35b_a3b --data /cpfs/datasets/llava-next-data/json \\
        --schema-adapter none --seq-length 8192 --mbs 4 --ga 1 \\
        --tp 2 --ep 16 --pp 1 --cp 1 --train-iters 20

Profiling: ``model_prof``/asys runs are launched through ``prof.sh`` with
``MODEL_PROFILE_RANGE=start,end``; ``MyStepCallback`` calls ``prof_iter(step)``
after each training step (see ``tune_llava_next_9b.sh``). The ``--enable-profiler``
and ``--enable-memory-snapshot`` flags below control Megatron's own PyTorch
profiler / memory snapshot independently.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess

import torch

from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_2b_sft_1gpu_h100_bf16_config,
    qwen35_vl_4b_sft_2gpu_h100_bf16_config,
    qwen35_vl_9b_sft_4gpu_h100_bf16_config,
    qwen35_vl_27b_sft_16gpu_h100_bf16_config,
    qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config,
)
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.vlm_step import forward_step as vlm_step


logger = logging.getLogger(__name__)

logging.getLogger("megatron.core.tensor_parallel.random").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# NUMA binding
# ---------------------------------------------------------------------------


def _get_gpu_numa_node(local_rank: int) -> int | None:
    """Query the NUMA node for the given GPU."""
    result = subprocess.run(
        ["nvidia-smi", "-i", str(local_rank), "--query-gpu=pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    nvidia_bus_id = result.stdout.strip()
    if not nvidia_bus_id:
        return None
    parts = nvidia_bus_id.lower().split(":")
    domain = parts[0][-4:]
    sysfs_addr = f"{domain}:{parts[1]}:{parts[2]}"
    numa_path = f"/sys/bus/pci/devices/{sysfs_addr}/numa_node"
    try:
        with open(numa_path) as f:
            node = int(f.read().strip())
            return node if node >= 0 else None
    except (FileNotFoundError, ValueError):
        return None


def _bind_to_gpu_numa():
    """Bind current process to the NUMA node of its GPU."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    numa_node = _get_gpu_numa_node(local_rank)
    if numa_node is None:
        return
    try:
        import ctypes

        libnuma = ctypes.CDLL("libnuma.so.1")
        libnuma.numa_run_on_node(numa_node)
        libnuma.numa_set_preferred(numa_node)
    except OSError as e:
        logger.info(f"NUMA binding skipped (libnuma not available): {e}")


_bind_to_gpu_numa()

try:
    from model_prof import prof_iter
except ImportError:

    def prof_iter(iter):  # noqa: D103
        pass


class MyStepCallback(Callback):
    """Profiling callback injected at the end of each training step."""

    def on_train_step_end(self, context):
        """Called automatically after each training step."""
        step = context.state.train_state.step
        prof_iter(step)


# ---------------------------------------------------------------------------
# Recipe config mapping
# ---------------------------------------------------------------------------

_RECIPE_MAP = {
    "2b": qwen35_vl_2b_sft_1gpu_h100_bf16_config,
    "4b": qwen35_vl_4b_sft_2gpu_h100_bf16_config,
    "9b": qwen35_vl_9b_sft_4gpu_h100_bf16_config,
    "27b": qwen35_vl_27b_sft_16gpu_h100_bf16_config,
    "35b_a3b": qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config,
}


def _get_recipe_config(model_size: str):
    """Return the recipe config builder for the requested Qwen3.5-VL size."""
    if model_size not in _RECIPE_MAP:
        raise ValueError(f"--model-size must be one of {list(_RECIPE_MAP.keys())}, got {model_size!r}")
    return _RECIPE_MAP[model_size]


# ---------------------------------------------------------------------------
# Feature enablers
# ---------------------------------------------------------------------------


def enable_tp_overlap(cfg: ConfigContainer) -> None:
    """Enable TP communication overlap."""
    cfg.model.sequence_parallel = True
    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
    else:
        cfg.comm_overlap.tp_comm_overlap = True


def enable_grad_overlap(cfg: ConfigContainer) -> None:
    """Overlap gradient reduce and parameter gather."""
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True


def enable_full_recompute(cfg: ConfigContainer) -> None:
    """Full activation recompute."""
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1


def enable_recompute(cfg: ConfigContainer) -> None:
    """Selective activation recompute on core_attn only."""
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["core_attn"]


def enable_cuda_graph_te(
    cfg: ConfigContainer,
    warmup_steps: int = 3,
    scope: list[str] | None = None,
) -> None:
    """CUDA graph: Transformer Engine scoped. Auto-detects scope for MoE vs dense."""
    if scope is None:
        num_experts = getattr(cfg.model, "num_moe_experts", None)
        scope = ["attn", "moe_router", "moe_preprocess"] if num_experts else ["attn"]
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = scope
    cfg.model.cuda_graph_warmup_steps = warmup_steps
    cfg.model.use_te_rng_tracker = True


def enable_full_iteration_graph(cfg: ConfigContainer, warmup_steps: int = 3) -> None:
    """Full iteration CUDA graph (entire fwd+bwd as one graph)."""
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_warmup_steps = warmup_steps
    cfg.model.use_te_rng_tracker = True
    cfg.ddp.check_for_nan_in_grad = False


def enable_fp8(cfg: ConfigContainer) -> None:
    """FP8 blockwise mixed precision (requires Hopper+ GPU)."""
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        fp8="te",
        fp8_recipe="tensorwise",
        fp8_param_gather=True,
    )


def enable_pytorch_profiler(
    cfg: ConfigContainer,
    step_start: int,
    step_end: int,
    output_dir: str,
) -> None:
    """PyTorch profiler: rank 0 only."""
    cfg.profiling.use_pytorch_profiler = True
    cfg.profiling.profile_step_start = step_start
    cfg.profiling.profile_step_end = step_end
    cfg.profiling.pytorch_profiler_collect_shapes = True
    cfg.profiling.pytorch_profiler_collect_callstack = True
    cfg.profiling.profile_ranks = [0]
    cfg.logger.tensorboard_dir = output_dir


def enable_full_activation_offload(cfg: ConfigContainer) -> None:
    """Enable layer-level CPU activation offloading."""
    cfg.model.cpu_offloading = True
    cfg.model.cpu_offloading_num_layers = max(1, getattr(cfg.model, "num_layers", 32) - 1)
    cfg.model.cpu_offloading_activations = True
    cfg.model.cpu_offloading_weights = False


def enable_fine_grained_activation_offload(cfg: ConfigContainer) -> None:
    """Enable fine-grained (TE module-level) activation offloading."""
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = [
        "core_attn",
        "qkv_linear",
        "attn_proj",
        "attn_norm",
        "mlp_norm",
    ]


# ---------------------------------------------------------------------------
# MoE config overrides
# ---------------------------------------------------------------------------


def _apply_moe_overrides(cfg: ConfigContainer, args: argparse.Namespace) -> None:
    """Override MoE config from CLI (recipe sets defaults, CLI takes priority)."""
    num_experts = getattr(cfg.model, "num_moe_experts", None)
    if not num_experts:
        return

    if args.ep is not None:
        cfg.model.expert_model_parallel_size = args.ep
    if args.etp is not None:
        cfg.model.expert_tensor_parallel_size = args.etp

    cfg.model.moe_token_dispatcher_type = args.moe_dispatcher_type
    if args.moe_dispatcher_backend:
        cfg.model.moe_flex_dispatcher_backend = args.moe_dispatcher_backend
    cfg.model.moe_grouped_gemm = args.moe_grouped_gemm
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = args.moe_permute_fusion
    cfg.model.use_transformer_engine_op_fuser = args.use_te_op_fuser
    cfg.model.moe_shared_expert_overlap = args.moe_shared_expert_overlap
    if hasattr(cfg.model, "moe_router_force_load_balancing"):
        cfg.model.moe_router_force_load_balancing = args.moe_force_load_balancing
    if hasattr(cfg.model, "moe_router_padding_for_fp8"):
        cfg.model.moe_router_padding_for_fp8 = False
    if hasattr(cfg.model, "overlap_moe_expert_parallel_comm"):
        cfg.model.overlap_moe_expert_parallel_comm = False
    if hasattr(cfg.model, "delay_wgrad_compute"):
        cfg.model.delay_wgrad_compute = False

    if args.moe_deepep_num_sms is not None and hasattr(cfg.model, "moe_deepep_num_sms"):
        cfg.model.moe_deepep_num_sms = args.moe_deepep_num_sms

    if args.enable_deepep_autotune:
        if hasattr(cfg.model, "moe_deepep_autotune"):
            cfg.model.moe_deepep_autotune = True
        cfg.model.moe_flex_dispatcher_backend = args.moe_dispatcher_backend

    logger.info(
        f"MoE config applied: num_experts={num_experts}, "
        f"dispatcher={args.moe_dispatcher_type}, backend={args.moe_dispatcher_backend}, "
        f"grouped_gemm={args.moe_grouped_gemm}, force_load_balancing={args.moe_force_load_balancing}"
    )


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> ConfigContainer:
    """Build ConfigContainer from CLI args and the variable-length data source."""
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    cp = args.cp
    tp = args.tp
    pp = args.pp
    dp = max(1, world_size // (tp * pp * cp))
    gbs = args.gbs if args.gbs is not None else max(args.mbs * dp * args.ga, 1)

    cfg = _get_recipe_config(args.model_size)()
    cfg.model.seq_length = args.seq_length
    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = pp
    cfg.model.virtual_pipeline_model_parallel_size = args.vpp
    cfg.model.context_parallel_size = cp
    # The H100 recipe enables the fused pre-GDR path, which requires
    # causal-conv1d. That extension has no PPU build, so keep the unfused FLA
    # path (the same default as the PPU recipes).
    if hasattr(cfg.model, "gdn_pre_gated_delta_rule_fusion"):
        cfg.model.gdn_pre_gated_delta_rule_fusion = False

    cfg.train.micro_batch_size = args.mbs
    cfg.train.global_batch_size = gbs
    cfg.train.train_iters = args.train_iters
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = args.log_interval
    cfg.logger.tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    cfg.logger.log_throughput_to_tensorboard = True
    cfg.logger.log_timers_to_tensorboard = True
    cfg.logger.log_memory_to_tensorboard = True

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.rerun_state_machine.check_for_nan_in_loss = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = args.enable_grad_overlap
    cfg.ddp.overlap_param_gather = args.enable_grad_overlap
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.nccl_ub = os.getenv("ENABLE_COPY_ENGINE", "0") == "1"

    # Variable-length JSONL source; packing happens in the Qwen collate and is
    # consumed by vlm_step through cu_seqlens/position_ids.
    data_jsonl = args.data_jsonl or os.path.join(args.data, "train.jsonl")
    schema_adapter = None if args.schema_adapter.lower() in ("", "none") else args.schema_adapter
    cfg.dataset.seq_length = args.seq_length
    cfg.dataset.source = HFDatasetSourceConfig(
        path_or_dataset="json",
        split="train",
        schema_adapter=schema_adapter,
        load_kwargs={"data_files": data_jsonl},
    )
    cfg.dataset.enable_in_batch_packing = True
    cfg.dataset.defer_in_batch_packing_to_step = False
    cfg.dataset.in_batch_packing_pad_to_multiple_of = 64
    cfg.dataset.num_workers = args.num_workers
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False

    # Checkpoint: finetune() needs a pretrained checkpoint, pretrain() does not.
    cfg.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    cfg.checkpoint.load = None
    if args.save_interval > 0:
        cfg.checkpoint.save = os.path.join(args.output_dir, "checkpoints")
        cfg.checkpoint.save_interval = args.save_interval
    else:
        cfg.checkpoint.save = None
        cfg.checkpoint.save_interval = 0

    # MoE overrides (CLI > recipe defaults)
    _apply_moe_overrides(cfg, args)
    # MTP routes are not imported/exported by the HF <-> Megatron conversion path.
    cfg.model.mtp_num_layers = None

    # CP > 1 required configuration
    if cp > 1:
        cfg.model.calculate_per_token_loss = True
        cfg.ddp.average_in_collective = False
        cfg.model.cross_entropy_loss_fusion = False

    cfg.validation.eval_micro_batch_size = None
    cfg.validation.eval_global_batch_size = None
    cfg.model.cuda_graph_impl = "none"

    # -- Apply advanced features --
    if args.enable_sp or args.enable_tp_overlap:
        cfg.model.sequence_parallel = True

    # GDN linear CP mode (chunkwise/headwise). "legacy" is the historical recipe
    # label for the zigzag + all-to-all path, which upstream Megatron calls
    # "headwise"; normalize it so Megatron only ever sees community values.
    if args.linear_cp_mode is not None:
        cfg.model.linear_cp_mode = "headwise" if args.linear_cp_mode == "legacy" else args.linear_cp_mode
        logger.info(f"Set linear_cp_mode={cfg.model.linear_cp_mode}")

    if args.cross_entropy_fusion_impl is not None:
        cfg.model.cross_entropy_fusion_impl = args.cross_entropy_fusion_impl
        logger.info(
            f"cross_entropy_fusion_impl overridden to '{args.cross_entropy_fusion_impl}' "
            f"(cross_entropy_loss_fusion={cfg.model.cross_entropy_loss_fusion})"
        )

    if args.disable_cross_entropy_fusion:
        cfg.model.cross_entropy_loss_fusion = False
        logger.info("cross_entropy_loss_fusion forced off by --disable-cross-entropy-fusion")

    if args.enable_grad_overlap:
        enable_grad_overlap(cfg)

    if args.enable_tp_overlap:
        enable_tp_overlap(cfg)

    if args.enable_full_recompute:
        enable_full_recompute(cfg)

    if args.enable_recompute:
        enable_recompute(cfg)

    if args.enable_cuda_graph_te:
        scope = None
        if args.cuda_graph_scope is not None:
            scope = [s.strip() for s in args.cuda_graph_scope.split(",") if s.strip()]
        enable_cuda_graph_te(cfg, warmup_steps=args.cuda_graph_warmup, scope=scope)

    if args.enable_vision_cuda_graph:
        cfg.model.vision_cuda_graph_impl = "transformer_engine"
        if args.vision_cuda_graph_scope is not None:
            cfg.model.vision_cuda_graph_scope = [
                s.strip() for s in args.vision_cuda_graph_scope.split(",") if s.strip()
            ]
        else:
            cfg.model.vision_cuda_graph_scope = []
        if args.max_vision_cuda_graph_seq_length is not None:
            cfg.model.max_vision_cuda_graph_seq_length = args.max_vision_cuda_graph_seq_length
        cfg.model.use_te_rng_tracker = True
        cfg.rng.te_rng_tracker = True

    if args.enable_full_iteration_graph:
        enable_full_iteration_graph(cfg)

    if args.moe_expert_rank_capacity_factor is not None:
        cfg.model.moe_expert_rank_capacity_factor = args.moe_expert_rank_capacity_factor
        cfg.model.moe_paged_stash = True
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.model.moe_flex_dispatcher_backend = "hybridep"

    if args.enable_full_offload:
        enable_full_activation_offload(cfg)

    if args.enable_fine_grained_offload:
        enable_fine_grained_activation_offload(cfg)

    if args.enable_vision_activation_offload:
        cfg.model.vision_activation_offload = True
        if args.vision_activation_offload_num_layers is not None:
            cfg.model.vision_activation_offload_num_layers = args.vision_activation_offload_num_layers

    if args.enable_fp8:
        enable_fp8(cfg)

    if args.enable_fsdp:
        cfg.dist.use_megatron_fsdp = True
        cfg.ddp.use_megatron_fsdp = True
        cfg.ddp.average_in_collective = False
        cfg.checkpoint.ckpt_format = "fsdp_dtensor"
        cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    if args.enable_profiler:
        enable_pytorch_profiler(
            cfg,
            step_start=args.profile_step_start,
            step_end=args.profile_step_end,
            output_dir=cfg.logger.tensorboard_dir,
        )

    if args.enable_memory_snapshot:
        cfg.profiling.record_memory_history = True
        cfg.profiling.memory_snapshot_path = args.memory_snapshot_path
        cfg.profiling.profile_step_start = args.profile_step_start
        cfg.profiling.profile_step_end = args.profile_step_end
        cfg.profiling.profile_ranks = [0]

    mode = "sft" if args.pretrained_checkpoint else "pretrain"
    logger.info(
        f"Qwen3.5-VL {args.model_size} variable-length: mode={mode} data={data_jsonl} "
        f"schema={schema_adapter} seq={args.seq_length} mbs={args.mbs} gbs={gbs} "
        f"ga={gbs // (args.mbs * dp)} dp={dp} tp={tp} pp={pp} cp={cp} "
        f"packing={cfg.dataset.enable_in_batch_packing} "
        f"defer={cfg.dataset.defer_in_batch_packing_to_step} "
        f"checkpoint={args.pretrained_checkpoint}"
    )

    return cfg


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
        description="Qwen3.5-VL variable-length training (must use torchrun)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Parallelism
    p.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    p.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    p.add_argument("--cp", type=int, default=1, help="Context parallel size")
    p.add_argument("--vpp", type=int, default=None, help="Virtual Pipeline parallel size")
    p.add_argument("--ep", type=int, default=None, help="Expert parallel size (MoE models only)")
    p.add_argument("--etp", type=int, default=None, help="Expert tensor parallel size (MoE models only)")

    # Model / training
    p.add_argument(
        "--model-size",
        type=str,
        default="9b",
        choices=list(_RECIPE_MAP.keys()),
        help="Qwen3.5-VL model size",
    )
    p.add_argument("--seq-length", type=int, default=8192)
    p.add_argument("--mbs", type=int, default=4, help="Micro batch size")
    p.add_argument("--ga", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--gbs", type=int, default=None, help="Global batch size (default: mbs * dp * ga)")
    p.add_argument("--train-iters", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--output-dir", default="./output", help="TensorBoard / checkpoint output root")

    # Variable-length dataset
    ds = p.add_argument_group("variable-length dataset")
    ds.add_argument("--data", default="/cpfs/datasets/cord-v2/json", help="Dataset directory")
    ds.add_argument(
        "--data-jsonl",
        default=None,
        help="Override the JSONL file (default: <data>/train.jsonl)",
    )
    ds.add_argument(
        "--schema-adapter",
        default="cord_v2",
        help="HF schema adapter; 'none' for typed conversations",
    )
    ds.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers")

    # Checkpoint
    ckpt = p.add_argument_group("checkpoint")
    ckpt.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help="Megatron checkpoint for SFT finetune; omit to run pretrain from random init",
    )
    ckpt.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Save a Megatron checkpoint every N steps; 0 disables saving",
    )

    # MoE configuration
    moe = p.add_argument_group("MoE configuration")
    moe.add_argument(
        "--moe-dispatcher-type",
        type=str,
        default="flex",
        choices=["alltoall", "allgather", "flex"],
        help="MoE token dispatcher type",
    )
    moe.add_argument(
        "--moe-dispatcher-backend",
        type=str,
        default="deepep",
        help="Backend for flex dispatcher (default: 'deepep')",
    )
    moe.add_argument(
        "--moe-grouped-gemm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable grouped GEMM for MoE",
    )
    moe.add_argument(
        "--moe-force-load-balancing",
        action="store_true",
        help="Force even token distribution across experts (recommended for benchmarking)",
    )
    moe.add_argument(
        "--moe-deepep-num-sms",
        type=int,
        default=None,
        help="Number of SMs for DeepEP communication kernels",
    )
    moe.add_argument(
        "--moe-shared-expert-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overlap shared expert compute with dispatch A2A",
    )
    moe.add_argument(
        "--moe-permute-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse token permutation ops during dispatch",
    )
    moe.add_argument(
        "--use-te-op-fuser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Transformer Engine operation fuser for advanced expert FFN fusions",
    )
    moe.add_argument(
        "--enable-deepep-autotune",
        action="store_true",
        help="Auto-tune DeepEP dispatch/combine chunk sizes at startup",
    )
    moe.add_argument(
        "--moe-expert-rank-capacity-factor",
        type=float,
        default=None,
        help="Expert rank capacity factor for static MoE buffer sizing",
    )

    # Advanced features
    feat = p.add_argument_group("advanced features")
    feat.add_argument("--enable-sp", action="store_true", help="Sequence Parallel")
    feat.add_argument("--enable-tp-overlap", action="store_true", help="TP comm overlap (auto-enables SP)")
    feat.add_argument("--enable-grad-overlap", action="store_true", help="Overlap grad-reduce + param-gather")
    feat.add_argument("--enable-full-recompute", action="store_true")
    feat.add_argument("--enable-recompute", action="store_true")
    feat.add_argument(
        "--enable-cuda-graph-te",
        action="store_true",
        help="CUDA graph TE scoped (auto-detects scope for MoE vs dense)",
    )
    feat.add_argument(
        "--enable-full-iteration-graph",
        action="store_true",
        help="Full iteration CUDA graph (entire fwd+bwd as one graph)",
    )
    feat.add_argument(
        "--cuda-graph-scope",
        type=str,
        default=None,
        help="Comma-separated CUDA graph scope modules (e.g. 'attn' for dense, "
        "'attn,moe_router,moe_preprocess' for MoE). Auto-detects if omitted.",
    )
    feat.add_argument("--cuda-graph-warmup", type=int, default=3, help="CUDA graph warmup steps")
    feat.add_argument(
        "--enable-vision-cuda-graph",
        action="store_true",
        help="Enable CUDA graph capture for the vision encoder (full-layer by default).",
    )
    feat.add_argument(
        "--vision-cuda-graph-scope",
        type=str,
        default=None,
        help="Comma-separated vision CUDA graph scope (default: full-layer). Examples: 'attn', 'attn,mlp'.",
    )
    feat.add_argument(
        "--max-vision-cuda-graph-seq-length",
        type=int,
        default=None,
        help="Max vision sequence length for CUDA graph padding (default: auto).",
    )
    feat.add_argument("--enable-full-offload", action="store_true",
                       help="Enable layer-level CPU activation offloading")
    feat.add_argument("--enable-fine-grained-offload", action="store_true",
                       help="Enable TE module-level activation offloading")
    feat.add_argument(
        "--enable-vision-activation-offload",
        action="store_true",
        help="Enable layer-level CPU activation offloading for the vision encoder",
    )
    feat.add_argument(
        "--vision-activation-offload-num-layers",
        type=int,
        default=None,
        help="Number of vision layers to offload (default: num_vision_layers - 1)",
    )
    feat.add_argument(
        "--linear-cp-mode",
        type=str,
        default=None,
        choices=[None, "chunkwise", "headwise", "legacy"],
        help="GDN CP strategy: chunkwise (FLA cp_context, no all-to-all), headwise (Ulysses), legacy (zigzag+a2a)",
    )
    feat.add_argument(
        "--cross-entropy-fusion-impl",
        type=str,
        choices=["native", "te"],
        default=None,
        help="Override model.cross_entropy_fusion_impl. 'te' uses TransformerEngine's "
        "triton cross entropy, which keeps logits in bf16 and saves only the in-place "
        "softmaxed logits for backward instead of several fp32 logits-sized buffers.",
    )
    feat.add_argument(
        "--disable-cross-entropy-fusion",
        action="store_true",
        help="Force model.cross_entropy_loss_fusion=False as well at CP=1. "
        "Diagnostic control for the CP sweep: CP>1 already disables fusion, "
        "so this separates the fusion loss from the CP cost.",
    )
    feat.add_argument("--enable-fp8", action="store_true", help="FP8 blockwise (Hopper+)")
    feat.add_argument(
        "--enable-fsdp",
        action="store_true",
        help="Enable Megatron FSDP (fully sharded data parallel). "
        "Shards params, grads, and optimizer state across DP ranks for lower memory.",
    )

    # Profiling (Megatron PyTorch profiler; model_prof/asys uses prof.sh instead)
    pr = p.add_argument_group("profiling")
    pr.add_argument("--enable-profiler", action="store_true")
    pr.add_argument("--enable-memory-snapshot", action="store_true",
                     help="Dump memory snapshot at profile_step_end")
    pr.add_argument("--memory-snapshot-path", type=str, default="/tmp/mem_snapshot.pickle")
    pr.add_argument("--profile-step-start", type=int, default=5)
    pr.add_argument("--profile-step-end", type=int, default=6)

    return p


def _apply_hydra_overrides(args: argparse.Namespace, overrides: list[str]) -> None:
    """Apply Hydra-style key=value overrides to an argparse Namespace.

    Keys must match argparse dest names (e.g. ``seq_length``, ``mbs``).
    Values are coerced to the type of the current default.
    """
    for override in overrides:
        if "=" not in override:
            raise argparse.ArgumentError(None, f"Invalid override (expected key=value): {override}")
        key, value = override.split("=", 1)
        if not hasattr(args, key):
            raise argparse.ArgumentError(None, f"Unknown config key: {key}")
        current = getattr(args, key)

        if isinstance(current, list):
            if value.startswith("[") and value.endswith("]"):
                setattr(args, key, [v.strip().strip('"').strip("'") for v in value[1:-1].split(",")])
            else:
                setattr(args, key, list(current) + [value])
        elif isinstance(current, bool):
            setattr(args, key, value.lower() in ("true", "1", "yes"))
        elif isinstance(current, int):
            setattr(args, key, int(value))
        elif isinstance(current, float):
            setattr(args, key, float(value))
        elif current is None:
            v_lower = value.lower()
            if v_lower in ("true", "false", "yes", "no"):
                setattr(args, key, v_lower in ("true", "1", "yes"))
            else:
                try:
                    if "." in value:
                        setattr(args, key, float(value))
                    else:
                        setattr(args, key, int(value))
                except ValueError:
                    setattr(args, key, value)
        else:
            setattr(args, key, type(current)(value))


def main() -> None:
    """Entry point for variable-length pretrain / SFT."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args, cli_overrides = parser.parse_known_args()
    if cli_overrides:
        _apply_hydra_overrides(args, cli_overrides)
    cfg = build_config(args)
    if args.pretrained_checkpoint:
        finetune(config=cfg, forward_step_func=vlm_step, callbacks=[MyStepCallback()])
    else:
        pretrain(cfg, vlm_step, callbacks=[MyStepCallback()])


if __name__ == "__main__":
    main()
