# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.5-VL 35B-A3B (MoE) MegatronMIMO benchmark with MockPerfVLMProvider and vision CP.

Uses ``MockPerfVLMProvider`` (processor-based, auto-detected image layout) instead
of an inline mock dataset, and supports vision CP via ``vision_dp_when_cp``.

When the language component has CP > 1, passing ``--enable-vision-cp`` distributes
image processing across CP ranks (each rank processes a subset of images, then
all-gathers the embeddings). This reduces peak vision activation memory.

Example 8-GPU run (TP=2, EP=2, CP=2, vision CP enabled, DeepEP default):

  python -m torch.distributed.run --standalone --nproc_per_node=8 \\
    train_qwen35_vl_mimo.py \\
      --hf-model Qwen/Qwen3.5-35B-A3B \\
      --component language=tp=2,cp=2,etp=2,dp=1,rank_offset=0 \\
      --component images=tp=1,dp=8,rank_offset=0 \\
      --seq-length 8192 --train-iters 5 --allow-random-init \\
      --enable-vision-cp

Example 8-GPU run (TP=2, EP=4, no CP, DeepEP default):

  python -m torch.distributed.run --standalone --nproc_per_node=8 \\
    train_qwen35_vl_mimo.py \\
      --hf-model Qwen/Qwen3.5-35B-A3B \\
      --component language=tp=2,etp=4,dp=1,rank_offset=0 \\
      --component images=tp=1,dp=8,rank_offset=0 \\
      --seq-length 4096 --train-iters 10 --allow-random-init

Use ``--moe-dispatcher-type alltoall`` to fall back to the alltoall dispatcher.
"""

from __future__ import annotations

import argparse
import logging
import os
import types
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
import torch.distributed as dist
from megatron.core.config import set_experimental_flag
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from transformers import AutoConfig

from megatron.bridge import AutoBridge
from megatron.bridge.data.megatron_mimo.dp_utils import get_megatron_mimo_sampling_info
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.data.vlm_datasets.mock_perf_provider import MockPerfVLMProvider
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DatasetBuildContext,
    LoggerConfig,
    MixedPrecisionConfig,
    ProfilingConfig,
    TrainingConfig,
)
from megatron.bridge.training.megatron_mimo_step import forward_step as megatron_mimo_forward_step
from megatron.bridge.training.pretrain_megatron_mimo import pretrain_megatron_mimo
from megatron.bridge.training.state import TrainState
from megatron.bridge.training.tokenizers.config import TokenizerConfig

from model_structure import install_model_structure


logger = logging.getLogger(__name__)

# Suppress benign "CPU RNG state changed within GPU RNG context" warning from MCore.
# This fires during model init when parameter reset_parameters() does CPU-side random
# ops inside the GPU RNG fork context. It does not affect training correctness.
logging.getLogger("megatron.core.tensor_parallel.random").setLevel(logging.ERROR)
import os
import subprocess


def get_gpu_numa_node(local_rank: int) -> int | None:
    """查询 GPU 对应的 NUMA 节点"""
    # 1. 通过 nvidia-smi 查询 GPU 的 PCI 总线地址
    result = subprocess.run(
        ["nvidia-smi", "-i", str(local_rank), "--query-gpu=pci.bus_id", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    nvidia_bus_id = result.stdout.strip()  # 例: "00000001:7F:00.0"
    if not nvidia_bus_id:
        return None

    # 2. 将 nvidia-smi 格式转换为 sysfs 格式
    #    nvidia-smi: 00000001:7F:00.0  (8位 domain)
    #    sysfs:      0001:7f:00.0      (4位 domain)
    parts = nvidia_bus_id.lower().split(":")  # ["00000001", "7f", "00.0"]
    domain = parts[0][-4:]                     # 取后4位: "0001"
    sysfs_addr = f"{domain}:{parts[1]}:{parts[2]}"  # "0001:7f:00.0"

    # 3. 从 sysfs 读取 NUMA 节点
    numa_path = f"/sys/bus/pci/devices/{sysfs_addr}/numa_node"
    try:
        with open(numa_path) as f:
            node = int(f.read().strip())
            return node if node >= 0 else None  # -1 表示非 NUMA 系统
    except (FileNotFoundError, ValueError):
        return None


def bind_to_gpu_numa():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    numa_node = get_gpu_numa_node(local_rank)
    if numa_node is None:
        print(f"[rank{local_rank}] No NUMA binding needed (non-NUMA system or node not found)")
        return

    print(f"[rank{local_rank}] Binding to NUMA node {numa_node}")
    try:
        import ctypes
        libnuma = ctypes.CDLL("libnuma.so.1")
        libnuma.numa_run_on_node(numa_node)
        libnuma.numa_set_preferred(numa_node)
    except OSError as e:
        # libnuma/numactl 均不可用时，放弃 NUMA 绑定以避免崩溃
        print(f"[rank{local_rank}] NUMA binding skipped (libnuma/numactl unavailable): {e}")
        return


bind_to_gpu_numa()
try:
    from model_prof import prof_iter
except ImportError:

    def prof_iter(iter):  # noqa: D103
        pass


# ---------------------------------------------------------------------------
# Component parsing & parallelism config
# ---------------------------------------------------------------------------

G_COMPONENT_KEY_TO_FIELD = {
    "tp": "tensor_model_parallel_size",
    "pp": "pipeline_model_parallel_size",
    "dp": "data_parallel_size",
    "cp": "context_parallel_size",
    "ep": "expert_model_parallel_size",
    "etp": "expert_tensor_parallel_size",
    "rank_offset": "rank_offset",
}

G_DEFAULT_COMPONENTS = [
    "language=tp=1,dp=1,rank_offset=0",
    "images=tp=1,dp=1,rank_offset=1",
]


def _parse_component_spec(raw: str) -> tuple[str, ModuleParallelismConfig]:
    """Parse a component spec string into a (name, config) pair."""
    if "=" not in raw:
        raise ValueError(f"Invalid --component {raw!r}; expected name=tp=N[,pp=N,dp=N,etp=N,cp=N,rank_offset=N]")
    name, _, payload = raw.partition("=")
    parsed: dict[str, int] = {}
    for item in payload.split(","):
        key, _, raw_value = item.partition("=")
        if key not in G_COMPONENT_KEY_TO_FIELD or not raw_value:
            raise ValueError(f"Invalid component field {item!r} in {raw!r}")
        parsed[G_COMPONENT_KEY_TO_FIELD[key]] = int(raw_value)
    return name, ModuleParallelismConfig(**parsed)


def _build_parallelism_config(
    component_specs: list[str],
    world_size: int,
) -> MegatronMIMOParallelismConfig:
    """Build MegatronMIMO parallelism config from component spec strings."""
    module_parallelisms: dict[str, ModuleParallelismConfig] = {}
    for raw in component_specs:
        name, parallelism = _parse_component_spec(raw)
        if name in module_parallelisms:
            raise ValueError(f"Duplicate --component for {name!r}")
        if parallelism.data_parallel_size is None:
            raise ValueError(f"Component {name!r} must set dp explicitly in {raw!r}.")
        module_parallelisms[name] = parallelism

    if MIMO_LANGUAGE_MODULE_KEY not in module_parallelisms:
        raise ValueError(f"Component layout must include {MIMO_LANGUAGE_MODULE_KEY!r}.")

    used_ranks = max(p.rank_offset + p.total_ranks for p in module_parallelisms.values())
    if used_ranks != world_size:
        raise ValueError(
            f"Component layout uses {used_ranks} ranks, but world_size is {world_size}. "
            "MegatronMIMO requires all ranks to be covered. "
            "Adjust --component tp/dp/etp/rank_offset to match exactly."
        )

    return MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)


def _patch_vision_model_for_mimo() -> None:
    """Wrap Qwen3VLVisionModel.forward in bf16 autocast for MIMO mode."""
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import (
            Qwen3VLVisionModel,
        )

        _orig_vision_fwd = Qwen3VLVisionModel.forward
        if not getattr(_orig_vision_fwd, "_mimo_patched", False):

            def _mimo_vision_fwd(self, *args, **kwargs):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return _orig_vision_fwd(self, *args, **kwargs)

            _mimo_vision_fwd._mimo_patched = True
            Qwen3VLVisionModel.forward = _mimo_vision_fwd
            logger.info("Patched Qwen3VLVisionModel.forward with bf16 autocast (MIMO mode).")
    except ImportError:
        logger.warning("Could not import Qwen3VLVisionModel; skipping vision forward patch.")


# ---------------------------------------------------------------------------
# MoE configuration
# ---------------------------------------------------------------------------


def _apply_moe_config(
    standard_provider: Any,
    parallelism_config: MegatronMIMOParallelismConfig,
    args: argparse.Namespace,
) -> None:
    """Apply MoE-specific configuration to the standard provider."""
    num_experts = getattr(standard_provider, "num_moe_experts", None)
    if not num_experts:
        logger.info("Model has no MoE experts; skipping MoE configuration.")
        return

    language_parallelism = parallelism_config.module_parallelisms.get(MIMO_LANGUAGE_MODULE_KEY)
    ep = language_parallelism.expert_model_parallel_size if language_parallelism else 1
    if ep > 1:
        standard_provider.expert_model_parallel_size = ep
        logger.info(f"Set expert_model_parallel_size={ep} (from language component ep)")
    etp = language_parallelism.resolved_expert_tensor_parallel_size if language_parallelism else 1
    if etp != getattr(standard_provider, "expert_tensor_parallel_size", etp):
        standard_provider.expert_tensor_parallel_size = etp
        logger.info(f"Set expert_tensor_parallel_size={etp} (from language component etp)")

    standard_provider.moe_token_dispatcher_type = args.moe_dispatcher_type
    if args.moe_dispatcher_backend:
        standard_provider.moe_flex_dispatcher_backend = args.moe_dispatcher_backend

    standard_provider.moe_grouped_gemm = args.moe_grouped_gemm
    standard_provider.moe_router_fusion = True
    standard_provider.moe_permute_fusion = args.moe_permute_fusion
    standard_provider.use_transformer_engine_op_fuser = args.use_te_op_fuser
    standard_provider.moe_shared_expert_overlap = args.moe_shared_expert_overlap
    if hasattr(standard_provider, "moe_router_force_load_balancing"):
        standard_provider.moe_router_force_load_balancing = args.moe_force_load_balancing
    if hasattr(standard_provider, "moe_router_padding_for_fp8"):
        standard_provider.moe_router_padding_for_fp8 = False
    if args.delay_wgrad and not args.enable_ep_overlap:
        raise ValueError("Delayed wgrad requires EP overlap")
    if args.enable_ep_overlap and (
        args.num_images != 0
        or args.moe_dispatcher_type != "flex"
        or args.moe_dispatcher_backend != "deepep"
        or args.moe_shared_expert_overlap
        or language_parallelism.tensor_model_parallel_size != 1
        or language_parallelism.context_parallel_size != 1
        or language_parallelism.pipeline_model_parallel_size != 1
    ):
        raise ValueError(
            "EP overlap requires pure text, language TP=CP=PP=1 "
            "and the DeepEP dispatcher without shared-expert overlap"
        )
    if hasattr(standard_provider, "overlap_moe_expert_parallel_comm"):
        standard_provider.overlap_moe_expert_parallel_comm = args.enable_ep_overlap
    if hasattr(standard_provider, "delay_wgrad_compute"):
        standard_provider.delay_wgrad_compute = args.delay_wgrad

    if args.moe_deepep_num_sms is not None and hasattr(standard_provider, "moe_deepep_num_sms"):
        standard_provider.moe_deepep_num_sms = args.moe_deepep_num_sms

    if args.enable_deepep_autotune:
        standard_provider.moe_deepep_autotune = True
        standard_provider.moe_flex_dispatcher_backend = args.moe_dispatcher_backend

    topk = getattr(standard_provider, "moe_router_topk", None)
    logger.info(
        f"MoE config applied: num_experts={num_experts}, topk={topk}, ep={etp}, "
        f"dispatcher={args.moe_dispatcher_type}, backend={args.moe_dispatcher_backend}, "
        f"grouped_gemm={args.moe_grouped_gemm}, force_load_balancing={args.moe_force_load_balancing}"
    )


# ---------------------------------------------------------------------------
# FLOPS injection for MIMO training
# ---------------------------------------------------------------------------


def _flops_forward_step(orig_forward_step: Callable) -> Callable:
    """Wrap a MIMO forward_step to inject profiling callback."""

    @wraps(orig_forward_step)
    def _wrapped(*args, **kwargs):
        # The EP-overlap schedule calls the forward step without GlobalState.
        if args and hasattr(args[0], "train_state"):
            prof_iter(args[0].train_state.step)
        return orig_forward_step(*args, **kwargs)

    return _wrapped


# ---------------------------------------------------------------------------
# Data iterator builder (MockPerfVLMProvider)
# ---------------------------------------------------------------------------


def _make_build_data_iterators(args: argparse.Namespace) -> Callable:
    """Create a closure that builds MIMO-compatible data iterators using MockPerfVLMProvider."""

    def _build_data_iterators(cfg, megatron_mimo_infra, *, train_state=None):
        if train_state is None:
            train_state = TrainState()

        if cfg.model._grids is None:
            raise ValueError("MegatronMIMOProvider._grids is None. Model must be built first.")

        sampler_dp_rank, sampler_dp_size, needs_data = get_megatron_mimo_sampling_info(
            cfg.model.megatron_mimo_parallelism_config,
            cfg.model._grids,
        )
        if not needs_data:
            return None, None

        train_samples = max(cfg.train.train_iters * cfg.train.global_batch_size, 100)

        mock_provider = MockPerfVLMProvider(
            seq_length=args.seq_length,
            hf_processor_path=args.hf_model,
            num_images=args.num_images,
            image_size=(args.image_size_w, args.image_size_h),
            output_format="mimo",
            modality_name="images",
            encoder_name="qwen_visual",
        )
        context = DatasetBuildContext(train_samples=train_samples, valid_samples=0, test_samples=0)
        train_ds, _, _ = mock_provider.build_datasets(context)

        train_loader = build_pretraining_data_loader(
            dataset=train_ds,
            consumed_samples=train_state.consumed_train_samples,
            dataloader_type="single",
            micro_batch_size=cfg.train.micro_batch_size,
            num_workers=args.num_workers,
            data_sharding=False,
            collate_fn=train_ds.collate_fn,
            pin_memory=True,
            persistent_workers=False,
            data_parallel_rank=sampler_dp_rank,
            data_parallel_size=sampler_dp_size,
            drop_last=True,
        )

        return iter(train_loader), None

    return _build_data_iterators


# ---------------------------------------------------------------------------
# Model provider builder
# ---------------------------------------------------------------------------


def _build_mimo_provider(
    hf_config: object,
    parallelism_config: MegatronMIMOParallelismConfig,
    args: argparse.Namespace,
) -> MegatronMIMOProvider:
    """Build a MegatronMIMOProvider from the standard Qwen3.5-VL bridge."""
    bridge = AutoBridge.from_hf_config(hf_config)
    standard_provider = bridge.to_megatron_provider(load_weights=False)
    standard_provider.seq_length = args.seq_length
    if hasattr(standard_provider, "language_max_sequence_length"):
        standard_provider.language_max_sequence_length = args.seq_length
    standard_provider.bf16 = True
    standard_provider.fp16 = False
    standard_provider.use_cpu_initialization = True
    standard_provider.bias_dropout_fusion = True
    if hasattr(standard_provider, "mtp_num_layers"):
        standard_provider.mtp_num_layers = None
    if hasattr(standard_provider, "_pack_sequences_in_batch"):
        standard_provider._pack_sequences_in_batch = False

    # MoE-only models (e.g. Qwen3.5 35B-A3B) have no dense FFN layers, so
    # ffn_hidden_size is None. Set to 0 so FLOPS calculation doesn't crash.
    if getattr(standard_provider, "ffn_hidden_size", None) is None:
        standard_provider.ffn_hidden_size = 0

    if args.enable_vision_cp:
        standard_provider.vision_dp_when_cp = True
        logger.info("Vision CP enabled: vision_dp_when_cp=True")

    if args.enable_vision_cuda_graph:
        standard_provider.vision_cuda_graph_impl = "transformer_engine"
        if args.vision_cuda_graph_scope is not None:
            standard_provider.vision_cuda_graph_scope = [
                s.strip() for s in args.vision_cuda_graph_scope.split(",") if s.strip()
            ]
        else:
            standard_provider.vision_cuda_graph_scope = []
        if args.max_vision_cuda_graph_seq_length is not None:
            standard_provider.max_vision_cuda_graph_seq_length = args.max_vision_cuda_graph_seq_length
        standard_provider.use_te_rng_tracker = True
        logger.info("Vision CUDA graph enabled: vision_cuda_graph_impl=transformer_engine")

    _apply_moe_config(standard_provider, parallelism_config, args)

    # GDN linear CP mode (chunkwise/headwise). "legacy" is the historical recipe label for
    # the zigzag + all-to-all path, which upstream Megatron calls "headwise"; normalize it so
    # Megatron's TransformerConfig only ever sees community values (it rejects "legacy").
    if args.linear_cp_mode is not None:
        linear_cp_mode = "headwise" if args.linear_cp_mode == "legacy" else args.linear_cp_mode
        standard_provider.linear_cp_mode = linear_cp_mode
        logger.info(f"Set linear_cp_mode={linear_cp_mode}")

    # GDN fused pre-gated-delta-rule (streamed Triton). Perf-only; requires causal-conv1d and is
    # non-deterministic, so it is off by default and never affects the validated unfused path.
    if args.gdn_fused_pre_gdr:
        if not hasattr(standard_provider, "gdn_pre_gated_delta_rule_fusion"):
            raise RuntimeError(
                "--gdn-fused-pre-gdr needs the gdn_pre_gated_delta_rule_fusion field; apply "
                "dockers/patches/qwen35-gdn-fused-pre-gdr.patch and rebuild the image."
            )
        standard_provider.gdn_pre_gated_delta_rule_fusion = True
        logger.info("Set gdn_pre_gated_delta_rule_fusion=True")

    provider = MegatronMIMOProvider.from_standard_provider(
        standard_provider=standard_provider,
        megatron_mimo_parallelism_config=parallelism_config,
    )
    provider.use_cpu_initialization = True
    provider.bf16 = True
    provider.fp16 = False

    return provider


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(
    model_provider: MegatronMIMOProvider,
    args: argparse.Namespace,
) -> ConfigContainer:
    """Build ConfigContainer for MIMO bench training."""
    language_parallelism = model_provider.megatron_mimo_parallelism_config.module_parallelisms.get(
        MIMO_LANGUAGE_MODULE_KEY
    )
    language_dp = language_parallelism.data_parallel_size if language_parallelism else 1
    # mbs is per-rank (same semantic as non-MIMO). Internally, MIMO loads
    # mbs_total = mbs * language_dp samples per microbatch and slices by language_dp.
    mbs_total = args.mbs * language_dp
    gbs = args.gbs if args.gbs is not None else mbs_total * args.ga
    enable_copy_engine = os.getenv("ENABLE_COPY_ENGINE", "0") == "1"
    if enable_copy_engine and args.no_nccl_ub:
        raise ValueError("--no-nccl-ub cannot be used with ENABLE_COPY_ENGINE=1")

    optimizer_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=5e-6,
        min_lr=5e-7,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        clip_grad=1.0,
        start_weight_decay=0.033,
        end_weight_decay=0.033,
    )

    logger_cfg = LoggerConfig()
    logger_cfg.log_interval = args.log_interval
    logger_cfg.log_timers_to_tensorboard = True
    if args.output_dir:
        logger_cfg.tensorboard_dir = args.output_dir

    profiling_cfg = ProfilingConfig()
    if args.enable_profiler:
        profiling_cfg.use_pytorch_profiler = True
        profiling_cfg.profile_step_start = args.profile_step_start
        profiling_cfg.profile_step_end = args.profile_step_end
        profiling_cfg.pytorch_profiler_collect_shapes = True
        profiling_cfg.pytorch_profiler_collect_callstack = True
        profiling_cfg.profile_ranks = [0]

    if args.enable_memory_snapshot:
        profiling_cfg.record_memory_history = True
        profiling_cfg.memory_snapshot_path = args.memory_snapshot_path
        profiling_cfg.profile_step_start = args.profile_step_start
        profiling_cfg.profile_step_end = args.profile_step_end
        profiling_cfg.profile_ranks = [0]

    cfg = ConfigContainer(
        train=TrainingConfig(
            micro_batch_size=mbs_total,
            global_batch_size=gbs,
            train_iters=args.train_iters,
            eval_interval=None,
            eval_iters=None,
            manual_gc=True,
            manual_gc_interval=100,
        ),
        model=model_provider,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        dataset=None,
        logger=logger_cfg,
        tokenizer=TokenizerConfig(),
        checkpoint=CheckpointConfig(
            save=None,
            save_interval=0,
            ckpt_format="torch_dist",
        ),
        profiling=profiling_cfg,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=args.enable_grad_overlap,
            overlap_param_gather=args.enable_grad_overlap,
            average_in_collective=True,
            data_parallel_sharding_strategy="optim_grads_params",
            use_distributed_optimizer=True,
            nccl_ub=enable_copy_engine,
        ),
    )
    cfg.mixed_precision = "bf16_mixed"
    cfg.dataset = types.SimpleNamespace(seq_length=args.seq_length)

    # -- Apply advanced features --
    if args.enable_sp or args.enable_tp_overlap:
        model_provider.standard_provider.sequence_parallel = True

    if args.enable_tp_overlap:
        std = model_provider.standard_provider
        if std is not None:
            std.sequence_parallel = True
        if cfg.comm_overlap is None:
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
        else:
            cfg.comm_overlap.tp_comm_overlap = True

    if args.enable_full_recompute:
        model_provider.standard_provider.recompute_granularity = "full"
        model_provider.standard_provider.recompute_method = "uniform"
        model_provider.standard_provider.recompute_num_layers = 1

    if args.enable_recompute:
        model_provider.standard_provider.recompute_granularity = "selective"
        model_provider.standard_provider.recompute_modules = ["core_attn"]

    if args.enable_cuda_graph_te:
        model_provider.standard_provider.cuda_graph_impl = "transformer_engine"
        # Determine cuda_graph_scope: explicit CLI arg takes priority, otherwise auto-detect.
        if args.cuda_graph_scope:
                    scope = [s.strip() for s in args.cuda_graph_scope.split(",") if s.strip()]
        else:
            num_experts = getattr(model_provider.standard_provider, "num_moe_experts", None)
            scope = ["attn", "moe_router", "moe_preprocess"] if num_experts else ["attn"]
        model_provider.standard_provider.cuda_graph_scope = scope
        model_provider.standard_provider.cuda_graph_warmup_steps = args.cuda_graph_warmup
        model_provider.standard_provider.use_te_rng_tracker = True

    if args.enable_full_iteration_graph:
        model_provider.standard_provider.cuda_graph_impl = "full_iteration"
        model_provider.standard_provider.cuda_graph_warmup_steps = args.cuda_graph_warmup
        model_provider.standard_provider.use_te_rng_tracker = True
        cfg.ddp.check_for_nan_in_grad = False
        cfg.rerun_state_machine.check_for_nan_in_loss = False

    if args.moe_expert_rank_capacity_factor is not None:
        model_provider.standard_provider.moe_expert_rank_capacity_factor = args.moe_expert_rank_capacity_factor
        model_provider.standard_provider.moe_paged_stash = True
        model_provider.standard_provider.use_transformer_engine_op_fuser = True
        model_provider.standard_provider.moe_flex_dispatcher_backend = "hybridep"

    if args.enable_full_offload:
        model_provider.standard_provider.cpu_offloading = True
        model_provider.standard_provider.cpu_offloading_num_layers = 31
        model_provider.standard_provider.cpu_offloading_activations = True
        model_provider.standard_provider.cpu_offloading_weights = False

    if args.enable_fine_grained_offload:
        model_provider.standard_provider.fine_grained_activation_offloading = True
        model_provider.standard_provider.offload_modules = [
            "core_attn",
            "qkv_linear",
            "attn_proj",
            "attn_norm",
            "mlp_norm",
        ]

    if args.enable_fp8:
        cfg.mixed_precision = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            fp8="te",
            fp8_recipe="tensorwise",
            fp8_param_gather=True,
        )

    if args.enable_vision_activation_offload:
        model_provider.vision_activation_offload = True
        if args.vision_offload_num_layers is not None:
            model_provider.vision_activation_offload_num_layers = args.vision_offload_num_layers

    if args.cross_entropy_fusion_impl is not None:
        model_provider.standard_provider.cross_entropy_fusion_impl = (
            args.cross_entropy_fusion_impl
        )
        logger.info(
            "cross_entropy_fusion_impl overridden to "
            f"'{args.cross_entropy_fusion_impl}' "
            "(cross_entropy_loss_fusion="
            f"{model_provider.standard_provider.cross_entropy_loss_fusion})"
        )

    if args.disable_cross_entropy_fusion:
        model_provider.standard_provider.cross_entropy_loss_fusion = False
        logger.info(
            "cross_entropy_loss_fusion forced off by --disable-cross-entropy-fusion"
        )

    # -- CP > 1 required configuration --
    language_cp = language_parallelism.context_parallel_size if language_parallelism else 1
    if language_cp > 1:
        model_provider.standard_provider.calculate_per_token_loss = True
        model_provider.standard_provider.cross_entropy_loss_fusion = False
        cfg.ddp.average_in_collective = False
        assert args.seq_length % (language_cp * 2) == 0, (
            f"seq_length ({args.seq_length}) must be divisible by 2 * context_parallel_size ({language_cp})"
        )
        logger.info(
            f"CP={language_cp} enabled for language module: "
            f"calculate_per_token_loss=True, cross_entropy_loss_fusion=False, "
            f"average_in_collective=False"
        )
        if args.enable_vision_cp:
            logger.info(
                f"Vision CP enabled: images will be split across {language_cp} CP ranks "
                f"and all-gathered after the vision encoder."
            )

    return cfg


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
        description="Qwen3.5-VL 35B-A3B (MoE) MIMO benchmark with MockPerfVLMProvider + vision CP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # MIMO component layout
    p.add_argument(
        "--component",
        action="append",
        default=None,
        help="Component layout: name=tp=N[,pp=N,dp=N,etp=N,cp=N,rank_offset=N]. "
        "Use etp (or ep) for expert parallelism in the language module.",
    )

    # Model
    p.add_argument("--hf-model", type=str, default="Qwen/Qwen3.5-35B-A3B", help="HF model id or local config path")
    p.add_argument("--seq-length", type=int, default=4096)
    p.add_argument(
        "--mbs",
        type=int,
        default=1,
        help="Per-rank micro batch size (same as non-MIMO mbs; MIMO total = mbs * language_dp)",
    )
    p.add_argument("--ga", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument(
        "--gbs",
        type=int,
        default=None,
        help="Global batch size (default: mbs * language_dp * ga, i.e. mbs_total * ga)",
    )
    p.add_argument("--train-iters", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--output-dir", default=None, help="TensorBoard output directory")
    p.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow training without pretrained checkpoint (perf-only).",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override text_config.num_hidden_layers for model scaling (e.g. 40 -> 20).",
    )
    p.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Override text_config.num_experts for model scaling (e.g. 256 -> 128). Must be divisible by language EP.",
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
        help="Number of SMs for DeepEP communication kernels (default: DeepEP library default)",
    )
    moe.add_argument(
        "--moe-shared-expert-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overlap shared expert compute with dispatch A2A (mutually exclusive with EP overlap)",
    )
    moe.add_argument(
        "--moe-permute-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse token permutation ops during dispatch (uses fused_indices_to_multihot + fused_permute_with_probs)",
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
        help="Auto-tune DeepEP dispatch/combine chunk sizes at startup (benchmarks on real hardware)",
    )

    # Mock dataset (MockPerfVLMProvider)
    ds = p.add_argument_group("mock VLM dataset (MockPerfVLMProvider)")
    ds.add_argument("--num-images", type=int, default=2, help="Number of mock images per sample.")
    ds.add_argument("--image-size-w", type=int, default=384, help="Mock image width.")
    ds.add_argument("--image-size-h", type=int, default=384, help="Mock image height.")
    ds.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers.")

    # Vision CP
    cp = p.add_argument_group("vision context parallelism")
    cp.add_argument(
        "--enable-vision-cp",
        action="store_true",
        help="Enable vision_dp_when_cp: split images across CP ranks in the vision encoder. "
        "Only effective when the language component has cp > 1.",
    )

    # Advanced features
    feat = p.add_argument_group("advanced features")
    feat.add_argument("--enable-sp", action="store_true", help="Sequence Parallel")
    feat.add_argument(
        "--linear-cp-mode",
        type=str,
        default=None,
        choices=[None, "chunkwise", "headwise", "legacy"],
        help="GDN CP strategy: chunkwise (FLA cp_context, no all-to-all), headwise (Ulysses), legacy (zigzag+a2a)",
    )
    feat.add_argument(
        "--gdn-fused-pre-gdr",
        action="store_true",
        help="Enable the streamed fused pre-gated-delta-rule Triton path. Requires causal-conv1d "
        "(cpp_functions.causal_conv1d_bwd_function). Non-deterministic; mutually exclusive with "
        "deterministic_mode. Perf-only: numerics differ slightly from the unfused path.",
    )
    feat.add_argument("--enable-tp-overlap", action="store_true", help="TP comm overlap (auto-enables SP)")
    feat.add_argument("--enable-grad-overlap", action="store_true", help="Overlap grad-reduce + param-gather")
    feat.add_argument(
        "--enable-ep-overlap",
        action="store_true",
        help="Overlap DeepEP dispatch/combine with MoE compute (MIMO schedule plan; "
        "requires pure text and language TP=CP=PP=1)",
    )
    feat.add_argument(
        "--delay-wgrad",
        action="store_true",
        help="Delay weight-gradient computation inside the EP overlap schedule",
    )
    feat.add_argument(
        "--no-nccl-ub",
        action="store_true",
        help="Keep NCCL userbuffers disabled; incompatible with ENABLE_COPY_ENGINE=1.",
    )
    feat.add_argument("--enable-full-recompute", action="store_true")
    feat.add_argument("--enable-recompute", action="store_true")
    feat.add_argument(
        "--enable-cuda-graph-te", action="store_true", help="CUDA graph TE scoped (attn + moe_router + moe_preprocess)"
    )
    feat.add_argument(
        "--enable-full-iteration-graph",
        action="store_true",
        help="Full iteration CUDA graph (entire fwd+bwd as one graph)",
    )
    feat.add_argument(
        "--cuda-graph-warmup",
        type=int,
        default=3,
        help="Eager iterations per module before CUDA graph capture. The profiling "
        "window is aligned to these steps, so both must come from the same value.",
    )
    feat.add_argument(
        "--moe-expert-rank-capacity-factor",
        type=float,
        default=None,
        help="Expert rank capacity factor for static MoE buffer sizing. Enables paged stashing for CUDA graph compatibility.",
    )
    feat.add_argument("--enable-full-offload", action="store_true")
    feat.add_argument("--enable-fine-grained-offload", action="store_true")
    feat.add_argument("--enable-vision-activation-offload", action="store_true",
                       help="Offload vision encoder activations to CPU via TE's native "
                       "cpu_offloading.  Each vision layer's saved-for-backward tensors "
                       "are offloaded with per-layer compute-transfer overlap.  "
                       "Automatically disables activation recomputation on the vision encoder.")
    feat.add_argument("--vision-offload-num-layers", type=int, default=None,
                       help="Number of vision encoder layers to offload (default: num_layers - 1).")
    feat.add_argument(
        "--cross-entropy-fusion-impl",
        type=str,
        choices=["native", "te"],
        default=None,
        help="Override model.cross_entropy_fusion_impl. 'te' uses TransformerEngine's "
        "triton cross entropy, which keeps logits in bf16 and saves only the in-place "
        "softmaxed logits for backward instead of several fp32 logits-sized buffers. "
        "Ignored when language CP > 1 (fusion is disabled there).",
    )
    feat.add_argument(
        "--disable-cross-entropy-fusion",
        action="store_true",
        help="Force model.cross_entropy_loss_fusion=False as well at CP=1. "
        "Diagnostic control for the CP sweep: CP>1 already disables fusion, "
        "so this separates the fusion loss from the CP cost.",
    )
    feat.add_argument("--enable-fp8", action="store_true")
    feat.add_argument(
        "--cuda-graph-scope",
        type=str,
        default=None,
        help="Comma-separated CUDA graph scope modules (e.g. 'attn' for dense, 'attn,moe_router,moe_preprocess' for MoE). "
        "If omitted, auto-detects based on model MoE config.",
    )
    feat.add_argument(
        "--enable-vision-cuda-graph",
        action="store_true",
        help="Enable CUDA graph capture for the vision encoder (full-layer by default).",
    )
    feat.add_argument(
        "--vision-cuda-graph-scope",
        type=str,
        default=None,
        help="Comma-separated vision CUDA graph scope (default: full-layer). "
        "Examples: 'attn', 'attn,mlp'.",
    )
    feat.add_argument(
        "--max-vision-cuda-graph-seq-length",
        type=int,
        default=None,
        help="Max vision sequence length for CUDA graph padding (default: auto).",
    )
    # Profiling
    pr = p.add_argument_group("profiling")
    pr.add_argument("--enable-profiler", action="store_true")
    pr.add_argument("--enable-memory-snapshot", action="store_true", help="Dump memory snapshot at profile_step_end")
    pr.add_argument("--memory-snapshot-path", type=str, default="/tmp/mem_snapshot.pickle")
    pr.add_argument("--profile-step-start", type=int, default=5)
    pr.add_argument("--profile-step-end", type=int, default=6)

    # Debug
    dbg = p.add_argument_group("debug")
    dbg.add_argument(
        "--debug-deepep",
        action="store_true",
        help="Log DeepEP dispatch/combine parameters on the first N calls (rank 0 only)",
    )
    dbg.add_argument(
        "--debug-deepep-log-calls",
        type=int,
        default=4,
        help="Number of dispatch/combine calls to log (default: 4, covers 2 fwd+bwd steps)",
    )

    return p


# ---------------------------------------------------------------------------
# DeepEP debug logging
# ---------------------------------------------------------------------------


def _patch_deepep_debug_logging(max_calls: int = 4) -> None:
    """Monkey-patch fused_a2a to log DeepEP dispatch/combine parameters on rank 0."""
    try:
        from megatron.core.transformer.moe import fused_a2a, token_dispatcher
    except ImportError:
        return

    orig_dispatch = getattr(fused_a2a, "fused_dispatch", None)
    orig_combine = getattr(fused_a2a, "fused_combine", None)
    if orig_dispatch is None or orig_combine is None:
        return

    call_counts = {"dispatch": 0, "combine": 0}

    def _logged_dispatch(x, token_indices, token_probs, num_experts, group, **kwargs):
        call_counts["dispatch"] += 1
        call_id = call_counts["dispatch"]
        rank = dist.get_rank()
        if call_id <= max_calls:
            send_counts = torch.zeros(num_experts, dtype=torch.long, device=token_indices.device)
            for col in range(token_indices.shape[1]):
                valid = token_indices[:, col] >= 0
                if valid.any():
                    send_counts.scatter_add_(
                        0, token_indices[:, col][valid], torch.ones_like(token_indices[:, col][valid])
                    )
            local_experts_per_rank = num_experts // group.size()
            my_rank_in_group = dist.get_group_rank(group, rank)
            my_node = my_rank_in_group // 8
            remote_tokens = 0
            for r in range(group.size()):
                if r // 8 != my_node:
                    start_e = r * local_experts_per_rank
                    end_e = start_e + local_experts_per_rank
                    remote_tokens += send_counts[start_e:end_e].sum().item()
            total_tokens = send_counts.sum().item()
            remote_bytes = remote_tokens * x.shape[1] * max(x.element_size(), 2)
            logger.info(
                f"[DeepEP DEBUG dispatch #{call_id} rank={rank}] "
                f"x.shape={list(x.shape)}, x.dtype={x.dtype}, "
                f"token_indices.shape={list(token_indices.shape)}, "
                f"num_experts={num_experts}, group_size={group.size()}, "
                f"topk={token_indices.shape[1]}, "
                f"total_send={total_tokens}, remote_send={remote_tokens}, "
                f"remote_bytes={remote_bytes}, remote_ratio={remote_tokens / max(total_tokens, 1):.1%}, "
                f"tokens_per_expert_send={send_counts.tolist()}, "
                f"kwargs={kwargs}",
            )
        return orig_dispatch(x, token_indices, token_probs, num_experts, group, **kwargs)

    def _logged_combine(x, group, handle, **kwargs):
        call_counts["combine"] += 1
        call_id = call_counts["combine"]
        rank = dist.get_rank()
        if call_id <= max_calls:
            logger.info(
                f"[DeepEP DEBUG combine #{call_id} rank={rank}] "
                f"x.shape={list(x.shape)}, x.dtype={x.dtype}, "
                f"group_size={group.size()}, "
                f"kwargs={kwargs}",
            )
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        result = orig_combine(x, group, handle, **kwargs)
        t1.record()
        if call_id <= max_calls:
            torch.cuda.synchronize()
            logger.info(
                f"[DeepEP DEBUG combine #{call_id} rank={rank}] elapsed={t0.elapsed_time(t1):.2f}ms",
            )
        return result

    fused_a2a.fused_dispatch = _logged_dispatch
    fused_a2a.fused_combine = _logged_combine
    token_dispatcher.fused_dispatch = _logged_dispatch
    token_dispatcher.fused_combine = _logged_combine
    if dist.get_rank() == 0:
        logger.info(f"[DeepEP DEBUG] Patched fused_dispatch/fused_combine (logging first {max_calls} calls)")


def _apply_hydra_overrides(args: argparse.Namespace, overrides: list[str]) -> None:
    """Apply Hydra-style key=value overrides to an argparse Namespace.

    Keys must match argparse dest names (e.g. ``seq_length``, ``enable_sp``,
    ``hf_model``).  Values are coerced to the type of the current default.
    ``component`` can be repeated to build the component list.
    """
    seen_keys: set[str] = set()
    for override in overrides:
        if "=" not in override:
            raise argparse.ArgumentError(None, f"Invalid override (expected key=value): {override}")
        key, _ = override.split("=", 1)
        seen_keys.add(key)

    # If the user overrides --component, replace the list built by the script.
    if "component" in seen_keys and args.component is not None:
        args.component.clear()

    for override in overrides:
        key, value = override.split("=", 1)
        if not hasattr(args, key):
            raise argparse.ArgumentError(None, f"Unknown config key: {key}")
        current = getattr(args, key)

        if key == "component":
            if value.startswith("[") and value.endswith("]"):
                setattr(args, key, [v.strip().strip('"').strip("'") for v in value[1:-1].split(",")])
            else:
                components = list(args.component) if args.component is not None else []
                components.append(value)
                setattr(args, key, components)
        elif isinstance(current, list):
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
                setattr(args, key, v_lower in ("true", "yes"))
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
    """Entry point for the MIMO benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args, cli_overrides = parser.parse_known_args()
    if cli_overrides:
        _apply_hydra_overrides(args, cli_overrides)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if args.enable_cuda_graph_te or args.enable_full_iteration_graph:
        torch.cuda.set_stream(torch.cuda.Stream())
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    set_experimental_flag(True)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.info(f"Distributed initialized: rank={rank}, world_size={world_size}")

    try:
        logger.info(f"Loading HF config from {args.hf_model}")
        hf_config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=True)

        text_config = getattr(hf_config, "text_config", hf_config)
        num_experts = getattr(text_config, "num_experts", None) or getattr(text_config, "num_local_experts", None)
        logger.info(f"Model architecture: num_experts={num_experts}")

        components = args.component or G_DEFAULT_COMPONENTS
        parallelism_config = _build_parallelism_config(components, world_size)
        logger.info(f"Component layout: {components}")

        _patch_vision_model_for_mimo()

        if args.debug_deepep:
            _patch_deepep_debug_logging(max_calls=args.debug_deepep_log_calls)

        logger.info("Building Qwen3.5-VL 35B-A3B MegatronMIMO provider (MoE, MockPerfVLMProvider)")
        model_provider = _build_mimo_provider(hf_config, parallelism_config, args)

        logger.info("Building training config")
        cfg = build_config(model_provider, args)

        logger.info("Launching pretrain_megatron_mimo")
        install_model_structure(source=os.path.basename(__file__))
        pretrain_megatron_mimo(
            cfg=cfg,
            forward_step_func=_flops_forward_step(megatron_mimo_forward_step),
            build_data_iterators_fn=_make_build_data_iterators(args),
        )
        logger.info("MIMO MoE bench with MockPerfVLMProvider PASSED")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
