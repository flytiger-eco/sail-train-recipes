# Alpamayo 2 Super Best Practices

Stage-1 VLM supervised fine-tuning recipes for **Alpamayo 2 Super** (34B vision-language-action
model: 32B VLM backbone + 2B diffusion expert) on **Zhenwu M890** (144 GiB/card).

The recipe trains the release-native [`nvidia/Alpamayo2-Super`](https://huggingface.co/nvidia/Alpamayo2-Super)
checkpoint on the [`nvidia/PhysicalAI-Autonomous-Vehicles`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
(PAI) dataset with Hugging Face Trainer + DeepSpeed ZeRO-3. It reproduces NVIDIA's
[`alpamayo-recipes`](https://github.com/NVlabs/alpamayo-recipes) `alpamayo2_sft` Stage-1 workflow and
adds a set of Zhenwu-tuned runtime optimizations.

> **Upstream sources are never edited in-tree.** Every change ships as a patch under
> [`../../dockers/patches/`](../../dockers/patches) and is applied with `git apply` at image-build
> time. The optimizations themselves are runtime monkey-patches registered through hydra callbacks,
> so the pinned upstream commits stay byte-for-byte intact.

## Files

| File | Description |
|------|-------------|
| `train_alpamayo2_super_full.sh` | Stage-1 VLM SFT launcher (single-node 8 GPU / dual-node 16 GPU), best-perf defaults |
| [`../../dockers/Dockerfile.Alpamayo2Super`](../../dockers/Dockerfile.Alpamayo2Super) | Image build: clones both repos at pinned commits, applies the two patches, builds the venv |
| [`../../dockers/patches/alpamayo2_recipes.patch`](../../dockers/patches/alpamayo2_recipes.patch) | 11-file patch to `alpamayo-recipes` (4 modified + 7 new callback modules) |
| [`../../dockers/patches/alpamayo2_core.patch`](../../dockers/patches/alpamayo2_core.patch) | 1-file patch to `alpamayo2` (`attn_implementation` propagation) |

## Performance Results

### Zhenwu M890 (144 GiB/card)

Common experiment setup:
- Model `nvidia/Alpamayo2-Super`, Stage-1 VLM cross-entropy SFT, BF16, expert frozen.
- Data PAI chunks 0-98 train / 99 validation, 7 cameras x 4 frames, `max_length=6144`, dynamic padding.
- DeepSpeed ZeRO-3 (no offload), `MBS=1`, `GA=16` (recipe default; global batch 128 on 8 GPU),
  FlashAttention-3 + all fusion switches ON, 4 comm channels, `expandable_segments:True`.
- Stack: torch 2.10 / CUDA 13 / deepspeed 0.19.1 / TransformerEngine 2.16 / triton 3.6.

| Topology | Global batch | Selective recompute (K of 64 text layers) | Step time | Throughput | Peak reserved / card (alloc) | Device peak (headroom) | Loss |
|---|---|---|---|---|---|---|---|
| 8 GPU (1 node)  | 128 | **K48** (default) | **97.95 s** | 1.31 samples/s | 101.6 GiB (94.8 GiB alloc) | 102.5 GiB (41.5 GiB) | 1.785 |
| 8 GPU (1 node)  | 128 | **K28** (faster)  | **92.26 s** | 1.39 samples/s | 125.3 GiB (119.1 GiB alloc) | 126.2 GiB (17.8 GiB) | 1.785 |
| 16 GPU (2 nodes)| 256 | **K0**  | GA=1: **5.72 s** (GA=16 pending re-measure) | 2.80 samples/s (GA=1) | 125.6 GiB (GA=1) | n/a | 1.816 (GA=1) |

- Step time is the steady-state median over the run (cold-start and export-spike steps excluded), measured
  at the recipe-default `GA=16`: one step = one optimizer step over 16 micro-batches (global batch 128 on 8 GPU).
- **K48 vs K28 (8 GPU)**: K28 recomputes 20 fewer text layers, so it is **5.8% faster** (92.26 vs 97.95 s/step)
  but holds ~24 GiB more activations (device peak 126.2 vs 102.5 GiB, headroom 17.8 vs 41.5 GiB). Both fit the
  144 GiB card; the script defaults to **K48** (safest headroom). Pin `CKPT_LAYERS=28` to trade headroom for speed.
- `GA=16` amortizes the ZeRO-3 optimizer step (param all-gather + reduce-scatter + Adam) over 16 micro-batches,
  so it reaches a **higher** per-sample throughput (1.31 samples/s at K48) than the old `GA=1` benchmark
  (1.22 samples/s) despite the larger per-step wall time: the optimizer cost is paid once per step, not per micro-batch.
- The 16 GPU row is the prior `GA=1` measurement (5.72 s/step, global batch 16, MFU ~50.5% FLOP-model estimate).
  At the new `GA=16` default its global batch becomes 256 and the step time must be **re-measured on a 2-node
  cluster** (this recipe's tuning host is a single 8-GPU node), so it is left as a `GA=1` reference for now.
- `K` = `CKPT_LAYERS` (how many of the 64 text layers stay recomputed); the script auto-selects
  **K48 for 1 node** and **K0 for >= 2 nodes**. Override with `CKPT_LAYERS=<0..64>`.

## Quick Start

### 1. Build the image & launch a container

The Dockerfile pins both upstream repos (`alpamayo-recipes @ 3f6504e2`, `alpamayo2 @ 9596749e`),
applies the two patches, and layers a `--system-site-packages` venv on the image's pre-baked
compiled stack (torch 2.10 / FA3 / TE 2.16 / triton 3.6) so nothing is rebuilt or version-bumped.

```bash
# Run from the repo root of sail-train-recipes
docker build -f dockers/Dockerfile.Alpamayo2Super -t alpamayo2-super:zhenwu .

# Single node (8 GPU)
docker run --gpus all --ipc=host --network=host \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 --init -dt \
  -v /path/to/models:/models -v /path/to/data:/data \
  --name=alpamayo2_sail_train \
  alpamayo2-super:zhenwu /bin/bash
```

For **dual-node RDMA**, additionally pass the InfiniBand devices and lock memory so cross-node
collectives use RDMA instead of falling back to TCP (about 6x slower):

```bash
docker run --gpus all --ipc=host --network=host \
  --device=/dev/infiniband --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 --init -dt \
  -v /path/to/models:/models -v /path/to/data:/data -v /shared:/shared \
  --name=alpamayo2_sail_train \
  alpamayo2-super:zhenwu /bin/bash
```

RDMA is active when all three hold inside the container: `ls /dev/infiniband` shows `uverbs*`,
`ulimit -l` is `unlimited`, and the training log prints `Using network IB` (non-zero `NET/IB` counts).

### 2. Download the model

`nvidia/Alpamayo2-Super` is a gated repository; authenticate first.

```bash
export HF_TOKEN=<your Hugging Face token>
hf auth login && hf auth whoami

huggingface-cli download nvidia/Alpamayo2-Super --local-dir /models/Alpamayo2-Super
export MODEL_PATH=/models/Alpamayo2-Super
```

A local snapshot must include `config.json`, the tokenizer/processor files, and all safetensors shards.

### 3. Download & prepare the data (PAI chunks 0-99)

The recipe ships `scripts/download_pai.py`. Download chunks 0-99 (half-open `0-100`), all seven
cameras, egomotion labels, and calibration:

```bash
cd /workspace/alpamayo-recipes
export DATA_PATH=/data/PhysicalAI-Autonomous-Vehicles

python scripts/download_pai.py \
  --chunk-ids 0-100 \
  --camera \
    camera_cross_left_120fov \
    camera_front_wide_120fov \
    camera_cross_right_120fov \
    camera_rear_left_70fov \
    camera_rear_tele_30fov \
    camera_rear_right_70fov \
    camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics \
  --labels egomotion \
  --output-dir "$DATA_PATH"
```

Chunk ranges are half-open: `0-100` downloads chunks 0-99. Training then uses `chunk_ids="0-99"`
(chunks 0-98) and validation uses `chunk_ids="99-100"` (held-out chunk 99).

### 4. Verify the data split

Run this before allocating GPUs (expects 9,881 train clips and 100 validation clips):

```bash
python - <<'PY'
import os
from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface
root = os.environ["DATA_PATH"]
train = PhysicalAIAVDatasetLocalInterface(root, chunk_ids="0-99")
val = PhysicalAIAVDatasetLocalInterface(root, chunk_ids="99-100")
print("train clips:", len(train.get_all_clip_ids()))   # expect 9881
print("val clips:", len(val.get_all_clip_ids()))        # expect 100
PY
```

### 5. Launch training

Copy `train_alpamayo2_super_full.sh` into the container (or mount this repo), then run it from
`best_practice/alpamayo2Super/`:

```bash
export MODEL_PATH=/models/Alpamayo2-Super
export DATA_PATH=/data/PhysicalAI-Autonomous-Vehicles

bash train_alpamayo2_super_full.sh
```

The defaults ARE the measured best config (ZeRO-3, all fusion switches ON, `CKPT_LAYERS` auto,
4 channels, `expandable_segments`, `MBS=1`/`GA=16`, `WARMUP_STEPS=4000`, `STOP_AFTER=30`, `SAVE_CKPT=0`) -
no extra performance flags needed. Preview the fully-resolved launch command any time with `DRY_RUN=1`.

To do a full training run (recipe-default schedule: MAX_STEPS=100000, warmup 4000 + cosine decay)
with checkpointing instead of a 30-step benchmark:

```bash
STOP_AFTER=0 SAVE_CKPT=1 SAVE_STEPS=500 \
OUTPUT_DIR=/shared/outputs/alpamayo2_stage1 \
bash train_alpamayo2_super_full.sh
```

Expected signals (matches upstream): finite Stage-1 total loss starting around 1.8 and trending down,
finite `grad_norm`, linear LR warmup, and the 9881/100 data split above.

## Single-node training (8 GPU, best performance)

The default invocation is the 8-GPU best config (`CKPT_LAYERS` auto-selects **K48**, `GA=16`, global batch 128):

```bash
export MODEL_PATH=/models/Alpamayo2-Super
export DATA_PATH=/data/PhysicalAI-Autonomous-Vehicles

NUM_GPUS=8 \
bash train_alpamayo2_super_full.sh
```

Measured (GA=16): **97.95 s/step**, 1.31 samples/s, peak reserved 101.6 GiB/card (94.8 GiB allocated),
device peak 102.5 GiB (41.5 GiB headroom). For ~5.8% more speed at the cost of headroom, pin
`CKPT_LAYERS=28`: **92.26 s/step**, 1.39 samples/s, device peak 126.2 GiB (17.8 GiB headroom).

## Dual-node training (16 GPU, best performance)

Multi-node is bound to the network fabric, so set the bootstrap NIC and RDMA HCAs per your cluster.
`CKPT_LAYERS` auto-selects **K0** when `NNODES >= 2` (global batch 256 at `GA=16`).

> **`OUTPUT_DIR` must be the SAME shared path on both nodes.** ZeRO-3 reassembles a full checkpoint
> only if all 16 ranks write one `paths.output_dir`; the script points the trainer at the shared
> `OUTPUT_DIR` and splits per-node logs / hydra / torchrun into `OUTPUT_DIR/node${NODE_RANK}/`.

Node 0 (master):
```bash
export MODEL_PATH=/models/Alpamayo2-Super
export DATA_PATH=/data/PhysicalAI-Autonomous-Vehicles

NNODES=2 NODE_RANK=0 MASTER_ADDR=<MASTER_NODE_IP> MASTER_PORT=29517 \
NCCL_SOCKET_IFNAME=<bootstrap-nic> NCCL_IB_HCA=<rdma-hca-0>,<rdma-hca-1>,<rdma-hca-2>,<rdma-hca-3> \
OUTPUT_DIR=/shared/outputs/alpamayo2_stage1 \
bash train_alpamayo2_super_full.sh
```

Node 1 (identical except `NODE_RANK=1`):
```bash
export MODEL_PATH=/models/Alpamayo2-Super
export DATA_PATH=/data/PhysicalAI-Autonomous-Vehicles

NNODES=2 NODE_RANK=1 MASTER_ADDR=<MASTER_NODE_IP> MASTER_PORT=29517 \
NCCL_SOCKET_IFNAME=<bootstrap-nic> NCCL_IB_HCA=<rdma-hca-0>,<rdma-hca-1>,<rdma-hca-2>,<rdma-hca-3> \
OUTPUT_DIR=/shared/outputs/alpamayo2_stage1 \
bash train_alpamayo2_super_full.sh
```

`MASTER_ADDR` may be a hostname or an `N-N-N-N` style node name; the script resolves it to an IPv4
address. Prior `GA=1` measurement: **5.72 s/step**, 2.80 samples/s, peak reserved 125.6 GiB/card, MFU ~50.5%.
At the new `GA=16` default the global batch becomes 256; the step time is **pending re-measurement on a
2-node cluster** (this recipe's tuning host is a single 8-GPU node).

## Patches applied (what & why)

Two patches, applied by the Dockerfile with `git apply` right after cloning the pinned commits.
Nothing is edited in-tree; remove the `COPY`/`git apply` lines and you get pristine upstream.

### `alpamayo2_recipes.patch` (11 files: 4 modified + 7 new)

Modified upstream files:

| File | Change | Why |
|------|--------|-----|
| `recipes/alpamayo2_sft/configs/sft_base.yaml` | `pad_to_fixed_length: true -> false` | Dynamic pad to batch-longest instead of fixed 6144: ~6% faster step, ~7 GiB lower peak, math-equivalent loss (padding is masked). |
| `recipes/alpamayo2_sft/models/trainable_alpamayo2_super.py` | add `stage1_supervised_logits` | Optional compact Stage-1 logits projection: run the LM head only on supervised positions. Off by default in the measured best. |
| `recipes/alpamayo2_sft/tests/test_trainable_alpamayo2_super.py` | tests for the above | Cover the compact projection path. |
| `recipes/alpamayo2_sft/train_hf.py` | wire the `callbacks.*` config group | Lets `+callbacks.<name>._target_=...` overrides instantiate and register with the Trainer. |

New callback modules (runtime monkey-patches, all under the `alpamayo2_sft` package):

| Module | Callback | Effect (enabled via) |
|--------|----------|----------------------|
| `stage1_triton_rope.py` | `TritonRoPECallback` | Fused rotary embedding for text + vision, bit-identical to HF on the `triton` backend (`TRITON_ROPE`, `TRITON_ROPE_BACKEND`). |
| `stage1_te_rmsnorm.py` | `TERMSNormCallback` | TransformerEngine fused RMSNorm fwd/bwd for every norm site (input/post-attn LN, q/k norm, final) (`TE_RMSNORM`). |
| `stage1_vision_fa3_varlen.py` | `VisionFA3VarlenCallback` | Vision tower uses concurrent `cu_seqlens` varlen under FA3 instead of per-chunk serial (`VISION_FA3_VARLEN`). |
| `stage1_selective_ckpt.py` | `SelectiveCheckpointCallback` | Selective recompute: keep K of 64 text layers recomputed, hold the rest as activations (`CKPT_LAYERS`). |
| `liger_patch.py` | `LigerSwiGLUCallback` | Liger fused SwiGLU on the 64 text MLPs; keeps gate/up/down Linears so param names / ZeRO-3 / checkpoints are unchanged (`LIGER_SWIGLU`). |
| `stage1_full_monitor.py` | `FullRunMonitor` | Writes `metrics_rank*.jsonl` (step time, memory), enforces `stop_after` / `save_at_stop`, and guards against non-finite loss. |
| `stage1_profiler.py` | `TorchProfilerCallback` | Optional `torch.profiler` trace over a step window on selected ranks (`PROF_ENABLE`). |

The five optimization callbacks patch disjoint symbols (RoPE / RMSNorm / vision-attention / SwiGLU /
checkpointing), so they compose with no ordering dependency and can be toggled independently.

### `alpamayo2_core.patch` (1 file)

| File | Change | Why |
|------|--------|-----|
| `src/alpamayo2_super/config.py` | propagate `attn_implementation` to sub-configs | So `+model.attn_implementation=flash_attention_3` reaches the text backbone, vision tower, and expert - not just the top-level config. |

## Environment variables reference

| Variable | Default | Description |
|------|--------|------|
| `MODEL_PATH` | (required) | Local snapshot of `nvidia/Alpamayo2-Super` |
| `DATA_PATH` | (required) | Local `nvidia/PhysicalAI-Autonomous-Vehicles` directory |
| `NUM_GPUS` | `8` | GPUs per node |
| `NNODES` | `1` | Number of nodes (1 = single-node, 2 = dual-node) |
| `NODE_RANK` | `0` | This node's rank |
| `MASTER_ADDR` | `127.0.0.1` | Rendezvous address (hostname / node name resolved to IPv4) |
| `MASTER_PORT` | `29517` | Rendezvous port |
| `OUTPUT_DIR` | `${CACHE_ROOT:-$HOME/alpamayo2}/outputs/alpamayo2_super_stage1` | Trainer output dir; must be a shared path for multi-node ZeRO-3 |
| `CONFIG_NAME` | `sft_stage1` | Hydra config (Stage-1 VLM SFT) |
| `TRAIN_CHUNKS` / `VAL_CHUNKS` | `0-99` / `99-100` | PAI chunk ranges (half-open) |
| `NUM_FRAMES` | `4` | Frames per camera |
| `MAX_LENGTH` | `6144` | Hard token cap per sample |
| `MAX_STEPS` | `100000` | LR-schedule horizon (recipe default; warmup 4000 = 4% of it) |
| `WARMUP_STEPS` | `4000` | LR warmup steps (recipe default) |
| `STOP_AFTER` | `30` | Early-stop step for benchmarking (`0` = run to `MAX_STEPS`) |
| `MBS` / `GRAD_ACCUM` | `1` / `16` | Per-device batch size / gradient accumulation (recipe default; global batch = NUM_GPUS x MBS x GA) |
| `DEEPSPEED_CONFIG` | `zero3.json` | DeepSpeed config name under `configs/deepspeed/` |
| `CKPT_LAYERS` | auto (48 single-node / 0 dual-node) | Selective recompute: text layers kept recomputed (0..64) |
| `GRAD_CKPT` | `1` | Trainer-level gradient checkpointing (vision tower); required by `CKPT_LAYERS` |
| `ATTN_IMPL` | `flash_attention_3` | Attention backend (`sdpa`/`eager`/`flash_attention_2`/`flash_attention_3`) |
| `VISION_FA3_VARLEN` | `1` | Vision tower concurrent varlen under FA3 |
| `TE_RMSNORM` | `1` | TransformerEngine fused RMSNorm |
| `TRITON_ROPE` | `1` | Fused RoPE (`0`/`1`/`text`/`vision`) |
| `TRITON_ROPE_BACKEND` | `triton` | RoPE backend (`triton`/`flash`/`liger`) |
| `LIGER_SWIGLU` | `1` | Liger fused SwiGLU on text MLPs |
| `PAD_TO_FIXED_LENGTH` | `false` | `false` = dynamic pad to batch-longest |
| `SUPERVISED_LM_HEAD` | `0` | Optional compact Stage-1 logits projection |
| `PCCL_MAX_NCHANNELS` / `NCCL_MAX_NCHANNELS` | `4` | Comm channel cap (also the `_P2P_` variants) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Allocator tuning |
| `NCCL_SOCKET_IFNAME` | (empty) | Multi-node bootstrap NIC |
| `NCCL_IB_HCA` | (empty) | Multi-node RDMA HCA list |
| `SAVE_CKPT` / `SAVE_STEPS` | `0` / `500` | Enable periodic checkpointing / interval |
| `RESUME_FROM` | `null` | Checkpoint dir to resume from |
| `PROF_ENABLE` | `0` | Enable `torch.profiler` capture |
| `PROF_START` / `PROF_END` | `20` / `21` | Profiler step window |
| `PROF_RANKS` | `[0]` | Ranks to profile |
| `PROF_WITH_STACK` | `0` | Include Python stack in the trace (larger, slower) |
| `CACHE_ROOT` | `${HOME}/alpamayo2` | Root for caches, temp, and default outputs |
| `PYTHON_BIN` | `python` | Interpreter (the image venv is already on `PATH`) |
| `DRY_RUN` | `0` | `1` = print the resolved launch command and exit |

## How the script works

1. Resolve the interpreter and the installed `alpamayo2_sft` package; locate the DeepSpeed config from it.
2. Validate `MODEL_PATH` / `DATA_PATH`, set cache dirs and offline mode.
3. Auto-select `CKPT_LAYERS` from `NNODES` (48 single-node, 0 dual-node) unless overridden.
4. Export comm/memory tuning (4 channels, `expandable_segments`, optional RDMA fabric).
5. Split output: shared `paths.output_dir` for ZeRO-3 checkpointing, per-node `node${NODE_RANK}/` for logs.
6. Build hydra overrides, registering the enabled optimization callbacks under `alpamayo2_sft.*`.
7. Print the config box, then launch `torch.distributed.run -m alpamayo2_sft.train_hf` (one code path
   for single- and dual-node).
8. After training, print a steady-state performance summary (median step time, throughput, peak memory,
   last loss) parsed from `metrics_rank0.jsonl`.
