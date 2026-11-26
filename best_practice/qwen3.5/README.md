# Qwen3.5-VL Best Practices

Training and MFU tuning recipes for **Qwen3.5-VL** on **PPU M890**.

## Experiment Setup

Qwen3.5 natively supports multimodal input through a ViT: images are encoded into vision tokens and merged with the text tokens before entering the LLM. Two workload types are used for performance analysis:

1. Image-heavy: fill the sequence with as many 1920x1080 images as needed to reach the target sequence length, padded with a small amount of text.
2. Text-only: pad pure text to the target sequence length.

All images and text are randomly generated for performance analysis.

| Item | Configuration |
|------|---------------|
| Hardware | PPU M890 (144G), ICN interconnect, 8 GPUs per node |
| Software | SAIL DOCKER 2v2: `pkg.flytiger-eco.com/docker_release/sdk2.2.0-cuda13.0-ubuntu24-py312-20261001` |
| Code | [sail-train-recipes](https://github.com/flytiger-eco/sail-train-recipes) |
| Precision | bf16 |
| Metric | Avg TFLOP/s/GPU over post-warmup iterations. Rows refreshed on 2026-11-03 use the median of iterations 12–20 (warmup and profiler steps excluded). |

### Optimization defaults

The `tune_*.sh` scripts apply the dense optimization set from the M890 image, including the chunkwise CP dispatch, by default. The commands below therefore only set the sweep axis (sequence length, image count) and, where a row differs from the script default, the parallel layout:

| Optimization | Enabled by | Applies to |
|--------------|------------|------------|
| FLA 0.5.2 | image | 2B / 4B / 9B / 27B |
| causal-conv1d 1.7.0 (sm89) | image | 2B / 4B / 9B / 27B |
| Chunkwise CP (`--linear-cp-mode chunkwise`) | script default (inert at CP=1) | 2B / 4B / 9B / 27B; 122B-A10B only at CP=1 — at CP>1 the script uses headwise because chunkwise deadlocks on that model |
| FLA kernel cache (`FLA_CACHE_MODE=default`) | script default | 2B / 4B / 9B / 27B / 122B-A10B |
| PCCL comm knobs (`PCCL_CTA_POLICY`, `PCCL_GRAPH_REGISTER=0`) | script default | 2B / 4B / 9B / 27B / 122B-A10B |
| PPU elementwise/broadcast opt, expandable segments | script default | all models |

MoE-only optimizations (the DeepEP flex dispatcher, forced router balancing, grouped GEMM) do not apply to these dense models; `tune_35b_a3b.sh` adds them.

## M890 Quick Start

Build and run the M890 image from the repository root:

```bash
docker build -f dockers/Dockerfile.Qwen35 -t ppu-qwen35:M890 .
docker run --privileged --ipc=host --pid=host --network=host -v /etc/environment:/etc/environment  --ulimit memlock=-1 --shm-size=10.24gb \
  --ulimit stack=67108864 --init -dt --cap-add=IPC_LOCK \
  -v $(pwd):/workspace/sail-train-recipes \
  --name=qwen35_sail_train \
  ppu-qwen35:M890 /bin/bash
```

`dockers/Dockerfile.Qwen35` no longer builds the Megatron-Bridge stack: it installs Megatron-LM-for-sail at a pinned commit, and the scripts drive `examples/multimodal_dev/pretrain_multimodal.py` directly.

## Performance Results

Image resolution: 1920x1080 unless stated otherwise. The mock provider does not resize, so image edges must be multiples of the ViT patch 16 and the per-sample budget must cover every merged patch token; the image-heavy commands below set a matching `IMAGE_SIZE_W`/`IMAGE_SIZE_H` and `IMAGE_SEQ_LENGTH`.

### Dense Models

**2B model** (1 node, 8 GPUs):

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 4096 | 0 | 1 | 8 | 196.0 |
| TP=1 DP=8 | 4096 | 2 | 1 | 8 | 147.3 |
| TP=1 DP=8 | 8192 | 0 | 1 | 8 | 216.3 |
| TP=1 DP=8 | 8192 | 4 | 1 | 8 | 154.4 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-2B"

SEQ_LEN=4096 NUM_IMAGES=0 bash tune_2b.sh

SEQ_LEN=4096 NUM_IMAGES=2 IMAGE_SIZE_W=448 IMAGE_SIZE_H=448 IMAGE_SEQ_LENGTH=1024 bash tune_2b.sh

SEQ_LEN=8192 NUM_IMAGES=0 bash tune_2b.sh

SEQ_LEN=8192 NUM_IMAGES=4 IMAGE_SIZE_W=960 IMAGE_SIZE_H=544 IMAGE_SEQ_LENGTH=4096 bash tune_2b.sh
```

**4B model** (1 node, 8 GPUs):

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 4096 | 0 | 1 | 8 | 204.0 |
| TP=1 DP=8 | 4096 | 2 | 1 | 8 | 164.4 |
| TP=1 DP=8 | 8192 | 0 | 1 | 8 | 219.3 |
| TP=1 DP=8 | 8192 | 4 | 1 | 8 | 173.2 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-4B"

SEQ_LEN=4096 NUM_IMAGES=0 bash tune_4b.sh

SEQ_LEN=4096 NUM_IMAGES=2 IMAGE_SIZE_W=448 IMAGE_SIZE_H=448 IMAGE_SEQ_LENGTH=1024 bash tune_4b.sh

SEQ_LEN=8192 NUM_IMAGES=0 bash tune_4b.sh

SEQ_LEN=8192 NUM_IMAGES=4 IMAGE_SIZE_W=960 IMAGE_SIZE_H=544 IMAGE_SEQ_LENGTH=4096 bash tune_4b.sh
```

**9B model** (1 node, 8 GPUs):

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 4096 | 0 | 1 | 8 | 249.3 |
| TP=1 DP=8 | 4096 | 2 | 1 | 8 | 182.4 |
| TP=1 DP=8 | 8192 | 0 | 1 | 8 | 262.1 |
| TP=2 DP=4 | 8192 | 4 | 1 | 8 | 180.1 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-9B"

SEQ_LEN=4096 NUM_IMAGES=0 bash tune_9b.sh

SEQ_LEN=4096 NUM_IMAGES=2 IMAGE_SIZE_W=448 IMAGE_SIZE_H=448 IMAGE_SEQ_LENGTH=1024 bash tune_9b.sh

SEQ_LEN=8192 NUM_IMAGES=0 bash tune_9b.sh

SEQ_LEN=8192 NUM_IMAGES=4 TP=2 IMAGE_SIZE_W=960 IMAGE_SIZE_H=544 IMAGE_SEQ_LENGTH=4096 bash tune_9b.sh
```

**27B model** (2 nodes, 16 GPUs, +FSDP):

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=2 DP=8 +FSDP | 4096 | 0 | 1 | 8 | 210.2 |
| TP=2 DP=8 +FSDP | 4096 | 2 | 1 | 8 | 181.7 |
| TP=2 DP=8 +FSDP | 8192 | 0 | 1 | 8 | **224.7** |
| TP=2 DP=8 +FSDP | 8192 | 4 | 1 | 8 | 193.1 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-27B"

# Run the same command on both nodes; only RANK differs. MASTER_ADDR is always the eth0 IP of the rank-0 node.
NNODES=2 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=4096 NUM_IMAGES=0 bash tune_27b.sh

NNODES=2 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=4096 NUM_IMAGES=2 IMAGE_SIZE_W=448 IMAGE_SIZE_H=448 IMAGE_SEQ_LENGTH=1024 bash tune_27b.sh

NNODES=2 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=8192 NUM_IMAGES=0 bash tune_27b.sh

NNODES=2 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=8192 NUM_IMAGES=4 IMAGE_SIZE_W=960 IMAGE_SIZE_H=544 IMAGE_SEQ_LENGTH=4096 bash tune_27b.sh
```

### MoE Models

**35B-A3B model** (2 nodes, 16 GPUs):

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 EP=16 DP=16 | 4096 | 0 | 1 | 8 | 117.9‡ |
| TP=1 EP=16 DP=16 | 4096 | 2 | 1 | 8 | 112.3 |
| TP=1 EP=16 DP=16 | 8192 | 0 | 1 | 8 | 133.2‡ |
| TP=1 EP=16 DP=16 | 8192 | 4 | 1 | 8 | 100.1 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-35B-A3B"

# Run the same command on both nodes; only RANK differs. MASTER_ADDR is always the eth0 IP of the rank-0 node.
NNODES=2 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=4096 NUM_IMAGES=0 bash tune_35b_a3b.sh

NNODES=2 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=4096 NUM_IMAGES=2 IMAGE_SIZE_W=448 IMAGE_SIZE_H=448 IMAGE_SEQ_LENGTH=1024 bash tune_35b_a3b.sh

NNODES=2 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=8192 NUM_IMAGES=0 bash tune_35b_a3b.sh
```

`‡` = measured on the previous (Megatron-Bridge) stack with the MIMO EP-overlap schedule (`ENABLE_EP_OVERLAP=1`), which this path does not have; reference-only. The TP=2 image row is reference-only too: EP=16 does not divide its DP=8.

> Multi-node runs: launch the same command on each node with the correct `RANK` and the same `MASTER_ADDR`/`MASTER_PORT`.
> All tuning scripts share the same environment variables. Override them as needed.
> Logs are written to `$OUTPUT_DIR/<tag>/train_node${RANK}.log`, with the tensorboard directory next to it; each run also writes `meta/run.env` (resolved profile mode, window and launch parameters), `meta/cmd.txt` (the exact command) under the same tag directory.
> Profiling is one switch: `PROFILE=torch` writes a torch profiler trace, while `PROFILE=asys` runs the entrypoint through `internal/capture_report.sh` and saves a model_prof report in `<tag>/profile/` instead. `PROFILE_STEPS=<start>-<end>` is 1-based and inclusive over training iterations and defaults to the last eager CUDA-graph warmup iteration (`5-5` with the default `CUDA_GRAPH_WARMUP_STEPS=5`): iteration 1 is skipped because it pays the dynamo/inductor and runtime compiles, and the graph is captured on the iteration after the last warmup one. The window is fed to the profiler as `--profile-step-start <start>-1`, so a one-iteration window still records one step. Profiling is off by default because it perturbs the captured steps; a capture run is kept for inspection, not as a performance measurement. The legacy `ENABLE_PROFILER=1` / `CAPTURE_ASYS=1` still map to `torch` / `asys`, but `PROFILE` wins if both are given.

## Long-Sequence Extension (Context Parallel)

Context Parallel (CP) splits the sequence across GPUs, extending the total sequence length while keeping the per-GPU sequence length fixed. The GDN context-parallel support ships in the pinned Megatron-LM-for-sail commit (the paths below are relative to that checkout); `dockers/Dockerfile.Qwen35` applies no CP patch.

- `ssm/gated_delta_net/gdn.py` (L131): `linear_cp_mode` dispatch — two CP modes are available: chunkwise (sequence sharding + CP-aware linear kernels, no head all-to-all; **recommended, and used by the CP rows refreshed on 2026-11-03**) and headwise (Ulysses-style head all-to-all). At CP=1 the mode has no effect. Exception: 122B-A10B at CP>1 must use headwise — chunkwise deadlocks at the first GDN layer (py-spy: ranks split between `build_cp_context` and MoE allgather, CP all-to-all watchdog on a 8192x3072 qkvzba transfer), so `tune_122b_a10b.sh` passes `headwise` for CP>1.
- `ssm/gdn_head_pack.py` (L211): vectorized head-pack CUDA kernel — used by the headwise path only. It packs head segments with 16-byte vectorized memory access before the headwise all-to-all, replacing the `index_select` gather with a single unsegmented a2a.
- `ssm/gated_delta_net/common.py` (L698): falls back to the generic head-perm path when a segment is not 16-byte aligned (e.g. the gate segment of 9B CP8), keeping every configuration usable.

> Note: the `ssm/gdn_head_pack.py` optimization has a large impact on compute efficiency **when the headwise mode is used**. The M890 has relatively few CUDA cores, so native kernels of this kind are at a disadvantage; the vectorized memory access mitigates the problem. However, when the linear head count per rank after CP sharding no longer satisfies the vectorization requirements, the code falls back to the original implementation and efficiency degrades.

> **How the CP rows below are launched.** Every `tune_*.sh` passes `--linear-cp-mode chunkwise` (except `tune_122b_a10b.sh` at CP>1, which passes `headwise`) itself, so the commands only vary the sequence length, the parallel layout and — for CP>1 — `CUDA_GRAPH_MODULES=` (empty):
>
> - `--linear-cp-mode chunkwise` is the Mcore default and is inert at CP=1, so the CP=1 row in each grid reproduces the base measurement.
> - The chunkwise CP path runs with TE CUDA graphs off: the 2B/4B/9B commands set `CUDA_GRAPH_MODULES=` (empty) for CP>1, and the 27B wrapper leaves graphs off by default. The 35B-A3B CP grid keeps EP=16, which no longer divides DP once CP>1, so only its CP=1 rows run on this path.

**2B model**

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 16384 | 0 | 1 | 8 | 219.6 |
| TP=1 CP=2 DP=4 | 32768 | 0 | 1 | 8 | 189.4† |
| TP=1 CP=4 DP=2 | 65536 | 0 | 1 | 8 | **169.1†** |
| TP=1 CP=8 DP=1 | 131072 | 0 | 1 | 8 | **163.0†** |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-2B"

SEQ_LEN=16384 NUM_IMAGES=0 bash tune_2b.sh

CUDA_GRAPH_MODULES= SEQ_LEN=32768 NUM_IMAGES=0 CP=2 bash tune_2b.sh

CUDA_GRAPH_MODULES= SEQ_LEN=65536 NUM_IMAGES=0 CP=4 bash tune_2b.sh

CUDA_GRAPH_MODULES= SEQ_LEN=131072 NUM_IMAGES=0 CP=8 bash tune_2b.sh
```

**4B model**

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 8192 | 0 | 1 | 8 | 219.3 |
| TP=1 CP=2 DP=4 | 16384 | 0 | 1 | 8 | 194.9† |
| TP=1 CP=4 DP=2 | 32768 | 0 | 1 | 8 | 178.2† |
| TP=1 CP=8 DP=1 | 65536 | 0 | 1 | 8 | **169.6†** |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-4B"

SEQ_LEN=8192 NUM_IMAGES=0 bash tune_4b.sh

CUDA_GRAPH_MODULES= SEQ_LEN=16384 NUM_IMAGES=0 CP=2 bash tune_4b.sh

CUDA_GRAPH_MODULES= SEQ_LEN=32768 NUM_IMAGES=0 CP=4 bash tune_4b.sh

CUDA_GRAPH_MODULES= SEQ_LEN=65536 NUM_IMAGES=0 CP=8 bash tune_4b.sh
```

**9B model**

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 8192 | 0 | 1 | 8 | 262.1 |
| TP=1 CP=2 DP=4 | 16384 | 0 | 1 | 8 | 238.5† |
| TP=1 CP=4 DP=2 | 32768 | 0 | 1 | 8 | 218.6† |
| TP=1 CP=8 DP=1 | 65536 | 0 | 1 | 8 | **205.2†** |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-9B"

SEQ_LEN=8192 NUM_IMAGES=0 bash tune_9b.sh

CUDA_GRAPH_MODULES= SEQ_LEN=16384 NUM_IMAGES=0 CP=2 bash tune_9b.sh

CUDA_GRAPH_MODULES= SEQ_LEN=32768 NUM_IMAGES=0 CP=4 bash tune_9b.sh

CUDA_GRAPH_MODULES= SEQ_LEN=65536 NUM_IMAGES=0 CP=8 bash tune_9b.sh
```

**27B model (TP=2, 8k base)**

CP extends the total sequence while keeping 8192 tokens per GPU per micro-batch:

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=2 CP=1 DP=8 | 8192 | 0 | 1 | 8 | **224.7** |
| TP=2 CP=2 DP=4 | 16384 | 0 | 1 | 8 | 210.6 |
| TP=2 CP=4 DP=2 | 32768 | 0 | 1 | 8 | 203.7 |
| TP=2 CP=8 DP=1 | 65536 | 0 | 1 | 8 | 192.2 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-27B"

# Run the same command on both nodes; only RANK differs. MASTER_ADDR is always the eth0 IP of the rank-0 node.
NNODES=2 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=8192 NUM_IMAGES=0 bash tune_27b.sh

NNODES=2 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=16384 NUM_IMAGES=0 CP=2 bash tune_27b.sh

NNODES=2 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=32768 NUM_IMAGES=0 CP=4 bash tune_27b.sh

NNODES=2 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=65536 NUM_IMAGES=0 CP=8 bash tune_27b.sh
```

The previous TP=1, 4k-base, headwise results at 4096/8192/16384/32768 were 228.6/216.9/193.9/184.3. They are kept for reference only and have not been re-measured on the current stack.

**35B-A3B model**

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 EP=16 CP=1 DP=16 | 4096 | 0 | 1 | 8 | 117.9‡ |
| TP=1 EP=16 CP=1 DP=16 | 8192 | 0 | 1 | 8 | 133.2‡ |
| TP=1 EP=16 CP=2 DP=8 | 16384 | 0 | 1 | 8 | 119.6† |
| TP=1 EP=16 CP=4 DP=4 | 32768 | 0 | 1 | 8 | 116.2† |
| TP=1 EP=16 CP=8 DP=2 | 65536 | 0 | 1 | 8 | 118.6† |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-35B-A3B"

# The CP=2/4/8 rows are reference-only on this path: EP=16 exhausts DP at 16
# ranks once CP>1, so the launcher rejects those layouts. The CP=1 rows are in
# the base grid above.
```

`†` = measured with TE CUDA graphs off (`CUDA_GRAPH_MODULES=`). The chunkwise CP path runs with graph off, and some configurations also hang during CUDA Graph capture (previously 9B CP=4/32768, 9B CP=8/65536).

`‡` = measured on the previous stack with the MIMO EP-overlap schedule (`ENABLE_EP_OVERLAP=1`); only the pure-text CP=1 rows satisfied its constraints, and the schedule does not exist on this path.

> **Refresh status (2026-11-03).** All dense rows were re-measured on the previous stack with the full optimization set above: the 2B/4B/9B text-only and image-heavy rows, their CP grids, and the 27B TP=2 8k-base grid including CP=8 at 65536. The 35B-A3B rows were re-measured on the same day: all four base rows, the two pure-text CP=1 rows of the CP table (with EP overlap), and the CP=2/4/8 rows (CUDA Graph off). Every 35B-A3B number above comes from that sweep.

All dense results above were measured on the previous (Megatron-Bridge) stack with the full dense optimization set, which the current scripts still apply by default: `FLA_CACHE_MODE=default`, `PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `NVTE_CPU_OFFLOAD_V1=1`, `NVTE_USE_CUTLASS_GROUPED_GEMM=1`, and the PCCL communication knobs (`PCCL_CTA_POLICY`, `PCCL_GRAPH_REGISTER=0`). The chunkwise CP dispatch is also passed by the scripts, so no command block needs to repeat it.

The 35B-A3B rows additionally run the MoE set: the DeepEP flex dispatcher (`--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep`), forced router balancing and MTP with one depth. The previous stack also ran `MEGATRON_MOE_TOPK_RECONSTRUCT=triton`, `--enable-grad-overlap` and the MIMO EP-overlap schedule; this path has none of the three. The PCCL copy engine (`ENABLE_COPY_ENGINE=1`) was measured on these rows and left off: on the 4096/0 row it cost about 5% (5.55 s / 111.7 TFLOP/s with CE versus 5.26 s / 117.9 TFLOP/s without).

## Qwen3.5-122B-A10B (MoE, FSDP)

122B-A10B is trained with FSDP; `tune_122b_a10b.sh` drives the same multimodal
entrypoint with the `122b_a10b` variant. The default layout is TP=1, PP=1,
EP=16 and DP=64 over eight nodes; the 8k and CP rows run TP=2
(`TP=2`; sequence parallelism is always on), which halves DP to 32 and shrinks the per-rank
activation without changing EP=16. FSDP shards parameters, gradients and
optimizer state across the DP ranks, so the fp32 master weights and Adam moments
that forced `--optimizer-cpu-offload` under DDP stay on the devices. gbs = MBS(1)
x DP x GA(2) = 128 at TP=1 and 64 at TP=2. A CP cell shrinks DP to 64/(TP·CP)
and runs the headwise GDN context-parallel path (`--linear-cp-mode headwise`)
because the chunkwise path deadlocks on this model at CP>1.

The script defaults to 256x256 images and one image per sample; the 8k 4-image
row below ran 1920x1080 on the previous stack; the mock provider does not resize, so its command uses
448x448 images for the 8k cell. The `Imgs` column is the count.

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 EP=16 DP=64 | 4096 | 0 | 1 | 2 | — |
| TP=1 EP=16 DP=64 | 4096 | 2 | 1 | 2 | — |
| TP=2 EP=16 DP=32 | 8192 | 0 | 1 | 2 | — |
| TP=2 EP=16 DP=32 | 8192 | 4 | 1 | 2 | — |
| TP=2 EP=16 CP=2 DP=16 | 16384 | 0 | 1 | 2 | — |
| TP=2 EP=16 CP=4 DP=8 | 32768 | 0 | 1 | 2 | — |
| TP=2 EP=16 CP=8 DP=4 | 65536 | 0 | 1 | 2 | — |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-122B-A10B"

# Run the same command on all eight nodes; only RANK differs. MASTER_ADDR is always the eth0 IP of the rank-0 node.
NNODES=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=4096 NUM_IMAGES=0 bash tune_122b_a10b.sh

NNODES=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> SEQ_LEN=4096 NUM_IMAGES=2 bash tune_122b_a10b.sh

NNODES=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> TP=2 SEQ_LEN=8192 NUM_IMAGES=0 bash tune_122b_a10b.sh

NNODES=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> TP=2 SEQ_LEN=8192 NUM_IMAGES=4 IMAGE_SIZE_W=448 IMAGE_SIZE_H=448 IMAGE_SEQ_LENGTH=2048 bash tune_122b_a10b.sh

NNODES=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> TP=2 SEQ_LEN=16384 NUM_IMAGES=0 CP=2 bash tune_122b_a10b.sh

NNODES=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> TP=2 EP=8 SEQ_LEN=32768 NUM_IMAGES=0 CP=4 bash tune_122b_a10b.sh

NNODES=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> TP=2 EP=4 SEQ_LEN=65536 NUM_IMAGES=0 CP=8 bash tune_122b_a10b.sh
```

On this path EP must divide DP, so the TP=2 cells shrink EP along with DP: CP=4
runs `EP=8` and CP=8 runs `EP=4` (the table keeps the intended EP=16 layout).

The 122B rows are the newest addition and the first sweep is still running, so
the TFLOP/s column is filled from that sweep rather than from the earlier DDP
runs. Activation memory is the binding constraint at this size, so the script
exposes the recompute policy (`full` from 8192 tokens up, `none` below, both
derived from `SEQ_LEN`; an explicit `RECOMPUTE=1/0` always wins) and keeps TE
CUDA graphs off unless `CUDA_GRAPH_MODULES` is set.

## Files

| File | Description |
|------|-------------|
| `tune_{2b,4b,9b,27b,35b_a3b,122b_a10b}.sh` | MFU tuning entry points. Each declares the variant, the parallel layout and the data shape, then calls `run_megatron`; training runs through `examples/multimodal_dev/pretrain_multimodal.py`. |
| `internal/common.sh` | Shared launcher sourced by the entry points: environment, validation, argv assembly, `run_megatron` and the `meta/` artifacts. |
| `internal/qwen_model_template.sh` | Qwen3.5 architecture table and every model-side flag. |
| `internal/parallel_template.sh` | Topology checks, parallel flags and Megatron-FSDP flags. |
| `internal/training_template.sh` | Optimizer/schedule/precision, TE CUDA graphs, logging, recompute, checkpoints. |
| `internal/smoke.sh` | End-to-end check (proxy variant, THD packing, vision CP, no checkpoint). |
| `internal/capture_report.sh` | Fail-closed `PROFILE=asys` capture wrapper. |
| `tune_varlen_*.sh`, `train_qwen35_varlen.py` | Variable-length SFT (LLaVA-NeXT JSONL, THD packing); Megatron-Bridge stack on `dockers/Dockerfile.Qwen35_varlen`, unchanged. |
| `../../dockers/Dockerfile.Qwen35` | Image for the entry points (replaces the Megatron-Bridge image): Megatron-LM-for-sail at a pinned commit + `../../dockers/patches/qwen35-multimodal.patch`. |

## Common Options

| Variable | Meaning |
|----------|---------|
| `HF_MODEL` | Hugging Face model path or id; supplies the tokenizer and processor. |
| `SEQ_LEN` | Packed sequence length (`--total-seq-length`). |
| `NUM_IMAGES` | Number of images per sample (`0` for text-only, which also drops the encoder). |
| `IMAGE_SIZE_W` / `IMAGE_SIZE_H` / `IMAGE_SEQ_LENGTH` | Mock image geometry (edges must be multiples of the ViT patch 16) and the vision-token budget. |
| `MBS` / `GA` | Micro-batch size and gradient accumulation; gbs = MBS x GA x DP. |
| `TP` / `EP` / `PP` / `CP` | Parallel sizes. EP must divide DP and dense wrappers require EP=1; FSDP and PP are mutually exclusive. |
| `CUDA_GRAPH_MODULES` | TE CUDA-graph module list; the wrappers set the per-model default and an empty value turns graphs off (the `†` configurations). |
| `USE_FSDP` | Megatron-FSDP toggle; defaults to the wrapper's `USE_FSDP_DEFAULT`. |
| `RECOMPUTE` | `1`/`0` full decoder recompute; `tune_122b_a10b.sh` defaults to `full` at `SEQ_LEN>=8192` and `none` below. |
| `MTP_NUM_LAYERS` | MTP prediction depths. |
| `DISPATCHER` / `FORCE_LOAD_BALANCING` | MoE dispatcher (`alltoall` or `flex`) and uniform router decisions for mock data. |
| `ENABLE_VISION_CP` / `USE_PACKED_SEQUENCE` | Split the batch images across CP ranks / THD packing. |
| `EXTRA_ARGS` | Additional entrypoint flags appended to the training command. The scripts already pass `--linear-cp-mode chunkwise`, so this is only needed for extra overrides. |
| `TUNE_DRY_RUN` | `1` prints the torchrun command line and exits without launching. |
| `PROFILE` / `PROFILE_STEPS` | `off`, `torch` or `asys`; the one-based, inclusive iteration window. |
| `CKPT_LOAD` / `CKPT_RESUME` / `CKPT_FORMAT` | Checkpoint loading. Bridge checkpoints are a GPT layout and do not load into this HybridModel. |
| `FLA_CACHE_MODE` | FLA kernel cache mode, defaults to `default`. |
| `ENABLE_COPY_ENGINE` | PCCL copy-engine mode; `0` (default) sets `PCCL_CTA_POLICY=0x1`, `1` sets `PCCL_CTA_POLICY=2` and enables the CE reduce-scatter. The 35B-A3B rows above run with it off. |
| `PCCL_RAS_ENABLE` | PCCL RAS thread, `0` by default: libpccl 2.2.0 RAS reads a corrupted host-side hash table after a cold inductor compile (`PCCL RAS` thread, segfault at `libpccl+0x1bfd4d`). Set `1` only when diagnosing PCCL itself. |
| `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL` | `0` by default; keeps inductor compilation synchronous so a forked compile worker cannot deadlock with PCCL comm threads. |

The scripts also set PPU/PyTorch optimizations such as `PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `NVTE_CPU_OFFLOAD_V1=1`, and `NVTE_USE_CUTLASS_GROUPED_GEMM=1` by default.
