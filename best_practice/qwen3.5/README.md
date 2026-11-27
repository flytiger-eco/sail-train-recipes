# Qwen3.5 Best Practices

Training, MFU tuning, and RL recipes for the **Qwen3.5** family on **PPU M890**.
The first part covers Qwen3.5-VL pre-training MFU tuning (Megatron-Bridge); the
GoLongRL section at the end covers text-only Qwen3.5-35B-A3B long-context GRPO
(slime 0.4.0 RL).

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
| Metric | Avg TFLOP/s/GPU, averaged over post-warmup iterations (iter > 10) |

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

## Performance Results

Image resolution: 1920x1080 unless stated otherwise.

### Dense Models

**2B model** (1 node, 8 GPUs):

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 4096 | 0 | 1 | 8 | 181.6 |
| TP=1 DP=8 | 4096 | 2 | 1 | 8 | 141.9 |
| TP=1 DP=8 | 8192 | 0 | 1 | 8 | 198.4 |
| TP=1 DP=8 | 8192 | 4 | 1 | 8 | 148.8 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-2B"

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=4096 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_2b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=2 SEQ_LEN=4096 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_2b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=8192 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_2b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=4 SEQ_LEN=8192 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_2b.sh
```

**4B model** (1 node, 8 GPUs):

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 4096 | 0 | 1 | 8 | 188.1 |
| TP=1 DP=8 | 4096 | 2 | 1 | 8 | 156.6 |
| TP=1 DP=8 | 8192 | 0 | 1 | 8 | 202.8 |
| TP=1 DP=8 | 8192 | 4 | 1 | 8 | 164.9 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-4B"

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=4096 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_4b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=2 SEQ_LEN=4096 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_4b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=8192 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_4b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=4 SEQ_LEN=8192 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_4b.sh
```

**9B model** (1 node, 8 GPUs):

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 4096 | 0 | 1 | 8 | 236.3 |
| TP=1 DP=8 | 4096 | 2 | 1 | 8 | 173.6 |
| TP=1 DP=8 | 8192 | 0 | 1 | 8 | 247.1 |
| TP=2 DP=4 | 8192 | 4 | 1 | 8 | 171.3 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-9B"

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=4096 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_9b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=2 SEQ_LEN=4096 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_9b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=8192 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_9b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=4 SEQ_LEN=8192 MBS=1 GA=8 BEST_LANG="language=tp=2,cp=1,ep=1,dp=4,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_9b.sh
```

**27B model** (2 nodes, 16 GPUs, +FSDP):

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=2 DP=8 +FSDP | 4096 | 0 | 1 | 8 | 201.3 |
| TP=2 DP=8 +FSDP | 4096 | 2 | 1 | 8 | 174.1 |
| TP=2 DP=8 +FSDP | 8192 | 0 | 1 | 8 | 214.8 |
| TP=2 DP=8 +FSDP | 8192 | 4 | 1 | 8 | 185.0 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-27B"

# Run the same command on both nodes; only RANK differs. MASTER_ADDR is always the eth0 IP of the rank-0 node.
NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=4096 MBS=1 GA=8 TP=2 CP=1 bash tune_27b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=2 SEQ_LEN=4096 MBS=1 GA=8 TP=2 CP=1 bash tune_27b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=8192 MBS=1 GA=8 TP=2 CP=1 bash tune_27b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=4 SEQ_LEN=8192 MBS=1 GA=8 TP=2 CP=1 bash tune_27b.sh
```

### MoE Models

**35B-A3B model** (2 nodes, 16 GPUs, MIMO):

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 EP=16 DP=16 | 4096 | 0 | 1 | 8 | 94.9 |
| TP=1 EP=16 DP=16 | 4096 | 2 | 1 | 8 | 102.0 |
| TP=1 EP=16 DP=16 | 8192 | 0 | 1 | 8 | 106.3 |
| TP=1 EP=16 DP=16 | 8192 | 4 | 1 | 8 | 92.6 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-35B-A3B"

# Run the same command on both nodes; only RANK differs. MASTER_ADDR is always the eth0 IP of the rank-0 node.
NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=4096 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=16,dp=16,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=16,rank_offset=0" bash tune_35b_a3b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=2 SEQ_LEN=4096 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=16,dp=16,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=16,rank_offset=0" bash tune_35b_a3b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=8192 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=16,dp=16,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=16,rank_offset=0" bash tune_35b_a3b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=4 SEQ_LEN=8192 MBS=1 GA=8 BEST_LANG="language=tp=2,cp=1,ep=16,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=2,cp=1,dp=8,rank_offset=0" bash tune_35b_a3b.sh
```

> Multi-node runs: launch the same command on each node with the correct `RANK` and the same `MASTER_ADDR`/`MASTER_PORT`.
> All tuning scripts share the same environment variables. Override them as needed.
> Logs are written to `$OUTPUT_DIR/<tag>/train_node${RANK}.log`.

## Long-Sequence Extension (Context Parallel)

Context Parallel (CP) splits the sequence across GPUs, extending the total sequence length while keeping the per-GPU sequence length fixed. The related optimizations ship as patches applied to the Megatron-LM submodule (`3rdparty/Megatron-LM`) by `dockers/Dockerfile.Qwen35`.

**`dockers/patches/qwen35-gdn-chunkwise-cp.patch`** (Megatron-LM side: GDN context-parallel backport + head-pack optimization):

- `ssm/gated_delta_net/gdn.py` (L131): `linear_cp_mode` dispatch — two CP modes are available: chunkwise (sequence sharding + CP-aware linear kernels) and headwise (Ulysses-style head all-to-all).
- `ssm/gdn_head_pack.py` (L211): vectorized head-pack CUDA kernel — packs head segments with 16-byte vectorized memory access before the headwise all-to-all, replacing the `index_select` gather with a single unsegmented a2a.
- `ssm/gated_delta_net/common.py` (L698): falls back to the generic head-perm path when a segment is not 16-byte aligned (e.g. the gate segment of 9B CP8), keeping every configuration usable.

**`dockers/patches/qwen35-gdn-fused-pre-gdr.patch`** (fused pre-GDR kernel, enabled with `--gdn-fused-pre-gdr`, off by default):

- `fusions/fused_pre_gated_delta_rule.py` (L2571): fuses causal_conv1d + SiLU after the GDN projection, QK l2-norm, and g/beta computation into a small number of triton kernels; the gate output is returned as a strided view without materialization, reducing elementwise kernels and intermediate copies.

> Note: the `ssm/gdn_head_pack.py` optimization has a large impact on compute efficiency. The M890 has relatively few CUDA cores, so native kernels of this kind are at a disadvantage; the vectorized memory access mitigates the problem. However, when the linear head count per rank after CP sharding no longer satisfies the vectorization requirements, the code falls back to the original implementation and efficiency degrades.

**2B model**

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 16384 | 0 | 1 | 8 | 202.8 |
| TP=1 CP=2 DP=4 | 32768 | 0 | 1 | 8 | 168.6 |
| TP=1 CP=4 DP=2 | 65536 | 0 | 1 | 8 | **138.6** |
| TP=1 CP=8 DP=1 | 131072 | 0 | 1 | 8 | **131.0** |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-2B"

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=16384 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_2b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=32768 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=2,ep=1,dp=4,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_2b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=65536 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=4,ep=1,dp=2,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_2b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=131072 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=8,ep=1,dp=1,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_2b.sh
```

**4B model**

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 8192 | 0 | 1 | 8 | 202.8 |
| TP=1 CP=2 DP=4 | 16384 | 0 | 1 | 8 | 178.0 |
| TP=1 CP=4 DP=2 | 32768 | 0 | 1 | 8 | 164.5 |
| TP=1 CP=8 DP=1 | 65536 | 0 | 1 | 8 | **140.1** |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-4B"

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=8192 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_4b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=16384 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=2,ep=1,dp=4,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_4b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=32768 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=4,ep=1,dp=2,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_4b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=65536 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=8,ep=1,dp=1,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_4b.sh
```

**9B model**

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=8 | 8192 | 0 | 1 | 8 | 247.1 |
| TP=1 CP=2 DP=4 | 16384 | 0 | 1 | 8 | 222.6 |
| TP=1 CP=4 DP=2 | 32768 | 0 | 1 | 8 | 207.0† |
| TP=1 CP=8 DP=1 | 65536 | 0 | 1 | 8 | **177.9†** |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-9B"

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=8192 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=1,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_9b.sh

NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=16384 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=2,ep=1,dp=4,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_9b.sh

ENABLE_CUDA_GRAPH=0 NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=32768 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=4,ep=1,dp=2,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_9b.sh

ENABLE_CUDA_GRAPH=0 NNODES=1 NPROC=8 IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=65536 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=8,ep=1,dp=1,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=8,rank_offset=0" bash tune_9b.sh
```

**27B model**

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 DP=16 | 4096 | 0 | 1 | 8 | 228.6 |
| TP=1 CP=2 DP=8 | 8192 | 0 | 1 | 8 | 216.9 |
| TP=1 CP=4 DP=4 | 16384 | 0 | 1 | 8 | 193.9 |
| TP=1 CP=8 DP=2 | 32768 | 0 | 1 | 8 | 184.3 |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-27B"

# Run the same command on both nodes; only RANK differs. MASTER_ADDR is always the eth0 IP of the rank-0 node.
NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=4096 MBS=1 GA=8 TP=1 CP=1 bash tune_27b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=8192 MBS=1 GA=8 TP=1 CP=2 bash tune_27b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=16384 MBS=1 GA=8 TP=1 CP=4 bash tune_27b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=32768 MBS=1 GA=8 TP=1 CP=8 bash tune_27b.sh
```

**35B-A3B model**

| Mode | Seq | Imgs | MBS | GA | Avg TFLOP/s/GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TP=1 EP=16 CP=1 DP=16 | 4096 | 0 | 1 | 8 | 94.9 |
| TP=1 EP=16 CP=1 DP=16 | 8192 | 0 | 1 | 8 | 106.3 |
| TP=1 EP=16 CP=2 DP=8 | 16384 | 0 | 1 | 8 | 93.5 |
| TP=1 EP=16 CP=4 DP=4 | 32768 | 0 | 1 | 8 | 91.9 |
| TP=1 EP=16 CP=8 DP=2 | 65536 | 0 | 1 | 8 | 95.0† |

Command:

```bash
export HF_MODEL="Qwen/Qwen3.5-35B-A3B"

# Run the same command on both nodes; only RANK differs. MASTER_ADDR is always the eth0 IP of the rank-0 node.
NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=4096 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=16,dp=16,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=16,rank_offset=0" bash tune_35b_a3b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=8192 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=1,ep=16,dp=16,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=16,rank_offset=0" bash tune_35b_a3b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=16384 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=2,ep=16,dp=8,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=16,rank_offset=0" bash tune_35b_a3b.sh

NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=32768 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=4,ep=16,dp=4,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=16,rank_offset=0" bash tune_35b_a3b.sh

ENABLE_CUDA_GRAPH=0 NNODES=2 NPROC=8 RANK=0 MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 NUM_IMAGES=0 SEQ_LEN=65536 MBS=1 GA=8 BEST_LANG="language=tp=1,cp=8,ep=16,dp=2,etp=1,pp=1,rank_offset=0" BEST_IMG="images=tp=1,cp=1,dp=16,rank_offset=0" bash tune_35b_a3b.sh
```

`†` = CUDA Graph off: these configurations hang during CUDA Graph capture and were re-run with `ENABLE_CUDA_GRAPH=0` (9B CP=4/32768, 9B CP=8/65536, 35B-A3B CP=8/65536). The corresponding launch commands already set the variable.

All results above were measured with `PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1`, which is set inside the tune scripts — no need to set it on the command line.

## Files

| File | Description |
|------|-------------|
| `train_qwen35_vl_mimo.py` | MegatronMIMO training entrypoint. |
| `train_qwen35_vl_vlm.py` | Standard VLM training entrypoint. |
| `tune_2b.sh` | MFU tuning: Qwen3.5-VL 2B dense (MIMO, 1 node). |
| `tune_4b.sh` | MFU tuning: Qwen3.5-VL 4B dense (MIMO, 1 node). |
| `tune_9b.sh` | MFU tuning: Qwen3.5-VL 9B dense (MIMO, 1 node). |
| `tune_27b.sh` | MFU tuning: Qwen3.5-VL 27B dense (VLM, 2 node). |
| `tune_35b_a3b.sh` | MFU tuning: Qwen3.5-VL 35B-A3B MoE (MIMO, 2 nodes). |

## Common Options

| Variable | Meaning |
|----------|---------|
| `HF_MODEL` | Hugging Face model path or local checkpoint. |
| `DATASET_ROOT` | Dataset root directory. |
| `SEQ_LEN` | Sequence length. |
| `NUM_IMAGES` | Number of images per sample (`0` for text-only). |
| `IMAGE_SIZE_W` / `IMAGE_SIZE_H` | Image resolution. |
| `MBS` / `GA` | Micro-batch size and gradient accumulation. |
| `BEST_LANG` / `BEST_IMG` | MIMO parallelism spec for language and image components. |
| `TP` / `CP` | Tensor / context parallel size (`tune_27b.sh`). |
| `ENABLE_CUDA_GRAPH` | CUDA Graph toggle, `1` by default; set `0` for the `†` configurations above. |

The scripts also set PPU/PyTorch optimizations such as `PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `NVTE_CPU_OFFLOAD_V1=1`, and `NVTE_USE_CUTLASS_GROUPED_GEMM=1` by default.

---

## GoLongRL Long-Context GRPO (slime 0.4.0 RL)

Long-context GRPO reinforcement-learning recipe for **Qwen3.5-35B-A3B** (hybrid Gated-Delta-Net + full-attention MoE) on a single node of **8x M890** (144 GiB per card), using **slime 0.4.0** (Megatron-LM training backend + SGLang rollout, colocated on the same cards, orchestrated by Ray).

This recipe covers the GoLongRL long-context band sweep (32k / 64k / 128k / 256k prompt context) and the PPU adaptation needed to run it: native GDN chunkwise context-parallel, the GDN weight-conversion gamma fix, TE fused cross-entropy, the contiguous data-plane single all-to-all, the GoLongRL reward plugin, the colocate `expandable_segments` IPC fix, and a hybrid-aware forward-FLOPs model.

### Experiment Setup / Baseline

| Item | Value |
|------|-------|
| Hardware | 8x M890, 144 GiB/card, single node |
| Base image | `pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.1-2.11.0-ubuntu24.04-cuda13.0-slime0.4.0-py312-20261111` |
| Framework | slime 0.4.0 (editable `/root/slime`), Megatron-LM (editable `/root/Megatron-LM`), sglang-for-sail (editable) |
| Model | Qwen3.5-35B-A3B — 40 layers (30 GDN linear-attn + 10 full-attn, `full_attention_interval=4`), hidden 2048, 256 experts top-8 |
| Dataset | [Kwai-Klear/GoLongRL](https://huggingface.co/datasets/Kwai-Klear/GoLongRL), banded by prompt token length |
| Precision | bf16 params, fp32 grad all-reduce + fp32 attention softmax, precision-aware optimizer |
| Workload | GRPO, `n_samples_per_prompt=8`, rollout batch 16, global batch 128, KL/entropy coef 0 |
| Metric | warm-step (step1/step2) actor train wall time, tokens/s, peak reserved memory; step0 excluded (cold JIT) |

Noise floor for matched-step comparisons is ±2%; conclusions use the mean of the
two warm steps. Memory is reported in GiB (`torch.cuda.max_memory_* / 1024**3`).

### Quick Start

#### 1. Build the image (from repository root)

```bash
docker build -f dockers/Dockerfile.Qwen35SlimeGoLongRL -t ppu-qwen35-golongrl:M890 .
# If your host needs a proxy for the math_verify pip install:
#   docker build --build-arg GIT_PROXY=http://<proxy>:<port> -f dockers/Dockerfile.Qwen35SlimeGoLongRL -t ppu-qwen35-golongrl:M890 .
```

The Dockerfile applies two combined patches — `qwen35-golongrl-megatron.patch` to
the Megatron-LM tree and `qwen35-golongrl-slime.patch` to the slime tree (the latter
includes the GoLongRL reward plugin under `slime_plugins/rm/` as new files) — then
installs `math_verify==0.9.0` and runs static landing checks (patch-effect anchors +
`py_compile`). The two patches were consolidated from the original per-feature set
and verified byte-identical to its sequential apply on the pristine base image.

#### 2. Create the container (host)

```bash
docker run -dit --name golongrl_m890 \
    --network=host --ipc=host --pid=host --privileged --init \
    --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
    --device=/dev/infiniband --device=/dev/alixpu_ctl --device=/dev/alixpu \
    -v /mnt:/mnt -v /cpfs:/cpfs \
    ppu-qwen35-golongrl:M890 /bin/bash
```

#### 3. Prepare weights and data (inside container)

Set the storage roots to your layout first (config block at the top of each
script); the defaults are neutral placeholders.

```bash
cd /path/to/sail-train-recipes/best_practice/qwen3.5
export MODEL_ROOT=/path/to/models NAS_ROOT=/path/to/golongrl
export HF_CKPT=${MODEL_ROOT}/Qwen3.5-35B-A3B

# TP2 weights for the 64k/128k bands:
bash golongrl_prepare_weights.sh
# TP1 weights for the 32k/256k bands (conversion TP must match training TP):
CONVERT_TP=1 SAVE_DIR=${MODEL_ROOT}/Qwen3.5-35B-A3B_slime_torch_dist_gdn_tp1 bash golongrl_prepare_weights.sh

# Band jsonl (HF download -> token counting -> banding -> slime jsonl):
bash golongrl_prepare_data.sh                 # all four bands; BAND=32k for a single band
```

#### 4. Train / tune (inside container)

`GDN_TORCH_DIST` auto-resolves to the `_tp${TP}` directory matching each band's training TP (32k/256k → TP1, 64k/128k → TP2); set it explicitly to override.

```bash
# 32k recommended config (TP1 CP4 EP8, DP2, no recompute):
BAND=32k bash tune_slime_golongrl.sh

# 128k recommended config (TP2 CP4 EP8, block-recompute 20 layers, expandable on):
BAND=128k bash tune_slime_golongrl.sh

# 256k (TP1 CP8 EP8, full-uniform recompute, expandable on; auto-resolves TP1 weights):
BAND=256k bash tune_slime_golongrl.sh
```

### Performance Results

Validated on 8x M890, bf16, GRPO `n=8`, rollout batch 16 / global batch 128.
All times are the mean of warm steps 1-2 (step0 cold-JIT excluded), read from slime's
`perf/*` metrics. The four time columns are nested calibers, not alternatives:
- **Rollout** = `rollout_time` — SGLang generation.
- **Actor train** = `actor_train_time` — Megatron fwd/bwd/optimizer only.
- **Train phase** = `train_time` — actor train + ref_log_probs + data_preprocess.
- **Full step** = `step_time` — end-to-end (rollout + train phase + weight sync + sleep/wake).

`Train tok/s` = `actor_train_tok_per_s` (training tokens/s, pairs with Actor train).
`Reserved` is the peak train-phase reserved memory per card (144 GiB capacity).

| Band | Parallelism | Recompute | Rollout (s) | Actor train (s) | Train phase (s) | Full step (s) | Train tok/s | Reserved (GiB) |
|------|-------------|-----------|------------:|----------------:|----------------:|--------------:|------------:|---------------:|
| 32k  | TP1 CP4 EP8 (DP2) | off          | 162.8 | 148.7  | 200.1  | 398.9  | 24938 | 124.0 |
| 64k  | TP2 CP4 EP8 (DP1) | off          | 244.2 | 381.0  | 507.5  | 772.7  | 20962 | 126.1 |
| 128k | TP2 CP4 EP8 (DP1) | block (20)   | 586.4 | 940.7  | 1205.1 | 1806.8 | 15986 | 129.5 |
| 256k | TP1 CP8 EP8 (DP1) | full-uniform | 945.4 | 2890.3 | 3581.6 | 4543.5 | 10210 | 136.1 |

Reproduce a row (inside container, after weights/data prep):

```bash
# 32k
BAND=32k NUM_ROLLOUT=3 EXP_TAG=perf32k bash tune_slime_golongrl.sh
# 128k
BAND=128k NUM_ROLLOUT=3 EXP_TAG=perf128k bash tune_slime_golongrl.sh
```

### Files

| File | Description |
|------|-------------|
| `tune_slime_golongrl.sh` | Training/tuning entry with per-band validated defaults. |
| `golongrl_prepare_weights.sh` | HF -> GDN torch_dist weight conversion. |
| `golongrl_prepare_data.sh` | GoLongRL HF -> banded slime jsonl (idempotent). |
| `../../dockers/Dockerfile.Qwen35SlimeGoLongRL` | Image build (base + adaptation patches + reward). |
| `../../dockers/patches/qwen35-golongrl-megatron.patch` | Megatron-LM adaptation (GDN CP + contiguous a2a). |
| `../../dockers/patches/qwen35-golongrl-slime.patch` | slime adaptation (GDN/TE-CE/IPC/FLOPs + reward plugin). |

### Environment Variables

| Variable | Meaning |
|----------|---------|
| `BAND` | Context band `32k` / `64k` / `128k` / `256k`; selects per-band parallelism + recompute defaults. |
| `MODEL_ROOT` / `NAS_ROOT` | Storage roots (override to your layout). |
| `HF_CKPT` / `GDN_TORCH_DIST` | HF snapshot / converted GDN weights (use `_tp1` for TP1 bands). |
| `RECOMPUTE` / `RECOMPUTE_NUM_LAYERS` | Recompute mode (`off`/`on`/`selective`/`block`) and block layer count. |
| `TP_OVERRIDE` / `CP_OVERRIDE` / `MTPG_OVERRIDE` / `RECOMPUTE_OVERRIDE` | Override per-band parallelism / max-tokens-per-gpu / recompute. |
| `NUM_ROLLOUT` / `SAVE_INTERVAL` / `EXP_TAG` | Rollouts / save cadence / ckpt-dir tag. |
| `OFFLOAD` | Optimizer CPU offload + D2H/H2D overlap (`on` by default). |
| `DP_ATTN` / `RADIX` | SGLang DP attention / radix prefix cache. |
| `GCS_PORT` / `DASH_PORT` | Ray ports (avoid the 10002-19999 worker range). |
| `EXTRA_TRAIN_ENV_JSON` | JSON object of extra Ray runtime `env_vars` (ablation knob). |

The script also defaults `PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1` (32k ablation: -3.0% train time), per-band `SLIME_ENABLE_EXPANDABLE_SEGMENTS` (requires the IPC patch), `SLIME_USE_TE_CROSS_ENTROPY=0`, and `MOE_GROUPED_GEMM=0`; override via the environment or `EXTRA_TRAIN_ENV_JSON`.
