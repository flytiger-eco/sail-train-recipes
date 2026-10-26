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
