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

## GoLongRL Long-Context RL with verl 0.9.1 (8x M890)

Single-node (8x ZW-M890P, 144GiB/card) GRPO recipe for Qwen3.5-35B-A3B on the GoLongRL long-context bands (32k/64k/128k/256k), built on verl 0.9.1 (Megatron actor + colocate sglang rollout). The GDN context-parallel path is switched from headwise to **chunkwise CP** (ported from the validated slime m890 adaptation): the outer data plane stays **zigzag** (Megatron-Bridge packs embeddings with align=tp*cp*2), and chunkwise GDN layers convert zigzag->contiguous internally via fla `cp_context`. Chunkwise removes the headwise `linear_num_value_heads % (tp*cp) == 0` constraint, unlocking TP1 combinations (TP1 auto-disables sequence parallel, +~40GB activations). verl loads HF weights directly through Megatron-Bridge — no offline torch_dist conversion; `main_ppo` starts its own ray head.

### Experiment Setup

| Item | Value |
|------|-------|
| Hardware | 8x ZW-M890P (144GiB/card), single node |
| Base image | `pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.1-2.11.0-ubuntu24.04-cuda13.0-sglang-verl0.9.1-py312-20261113` |
| Frameworks | verl 0.9.1 (editable `/opt/verl`), megatron-core 0.18.0+ba7b5ebce, megatron-bridge 0.5.2, TE 2.16+v0.1.0.ppu2.2.0, sglang 0.5.12+v0.2.0.ppu2.2.0, fla 0.5.2, torch 2.11.0 PPU |
| TMS | zhuzilin `torch_memory_saver` 0.0.9.post1+slime @bdab8c5 (cu13 .so; VMM preload hooks coexist with `expandable_segments`) |
| Model | Qwen3.5-35B-A3B (HF; Megatron-Bridge provider, `vanilla_mbridge=False`) |
| Data | [Kwai-Klear/GoLongRL](https://huggingface.co/datasets/Kwai-Klear/GoLongRL), banded parquets `golongrl-verl-bands/band{32k,64k,128k,256k}/{train,val}.parquet` |
| Precision | bf16 train / rollout; GRPO, lr 1e-6, batch 16 x n8, mini-batch 16 |

Band configs (validated defaults; rollout always TP8-EP8; `expandable_segments:True` on for all bands):

| Band | Prompt/Resp | TP | CP | EP | DP | Recompute | Offload frac | Engine util |
|------|-------------|----|----|----|----|-----------|--------------|-------------|
| 32k  | 28600/4096  | 1 | 4 | 8 | 2 | off | 1.0 | 0.7 |
| 64k  | 61380/4096  | 2 | 4 | 8 | 1 | off | 1.0 | 0.7 |
| 128k | 122800/8192 | 2 | 4 | 8 | 1 | block(20) | 1.0 | 0.7 |
| 256k | 253900/8192 | 1 | 8 | 8 | 1 | full-uniform | 0.8 | 0.55 |

### Quick Start

```bash
# 1. Build the image (from repository root; CPU/network job, no GPU needed)
docker build -f dockers/Dockerfile.Qwen35VerlGoLongRL -t ppu-qwen35-golongrl-verl:M890 .

# 2. Start the container (shared host: never ray stop; --pid=host caveat)
docker run -d --name golongrl_verl --privileged --shm-size=64g \
    --network=host --ipc=host --init --pid=host \
    --device /dev/infiniband --device /dev/alixpu_ctl --device /dev/alixpu \
    -v /mnt:/mnt -v /path/to/workspace:/workspace \
    ppu-qwen35-golongrl-verl:M890 sleep infinity

# 3. Prepare weights (HF download + integrity check; no TP conversion) and
#    data (B1 full tokenize ~1-2h; idempotent, resumable)
docker exec golongrl_verl bash /workspace/best_practice/qwen3.5/golongrl_prepare_weights_verl.sh
docker exec golongrl_verl bash -c 'DATA_DIR=/workspace/golongrl/datas bash /workspace/best_practice/qwen3.5/golongrl_prepare_data_verl.sh'

# 4. Train (preflight refuses unless all 8 cards are idle)
docker exec golongrl_verl bash -c 'BAND=32k NAS_ROOT=/workspace/golongrl bash /workspace/best_practice/qwen3.5/tune_verl_golongrl.sh'
# Other bands / probes:
#   BAND=256k bash tune_verl_golongrl.sh
#   BAND=32k TP_OVERRIDE=2 CP_OVERRIDE=4 EXP_TAG=dp1 bash tune_verl_golongrl.sh
#   BAND=128k RECOMPUTE_OVERRIDE=off bash tune_verl_golongrl.sh   # memory-boundary probe
```

### Performance Results

Validated on 8x M890, bf16, GRPO `n=8`, rollout batch 16 / global batch 128. All times are the mean of warm steps 2-5 (step 1 cold-JIT excluded), read from verl's `timing_s/*` metrics. The three time columns are nested calibers, not alternatives:
- **Rollout** = `timing_s/gen` — sglang generation.
- **Actor train** = `timing_s/update_actor` — Megatron fwd/bwd/optimizer only.
- **Full step** = `timing_s/step` — end-to-end (rollout + old_log_prob + ref + actor train + weight sync).

`Step tok/s` = `perf/throughput` (end-to-end tokens/s, pairs with Full step). `Reserved` is the peak actor reserved memory per card (`actor/perf/max_memory_reserved_gb`, GiB = `max_memory_reserved / 1024**3`; 144 GiB capacity). rollout↔actor logprob pearson ≥ 0.9985 across all runs confirms the chunkwise CP forward matches the sglang rollout numerically.

| Band | Parallelism | Recompute | Rollout (s) | Actor train (s) | Full step (s) | Step tok/s | Reserved (GiB) |
|------|-------------|-----------|------------:|----------------:|--------------:|-----------:|--------------:|
| 32k  | TP1 CP4 EP8 (DP2) | off          | 122.4 | 171.4  | 420.5  | 1102 | 113.7 |
| 64k  | TP2 CP4 EP8 (DP1) | off          | 216.9 | 440.9  | 964.6  | 1038 | 109.6 |
| 128k | TP2 CP4 EP8 (DP1) | block (20)   | 533.8 | 1041.4 | 2199.5 | 856  | 135.0 |
| 256k | TP1 CP8 EP8 (DP1) | full-uniform | 1188.6 | 3165.0 | 5922.9 | 642  | 135.1 |

Reproduce a row (inside the container, after weights/data prep):

```bash
# 32k
BAND=32k EXP_TAG=perf32k NAS_ROOT=/workspace/golongrl bash tune_verl_golongrl.sh
# 128k
BAND=128k EXP_TAG=perf128k NAS_ROOT=/workspace/golongrl bash tune_verl_golongrl.sh
```

### Files

| File | Description |
|------|-------------|
| `tune_verl_golongrl.sh` | Training/tuning entry with per-band validated defaults, preflight, and `TUNE_STATUS: OK/FAILED` marker. |
| `golongrl_prepare_weights_verl.sh` | HF weight presence check / download fallback / architecture integrity validation. |
| `golongrl_prepare_data_verl.sh` | GoLongRL download -> token count -> band windows -> verl 5-column parquets (idempotent). |
| `../../dockers/Dockerfile.Qwen35VerlGoLongRL` | Image build (official verl 0.9.1 base + TMS + adaptation patches + final guard). |
| `../../dockers/patches/verl-golongrl-m890/` | Patches 01-09 + idempotent `apply_patches.sh` (verl git tree + megatron/sglang site-packages). |

### Environment Variables

| Variable | Meaning |
|----------|---------|
| `BAND` | Context band `32k` / `64k` / `128k` / `256k`; selects per-band validated defaults. |
| `NAS_ROOT` | Root for models/datas/logs (`/workspace/golongrl` by default). |
| `TP_OVERRIDE` / `CP_OVERRIDE` / `EP_OVERRIDE` | Actor parallelism overrides for probes (TP*CP must divide 8, EP<=8). |
| `RECOMPUTE_OVERRIDE` | Recompute override: `off` / `on` / `block` / `selective`. |
| `TOTAL_STEPS` | Training steps for the smoke/tuning run (`5` by default). |
| `EXP_TAG` | Experiment tag in log/experiment names. |

See the script header for the remaining knobs (offload fraction, engine memory util, CP mode, extra hydra overrides).
