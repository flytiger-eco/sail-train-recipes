# Qwen3-Omni-30B-A3B-Instruct Best Practices

Training and MFU tuning recipes for **Qwen3-Omni-30B-A3B-Instruct** on **PPU 890P/810E**.

## Files

| File | Description |
|------|-------------|
| `train_qwen3Omni_thinker_full.sh` | Qwen3-Omni-30B-A3B thinker-only training (single-node / multi-node) |
| `prepare_videomme_data.py` | Video-MME data preparation script (video trimming + JSONL generation) |

## Performance Results

### 890P

Experiment Setup:
- 1 node for seq_len 8k (8 PPU 890P per node)
- 2 nodes for seq_len 16k (16 PPU 890P in total)

| Model | SEQ_LEN | NUM_IMAGES | IMAGE_RESOLUTION | MBS | GA | PARALLEL_MODE | Avg TFLOPS/s | Notes |
|---|---|---|---|---|---|---|---|---|
| 30B-A3B | 8192 | 10 | 1280 x 720 | 1 | 8 | EP=8,CP=1 | 100 | |
| 30B-A3B | 16384 | 20 | 1280 x 720 | 1 | 16 | EP=8,CP=2 | 98 | |

### 810E

Experiment Setup:
- 1 node for both seq_len (16 PPU 810E per node)
- Dataset `test_video_only.jsonl` (video), `RECOMPUTE_NUM_LAYERS=12`, few iterations; TP=1 / EP=8 / ETP=1, `use_distributed_optimizer=True`, no offload / PAO / fp8, freeze vision+audio and train thinker text only, flash_attention_3 + full recompute

| Model | SEQ_LEN | NUM_IMAGES | IMAGE_RESOLUTION | MBS | GA | PARALLEL_MODE | Avg TFLOPS/s | Notes |
|---|---|---|---|---|---|---|---|---|
| 30B-A3B | 8192 | 10 | 1280 x 720 | 1 | 1 | EP=8,CP=1 | 42 | Peak 77.1GB/96GB; 20/20 iters |
| 30B-A3B | 16384 | 20 | 1280 x 720 | 1 | 1 | EP=8,CP=1 | 44 | Peak 97.85GB (~91GiB), headroom ~5GiB; 3/3 iters |

## 890P Quick Start

### 1. Build image & Launch container

```bash
# Run from repo root of sail-train-recipes
# This image supports both PPU 890P and 810E
docker build -f dockers/Dockerfile.Qwen3Omni -t ppu-qwen3omni:890p .
docker run --privileged --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --init -dt \
  -v /path/to/data:/data \
  --name=qwen3omni_sail_train \
  ppu-qwen3omni:890p /bin/bash
```

### 2. Download model

Use modelscope to download Qwen3-Omni-30B-A3B-Instruct:

```bash
export MODELSCOPE_CACHE=/path/to/models
modelscope download --model Qwen/Qwen3-Omni-30B-A3B-Instruct
```

Or use HuggingFace mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local-dir /path/to/models/Qwen3-Omni-30B-A3B-Instruct
```

Set the model path after download:

```bash
export HF_MODEL_PATH=/path/to/models/Qwen3-Omni-30B-A3B-Instruct
```

### 3. Download & Prepare data

#### 3a. Download Video-MME dataset

```bash
export HF_ENDPOINT=https://hf-mirror.com

# Download dataset (parquet + subtitles)
huggingface-cli download lmms-lab/Video-MME --repo-type dataset \
  --local-dir /path/to/data/Video-MME
```

#### 3b. Generate training JSONL from raw data

Install `ffmpeg` inside the container as root before preparing data:

```bash
apt-get update && apt-get install -y ffmpeg
```

Use `prepare_videomme_data.py` to process raw videos, trim to the first 10 seconds and generate JSONL:

```bash
python3 prepare_videomme_data.py \
  --video-dir    /path/to/data/Video-MME-uncom/data \
  --parquet      /path/to/data/Video-MME/videomme/test-00000-of-00001.parquet \
  --subtitle-dir /path/to/data/Video-MME/subtitle \
  --output-dir   /path/to/data/videomme_mbridge/test \
  --trim-seconds 10 \
  --fps 2.0
```

Output structure:

```
/path/to/data/videomme_mbridge/test/
├── videos/      (900 mp4 files, 10 seconds each)
├── audios/      (900 wav files, 16kHz mono, 10 seconds each)
└── test.jsonl   (2700 entries, OpenAI conversation format)
```

#### 3c. Use pre-processed data (recommended)

If you already have pre-processed data, simply set the path:

```bash
export TRAIN_JSONL=/path/to/data/videomme_mbridge/test/test.jsonl
```

Data format example:

```json
{
  "conversation": [
    {
      "role": "user",
      "content": [
        {"type": "video", "video": "/path/to/videos/xxx.mp4"},
        {"type": "audio", "audio": "/path/to/audios/xxx.wav"},
        {"type": "text",  "text": "[subtitle]\n[00:00:00.220] ..."},
        {"type": "text",  "text": "Question...\nA. ...\nB. ...\nC. ...\nD. ..."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "C"}]
    }
  ],
  "metadata": {
    "data_source": "Video-MME",
    "video_id": "xxx",
    "question_id": "001-1",
    "domain": "Knowledge",
    "task_type": "Counting Problem"
  }
}
```

### 4. One-command training launch

```bash
export HF_MODEL_PATH=/path/to/models/Qwen3-Omni-30B-A3B-Instruct
export TRAIN_JSONL=/path/to/data/videomme_mbridge/test/test.jsonl

bash train_qwen3Omni_thinker_full.sh
```

The script defaults to PPU optimal config (`ATTENTION_BACKEND=flash`, `RECOMPUTE_MODULES=null`, `MULTIMODAL_ATTN_IMPL=flash_attention_3`), no need to specify additional performance parameters.

To customize:

```bash
SEQ_LENGTH=8192 GLOBAL_BATCH_SIZE=8 MICRO_BATCH_SIZE=1 \
TRAIN_ITERS=100 \
bash train_qwen3Omni_thinker_full.sh
```

## 810E Quick Start

### 1. Build image

```bash
# Run from repo root of sail-train-recipes
# Get current IPv4 first: getent ahostsv4 github.com (IP may change, re-fetch if expired)
docker build --network=host --add-host github.com:20.205.243.166 \
  -f dockers/Dockerfile.Qwen3Omni \
  -t ppu-qwen3omni:810e .
```

### 2. Launch container

```bash
docker run --privileged --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --init -dt \
  -v /mnt:/mnt \
  --name=qwen3omni_810e \
  ppu-qwen3omni:810e /bin/bash
```

### 3. Launch training (16-GPU tested config)

```bash
docker exec -it qwen3omni_810e bash

export HF_MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct
export TRAIN_JSONL=/path/to/test_video_only.jsonl

NUM_GPUS=16 EXPERT_MODEL_PARALLEL_SIZE=8 GLOBAL_BATCH_SIZE=16 \
MOE_TOKEN_DISPATCHER_TYPE=flex RECOMPUTE_NUM_LAYERS=12 \
SEQ_LENGTH=8192 \
bash train_qwen3Omni_thinker_full.sh
```

### 4. Configuration guide (Parallelism / offload / CP selection)

> See "Performance Results - 810E" at the top for measured performance and memory data. This section only covers configuration choices.

- **No need for TP/ETP/offload/FSDP**: With 16 GPUs, TP=1 yields dense-DP=16, and the distributed optimizer shards optimizer state across 16 GPUs, significantly reducing per-card memory usage. Both 8k and 16k can run directly. (By contrast, in the 8-GPU era dense-DP was only 2 and the optimizer was sharded across only 2 GPUs; TP=1/EP=8 would OOM, forcing TP=4/EP=2/ETP=4/SP/offload just to barely fit.)
- **Single-node works without CP even at 16k**: flash attention keeps attention memory growing linearly with sequence + full recompute, fitting 16k within 96GiB; when headroom is thin, increase `RECOMPUTE_NUM_LAYERS` first (e.g., 24/48) as fallback. Activation linear extrapolation up to **~32k before needing CP>=2**. (Note: The 890P 16k official benchmark uses 2 nodes x 16 GPUs + CP=2, which is a different path from the single-node 16-GPU CP=1 setup here.)

## Single-node training

Single-node commands are **hardware-agnostic**. The only machine-specific adjustment is `NUM_GPUS` (= local available GPU count): 890P single-node typically has 8 GPUs, 810E single-node has 16 GPUs. See respective Quick Start above for other parallel parameters. No need to duplicate commands per hardware.

890P (8 GPUs, PPU optimal config):
```bash
export HF_MODEL_PATH=/path/to/models/Qwen3-Omni-30B-A3B-Instruct
export TRAIN_JSONL=/path/to/data/videomme_mbridge/test/test.jsonl

ATTENTION_BACKEND=flash OVERLAP_PARAM_GATHER=True OVERLAP_GRAD_REDUCE=True RECOMPUTE_GRANULARITY=full RECOMPUTE_METHOD=block RECOMPUTE_NUM_LAYERS=26 VIT_GRADIENT_CHECKPOINTING=False PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=True  MULTIMODAL_ATTN_IMPL=flash_attention_3 bash train_qwen3Omni_thinker_full.sh
```

810E (16 GPUs, tested config, see "810E Quick Start" for details):
```bash
NUM_GPUS=16 EXPERT_MODEL_PARALLEL_SIZE=8 GLOBAL_BATCH_SIZE=16 \
MOE_TOKEN_DISPATCHER_TYPE=flex RECOMPUTE_NUM_LAYERS=12 \
bash train_qwen3Omni_thinker_full.sh
```

> Warning: Without `NUM_GPUS`, it uses default value **8**; on a 16-GPU 810E machine this only uses half the GPUs. You must explicitly set `NUM_GPUS=16`.

## Dual-node training (890P x 2, ICN interconnect)

> Multi-node training is bound to network interfaces / interconnect topology, and is truly hardware-dependent. Therefore, commands are provided per platform. Below is an 890P x 2 example; 810E multi-node requires per `NCCL_SOCKET_IFNAME` and interconnect method for adaptation.

Node 0 (MASTER):
```bash
export HF_MODEL_PATH=/path/to/models/Qwen3-Omni-30B-A3B-Instruct
export TRAIN_JSONL=/path/to/data/videomme_mbridge/test/test.jsonl

NNODES=2 NODE_RANK=0 MASTER_ADDR=<MASTER_ADDR_IP> MASTER_PORT=29513 \
NCCL_SOCKET_IFNAME=bond1 \
EXPERT_MODEL_PARALLEL_SIZE=8 GLOBAL_BATCH_SIZE=16 \
bash train_qwen3Omni_thinker_full.sh
```

Node 1:
```bash
export HF_MODEL_PATH=/path/to/models/Qwen3-Omni-30B-A3B-Instruct
export TRAIN_JSONL=/path/to/data/videomme_mbridge/test/test.jsonl

NNODES=2 NODE_RANK=1 MASTER_ADDR=<MASTER_ADDR_IP> MASTER_PORT=29513 \
NCCL_SOCKET_IFNAME=bond1 \
EXPERT_MODEL_PARALLEL_SIZE=8 GLOBAL_BATCH_SIZE=16 \
bash train_qwen3Omni_thinker_full.sh
```

## Environment variables reference

| Variable | Default | Description |
|------|--------|------|
| `HF_MODEL_PATH` | (required) | Local path to HF model |
| `TRAIN_JSONL` | (required) | Path to training data JSONL |
| `SEQ_LENGTH` | `8192` | Sequence length |
| `TRAIN_ITERS` | `20` | Training iterations |
| `GLOBAL_BATCH_SIZE` | `8` | Global batch size |
| `MICRO_BATCH_SIZE` | `1` | Per-GPU micro batch size |
| `TENSOR_PARALLEL_SIZE` | `1` | TP parallelism size |
| `PIPELINE_PARALLEL_SIZE` | `1` | PP parallelism size |
| `EXPERT_MODEL_PARALLEL_SIZE` | `8` | MoE EP parallelism size |
| `SEQUENCE_PARALLEL` | `False` | Sequence parallel |
| `MOE_TOKEN_DISPATCHER_TYPE` | `flex` | MoE token dispatcher (`flex`=DeepEP; fallback to `alltoall`/`allgather`) |
| `MOE_FLEX_DISPATCHER_BACKEND` | `deepep` | flex dispatcher backend (`deepep`/`hybridep`/`ncclep`) |
| `MOE_GROUPED_GEMM` | `True` | MoE grouped GEMM (companion to DeepEP) |
| `MOE_PERMUTE_FUSION` | `True` | Fused permutation in dispatch phase |
| `MOE_DEEPEP_NUM_SMS` | (empty) | Number of SMs occupied by DeepEP communication kernel (optional) |
| `ATTENTION_BACKEND` | `flash` | LLM attention backend (PPU requires flash) |
| `MULTIMODAL_ATTN_IMPL` | `flash_attention_3` | Multimodal attention implementation |
| `RECOMPUTE_GRANULARITY` | `full` | Activation recompute granularity |
| `RECOMPUTE_METHOD` | `block` | Recompute method |
| `RECOMPUTE_NUM_LAYERS` | `26` | Number of recompute layers |
| `RECOMPUTE_MODULES` | `null` | Recompute modules |
| `VIT_GRADIENT_CHECKPOINTING` | `False` | ViT gradient checkpointing |
| `OPTIMIZER_CPU_OFFLOAD` | `False` | Optimizer CPU offload |
| `OVERLAP_GRAD_REDUCE` | `True` | Overlap gradient reduce |
| `OVERLAP_PARAM_GATHER` | `True` | Overlap parameter gather |
| `FREEZE_LANGUAGE_MODEL` | `False` | Freeze language model |
| `FREEZE_VISION_MODEL` | `True` | Freeze vision model |
| `FREEZE_AUDIO_MODEL` | `True` | Freeze audio model |
| `DATASET_NUM_WORKERS` | `8` | Number of data loading workers |
| `NNODES` | `1` | Number of nodes |
| `NODE_RANK` | `0` | Current node rank |
| `MASTER_ADDR` | `127.0.0.1` | Master node address |
| `MASTER_PORT` | `29500` | Master node port |
| `MBRIDGE_ROOT` | `/workspace/Megatron-Bridge` | Megatron-Bridge root directory |
| `DRY_RUN` | `0` | Set to 1 to print commands only without execution |

## How the script works

1. Locate Megatron-Bridge root directory (`MBRIDGE_ROOT`), set `PYTHONPATH`
2. Create thinker-only model mirror (disable audio output / talker) to avoid loading unnecessary modules during training
3. Build training config via `run_recipe.py --recipe qwen3_omni_30b_a3b_sft_hf_json_config`
4. Launch distributed training using `torchrun`
5. Output performance summary after training (iter time, tokens/sec, TFLOP/s/GPU)
