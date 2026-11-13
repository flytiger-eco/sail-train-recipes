# Qwen3.8-Flash-Next Best Practices

Single-node training recipe for **Qwen3.8-Flash-Next** (125B total / A6B activated
MoE) on **PPU M890**, as a baseline for later MFU tuning.

## Why this recipe is not on Megatron-Bridge

The model is a hybrid stack: 36 of 48 decoder layers are Gated DeltaNet and every
fourth layer is QSA block-sparse attention (`indexer_budget` 2048), on top of
Hyper-Connections gated residual streams (4 streams) and a PLE / N-gram embedding
whose table is far larger than the transformer body. NVIDIA Megatron-LM main has
GDN and hyper-connections, but no QSA and no PLE/N-gram layer, and
`megatron.bridge.models.qwen` has no Qwen3.8-Flash-Next model. The public recipe
therefore runs the model through **ms-swift's `megatron sft` CLI**, which uses
ModelScope's **mcore-bridge** for the model (`qwen4_exp`) and **megatron-core** as
the training engine. `dockers/Dockerfile.Qwen38NextFlash` builds that stack; it is
unrelated to `dockers/Dockerfile.Qwen35`, which builds NVIDIA Megatron-Bridge.

| Item | Configuration |
|------|---------------|
| Hardware | PPU M890P (147 GB), ICN interconnect, 8 PPUs per node |
| Software | `ppu-qwen38-next-flash:M890` (`dockers/Dockerfile.Qwen38NextFlash`) |
| Code | ms-swift 4.6.0 + mcore-bridge 1.6.5 + megatron-core 0.18.2 |
| Precision | bf16 |
| Fine-tuning | LoRA r8 / alpha 32 on `in_proj out_proj linear_proj linear_qkv` |
| Layout | TP=2, PP=2 (24/24 layers), EP=4, ETP=1, DP=1, sequence parallel, padding-free |
| Metric | `train_speed(s/it)` and `memory(GiB)` from the CLI's per-step `logging.jsonl`; tokens/s and TFLOP/s/GPU derived, see [Metrics](#metrics) |

## M890 Quick Start

```bash
docker build -f dockers/Dockerfile.Qwen38NextFlash -t ppu-qwen38-next-flash:M890 .
docker run --ipc=host --network=host \
  --device=/dev/alixpu --device=/dev/alixpu_ctl --device=/dev/alixpu-caps \
  --device=/dev/alixpu-caps-imex-channels \
  --device=/dev/alixpu_ppu0 --device=/dev/alixpu_ppu1 --device=/dev/alixpu_ppu2 \
  --device=/dev/alixpu_ppu3 --device=/dev/alixpu_ppu4 --device=/dev/alixpu_ppu5 \
  --device=/dev/alixpu_ppu6 --device=/dev/alixpu_ppu7 \
  --ulimit memlock=-1 --shm-size=10.24gb --ulimit stack=67108864 --init -dt \
  -v /path/to/models:/models \
  -v $(pwd):/workspace/sail-train-recipes \
  --name=qwen38_next_flash ppu-qwen38-next-flash:M890 /bin/bash
```

All eight `alixpu_ppuN` nodes have to be passed explicitly; with a partial list
`ppu-smi` prints nothing and the ranks fail at device init.

Then, inside the container:

```bash
export HF_MODEL=/models/Qwen3.8-Flash-Next
SEQ_LEN=8192 TRAIN_ITERS=40 bash best_practice/qwen3.8-next-flash/tune_next_flash.sh
```

The checkpoint must be the released HF-format directory (config, tokenizer and
safetensors shards). `tune_next_flash.sh` generates its own fixed-length data with
`bench_data.py`, so the token count per step is exact: `SEQ_LEN * MBS * GA`.

`PLE_CPU_OFFLOAD=1` (the script default) keeps the N-gram table in host memory.
With the table on the PPU the run does not fit on one node; the flag freezes that
table, which is the intended behaviour for the released checkpoint.

## Baseline Results

One node, 8 M890P, 8192-token samples, MBS=1, GA=8 (65536 tokens per optimizer
step), LoRA r8, 40 optimizer steps. Step time is the median of iterations 12–20
(the per-step `train_speed(s/it)` the CLI logs; the log truncates to whole
seconds, so the mean of iterations 2–40, 24.5 s, is 1% faster). Step 1 is
excluded: it carries the Triton JIT of the QSA / PLE / GDN kernels.

| Seq | MBS | GA | TP | PP | EP | Step time (s) | Tokens/s | Tokens/s/GPU | TFLOP/s/GPU | Memory (GiB/PPU) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8192 | 1 | 8 | 2 | 2 | 4 | 25.0 | 2,621 | 328 | 11.8 | 46.1 |

All 40 steps take 18 m 52 s (1132 s), loss 0.0399 → 0.0021, grad norm settling
around 0.03. Every number above comes from the published image
`ppu-qwen38-next-flash:M890` running `tune_next_flash.sh` unchanged.

## Measurement Breakdown

Wall-clock phases of the 40-step run on a warm node (weights and the base image
already in page cache):

| Phase | Wall time |
|-------|-----------|
| Bench data generation (320 samples x 8192 tokens, `bench_data.py`) | 38 s |
| Checkpoint load (336 GB HF weights, 8 ranks) + model build | ~15 s |
| First step (Triton cache warm; 9 m 12 s from a cold container) | 175 s |
| Steady-state step (mean of iterations 2–40) | 24.5 s |
| Adapter save (189 MB, `--merge_lora false`) | 38 s |
| Merged safetensors export (331 GB, 118 shards) — measured separately | 2 m 47 s |
| **40-step run, end to end** | **20 m 23 s** |

The 331 GB merged export is off by default in `tune_next_flash.sh`
(`MERGE_LORA=false`): it rewrites the whole checkpoint to produce a deployable
model, which a throughput measurement does not need.

## Metrics

* **Tokens/s** — `SEQ_LEN * MBS * GA / step_time`. With padding-free training and
  equal-length samples this is exact; a padded run would count the pad.
* **TFLOP/s/GPU** — `6 * N_activated * tokens_per_second / 8`, with
  `N_activated = 6e9` (the A6B figure). This is the standard training-FLOPs
  convention and is a **lower bound** here: it does not count the QSA indexer and
  sparse attention, the GDN chunked kernels, the N-gram gather, or the padding
  that grouped GEMM applies to experts with few tokens. `megatron-core`'s own
  `throughput per GPU (TFLOP/s/GPU)` counter is not reachable from the ms-swift
  CLI (its `--megatron-extra-kwargs` feeds `TransformerConfig`, not the logger
  config), so the CI metric is computed from the step time instead.
* **Memory** — `memory(GiB)` in `logging.jsonl`, the peak allocated per rank. It
  excludes the N-gram table, which `PLE_CPU_OFFLOAD=1` keeps in host memory.

## What this baseline does not cover

* **Cold start.** The first step of a freshly started container is 9 m 12 s
  (Triton kernels and the weight page cache both cold) instead of 175 s. A
  latency-sensitive job should warm the container before measuring.

* **Long context.** 8192 tokens is the sweep point; the QSA budget (2048) is
  exercised, the model's 262k maximum is not.
* **Parallelism sweeps.** Only TP=2 / PP=2 / EP=4 was measured. PP=2 with
  `--decoder_first_pipeline_num_layers 12` (the value the upstream practice guide
  publishes) splits 48 layers 12/36 and is visibly slower than the 24/24 split
  used here.
* **`causal_conv1d` C++ extension.** It is not installed, so GDN's depthwise
  convolution runs through the FLA torch fallback. flash-linear-attention itself
  is required and is installed.
* **Full fine-tuning.** LoRA r8 only; the frozen 125B body still streams through
  every step.
