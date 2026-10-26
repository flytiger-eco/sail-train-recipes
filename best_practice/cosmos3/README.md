# Cosmos3 Nano / Super M890 Performance Recipe

This directory contains only the reproduction instructions. The actual training entry points, TOML configurations, and dependencies are added to the image by `Dockerfile.Cosmos3` and `cosmos3.patch`.

## Baseline

- Source: <https://github.com/NVIDIA/cosmos-framework.git>
- Commit: `300faa14daab3910be9d303c31708a0a1d6e4371`
- Dockerfile: `dockers/Dockerfile.Cosmos3`
- Patch: `dockers/patches/cosmos3.patch`
- Hardware: One node with 8 M890 144 GiB cards
- Input: 1,000 synthetic videos at 1920×1200 resolution with 81 frames each
- Nano: 8-card FSDP, CP=1
- Super: DP=4, CP=2

`torch.compile` is used only for the language component; compilation is disabled for the VAE and vision components. `PYTORCH_ALLOC_CONF=expandable_segments:True` is set only by the Super launcher and is not set globally in the Dockerfile or the Nano launcher.

## Training Entry Points

The training TOML files and launch logic are provided under `/workspace/cosmos-framework/examples` in the image:

- Nano: `launch_sft_vision_nano_1920_1200_profile.sh`
- Super: `launch_sft_vision_super_1920_1200_sft.sh`

## Build the Image

```bash
cd /path/to/sail-train-recipes
export BASE_IMAGE=pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.0-2.10.0-ubuntu24.04-cuda13.0-py312-20261001
DOCKER_BUILDKIT=0 docker build --network=host \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f dockers/Dockerfile.Cosmos3 \
  -t ppu-cosmos3:M890 .
```

The Dockerfile explicitly uses Alibaba Cloud mirrors for Ubuntu and PyPI. If the FlyTiger tag above has not yet been published to the current registry, replace only `BASE_IMAGE` with a compatible base image containing the SAIL SDK, PyTorch, Python, and CUDA.

## Create the Container

Mount the complete assets root directory at `/workspace/cosmos3-assets`:

```bash
docker run --privileged --name cosmos3_train \
  --network=host --ipc=host --pid=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --init -dit \
  -v /path/to/cosmos3-assets:/workspace/cosmos3-assets \
  ppu-cosmos3:M890 /bin/bash
```

## Prepare the Assets

Run the following commands inside the `cosmos3_train` container. The checkpoints and data are written directly to the mounted directory, which must match the `ASSET_ROOT` used during training:

```bash
export ASSET_ROOT=/workspace/cosmos3-assets/examples
mkdir -p "$ASSET_ROOT/checkpoints"
cd /workspace/cosmos-framework

uvx --from modelscope-hub modelscope download \
  Wan-AI/Wan2.2-TI2V-5B \
  Wan2.2_VAE.pth \
  --local-dir "$ASSET_ROOT/checkpoints/wan22_vae"

for BASE_CHECKPOINT_NAME in Cosmos3-Nano Cosmos3-Super; do
  python -m cosmos_framework.scripts.convert_model_to_dcp \
    -o "$ASSET_ROOT/checkpoints/$BASE_CHECKPOINT_NAME" \
    --checkpoint-path "$BASE_CHECKPOINT_NAME"
done

cd /workspace/cosmos3-assets
python /workspace/cosmos-framework/scripts/generate_synthetic_data.py
```

The resulting directory structure should be:

```text
/workspace/cosmos3-assets/examples/
├── checkpoints/Cosmos3-Nano/model/.metadata
├── checkpoints/Cosmos3-Super/model/.metadata
├── checkpoints/wan22_vae/Wan2.2_VAE.pth
└── data/synthetic_1k_1920x1200/train/video_dataset_file.jsonl
```

## Run Sequentially

Both Nano and Super use all 8 cards, so they must be run sequentially inside the container:

```bash
cd /workspace/cosmos-framework
export ASSET_ROOT=/workspace/cosmos3-assets/examples
export DATASET_PATH="$ASSET_ROOT/data/synthetic_1k_1920x1200"
export WAN_VAE_PATH="$ASSET_ROOT/checkpoints/wan22_vae/Wan2.2_VAE.pth"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8

export BASE_CHECKPOINT_PATH="$ASSET_ROOT/checkpoints/Cosmos3-Nano"
export MASTER_PORT=50012
export OUTPUT_ROOT=/workspace/cosmos3-assets/outputs/nano_sft
bash examples/launch_sft_vision_nano_1920_1200_profile.sh

export BASE_CHECKPOINT_PATH="$ASSET_ROOT/checkpoints/Cosmos3-Super"
export MASTER_PORT=50013
export OUTPUT_ROOT=/workspace/cosmos3-assets/outputs/super_sft
bash examples/launch_sft_vision_super_1920_1200_sft.sh
```

## Measured Results

| Model | Step 1 | Average of Steps 2–10 | Additional Steady-State Result | Peak Memory per Card |
|---|---:|---:|---:|---:|
| Cosmos3 Nano | 292.40 s | 58.39 s | Steps 3–10: 57.37 s | 66,518 MiB (64.96 GiB) |
| Cosmos3 Super | 212.50 s | 95.69 s | Stable from Steps 2–10 | 105,599 MiB (103.12 GiB) |

Step 2 of the Nano run still includes a small amount of trailing compilation overhead, so the steady-state average for Steps 3–10 is also reported. Memory usage was sampled with `ppu-smi` every two seconds, and the maximum across all cards and all samples is reported.

The raw logs are located in the assets output directory:

- Nano: `outputs/repro_nano_final/logs/vision_sft_nano_sft.log`
- Super: `outputs/repro_super_final/logs/vision_sft_super_sft_sft.log`
- Memory: `outputs/repro_nano_final_memory.log`, `outputs/repro_super_final_memory.log`
