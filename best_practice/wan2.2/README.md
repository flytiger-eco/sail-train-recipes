# Wan 2.2 5B M890 Performance Recipe

This directory contains only the reproduction instructions. The training entry point and dependencies are added to the image by `Dockerfile.Wan2.2` and `wan2.2.patch`.

## Baseline

- Source: <https://github.com/aigc-apps/VideoX-Fun.git>
- Commit: `67e3b4b8184ae8ff61ab3e3fbdb68bb817f41515`
- Dockerfile: `dockers/Dockerfile.Wan2.2`
- Patch: `dockers/patches/wan2.2.patch`
- Hardware: One node with 8 M890 144 GiB cards
- Model: Wan2.2-TI2V-5B
- Input: X-Fun-Videos-Demo at 960 resolution with 49 frames
- Parallelism: Accelerate FSDP FULL_SHARD
- Precision: BF16

## Build the Image

```bash
cd /path/to/sail-train-recipes
export BASE_IMAGE=pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.0-2.10.0-ubuntu24.04-cuda13.0-py312-20261001
DOCKER_BUILDKIT=0 docker build --network=host \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f dockers/Dockerfile.Wan2.2 \
  -t ppu-wan2.2:M890 .
```

The Dockerfile uses Alibaba Cloud mirrors for Ubuntu and PyPI. If the FlyTiger tag has not yet been published, replace only `BASE_IMAGE` with a compatible base image containing the SAIL SDK, PyTorch, Python, and CUDA.

## Create the Container

```bash
docker run --privileged --name wan2_2_train \
  --network=host --ipc=host --pid=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --init -dit \
  -v /path/to/wan2.2-assets:/workspace/wan2.2-assets \
  ppu-wan2.2:M890 /bin/bash
```

## Prepare the Assets

Run the following commands inside the container. The downloaded files are written directly to the mounted directory:

```bash
export ASSET_ROOT=/workspace/wan2.2-assets
mkdir -p "$ASSET_ROOT/models" "$ASSET_ROOT/datasets"

uvx --from modelscope-hub modelscope download \
  Wan-AI/Wan2.2-TI2V-5B \
  --local-dir "$ASSET_ROOT/models/Wan2.2-TI2V-5B"

uvx --from modelscope-hub modelscope download \
  --repo-type dataset \
  PAI/X-Fun-Videos-Demo \
  --local-dir "$ASSET_ROOT/datasets/X-Fun-Videos-Demo"
```

The assets directory should contain:

```text
/workspace/wan2.2-assets/
├── models/Wan2.2-TI2V-5B/config.json
├── datasets/X-Fun-Videos-Demo/metadata_add_width_height.json
└── datasets/X-Fun-Videos-Demo/train/*.mp4
```

## Run Training

The first run in a new container incurs one-time kernel compilation overhead. Complete a 2-step warmup first, then run the formal benchmark in the same container:

```bash
cd /workspace/VideoX-Fun
export MODEL_NAME=/workspace/wan2.2-assets/models/Wan2.2-TI2V-5B
export DATASET_NAME=/workspace/wan2.2-assets/datasets/X-Fun-Videos-Demo
export DATASET_META_NAME="$DATASET_NAME/metadata_add_width_height.json"

export OUTPUT_DIR=/workspace/wan2.2-assets/outputs/warmup
export MAX_TRAIN_STEPS=2
bash scripts/wan2.2/train_ti2v_5b_singleview.sh

export OUTPUT_DIR=/workspace/wan2.2-assets/outputs/ti2v_5b_singleview
export MAX_TRAIN_STEPS=50
bash scripts/wan2.2/train_ti2v_5b_singleview.sh
```

The launcher is configured to use 8 cards, BF16, FSDP FULL_SHARD, 960 resolution, 49 frames, `train_batch_size=1`, and `gradient_accumulation_steps=1`. It also enables `PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=True`. Each training run saves an FSDP checkpoint of approximately 66 GiB.

## Measured Results

The following results were measured over 50 steps after completing the 2-step warmup with the Dockerfile, image, and launcher described above:

| Metric | Result |
| :--- | ---: |
| Training time for the first 50 steps | 5 min 25 s |
| Average time over the first 50 steps | 6.50 s/step |
| First step | 9.38 s |
| Average time for Steps 2–50 | Approximately 6.44 s/step |
| Peak memory per card | 28,976 MiB (28.30 GiB) |
| Loss range / mean / median | 0.147–0.590 / 0.2623 / 0.2305 |

Training completed successfully and saved `checkpoint-50`, with no OOM, NaN, or abnormal termination. The smoothed rate at the end of the progress bar was 6.52 s/step, which is consistent with the documented steady-state performance baseline. The reported training time excludes warmup, model initialization, and FSDP checkpoint saving after training.
