# FastWAM: Eight-GPU LIBERO / RoboTwin Training on M890

This recipe provides image-building and training entry points through [Dockerfile.FastWAM](../../dockers/Dockerfile.FastWAM) and [fastwam.patch](../../dockers/patches/fastwam.patch). The full FastWAM guide is maintained under `docs/training-finetuning/` in the separate `best_practice_doc` repository.

## Baseline and Patch

| Item | Configuration |
| :--- | :--- |
| Official source | <https://github.com/yuantianyuan01/FastWAM> |
| Public baseline | `7faa71108368fbb3b6885649f112af607427a2d4` |
| Patch origin | Validated local adaptation `f2b055e815a4c3613b38b758f44b37f1475ef038`; not a public release branch |
| Training | Single node, 8 M890 GPUs, BF16, Accelerate + ZeRO-1, DP=8 |
| Batch / data loading | 16 per GPU, gradient accumulation=1, global batch=128; 8 workers per rank, CPU TorchCodec |

The patch includes RoboTwin data-path optimizations, seeding before model initialization, two eight-GPU NUMA launchers, and early rank/worker binding. The profiler remains disabled by default. It excludes personal wrappers, analysis documents, and uncommitted tests. The launchers retain the model objectives and delegate to the original `train_zero1.sh`.

## Build an Image (Optional)

Use a prepared training image if one is available. To build your own, use Docker 26+ / BuildKit with its built-in Dockerfile frontend supporting RUN heredocs.

`BASE_IMAGE` must contain Python 3.12, an SDK 2.2-compatible PPU stack, torch `2.13.0`, compatible torchvision/triton, CPU TorchCodec `0.17.0+cpu`, and PPU-compatible DeepSpeed `0.18.7`.

```bash
cd /path/to/sail-train-recipes
: "${BASE_IMAGE:?Set an image address or local tag containing the validated PPU stack}"
DOCKER_BUILDKIT=1 docker build --network=host \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f dockers/Dockerfile.FastWAM -t ppu-fastwam:M890 .
```

The Dockerfile pins the official baseline, checks and applies the patch, and validates Bash/Python syntax and dependency imports. Upstream application dependencies are installed with constraints protecting the existing PPU stack. FastWAM uses `--no-deps --no-build-isolation`, without installing the NVIDIA torch/vision/codec versions declared upstream. The build does not use devices or run environment-recovery scripts; a conflict-free `pip check` is not claimed.

## Create a Container

Use the prepared image address or local tag as `FASTWAM_IMAGE`. The example defaults to the local build tag.

```bash
export FASTWAM_IMAGE="${FASTWAM_IMAGE:-ppu-fastwam:M890}"
docker run --privileged --name fastwam_train \
  --network=host --ipc=host --pid=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --init -dit \
  -v /path/to/fastwam-assets:/workspace/fastwam-assets \
  "$FASTWAM_IMAGE" /bin/bash
```

## Run Inside the Container

Run the following commands in the training container. Preflight requires 8 M890 GPUs, 128 available physical CPU cores, and CPU-affinity/MPOL_BIND permissions. It reads the actual topology rather than hard-coding CPU IDs. Run the two tasks sequentially on the eight-GPU node.

```bash
cd /workspace/FastWAM
export ASSET_ROOT=/workspace/fastwam-assets
export DIFFSYNTH_MODEL_BASE_PATH="$ASSET_ROOT/models"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/train_LIBERO.sh --check
bash scripts/train_RoboTwin.sh --check

# Enable both M890 optimizations; launchers inherit these settings unchanged.
export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1
export PYTORCH_ENABLE_PPU_FLEX_FLASH_ATTENTION=1
COMMON_ARGS=(
  model.model_id=Wan2.2-TI2V-5B
  model.tokenizer_model_id=Wan2.2-TI2V-5B
  model.redirect_common_files=false
  model.load_text_encoder=false
  "model.action_dit_pretrained_path=$ASSET_ROOT/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
)
LIBERO_DATA="$ASSET_ROOT/data/libero/lerobot_v30"
LIBERO_DIRS="[$LIBERO_DATA/libero_spatial_no_noops_lerobot,$LIBERO_DATA/libero_object_no_noops_lerobot,$LIBERO_DATA/libero_goal_no_noops_lerobot,$LIBERO_DATA/libero_10_no_noops_lerobot]"
ROBOTWIN_DATA="$ASSET_ROOT/data/robotwin/robotwin2.0"

bash scripts/train_LIBERO.sh "${COMMON_ARGS[@]}" \
  "data.train.dataset_dirs=$LIBERO_DIRS" \
  "data.train.text_embedding_cache_dir=$ASSET_ROOT/text_embeds_cache/libero" \
  "+data.train.pretrained_norm_stats=$ASSET_ROOT/stats/libero_dataset_stats.json" \
  wandb.enabled=false profiler.enabled=false

bash scripts/train_RoboTwin.sh "${COMMON_ARGS[@]}" \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  "data.val.dataset_dirs=[$ROBOTWIN_DATA]" \
  "data.train.text_embedding_cache_dir=$ASSET_ROOT/text_embeds_cache/robotwin" \
  "data.val.text_embedding_cache_dir=$ASSET_ROOT/text_embeds_cache/robotwin" \
  "data.train.pretrained_norm_stats=$ASSET_ROOT/stats/robotwin_dataset_stats.json" \
  "data.val.pretrained_norm_stats=$ASSET_ROOT/stats/robotwin_dataset_stats.json" \
  save_every=2500 wandb.enabled=false profiler.enabled=false
```

LIBERO retains 10 epochs, eval=200, and save=2000. RoboTwin retains 5 epochs and eval=500; its launcher defaults to save=0, while the example restores save=2500. Do not change the scheduler by overriding `max_steps` for short runs. To disable both optimizations, unset `PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT` and `PYTORCH_ENABLE_PPU_FLEX_FLASH_ATTENTION`. Start each mode with the same weights, seed, data order, and statistics.

## Performance

| Dataset | M890 EW + FFA seconds/step | Throughput (samples/s) |
| :--- | ---: | ---: |
| LIBERO | 1.5083 | 84.86 |
| RoboTwin | 1.8133 | 70.59 |
