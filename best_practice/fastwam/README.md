# FastWAM：M890 八卡 LIBERO / RoboTwin 训练 recipe

本目录只保留镜像构建和训练入口说明。环境由 [Dockerfile.FastWAM](../../dockers/Dockerfile.FastWAM) 和 [fastwam.patch](../../dockers/patches/fastwam.patch) 固化；完整算法、模型/数据下载、文本缓存、归一化统计、性能表和前 50 步 loss 对照维护在独立 `best_practice_doc` 仓库的 `docs/training-finetuning/FastWAM_LIBERO_RoboTwin最佳实践.md`。

## 基线与补丁

| 项目 | 配置 |
| :--- | :--- |
| 官方源码 | <https://github.com/yuantianyuan01/FastWAM> |
| 公开基线 | `7faa71108368fbb3b6885649f112af607427a2d4` |
| patch 来源 | 本地已验证适配 `f2b055e815a4c3613b38b758f44b37f1475ef038`；不是公开发布分支 |
| 训练 | 单机 8 张 M890，BF16，Accelerate + ZeRO-1，DP=8 |
| batch / 数据加载 | 每卡 16，梯度累积 1，global batch=128；每 rank 8 worker、CPU TorchCodec |

patch 包含 RoboTwin 数据路径优化、模型初始化前播种、两个八卡 NUMA 入口及早期 rank/worker 绑定，并保留默认关闭的 Profiler。它不含个人路径包装脚本、分析文档或未提交测试，不改模型目标或另写训练循环；入口继续委托原 `train_zero1.sh`。

## 构建镜像

构建入口使用 Docker 26+ / BuildKit 的内置 Dockerfile frontend（支持 RUN heredoc）。

`BASE_IMAGE` 必须为交付方已准备好的 Python 3.12 / SDK 2.2 兼容环境，包含 PPU torch `2.13.0`（`torch.version.git_version=9a1cf3bee85a2c725e1e736594b524ebfcca6ba0`）、兼容 torchvision/triton、CPU TorchCodec `0.17.0+cpu`、PPU 可用 DeepSpeed `0.18.7`。当前未核实该 torch wheel 的公共下载地址；历史基础镜像不等于后续 FFA 验证构建，不能仅凭 torch 版本号替代 git version。

```bash
cd /path/to/sail-train-recipes
: "${BASE_IMAGE:?请设置包含上述已验证PPU计算栈的交付镜像地址或本地tag}"
DOCKER_BUILDKIT=1 docker build --network=host \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f dockers/Dockerfile.FastWAM -t ppu-fastwam:M890 .
```

Docker 固定官方基线，先 `git apply --check` 再应用 patch，校验 Bash/Python 语法及依赖导入。应用依赖按上游固定版本安装并用 constraints 保护已有计算栈；FastWAM 本体用 `--no-deps --no-build-isolation`，不安装上游声明的 NVIDIA torch/vision/codec。构建时不使用设备、不调用环境恢复脚本，也不宣称整个镜像 `pip check` 无冲突。

仅校验公开源码与补丁时，可加 `--target fastwam-source`，并将 `BASE_IMAGE` 设置为历史基础镜像 `pkg.flytiger-eco.com/docker_release/llm:sdk2.2.0-pytorch2.13.0-ubuntu24.04-cuda13.0-vllm0.27.0-py312-20261001`；这不是完整训练镜像验收。

## 创建容器与准备资产

```bash
docker run --privileged --name fastwam_train \
  --network=host --ipc=host --pid=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --init -dit \
  -v /path/to/fastwam-assets:/workspace/fastwam-assets \
  ppu-fastwam:M890 /bin/bash
```

资产不打入镜像。按完整最佳实践准备 Wan2.2-TI2V-5B、ActionDiT 初始化、两个数据集、各自匹配的 T5 缓存与 `dataset_stats.json`。LIBERO 使用 LeRobot v3，RoboTwin 使用 v2.1，读取器/统计/缓存不可互换；首次统计可设为 `null` 并等待初始化完成，对照实验须复用生成的统计。

## 容器内启动

在已激活 SDK 的容器终端执行。预检需八张 M890、共 128 个可用物理核及 CPU/MPOL_BIND 权限；读取真实拓扑，不硬编码 CPU 编号。两任务顺序运行，不在同一八卡节点并发启动。

```bash
cd /workspace/FastWAM
export ASSET_ROOT=/workspace/fastwam-assets
export DIFFSYNTH_MODEL_BASE_PATH="$ASSET_ROOT/models"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/train_LIBERO.sh --check
bash scripts/train_RoboTwin.sh --check

# 正式双开；启动器继承终端开关，不自动设置或清除。
export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1
export PYTORCH_ENABLE_PPU_FLEX_FLASH_ATTENTION=1
export TORCH_FLEX_FLASH_SDPA_ENABLED=1
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

LIBERO 保留 10 epoch、eval=200、save=2000；RoboTwin 保留 5 epoch、eval=500，入口默认 save=0，完整训练上例显式恢复 save=2500。不要为短测修改 `max_steps` 而改变 scheduler。关闭算子优化时 `unset` 三个开关，四模式从相同权重、seed、数据顺序和统计分别启动。

## 已有结果与验收边界

| 数据集 | M890 双开 秒/step | 吞吐 samples/s |
| :--- | ---: | ---: |
| LIBERO | 1.5083 | 84.86 |
| RoboTwin | 1.8133 | 70.59 |

以上为已有四模式记录中的双开结果，不是新 Docker 镜像重测。RoboTwin 统计 21–100 步全部 80 步；LIBERO 原表窗口未完整记录。完整消融、两组 H200 基线及误差限制只维护在最佳实践文档，不在 recipes 发布性能图或 loss 数据副本。

运行时仍需核对 torch git version、八卡可见性、实际视频 CPU 解码和 NUMA 预检。已有千步训练/SDPA 单测不等于完整收敛或仿真任务成功率验收；新增 Docker 的完整构建及训练须单独记录，不能引用旧性能数据代替。
