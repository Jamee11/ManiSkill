# Tracker-conditioned DiT4DiT 训练与评估命令（2026-07-21）

本节在修改评估代码前写入，固定当前已经执行的训练合同，以及接下来只复用现有 evaluator 所需的最小改动。后面的“采集命令留存”作为数据 provenance 附录保留。

## 1. 当前训练合同

- DiT4DiT：`/remote-home/jinminghao/DiT4DiT`
- 数据：`/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real/lerobot/train/learned_tracker`
- run：`checkpoints/maniskill_eort_v3_policy/maniskill_eort_v3_push_pick_front_wrist_tracker_fullft_40k`
- joint PushCube + PickCube；只读取 `front` 与 `wrist`，每个时刻横向拼成 `(3,224,448)`。
- Action DiT state：`16D` robot sin/cos + `17D` learned-tracker message + `1D` valid = `34D`。
- Video DiT 与 Action DiT 全参训练；只冻结 text encoder/VAE；无 LoRA、独立 object token 或 object-dynamics head。
- loss：`action_dit_loss + future_video_loss`；tracker 已离线冻结，不属于 policy optimizer。

### 一步 smoke

```bash
cd /remote-home/jinminghao/DiT4DiT
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONNOUSERSITE=1 \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
BASE_MODEL=/remote-home/jinminghao/ckpts/Cosmos-Predict2.5-2B \
NUM_PROCESSES=4 PER_DEVICE_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_TRAIN_STEPS=1 NUM_WARMUP_STEPS=1 LOGGING_FREQUENCY=1 SAVE_INTERVAL=1 \
WANDB_MODE=offline RUN_ID=maniskill_eort_v3_front_wrist_fullft_smoke_1step \
bash examples/RLBench_EORT/train_files/run_maniskill_eort_v3_joint_official_minimal.sh
```

### 40k 正式训练

```bash
cd /remote-home/jinminghao/DiT4DiT
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONNOUSERSITE=1 \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
BASE_MODEL=/remote-home/jinminghao/ckpts/Cosmos-Predict2.5-2B \
NUM_PROCESSES=4 PER_DEVICE_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=4 \
MAX_TRAIN_STEPS=40000 SAVE_INTERVAL=5000 LOGGING_FREQUENCY=10 WANDB_MODE=offline \
RUN_ID=maniskill_eort_v3_push_pick_front_wrist_tracker_fullft_40k \
bash examples/RLBench_EORT/train_files/run_maniskill_eort_v3_joint_official_minimal.sh \
> logs/maniskill_eort_v3_policy/front_wrist_tracker_fullft_40k.log 2>&1
```

2026-07-22 04:14 UTC 审计时已有 `steps_5000/10000/15000/20000_pytorch_model.pt`；40k 训练仍在继续。评估只使用已经原子写完的 `.pt`，当前推荐先以 `steps_20000_pytorch_model.pt` 做流程验证。

### 2026-07-27：无 tracker 的原版式配对 baseline

该 baseline 用于回答“learned-tracker condition 是否真正带来收益”。它保持 PushCube + PickCube、front + wrist、训练样本、Video DiT hidden-state condition、action chunk、loss、全参训练方式和优化参数不变，只移除 Action DiT 的 tracker message：

```text
tracker 版本：Action DiT state = 16D robot sin/cos + 17D tracker message + 1D valid = 34D
baseline 版本：Action DiT state = 16D robot sin/cos
```

这里仍读取 `lerobot/train/learned_tracker`，是为了让两组实验使用完全相同的视频、动作、episode 和 split。数据字段的顺序是 robot state 在前、tracker condition 在后；设置 `STATE_DIM=16` 和 `MAX_STATE_DIM=16` 后，loader 只保留前 16D robot sin/cos，后 18D tracker 字段不会进入模型。训练时也不会加载或运行 tracker。实测 loader 合同为：

```text
dataset samples: 78326
state:           (1,16)
action chunk:    (8,7)
video:           5 × (3,224,448)
```

这是一组“原版 DiT4DiT conditioning 逻辑 + ManiSkill 数据适配”的 robot-only 配对 baseline；不是声称整个工程与上游仓库逐文件完全相同。Video DiT hidden states 仍然送入 Action DiT，future-video loss 仍然监督，Video DiT 与 Action DiT 均全参训练，仅冻结 text encoder 和 VAE。

完整 4-GPU、40k 启动命令：

```bash
cd /remote-home/jinminghao/DiT4DiT
mkdir -p logs/maniskill_eort_v3_policy

PATH=/remote-home/jinminghao/miniconda3/envs/cosmos/bin:$PATH \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONNOUSERSITE=1 \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
BASE_MODEL=/remote-home/jinminghao/ckpts/Cosmos-Predict2.5-2B \
DATA_ROOT_DIR=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real/lerobot/train/learned_tracker \
DATA_MIX=maniskill_eort_push_pick_v3_sim2real_learned_tracker_lerobot \
MANISKILL_EORT_OBJECT_DROPOUT_PROB=1 \
STATE_DIM=16 \
MAX_STATE_DIM=16 \
OBJECT_TOKEN_START=0 \
OBJECT_TOKEN_DIM=0 \
OBJECT_DYNAMICS_ENABLED=false \
ACTION_TYPE=metric_robot_base_delta_pose_controller_command \
ACTION_NORMALIZATION=q99 \
ACTION_VIDEO_FREQ_RATIO=2 \
COMPUTE_FUTURE_VIDEO_LOSS=true \
COSMOS_TRAINING_MODE=joint \
COSMOS_DETACH_HIDDEN=true \
BACKBONE_TUNING_MODE=default \
FREEZE_MODULES=backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae \
VIDEO_BACKEND=decord \
LOAD_ALL_DATA_FOR_TRAINING=false \
NUM_WORKERS=8 \
NUM_PROCESSES=4 \
PER_DEVICE_BATCH_SIZE=3 \
GRADIENT_ACCUMULATION_STEPS=4 \
MAX_TRAIN_STEPS=40000 \
SAVE_INTERVAL=10000 \
LOGGING_FREQUENCY=100 \
WANDB_MODE=offline \
WANDB_PROJECT=DiT4DiT_maniskill_eort_v3 \
RUN_ROOT_DIR=/remote-home/jinminghao/DiT4DiT/checkpoints/maniskill_eort_v3_policy \
RUN_ID=maniskill_eort_v3_push_pick_front_wrist_robot_only_fullft_40k_bs3acc4 \
nohup bash examples/RLBench_EORT/train_files/run_rlbench_lerobot.sh \
> logs/maniskill_eort_v3_policy/front_wrist_robot_only_fullft_40k_bs3acc4.log 2>&1 &

echo "PID=$!"
```

有效 batch size 为 `4 GPUs × 3 samples/GPU × 4 accumulation = 48`，与对应的 tracker run 保持一致。查看日志：

```bash
tail -f logs/maniskill_eort_v3_policy/front_wrist_robot_only_fullft_40k_bs3acc4.log
```

底层 `run_rlbench_lerobot.sh` 直接调用 `accelerate`，只设置 `PYTHON_BIN` 不会自动修改 `PATH`，因此命令首行必须保留：

```bash
PATH=/remote-home/jinminghao/miniconda3/envs/cosmos/bin:$PATH
```

若日志显示 `Object dynamics: enabled=false, target=34+18`，其中 `target=34+18` 只是未启用 head 的默认打印值；以 `enabled=false` 为准，不会建立或训练 object-dynamics head。不要用 `run_maniskill_eort_v3_joint_official_minimal.sh` 启动该 baseline，因为该 wrapper 固定覆盖为 learned-tracker 数据合同和 `34D state`。

### robot-only baseline 闭环评估（严格无 tracker / 无 oracle）

实现复用已有 server、图像预处理、动作反归一化与闭环 evaluator；新增唯一编排脚本：

```text
DiT4DiT/examples/RLBench_EORT/eval_files/run_maniskill_eort_robot_only_baseline_eval.sh
```

评估数据流严格与训练的 `STATE_DIM=16` / `MAX_STATE_DIM=16` 对齐：

```text
front RGB + wrist RGB ──> Video DiT
8D robot state ──> sin/cos ──> 16D Action DiT state

tracker、front RGB-D、simulator object/goal pose、segmentation、contact 均不读取
```

`robot_only` evaluator 只构造零值的 17D/1D 占位以复用观测接口；client 在此模式只取前 8D robot state 并输出 `(1,16)`，不会归一化或传入 object/valid 字段。因此 `episodes.jsonl` 中的 `valid_condition_steps=0` 是预期行为，不表示 tracker 失败。

正式 50-seed 命令（每个 episode 自动保存 front+wrist MP4）：

```bash
cd /remote-home/jinminghao/DiT4DiT
CKPT=checkpoints/maniskill_eort_v3_policy/maniskill_eort_v3_push_pick_front_wrist_robot_only_fullft_40k_bs3acc4/checkpoints/steps_20000_pytorch_model.pt \
MODEL_GPU=7 SIM_GPU=6 PORT=10093 EPISODES=50 STAGE=all \
bash examples/RLBench_EORT/eval_files/run_maniskill_eort_robot_only_baseline_eval.sh
```

`STAGE=all|push|pick` 可只重跑一个任务；`SEED_START=2000000`、`REPLAN_EVERY=8`、`MAX_STEPS=200`、`DDIM_STEPS=10` 默认与 tracker 评估相同。脚本为每个运行新建输出目录，写入 `push/`、`pick/` 的 `summary.json`、`episodes.jsonl`、`run_config.json` 和逐 episode MP4；模型 server 最多等待 `SERVER_WAIT_SECONDS=600` 秒并在结束时只终止自身启动的 PID。

已用 10k checkpoint、seed `2000000` 做真实 smoke：Push 76 步成功、Pick 132 步成功，均保存双视角视频，且 `valid_condition_steps=0`。

## 2. 评估流程：复用现有框架，最小补齐

复用以下已有文件，不新建第二套 server/client：

- model server：`DiT4DiT/deployment/model_server/server_policy.py`
- server launcher：`examples/RLBench_EORT/eval_files/run_maniskill_eort_policy_server.sh`
- state/action adapter：`examples/RLBench_EORT/eval_files/maniskill_model_client.py`
- held-out open-loop：`examples/RLBench_EORT/eval_files/eval_maniskill_eort_openloop.py`
- simulator closed-loop：`examples/RLBench_EORT/eval_files/eval_maniskill_eort.py`
- eval launcher：`examples/RLBench_EORT/eval_files/run_maniskill_eort_eval.sh`

开始修改前确认的缺口与拟实施的最小修复：

1. evaluator 默认还是旧 `PushCubeEORT-v1` + `base_camera`；改为训练数据使用的 `PushCubeEORTSim2Real-v1` / `PickCubeEORTSim2Real-v1`。
2. policy 当前只收到 front；改为明确读取 `front_camera` + `hand_camera`，沿用现有 client 的横向拼接逻辑。`right_shoulder_camera` 不进入 policy。
3. frozen tracker 条件仍只由 front RGB-D 产生，与离线 learned-tracker export 一致；wrist 只进入 Video DiT。
4. online tracker geometry 与 robot state 显式转换到 `robot_base`；相机直接复用 `EORTSim2RealV3Mixin`，不得复制或另写位姿。
5. client 接受当前 checkpoint 的 `metric_robot_base_delta_pose_controller_command`，再转回 Panda `pd_ee_delta_pose` normalized command。
6. open-loop 同时读取 front/wrist，并在 held-out learned-tracker test split 计算 action chunk error；closed-loop 记录 success、condition valid、clipping 和双视角 MP4。

### 相机与场景合同

- front：`front_camera`，真实序列号 `344522302193`。
- wrist：`hand_camera`，真实序列号 `339222070579`。
- 禁用：`right_shoulder_camera`。
- language 必须逐字使用训练 metadata：Push 为 `push cube to goal`，Pick 为 `pick cube to elevated goal`。
- 相机内外参与每 episode ±1 cm / ±2° jitter 直接来自 `mani_skill/envs/tasks/tabletop/eort_visual_variants.py::EORTSim2RealV3Mixin`。
- Push v3 目标中心偏移使用训练时的 `goal_center_offset_x=0.15 m`；eval 不重新定义场景几何。

### 20k checkpoint 的实际闭环数据流

```text
ManiSkill reset(test seed)
  ├─ front RGB-D ──> frozen learned tracker
  │                    ├─ object/goal heatmap + visibility
  │                    └─ depth + calibration 回投到 robot-base
  │                         └─ 17D object condition + 1D valid
  ├─ 8D robot state ──> sin/cos ──> 16D robot condition
  └─ front RGB + wrist RGB
       └─ 各自 256→224，横向拼接为 (3,224,448)

language + front/wrist + (16D robot + 17D tracker + 1D valid)
  └─ Video DiT：预测 future-video flow，并把 hidden states 交给 Action DiT
       └─ Action DiT：从随机噪声去噪得到 (8,7) action chunk
            └─ 反归一化为 robot-base metric action
                 [Δxyz_m, Δrotvec_rad, gripper_open_fraction]
                 └─ Panda adapter 转成 pd_ee_delta_pose command
                      └─ 连续执行完整 8 步，再重新观测与规划
```

关键边界：

- learned tracker 已冻结，不随 policy eval 更新；它只读取当前 front RGB-D。
- Video DiT hidden states 仍然是 Action DiT 的条件；tracker message 只是加入原有 state prefix，没有替换 Video DiT。
- policy 输入不包含 segmentation、actor ID、simulator pose、contact、phase 或 future label。
- `--tracker-overlay` 只修改保存的视频副本，不修改送入模型的 RGB。
- Action DiT 从 `torch.randn` 初始化；固定 simulator seed 仍不代表固定 model-noise seed。

### 已实现并验证的命令

数据/环境 check-only（无需 model server）：

```bash
cd /remote-home/jinminghao/DiT4DiT
PYTHONNOUSERSITE=1 /remote-home/jinminghao/miniconda3/envs/cosmos-libero/bin/python \
examples/RLBench_EORT/eval_files/eval_maniskill_eort_openloop.py \
--dataset-root /remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real/lerobot/test/learned_tracker \
--condition learned_tracker --check-only
```

启动 checkpoint server：

```bash
cd /remote-home/jinminghao/DiT4DiT
CUDA_DEVICE=<model_gpu> MODEL_PYTHON=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
PORT=10093 bash examples/RLBench_EORT/eval_files/run_maniskill_eort_policy_server.sh \
checkpoints/maniskill_eort_v3_policy/maniskill_eort_v3_push_pick_front_wrist_tracker_fullft_40k/checkpoints/steps_<N>_pytorch_model.pt
```

held-out open-loop action 评估（需要先启动上面的 model server）：

```bash
cd /remote-home/jinminghao/DiT4DiT
PYTHONNOUSERSITE=1 /remote-home/jinminghao/miniconda3/envs/cosmos-libero/bin/python \
examples/RLBench_EORT/eval_files/eval_maniskill_eort_openloop.py \
--dataset-root /remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real/lerobot/test/learned_tracker \
--checkpoint checkpoints/maniskill_eort_v3_policy/maniskill_eort_v3_push_pick_front_wrist_tracker_fullft_40k/checkpoints/steps_<N>_pytorch_model.pt \
--condition learned_tracker --episodes-per-dataset 10 --frame-stride 10 \
--output eval_outputs/maniskill_eort_v3/steps_<N>_openloop
```

closed-loop：

```bash
cd /remote-home/jinminghao/DiT4DiT
CUDA_DEVICE=<sim_tracker_gpu> bash examples/RLBench_EORT/eval_files/run_maniskill_eort_eval.sh \
--checkpoint checkpoints/maniskill_eort_v3_policy/maniskill_eort_v3_push_pick_front_wrist_tracker_fullft_40k/checkpoints/steps_<N>_pytorch_model.pt \
--env-id PushCubeEORTSim2Real-v1 --condition learned_tracker \
--tracker-checkpoint checkpoints/maniskill_eort_v3_push_cube_shared_rgbd_tracker_cache_v1_run2/best.pt \
--seed-start 2000000 --episodes 50 --replan-every 8 --max-steps 200 \
--output <new_output_dir>
```

PickCube 使用 `PickCubeEORTSim2Real-v1` 和 Pick tracker checkpoint。仿真正式评估执行完整 8-step action chunk，最大 200 步；每个输出目录必须是新目录，test seed 不参与调参。挑选 case 可额外传 `--tracker-overlay`，只在保存的 front 视频上绘制 object/goal 像素、visibility、valid 和 robot-base relation，不改变 policy 输入。

### 20k checkpoint 可直接复制的完整 eval runbook

固定路径：

```bash
cd /remote-home/jinminghao/DiT4DiT
export CKPT=/remote-home/jinminghao/DiT4DiT/checkpoints/maniskill_eort_v3_policy/maniskill_eort_v3_push_pick_front_wrist_tracker_fullft_40k/checkpoints/steps_20000_pytorch_model.pt
export DATASET=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real/lerobot/test/learned_tracker
export PORT=10093
```

终端 A：使用训练一致的 `cosmos` 环境启动 model server。20k checkpoint 约 20 GB，实际验证时放在 GPU 7：

```bash
cd /remote-home/jinminghao/DiT4DiT
CUDA_DEVICE=7 \
MODEL_PYTHON=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
PORT="$PORT" \
bash examples/RLBench_EORT/eval_files/run_maniskill_eort_policy_server.sh "$CKPT"
```

看到 server listening 后，终端 B 运行 held-out open-loop：

```bash
cd /remote-home/jinminghao/DiT4DiT
PYTHONNOUSERSITE=1 \
/remote-home/jinminghao/miniconda3/envs/cosmos-libero/bin/python \
examples/RLBench_EORT/eval_files/eval_maniskill_eort_openloop.py \
  --dataset-root "$DATASET" \
  --checkpoint "$CKPT" \
  --condition learned_tracker \
  --episodes-per-dataset 10 \
  --frame-stride 10 \
  --host localhost --port "$PORT" \
  --output eval_outputs/maniskill_eort_v3/steps_20000_openloop_full
```

终端 B 正式 Push closed-loop；GPU 6 只运行 simulator + Push tracker，GPU 7 保留给 server：

```bash
cd /remote-home/jinminghao/DiT4DiT
CUDA_DEVICE=6 PORT="$PORT" \
bash examples/RLBench_EORT/eval_files/run_maniskill_eort_eval.sh \
  --checkpoint "$CKPT" \
  --env-id PushCubeEORTSim2Real-v1 \
  --condition learned_tracker \
  --tracker-checkpoint checkpoints/maniskill_eort_v3_push_cube_shared_rgbd_tracker_cache_v1_run2/best.pt \
  --seed-start 2000000 --episodes 50 \
  --replan-every 8 --max-steps 200 --ddim-steps 10 \
  --output eval_outputs/maniskill_eort_v3/steps_20000_push_closedloop_50_replan8_max200
```

终端 B 正式 Pick closed-loop：

```bash
cd /remote-home/jinminghao/DiT4DiT
CUDA_DEVICE=6 PORT="$PORT" \
bash examples/RLBench_EORT/eval_files/run_maniskill_eort_eval.sh \
  --checkpoint "$CKPT" \
  --env-id PickCubeEORTSim2Real-v1 \
  --condition learned_tracker \
  --tracker-checkpoint checkpoints/maniskill_eort_v3_pick_cube_shared_rgbd_tracker_cache_v1_run2/best.pt \
  --seed-start 2000000 --episodes 50 \
  --replan-every 8 --max-steps 200 --ddim-steps 10 \
  --output eval_outputs/maniskill_eort_v3/steps_20000_pick_closedloop_50_replan8_max200
```

只生成少量 tracker overlay 诊断视频时，使用新的输出目录并附加：

```bash
--episodes 3 --tracker-overlay --output <new_tracker_overlay_output_dir>
```

结果文件：

```text
summary.json     聚合 success、steps、tracker-valid、action clipping
episodes.jsonl  每个 seed 的逐 episode 结果
run_config.json 完整协议
episode_*.mp4   front+wrist H.264/yuv420p 视频
```

评估完成后在终端 A 使用 `Ctrl-C` 停止 server。不得用模糊的 `pkill python`，以免误杀其他训练；如果 server 是后台启动，只终止启动命令打印出的对应 PID。输出目录采用 `exist_ok=False`，重跑必须换新目录，不覆盖原结果。

### 2026-07-22 实际验证结果

- 语法、focused client/geometry tests 与 `git diff --check` 通过。
- held-out check-only 同时读到 Push/Pick：raw state `(26,)`、metric robot-base action `(7,)`、front/wrist 各 `(256,256,3)`；right-shoulder 未进入 policy。
- 使用 production Push test seed `2000000` 做真实 Sim2Real v3 replay：52 步 success，learned tracker condition 52/52 valid，动作零裁剪。
- replay 视频为 front+wrist 横向合成，H.264/yuv420p、`512×256`、20 FPS、53 帧。
- 在线 tracker 已去除对离线 converter/pandas 的隐式依赖，`maniskill-eort-v1` 环境可以直接加载 227 KB tracker checkpoint。
- policy server 已在 GPU 7 真实加载 20k checkpoint。held-out open-loop 共 146 samples：first-action arm L2 `0.005925`，chunk arm L2 `0.007187`，gripper wrong rate `0.004281`。
- 正式 closed-loop 使用 test seeds `2000000..2000049`、`replan_every=8`、`max_steps=200`、10 DDIM steps：Push `49/50=98%`，Pick `41/50=82%`，两者均零动作裁剪。此前 `replan_every=1` 只反复执行 chunk 首动作，属于无效诊断结果，不进入结论。
- formal MP4 与 tracker overlay MP4 均为 H.264/yuv420p、`512x256`。结果目录分别为 `eval_outputs/maniskill_eort_v3/steps_20000_{push,pick}_closedloop_50_replan8_max200` 和 `eval_outputs/maniskill_eort_v3/tracker_overlay_cases`。
- 当前 Action DiT 采样由 `torch.randn` 初始化，同一 simulator seed 重跑可能变化；50-seed 成功率是随机策略估计。仿真结果不替代真实相机标定、tracker invalid handling 和硬件安全验证。

## 附录：EORT v3 采集命令留存

工作目录：`/remote-home/jinminghao/WAMs/ManiSkill`。

## 固定环境与安全约束

```bash
cd /remote-home/jinminghao/WAMs/ManiSkill
export PYTHONNOUSERSITE=1
export MANISKILL_EORT_CONTROLLER_REPLAY_MAX_RETRY=2
```

- Python：`/remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python`
- 标准输出根：`/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real`
- 每个 shard 默认不可覆盖；同一个 `task/split` 只能有一个 writer。
- `CONTROLLER_REPLAY_MAX_RETRY=2` 表示 controller replay 最多三次总尝试。collector 在导出前要求 source/controller 数量均等于请求数量，且全部 terminal success。

## 预览计划（不启动采集）

```bash
MANISKILL_EORT_CUDA_VISIBLE_DEVICES=7 \
  bash scripts/eort/collect_production_matrix_v3_sim2real.sh
```

## 单个正式 shard

将 `<gpu>`、`<task>`、`<split>`、`<seed>` 和 `<count>` 替换为表中的值。不同 shard 可在不同 GPU 并行执行；不得对同一输出目录并行执行。

| split | seed | count |
| --- | ---: | ---: |
| train | 0 | 500 |
| val | 1000000 | 100 |
| test | 2000000 | 200 |

```bash
PYTHONNOUSERSITE=1 \
MANISKILL_EORT_CUDA_VISIBLE_DEVICES=<gpu> \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
MANISKILL_EORT_TASK=<task> \
MANISKILL_EORT_SPLIT=<split> \
MANISKILL_EORT_START_SEED=<seed> \
NUM_TRAJ=<count> \
MANISKILL_EORT_CONTROLLER_REPLAY_MAX_RETRY=2 \
bash scripts/eort/collect_objectcentric_v3_sim2real.sh
```

示例：PushCube/test，GPU 1。

```bash
PYTHONNOUSERSITE=1 MANISKILL_EORT_CUDA_VISIBLE_DEVICES=1 \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
MANISKILL_EORT_TASK=push_cube MANISKILL_EORT_SPLIT=test \
MANISKILL_EORT_START_SEED=2000000 NUM_TRAJ=200 \
MANISKILL_EORT_CONTROLLER_REPLAY_MAX_RETRY=2 \
bash scripts/eort/collect_objectcentric_v3_sim2real.sh
```

## 矩阵模式

矩阵脚本按 task、split 顺序执行；需要跨 GPU 并行时，优先使用上面的单 shard 命令。

```bash
PYTHONNOUSERSITE=1 MANISKILL_EORT_CUDA_VISIBLE_DEVICES=7 \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
MANISKILL_EORT_MATRIX_TASKS=pick_cube \
MANISKILL_EORT_MATRIX_EXECUTE=1 \
MANISKILL_EORT_CONTROLLER_REPLAY_MAX_RETRY=2 \
bash scripts/eort/collect_production_matrix_v3_sim2real.sh
```

## 当前并行任务

| GPU | 工作 | 根目录 | 备注 |
| --- | --- | --- | --- |
| 1 | PushCube/train repair → merge | 标准根的 `push_cube/train/sim2real/repair_*` | 原 seed 354 的 controller replay 三次均失败；保留有效 499 条，另采 1 条 seed 5,000,000 成功样本后合并。 |
| 2 | PickCube/test | 标准根 | 200 条。 |
| 3 | PushCube/val replacement | `/remote-home/jinminghao/datasets/maniskill_eort_v3_replacement_push_val_20260721T0310Z` | 完整后再以保留旧 28 条诊断的方式提升到标准根。 |
| 7 | PickCube/train、val | 标准根 | train 完成后顺序执行 val；test 已由 GPU 2 独立执行。 |

## PushCube/train 缺失样本修复命令

原始 500 条 source 的全量 replay 在 seed 354（原日志 episode 342）稳定失败；不能直接插入旧 HDF5，因为 ManiSkill 会将保存的 replay group 重新编号。当前采用“499 条成功 controller + 1 条独立成功补充轨迹”的可审计合并方案。

补充轨迹：

```bash
PYTHONNOUSERSITE=1 MANISKILL_EORT_CUDA_VISIBLE_DEVICES=1 \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_v3_push_train_makeup_20260721T0320Z \
MANISKILL_EORT_TASK=push_cube MANISKILL_EORT_SPLIT=train \
MANISKILL_EORT_START_SEED=5000000 NUM_TRAJ=1 \
MANISKILL_EORT_MAX_ATTEMPTS=5 \
MANISKILL_EORT_CONTROLLER_REPLAY_MAX_RETRY=2 \
bash scripts/eort/collect_objectcentric_v3_sim2real.sh
```

合并（必须使用新目录，不覆盖旧 controller）：

```bash
PYTHONNOUSERSITE=1 /remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python -c '
from mani_skill.trajectory.merge_trajectory import merge_trajectories
merge_trajectories(
    "<combined_controller>.h5",
    ["<retained_499_controller>.h5", "<makeup_1_controller>.h5"],
)
'
```

合并后必须验证 500 个 group、全部 terminal success，再重新运行 deriver、preview、LeRobot exporter 和 audit；旧 499 条 controller/sidecar/LeRobot 只能移入诊断目录保留，不能删除。

## 运行状态与最终审计

```bash
tmux ls | rg 'eort-v3'
tr '\r' '\n' < /tmp/<job>.log | tail -n 30
cat /tmp/<job>.status
```

每个 task 在所有 split 完成后运行：

```bash
PYTHONNOUSERSITE=1 /remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python \
  scripts/eort/audit_collection_v3.py \
  --root /remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
  --task <push_cube|pick_cube> \
  --splits train,val,test \
  --expected-counts train=500,val=100,test=200
```
