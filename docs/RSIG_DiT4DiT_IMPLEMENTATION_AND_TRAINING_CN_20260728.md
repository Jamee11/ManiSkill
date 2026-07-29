# RSIG-DiT4DiT v1 实现、训练与验证手册

更新时间：2026-07-29 UTC

本文是当前 RSIG-DiT4DiT 的**唯一总入口文档**，用于回答：

- 当前相对原版 DiT4DiT 改了什么；
- 数据如何从 ManiSkill 进入模型；
- RSIG 如何同时作用于 Video DiT 和 Action DiT；
- v3 干净数据与 v4 彩色干扰物数据如何训练；
- RSIG、robot-only baseline 如何在干净/干扰环境闭环评估；
- 当前哪些结果已经验证，哪些仍未完成。

若命令与早期计划文档冲突，以本文和仓库当前脚本为准。

前向、反向、detach边界和DeepSpeed梯度诊断的逐参数说明见
`docs/RSIG_DiT4DiT_FORWARD_BACKWARD_AND_GRADIENT_FLOW_CN_20260728.md`。

## 1. 当前已经实现到哪里

当前代码已经形成一条可切换、无 memory、无 Goal Slot、无 phase 输入的完整工程链路：

```text
v3 raw HDF5 + robot-base sidecar
  → 离线导出 front+wrist LeRobot
  → robot 8D经sin/cos得到16D
  → 44D simulator GT只作为RSIG auxiliary targets
  → Cosmos front+wrist + language得到detached H18
  → EEF/Object Role Slots
  → continuous Interaction Token
  → h1/h4/h8 EEF/Object direction + moving
  → dense H18 + 6个RSIG tokens送入Action DiT
  → 8×7D action chunk
```

第二次 future-video flow-matching forward 可在 live H18 使用同一个 RSIG module：

```text
H18' = H18 + video_alpha × role/interaction cross-attention residual
video_alpha初始化为0
```

这里不是“只挂 auxiliary loss”：

- `Role/Interaction → 3个RSIG token → Action DiT → action_dit_loss`，Action主损失直接反传到role/interaction及其Action投影。
- `Role/Interaction → block-17 zero-init residual → Video DiT → future_video_loss`，Video主损失第一步先反传到`video_alpha`；gate离开0后继续反传到video adapter和共享role/interaction。
- 3个Plan token会进入Action DiT并改变预测；v1在Plan features处stop-gradient，Plan head只由direction/moving supervision训练，而`plan_to_hidden`由Action loss训练。这一点不应误报为Action loss训练Plan predictor。
- `COSMOS_DETACH_HIDDEN=true`仍保持官方最小改动边界：Action loss不经H18回传Video DiT；Video DiT由`future_video_loss`全参更新。

训练 state 是 `(1,60)`：

```text
前16D：唯一部署输入，robot sin/cos state
后44D：只计算auxiliary loss，送Action前物理切除
```

推理严格接收 `(1,16)`，不构造44个零占位。目标tail任意改变不会改变RSIG预测token或Action condition。

截至 2026-07-29 的实际状态：

- v3 Push/Pick 联合 RSIG 已训练并保存 `steps_10000_pytorch_model.pt`；
- 该 checkpoint 已完成干净环境 Push/Pick 各3次闭环，结果均为 `3/3`，零动作裁剪；
- eval 已修复 standalone BF16 dtype，并生成带 RSIG 数值叠加的 H.264 视频；
- 正式干净环境 `50+50` 尚未完成：两次启动都在模型搬运阶段被同机并发任务动态占满显存，rollout 未开始；
- v4 彩色干扰数据已完整导出并通过 loader gate，但 v4 RSIG 正式训练 checkpoint 尚未产生；
- v4 彩色干扰闭环环境、真实 test seed manifest 和 robot-only/RSIG eval 入口已搭好，尚未取得正式成功率。

## 2. 44D label顺序

```text
0:8    EEF/Object × front/wrist normalized UV
8:12   四个per-view valid
12:15  robot-base ee_to_object
15     physical contact
16     current object moving
17:26  EEF h1/h4/h8 unit direction
26:35  Object h1/h4/h8 unit direction
35:38  EEF moving
38:41  Object moving
41:44  shared future valid
```

Object UV来自oracle segmentation mask centroid；EEF UV由TCP world point和相机内外参投影。current object-moving由robot-base linear velocity是否超过`fps × 0.001 m/s`判断。未来moving由位移是否超过1 mm判断；静止或invalid帧不计算direction loss。

## 3. 模型与开关

主开关：

```text
RSIG_ENABLED=true|false
RSIG_VIDEO_INJECTION=true|false
RSIG_PLAN_TOKENS_ENABLED=true|false
RSIG_EEF_DIRECTION_ENABLED=true|false
```

含义：

- `RSIG_ENABLED=false`：完全走原有baseline/tracker/object-dynamics代码。
- `RSIG_VIDEO_INJECTION=false`：A3 no role-to-video residual。
- `RSIG_PLAN_TOKENS_ENABLED=false`：A1 no direction-plan condition，保持condition token shape不变。
- `RSIG_EEF_DIRECTION_ENABLED=false`：A2 object-direction-only。

五项auxiliary loss及总scale均可通过环境变量独立设定：

```text
RSIG_ROLE_WEIGHT
RSIG_RELATION_WEIGHT
RSIG_CONTACT_WEIGHT
RSIG_MOVING_WEIGHT
RSIG_DIRECTION_WEIGHT
RSIG_LOSS_SCALE
```

初始值1只用于20-step量级检查，不应未经检查直接视为最终40k权重。

## 4. 数据导出

```bash
cd /remote-home/jinminghao/DiT4DiT

PYTHONNOUSERSITE=1 \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos-libero/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
PARALLEL_JOBS=6 \
bash examples/RLBench_EORT/train_files/export_maniskill_eort_v3_rsig_v1_all.sh
```

输出为：

```text
.../lerobot/{train,val,test}/rsig_v1/
  maniskill_eort_push_cube_256_sim2real_robot_base_rsig_v1_lerobot
  maniskill_eort_pick_cube_256_sim2real_robot_base_rsig_v1_lerobot
```

只编码front和wrist，不重复写right-shoulder视频。已有learned-tracker、oracle和robot-only数据不会被修改；目标目录存在时exporter默认拒绝覆盖。

## 5. 训练前loader gate

```bash
cd /remote-home/jinminghao/DiT4DiT

PYTHONNOUSERSITE=1 \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos-libero/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
CHECK_ONLY=1 \
bash examples/RLBench_EORT/train_files/run_maniskill_eort_v3_joint_rsig_v1.sh
```

必须输出：

```text
state  (1,60)
action (8,7)
image  5 × (3,224,448)
```

## 6. 单GPU一步forward/backward smoke

```bash
cd /remote-home/jinminghao/DiT4DiT
mkdir -p logs/maniskill_eort_v3_policy

CUDA_VISIBLE_DEVICES=4 \
PYTHONNOUSERSITE=1 \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
BASE_MODEL=/remote-home/jinminghao/ckpts/Cosmos-Predict2.5-2B \
nohup bash examples/RLBench_EORT/train_files/run_maniskill_eort_v3_joint_rsig_v1_smoke.sh \
> logs/maniskill_eort_v3_policy/rsig_v1_smoke_1step.log 2>&1 &

echo "PID=$!"
```

必须检查：

```text
action_dit_loss
future_video_loss
rsig_role/relation/contact/moving/direction_loss
total_loss
rsig_diag/action_role_projection_grad_norm > 0
rsig_diag/action_interaction_projection_grad_norm > 0
rsig_diag/action_plan_projection_grad_norm > 0
rsig_diag/video_alpha_grad_norm > 0
rsig_diag/video_hidden_dim_* 与真实H18布局一致
rsig_diag/video_alpha_before_step != video_alpha_after_step
无shape/dtype/OOM/NaN错误
final_model可保存
```

Plan predictor stop-gradient与`plan_to_hidden`非零梯度由focused unit test验证，不使用ZeRO-2清理后的`.grad` presence作运行时证据。smoke launcher默认使用`NUM_WARMUP_STEPS=0`，因此一步即可观察gate更新；它关闭额外DDIM eval和中间checkpoint，只保留训练结束时的`final_model`。

## 7. 4 GPU 20-step校准

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONNOUSERSITE=1 \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
BASE_MODEL=/remote-home/jinminghao/ckpts/Cosmos-Predict2.5-2B \
RUN_ID=maniskill_eort_v3_push_pick_rsig_v1_calibration20 \
nohup bash examples/RLBench_EORT/train_files/run_maniskill_eort_v3_joint_rsig_v1_calibration_20step.sh \
> logs/maniskill_eort_v3_policy/rsig_v1_calibration20.log 2>&1 &
```

launcher固定4 GPU、batch 1、accumulation 4、20步、warmup 5000、decord、无DDIM eval、每步诊断，并在step 20保存checkpoint。当前loss日志是rank-0最后一个accumulation microbatch，只适合finite/数量级判断，不是全局effective-batch均值；梯度诊断现使用backward期间的当前rank临时autograd hook，不再聚合ZeRO完整梯度，但该run仍包含逐步诊断，不能用于正式吞吐结论。根据Action/Video主loss、五项raw auxiliary loss、路径专属梯度、`video_alpha`、外部完整optimizer-step wall time和峰值显存确定最终权重。然后用20-step checkpoint验证save/resume，不先启动40k。

## 8. 4 GPU正式训练

完成上述gate后：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONNOUSERSITE=1 \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
BASE_MODEL=/remote-home/jinminghao/ckpts/Cosmos-Predict2.5-2B \
NUM_PROCESSES=4 \
PER_DEVICE_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=3 \
MAX_TRAIN_STEPS=40000 \
SAVE_INTERVAL=10000 \
LOGGING_FREQUENCY=100 \
WANDB_MODE=offline \
RUN_ID=maniskill_eort_v3_push_pick_front_wrist_rsig_v1_planproj_fullft_40k_bs4acc3 \
nohup bash examples/RLBench_EORT/train_files/run_maniskill_eort_v3_joint_rsig_v1.sh \
> logs/maniskill_eort_v3_policy/rsig_v1_planproj_fullft_40k_bs4acc3.log 2>&1 &

echo "PID=$!"
```

这仍是Video DiT和Action DiT全参数训练；只冻结原版已冻结的text encoder与VAE，不挂LoRA。

## 9. 闭环eval与可视化

模型server继续读取checkpoint内保存的RSIG配置：

```bash
CUDA_DEVICE=4 PORT=10093 \
MODEL_PYTHON=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
bash examples/RLBench_EORT/eval_files/run_maniskill_eort_policy_server.sh \
  /path/to/RSIG_CHECKPOINT.pt
```

客户端只发送front+wrist、task text和8D raw robot state；内部变为16D：

```bash
CUDA_DEVICE=5 \
bash examples/RLBench_EORT/eval_files/run_maniskill_eort_eval.sh \
  --checkpoint /path/to/RSIG_CHECKPOINT.pt \
  --condition rsig \
  --rsig-overlay \
  --env-id PushCubeEORTSim2Real-v1 \
  --episodes 50 \
  --seed-start 2000000 \
  --replan-every 8 \
  --max-steps 200 \
  --port 10093 \
  --output /path/to/rsig_push_eval
```

overlay显示front预测的EEF/Object位置、contact/object-moving概率和h1/h4/h8双运动方向；heatmap peak不是校准visibility概率。

更推荐直接使用会管理 server 生命周期的统一入口：

```bash
CKPT=/path/to/RSIG_CHECKPOINT.pt \
MODEL_GPU=4 \
SIM_GPU=5 \
EPISODES=50 \
STAGE=all \
bash examples/RLBench_EORT/eval_files/run_maniskill_eort_rsig_eval.sh
```

默认协议固定为：

```text
Push + Pick
replan_every = 8
max_steps = 200
DDIM steps = 10
front + wrist
RSIG overlay = enabled
```

server 必须使用 BF16。模型卡应稳定空闲至少约24 GiB；“启动前看起来有空间”不足以保证模型搬运期间不会被并发任务抢占。

## 10. 当前最需要警惕的点

1. v3 10k 已证明 checkpoint 可做 RSIG BF16 forward 和闭环动作，但3条/任务不构成正式成功率。
2. auxiliary loss默认权重仍缺少完整20-step量级与40k稳定性记录。
3. `video_alpha=0`保证初始化不破坏baseline，但必须持续记录gate和Video侧梯度，不能只看RSIG auxiliary loss。
4. Push/Pick干净环境接近ceiling，主要结论必须依赖v4 distractor、camera/physics/OOD和配对baseline。
5. RSIG role label来自simulator oracle监督；部署输入不含oracle不等于已经解决真机grounding。
6. v4只有六种训练内颜色；它测试未见seed/布局和干扰鲁棒性，不等于开放词汇或未见颜色泛化。
7. Action DiT推理从随机噪声开始，同一环境seed重跑仍可能变化；严格模型比较必须固定或重复model-noise。
8. 当前显存竞争会让server在模型搬运阶段OOM；OOM前未开始rollout时不能误报为策略失败。

## 11. 一张图看清当前完整链路

```text
ManiSkill successful episode
  ├─ front RGB 256×256
  ├─ wrist RGB 256×256
  ├─ robot raw state 8D
  ├─ robot-base metric action 7D
  └─ simulator-only RSIG target 44D
                 │
                 ▼
          LeRobot train split
  5个时序帧，每帧front+wrist横拼
          image: 5 × (3,224,448)
          action: (8,7)
          state:  (1,60)
                 │
                 ├─ 前16D：8D robot → sin/cos
                 └─ 后44D：只作为监督，进入策略前切除
                 │
                 ▼
     Front/Wrist + Instruction + 16D Robot
                 │
                 ▼
        Cosmos Video DiT / H18
                 │
          ┌──────┴────────┐
          ▼               ▼
  RSIG Role/Interaction   原Video hidden
  ├─ EEF Role Token       │
  ├─ Object Role Token    │
  ├─ Interaction Token    │
  └─ h1/h4/h8 Plan Tokens│
          │               │
          ├─ 6 tokens ────┼──────────────┐
          │               │              ▼
          └─ zero-init residual ──► Video DiT
                                    │
                                    └─ Future Video Loss

        [Video hidden ; 6 RSIG tokens] + Robot State
                            │
                            ▼
                       Action DiT
                            │
                            ▼
        8 × 7D robot-base metric action chunk
```

总损失：

```text
L_total = L_action + λ_video L_future_video + λ_rsig L_rsig

L_rsig =
  w_role      L_role
  + w_relation  L_relation
  + w_contact   L_contact
  + w_moving    L_moving
  + w_direction L_direction
```

关键梯度边界：

- Action loss更新Action DiT、Role/Interaction投影和Plan-to-Action投影；
- Plan predictor在送入Action token前stop-gradient，只由direction/moving监督训练；
- `COSMOS_DETACH_HIDDEN=true`时Action loss不穿过H18更新Video DiT；
- Future Video loss更新Video DiT，并通过zero-init gate逐步启用RSIG Video residual；
- RSIG loss不是唯一作用路径，RSIG token和Video residual都进入主任务计算。

## 12. 数据合同与train/val/test

### 12.1 每个样本

| 字段 | Shape | 含义 |
| --- | --- | --- |
| front+wrist时序图像 | `5 × (3,224,448)` | `t={0,2,4,6,8}`，每个时刻横向拼接两视角 |
| raw robot state | `8D` | TCP `xyz+rpy`、pad、gripper |
| policy robot state | `16D` | `sin(8D)+cos(8D)` |
| RSIG target | `44D` | 只用于训练监督 |
| train state | `(1,60)` | `16D policy + 44D target` |
| inference state | `(1,16)` | 只含可部署robot state |
| action target | `(8,7)` | robot-base `Δxyz+Δrotvec+gripper` |

### 12.2 v3与v4

| 项目 | v3 clean | v4 color distractors |
| --- | --- | --- |
| 环境 | `*EORTSim2Real-v1` | `*EORTColorDistractors-v1` |
| 目标 | 单目标 | 彩色目标cube |
| 干扰物 | 无人工干扰物 | 3个物理干扰物 |
| 指令 | generic task text | 六种颜色指令 |
| train episodes | Push 500 + Pick 500 | Push 1000 + Pick 1000 |
| val episodes | 独立1M seed段 | Push 200 + Pick 200 |
| test episodes | 独立2M seed段 | Push 400 + Pick 400 |
| 模型/shape/action | RSIG-v1 | 完全相同 |

train用于优化；val用于选择checkpoint和调参；test只用于最终报告。不能随机拆同一episode的帧，否则相邻帧和同一场景会泄漏到不同split。

v4闭环必须读取采集时保存的真实test seed manifest。因为collector只保留成功规划轨迹，400个seed并非严格连续：

```text
Push: 2000000 ... 2000409，共400个唯一成功seed
Pick: 2000000 ... 2000407，共400个唯一成功seed
```

## 13. 当前数据路径、mixture与checkpoint

### 13.1 v3 RSIG

```text
数据：
/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real/lerobot/train/rsig_v1

mixture：
maniskill_eort_push_pick_v3_sim2real_rsig_v1_lerobot

当前checkpoint：
/remote-home/jinminghao/DiT4DiT/checkpoints/maniskill_eort_v3_policy/
  maniskill_eort_v3_push_pick_front_wrist_rsig_v1_planproj_fullft_40k_bs4acc3/
  checkpoints/steps_10000_pytorch_model.pt
```

该run配置为每卡batch 4、gradient accumulation 3、目标40k；当前只确认10k checkpoint存在。

### 13.2 v4 RSIG

```text
数据：
/remote-home/jinminghao/datasets/maniskill_eort_color_distractors_v4_20260728/
  lerobot/train/rsig_v1

mixture：
maniskill_eort_push_pick_v4_color_distractors_rsig_v1_lerobot

预期checkpoint根目录：
/remote-home/jinminghao/DiT4DiT/checkpoints/
  maniskill_eort_v4_color_distractors_policy
```

### 13.3 robot-only baseline

```text
/remote-home/jinminghao/DiT4DiT/checkpoints/maniskill_eort_v3_policy/
  maniskill_eort_v3_push_pick_front_wrist_robot_only_fullft_40k_bs3acc4/
  checkpoints/steps_40000_pytorch_model.pt
```

robot-only推理只使用front+wrist、instruction和16D robot state，不使用tracker、oracle object condition或RSIG。

## 14. 完整训练命令

### 14.1 v3 loader gate

```bash
cd /remote-home/jinminghao/DiT4DiT

PYTHONNOUSERSITE=1 \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
CHECK_ONLY=1 \
bash examples/RLBench_EORT/train_files/run_maniskill_eort_v3_joint_rsig_v1.sh
```

### 14.2 v3四卡正式训练

```bash
cd /remote-home/jinminghao/DiT4DiT
mkdir -p logs/maniskill_eort_v3_policy

CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real \
BASE_MODEL=/remote-home/jinminghao/ckpts/Cosmos-Predict2.5-2B \
NUM_PROCESSES=4 \
PER_DEVICE_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=3 \
MAX_TRAIN_STEPS=40000 \
SAVE_INTERVAL=10000 \
LOGGING_FREQUENCY=100 \
VIDEO_BACKEND=decord \
NUM_WORKERS=8 \
WANDB_MODE=offline \
RUN_ID=maniskill_eort_v3_push_pick_front_wrist_rsig_v1_planproj_fullft_40k_bs4acc3 \
nohup bash examples/RLBench_EORT/train_files/run_maniskill_eort_v3_joint_rsig_v1.sh \
> logs/maniskill_eort_v3_policy/rsig_v1_planproj_fullft_40k_bs4acc3.log 2>&1 &

echo "PID=$!"
```

### 14.3 v4 loader gate

```bash
cd /remote-home/jinminghao/DiT4DiT

PYTHONNOUSERSITE=1 \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_color_distractors_v4_20260728 \
CHECK_ONLY=1 \
bash examples/RLBench_EORT/train_files/run_maniskill_eort_v4_color_distractors_joint_rsig_v1.sh
```

### 14.4 v4四卡20-step gate

```bash
cd /remote-home/jinminghao/DiT4DiT
mkdir -p logs/maniskill_eort_v4_color_distractors_policy

CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_color_distractors_v4_20260728 \
BASE_MODEL=/remote-home/jinminghao/ckpts/Cosmos-Predict2.5-2B \
NUM_PROCESSES=4 \
PER_DEVICE_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
MAX_TRAIN_STEPS=20 \
NUM_WARMUP_STEPS=0 \
SAVE_INTERVAL=20 \
LOGGING_FREQUENCY=1 \
NUM_WORKERS=0 \
RSIG_GRAD_DIAGNOSTICS=true \
RSIG_GRAD_DIAGNOSTICS_FREQUENCY=1 \
WANDB_MODE=offline \
RUN_ID=maniskill_eort_v4_color_distractors_rsig_v1_cal20 \
nohup bash examples/RLBench_EORT/train_files/run_maniskill_eort_v4_color_distractors_joint_rsig_v1.sh \
> logs/maniskill_eort_v4_color_distractors_policy/rsig_v1_cal20_4xa100.log 2>&1 &

echo "PID=$!"
```

### 14.5 v4四卡40k

```bash
cd /remote-home/jinminghao/DiT4DiT
mkdir -p logs/maniskill_eort_v4_color_distractors_policy

CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHON_BIN=/remote-home/jinminghao/miniconda3/envs/cosmos/bin/python \
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_color_distractors_v4_20260728 \
BASE_MODEL=/remote-home/jinminghao/ckpts/Cosmos-Predict2.5-2B \
NUM_PROCESSES=4 \
PER_DEVICE_BATCH_SIZE=3 \
GRADIENT_ACCUMULATION_STEPS=4 \
MAX_TRAIN_STEPS=40000 \
SAVE_INTERVAL=10000 \
LOGGING_FREQUENCY=100 \
VIDEO_BACKEND=decord \
NUM_WORKERS=8 \
RSIG_GRAD_DIAGNOSTICS=false \
WANDB_MODE=offline \
RUN_ID=maniskill_eort_v4_color_distractors_push_pick_rsig_v1_fullft_40k_bs3acc4 \
nohup bash examples/RLBench_EORT/train_files/run_maniskill_eort_v4_color_distractors_joint_rsig_v1.sh \
> logs/maniskill_eort_v4_color_distractors_policy/rsig_v1_fullft_40k_bs3acc4.log 2>&1 &

echo "PID=$!"
```

若20-step证明batch 3超显存，使用：

```text
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=12
```

## 15. 完整闭环评估命令

### 15.1 v3干净环境：RSIG checkpoint

```bash
cd /remote-home/jinminghao/DiT4DiT
mkdir -p logs/maniskill_eort_v3_policy

CKPT=$PWD/checkpoints/maniskill_eort_v3_policy/maniskill_eort_v3_push_pick_front_wrist_rsig_v1_planproj_fullft_40k_bs4acc3/checkpoints/steps_10000_pytorch_model.pt \
MODEL_GPU=4 \
SIM_GPU=5 \
EPISODES=50 \
STAGE=all \
REPLAN_EVERY=8 \
MAX_STEPS=200 \
STAMP=rsig10k_clean_test50 \
nohup bash examples/RLBench_EORT/eval_files/run_maniskill_eort_rsig_eval.sh \
> logs/maniskill_eort_v3_policy/rsig10k_clean_test50.log 2>&1 &

echo "PID=$!"
```

### 15.2 v4彩色干扰环境：RSIG checkpoint

旧v3 checkpoint和未来v4 checkpoint共用同一入口，只替换`CKPT`：

```bash
cd /remote-home/jinminghao/DiT4DiT
mkdir -p logs/maniskill_eort_v4_color_distractors

CKPT=/path/to/RSIG_CHECKPOINT.pt \
MODEL_GPU=4 \
SIM_GPU=5 \
EPISODES=50 \
STAGE=all \
REPLAN_EVERY=8 \
MAX_STEPS=200 \
STAMP=rsig_v4_test50 \
nohup bash examples/RLBench_EORT/eval_files/run_maniskill_eort_v4_color_distractors_rsig_eval.sh \
> logs/maniskill_eort_v4_color_distractors/rsig_v4_test50.log 2>&1 &

echo "PID=$!"
```

该wrapper自动：

- 切换到`Push/PickCubeEORTColorDistractors-v1`；
- 读取Push/Pick各自真实test seed manifest；
- 按`target_color_id`生成六种颜色指令；
- 启用BF16 server和RSIG overlay；
- 将输出写入`eval_outputs/maniskill_eort_v4_color_distractors`。

### 15.3 v4彩色干扰环境：干净数据训练的robot-only baseline

先用短变量保存manifest，避免终端复制时长路径被换行拆断：

```bash
cd /remote-home/jinminghao/DiT4DiT
mkdir -p logs/maniskill_eort_v4

DATA=/remote-home/jinminghao/datasets/maniskill_eort_color_distractors_v4_20260728
PUSH_MANIFEST="$DATA/push_cube/test/color_distractors/raw/PushCubeEORTColorDistractors-v1/motionplanning/push_cube_test_color_distractors_seed2000000_n400.seed_manifest.json"
PICK_MANIFEST="$DATA/pick_cube/test/color_distractors/raw/PickCubeEORTColorDistractors-v1/motionplanning/pick_cube_test_color_distractors_seed2000000_n400.seed_manifest.json"

test -f "$PUSH_MANIFEST" && test -f "$PICK_MANIFEST"
```

```bash
CKPT=$PWD/checkpoints/maniskill_eort_v3_policy/maniskill_eort_v3_push_pick_front_wrist_robot_only_fullft_40k_bs3acc4/checkpoints/steps_40000_pytorch_model.pt \
CONDITION=robot_only \
RSIG_OVERLAY=0 \
MODEL_GPU=4 \
SIM_GPU=5 \
EPISODES=50 \
STAGE=all \
REPLAN_EVERY=8 \
MAX_STEPS=200 \
EVAL_VARIANT=color_distractors_v4 \
OUTPUT_ROOT=$PWD/eval_outputs/maniskill_eort_v4_color_distractors \
PUSH_ENV_ID=PushCubeEORTColorDistractors-v1 \
PICK_ENV_ID=PickCubeEORTColorDistractors-v1 \
PUSH_SEED_MANIFEST="$PUSH_MANIFEST" \
PICK_SEED_MANIFEST="$PICK_MANIFEST" \
STAMP=robot_only_40k_v4_test50 \
nohup bash examples/RLBench_EORT/eval_files/run_maniskill_eort_robot_only_baseline_eval.sh \
> logs/maniskill_eort_v4/robot_only_40k_v4_test50.log 2>&1 &

echo "PID=$!"
```

### 15.4 结果与视频审计

```bash
tail -f /path/to/eval.log
find /path/to/eval_output -name summary.json -print -exec cat {} \;
find /path/to/eval_output -name episodes.jsonl -print
find /path/to/eval_output -name '*.mp4' | wc -l
```

每个episode记录：

```text
seed
instruction
steps
success
clipped_actions
valid_condition_steps
```

RSIG视频为front+wrist横拼的H.264 MP4。front overlay显示：

```text
EEF/Object role位置与heatmap peak
contact probability
object-moving probability
h1/h4/h8的EEF/Object robot-base XY方向与moving probability
```

这些是模型预测，不是GT；heatmap peak不能称为校准visibility。

## 16. 公平实验矩阵

最小主实验：

| 训练 | 测试环境 | 目的 |
| --- | --- | --- |
| clean robot-only | clean | 原版式ID baseline |
| clean RSIG | clean | 确认RSIG不破坏基础能力 |
| clean robot-only | v4 distractor test | 无结构条件的OOD baseline |
| clean RSIG | v4 distractor test | RSIG零样本抗干扰 |
| v4 robot-only | v4 distractor test | 数据增强本身收益 |
| v4 RSIG | v4 distractor test | 完整方法 |

所有比较必须保持：

```text
相同front+wrist
相同Video/Action全参训练
相同训练step与global batch
相同test seed manifest
replan=8
max_steps=200
相同DDIM steps
报告零/非零action clipping
```

建议至少报告：

```text
Push success / 50
Pick success / 50
总体 success / 100
平均完成步数
action clipping
按六种颜色分组结果
模型随机噪声重复或置信区间
```

## 17. 当前已有结果与未完成事项

### 已有证据

- v4六组LeRobot数据：3200 episodes、223113 frames、6400个front/wrist视频；
- v3 RSIG 10k checkpoint存在；
- v3 clean闭环smoke：Push `3/3`、Pick `3/3`，零action clipping；
- clean robot-only 40k正式闭环：Push `50/50`、Pick `50/50`，零action clipping；这是单次随机Action采样结果，不代表OOD能力；
- RSIG standalone BF16 dtype问题已修复；
- RSIG overlay已验证生成H.264、`512×256`、`yuv420p`视频；
- robot-only 40k checkpoint完整存在；
- v4环境ID、颜色指令和两个400-seed manifest已接入闭环launcher。

### 尚未完成

- v3 RSIG正式clean `50+50`；
- clean robot-only在v4上的`50+50`；
- clean RSIG在v4上的`50+50`；
- v4 RSIG 20-step loss/梯度校准；
- v4 RSIG 40k训练与正式评估；
- 固定或重复Action DiT model-noise；
- 真机front/wrist标定、时延、安全和真实grounding。

## 18. 代码位置

```text
ManiSkill数据与环境：
/remote-home/jinminghao/WAMs/ManiSkill

DiT4DiT模型与训练：
/remote-home/jinminghao/DiT4DiT
```

核心文件：

```text
DiT4DiT/model/modules/rsig.py
  RSIG module、6 tokens、44D loss、Video residual

DiT4DiT/model/framework/DiT4DiT.py
  60D/16D切分、Video/Action接入、总前向

DiT4DiT/model/modules/vlm/Cosmos25.py
  live Video H18 hook

DiT4DiT/training/train.py
  Action/Video/RSIG总loss和梯度诊断

examples/RLBench_EORT/train_files/
  run_maniskill_eort_v3_joint_rsig_v1.sh
  run_maniskill_eort_v4_color_distractors_joint_rsig_v1.sh

examples/RLBench_EORT/eval_files/
  eval_maniskill_eort.py
  run_maniskill_eort_rsig_eval.sh
  run_maniskill_eort_v4_color_distractors_rsig_eval.sh
  run_maniskill_eort_robot_only_baseline_eval.sh
```

## 19. 其他文档如何使用

本文负责“当前事实与执行命令”。其他文档只作专项参考：

- `RSIG_DiT4DiT_FORWARD_BACKWARD_AND_GRADIENT_FLOW_CN_20260728.md`：逐参数梯度与DeepSpeed诊断；
- `RSIG_DiT4DiT_FINAL_FEASIBLE_PLAN_CN_20260727.md`：为什么删除memory、Goal Slot和phase；
- `RSIG_DiT4DiT_Problems_and_Solutions_CN.md`：研究问题与方案演化；
- `TRACKER_BASED_TRAINING_COMMANDS_20260721.md`：早期learned-tracker与robot-only基线历史；
- DiT4DiT侧`MANISKILL_EORT_V4_COLOR_DISTRACTORS_RSIG_DATA_AND_TRAINING_CN_20260728.md`：v4数据处理与数量审计。

执行新训练或评估前，仍必须先读取：

```text
/remote-home/jinminghao/WAMs/ManiSkill/docs/思考与隐患.md
/remote-home/jinminghao/DiT4DiT/docs/思考与隐患.md
```
