# RSIG-DiT4DiT v1 实现、训练与验证手册

更新时间：2026-07-28 03:27:43 UTC

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
- 3个Plan token会进入Action DiT并改变预测，但v1按既定因果边界在该分支stop-gradient；Plan head只由direction/moving supervision训练。这一点是有意设计，不应误报为Action loss训练Plan predictor。
- `COSMOS_DETACH_HIDDEN=true`仍保持官方最小改动边界：Action loss不经H18回传Video DiT；Video DiT由`future_video_loss`全参更新。

训练 state 是 `(1,60)`：

```text
前16D：唯一部署输入，robot sin/cos state
后44D：只计算auxiliary loss，送Action前物理切除
```

推理严格接收 `(1,16)`，不构造44个零占位。目标tail任意改变不会改变RSIG预测token或Action condition。

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
rsig_diag/video_alpha_grad_norm > 0
rsig_diag/video_hidden_dim_* 与真实H18布局一致
rsig_diag/video_alpha_before_step != video_alpha_after_step
无shape/dtype/OOM/NaN错误
final_model可保存
```

Plan stop-gradient由focused unit test验证，不使用ZeRO-2清理后的`.grad` presence作运行时证据。smoke launcher默认使用`NUM_WARMUP_STEPS=0`，因此一步即可观察gate更新；它关闭额外DDIM eval和中间checkpoint，只保留训练结束时的`final_model`。

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
PER_DEVICE_BATCH_SIZE=3 \
GRADIENT_ACCUMULATION_STEPS=4 \
MAX_TRAIN_STEPS=40000 \
SAVE_INTERVAL=10000 \
LOGGING_FREQUENCY=100 \
WANDB_MODE=offline \
RUN_ID=maniskill_eort_v3_push_pick_front_wrist_rsig_v1_fullft_40k \
nohup bash examples/RLBench_EORT/train_files/run_maniskill_eort_v3_joint_rsig_v1.sh \
> logs/maniskill_eort_v3_policy/rsig_v1_fullft_40k.log 2>&1 &

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

## 10. 当前最需要警惕的点

1. 最大不确定性是完整Cosmos checkpoint的H18真实layout和第二次flow-matching hook，synthetic测试不能替代GPU forward/backward。
2. auxiliary loss默认权重尚未校准，40k前必须先做20-step。
3. `video_alpha=0`保证初始化不破坏baseline，但也可能长期不动；必须记录它和role-to-video梯度。
4. Push/Pick ID baseline接近ceiling，最终贡献需要OOD camera/occlusion/physics/distractor和更复杂任务，不能只报告ID success。
5. RSIG role label仍来自simulator oracle supervision；部署输入不含oracle，但这不自动证明真实检测/跟踪鲁棒。
6. instruction文本在当前两任务中变化有限，不能据此声称开放词汇grounding。
