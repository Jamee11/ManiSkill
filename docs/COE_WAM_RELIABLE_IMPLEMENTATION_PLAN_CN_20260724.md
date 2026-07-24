# COE-WAM 可靠实施规划

日期：2026-07-24

状态：实施前设计冻结；尚未修改 COE-WAM 模型、训练或 counterfactual collector
主仓库：

```text
ManiSkill : /remote-home/jinminghao/WAMs/ManiSkill
DiT4DiT  : /remote-home/jinminghao/DiT4DiT
```

本文把前期 object-centric、tracker、DiT4DiT dynamics、Causal-RIFT/CARE 和
FlowWAM 讨论收敛为一套可逐步验证、失败可停止、不会破坏现有 baseline 的
COE-WAM 实施方案。

本文是规划，不表示以下模块已经实现。后续每次进入模型、数据或训练代码修改前，
必须先按项目规则说明精确变更并取得用户确认。

---

## 0. 一页结论

### 0.1 最终研究问题

COE-WAM 不再研究“怎样再加入一个 object condition”或“怎样再预测一种 flow”，
而研究：

> 给定同一个当前状态和不同的候选动作，模型能否识别这些动作将怎样改变任务物体，
> 并利用这种 object effect 改善动作选择、OOD 和 sim2real？

形式上是：

```text
p(object effect | current deployable observation, candidate action)
```

而不是当前 observation-only dynamics 学到的：

```text
p(expert future | current observation, instruction)
```

### 0.2 最可靠的实现顺序

```text
冻结并同步现有 baseline
→ 证明当前 tracker condition 被 policy 使用且可能有独立价值
→ 验证 simulator same-state restore
→ 小规模采集 same-state multi-action branch
→ 离线训练最小 action-conditioned effect predictor
→ 先测 action sensitivity 和 oracle candidate upper bound
→ 再接 K=4 sample–predict–select
→ 确认闭环收益后才联合训练 Video DiT/Action DiT
→ 最后才考虑 role query、uncertainty、optical-flow auxiliary 和更多任务
```

### 0.3 第一版明确不做

- 不复刻 FlowWAM 的 RGB/optical-flow 双流生成；
- 不增加第二个 Video DiT；
- 不做 Action DiT draft-refine；
- 不修改 Cosmos prompt token；
- 不引入 Slot Attention、DETR、ControlNet 或新的大型视觉 backbone；
- 不用 simulator actor ID、segmentation、GT pose、phase 作为 policy 输入；
- 不在第一版训练 uncertainty head；
- 不覆盖现有 v3 HDF5、LeRobot、tracker checkpoint 或 40k policy checkpoint；
- 不把 PushCube/PickCube 的结果表述成通用 object-centric sim2real。

---

## 1. 当前可作为 baseline 的真实状态

## 1.1 ManiSkill 数据与感知

正式数据根：

```text
/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real
```

现有数据：

| task | train | val | test |
| --- | ---: | ---: | ---: |
| PushCube | 500 episodes | 100 | 200 |
| PickCube | 500 episodes | 100 | 200 |

当前数据合同：

- 三相机 raw RGB-D/segmentation 为 256×256；
- policy 只使用 front+wrist RGB；
- tracker 当前只使用 front RGB-D；
- canonical 几何和 action 为 robot-base；
- action 为 `(8,7)` chunk：

```text
[Δx_m, Δy_m, Δz_m,
 Δrotvec_x_rad, Δrotvec_y_rad, Δrotvec_z_rad,
 gripper_open_fraction]
```

tracker 已离线训练并冻结：

| task | val pixel error | val visibility accuracy |
| --- | ---: | ---: |
| PushCube | 8.654 px | 0.99705 |
| PickCube | 10.444 px | 0.99378 |

tracker 内部 heatmap 为 `(B,2,128,128)`；policy 最终读取的是每步：

```text
17D learned object message + 1D valid
```

不能把 heatmap resolution 当成 policy condition shape。

## 1.2 当前 DiT4DiT baseline

当前 official-minimal 路径：

```text
front+wrist 5 pixel frames
→ Video DiT predictive hidden
→ Action DiT
→ 8×7D action chunk
```

Action DiT 另外读取：

```text
16D robot sin/cos state
+ 17D learned tracker message
+ 1D observation valid
= 34D state
→ 一个 state token
```

训练合同：

- Video DiT 与 Action DiT 全参数训练；
- text encoder 和 VAE 冻结；
- 无 LoRA；
- future-video flow-matching loss 保留；
- tracker 冻结；
- Push/Pick 联合训练；
- front+wrist，right shoulder 不进入 policy。

当前最强已完成 checkpoint：

```text
/remote-home/jinminghao/DiT4DiT/checkpoints/maniskill_eort_v3_policy/
maniskill_eort_v3_push_pick_front_wrist_tracker_fullft_40k_bs3acc2/
checkpoints/steps_40000_pytorch_model.pt
```

同一批 50 个 test seeds、`replan=8`、`max_steps=200` 的已有结果：

| task | success | tracker valid |
| --- | ---: | ---: |
| PushCube | 49/50 = 98% | 3608/3608 = 100% |
| PickCube | 48/50 = 96% | 4416/4811 = 91.79% |

这说明：

1. 当前 baseline 已经很强，Push/Pick ID 存在明显 ceiling；
2. 新方法不能只看训练 loss 或 Push ID；
3. 必须使用 Pick failure、tracker-invalid、相机/物理 OOD 和更难任务；
4. 新旧方法必须固定 simulator seed 和 model-noise protocol 做 paired evaluation。

## 1.3 当前 Git 基线

本次规划开始前：

```text
ManiSkill branch : agent/maniskill-eort-oracle
ManiSkill HEAD   : 4764ea5
remote           : git@github.com:Jamee11/ManiSkill.git

DiT4DiT branch   : agent/rlbench-eort-oracle
DiT4DiT HEAD     : 2ad7019
remote           : git@github.com:Jamee11/dit4wam.git
```

两个分支均已跟踪 Jamee11 远端。当前工作树中的源码、规划和实验记录应分别形成明确
checkpoint commit；数据集、20GB 模型权重、`eval_outputs/` 和本地工具配置不提交
GitHub。

---

## 2. 对旧方案必须做的修正

## 2.1 当前 object dynamics 不是 COE-WAM

现有 DiT4DiT `object_dynamics_head`：

```text
mean(Video hidden) + current state
→ Linear
→ future object pose delta
```

它有三个问题：

1. 不读取 candidate action；
2. 只在成功 expert trajectory 上学习；
3. 更可能学习“当前状态下 expert 接下来会成功做什么”，而不是“换一个 action 后
   object 会怎样”。

因此不能直接把 `OBJECT_DYNAMICS_ENABLED=true` 当成 COE-WAM。

## 2.2 现有 18D future target 也不是最终 effect target

当前 18D 是：

```text
h={1,4,8} × [object Δxyz + object Δrotvec]
```

它可作为历史 dynamics baseline，但第一版 COE 需要重新定义监督与 scorer 的关系。

## 2.3 早先的 6D relation target 存在捷径和冗余

早先建议每个 horizon 预测：

```text
Δ(ee_to_object xyz)      3
Δ(object_to_goal xyz)    3
```

这个设计不够稳妥：

1. `Δee_to_object` 会随着 EEF 自身动作变化。即使 object 完全不动，模型也可通过
   action kinematics 得到较低误差，未必学到 object response。
2. 当前 Push/Pick 的 goal 静止，因此
   `Δ(object_to_goal) = -Δobject_position`，与 object displacement 重复。
3. scorer 可能偏爱“夹爪靠近物体”，但这不等于物体被推动、抓住或送向 goal。

修正：

> 第一版 primary effect 必须直接监督 action 所造成的 object response；
> EEF relation 作为 current context 和诊断，不作为 primary effect target。

## 2.4 uncertainty 不应在第一版强行实现

当前 simulator branch 在固定状态、固定 action、固定物理版本下应接近确定性。立即
输出 learned variance 容易得到不可解释的方差塌缩，也增加 loss 权重和 calibration
问题。

第一版只做 deterministic effect + valid/contact/grasp。只有 effect prediction 和
candidate ranking 已成立后，再用三 seed effect-head ensemble 或其他简单方式估计
epistemic disagreement。

## 2.5 不能直接让 effect loss 更新 Video DiT

当前 Cosmos layer hook 会把抓取的 hidden `detach()`。即使配置
`detach_hidden=false`，hook 输出仍然断开梯度。

因此实施分两阶段：

1. 第一版明确冻结 Video hidden，只训练小 effect head；
2. 只有独立 effect gate 成立，才单独修改 hidden gradient contract，并做
   gradient-presence、显存和 future-video-loss 回归。

这避免一开始同时改数据、模型、梯度和 policy。

---

## 3. 修订后的 COE-WAM 定义

## 3.1 当前状态

部署可用输入：

```text
front RGB-D
→ frozen tracker
→ object/goal current relation + visibility/valid

front+wrist RGB + language
→ Video DiT
→ predictive hidden

robot state
→ robot-base TCP/gripper state
```

不得作为输入：

```text
simulator actor ID
segmentation mask
GT object/goal pose
future object target
contact/phase/is_grasped GT
occluder_active
```

这些只能作为监督、分层采样或 QA。

## 3.2 Candidate action

固定为与当前 policy 完全一致的：

```text
candidate_action_metric: (K,8,7)
```

同时保存：

```text
candidate_action_normalized: (K,8,7)
```

原因：

- metric action 用于跨机器人语义、replay 和数据审计；
- normalized action 与 Action DiT 输出分布一致，供网络输入；
- 二者必须可逆核对，不能只保存 normalized 数值。

## 3.3 第一版 object-effect target

当前 cube 任务的最小 target：

```text
object_delta_pos_base_m: (K,3,3)
                         K branches
                         horizons h={1,4,8}
                         xyz meters

effect_valid:            (K,3)
contact:                 (K,3,1)
grasped:                 (K,3,1)  # Pick 有效；Push mask=false
```

可由 target 派生但不重复监督：

```text
predicted_object_pos
predicted_object_to_goal
goal_distance_after
goal_progress
object_motion_magnitude
```

第一版不监督：

- `Δee_to_object`：保留作 context/diagnostic；
- `Δobject_to_goal`：静态 goal 下与 object delta 重复；
- object rotation：两个 cube 任务不是主要任务变量；
- uncertainty：待 effect gate 通过；
- future RGB：数据和 I/O 成本高，当前研究问题不需要。

扩展到插入、旋转、工具或 articulated object 时，再按任务 schema 增加 rotation 或
joint-state target；不要现在创建一个大量 padding 的“万能 object state”。

## 3.4 最小 effect predictor

第一版复用已有数据与 hidden，不引入新的 transformer：

```text
pooled Video hidden                     2048D
+ current deployable state                34D
+ flattened normalized action chunk       56D
                                          ─────
                                           2138D
→ Linear
→ SiLU
→ Linear
→ object delta + contact/grasp logits
```

这是一个约百万参数量级的小 MLP。它足以检验 action conditioning；若这个最小版本
都无法通过 action-shuffle 和 within-state ranking，不应先增加更复杂结构。

必须同时做三个输入消融：

```text
A0: action only
A1: current state + action
A2: Video hidden + current state + action
```

只有 `A2 > A1`，才说明 predictive Video hidden 对 effect prediction 有增量价值。

## 3.5 推理时 sample–predict–select

```text
current observation
→ Video DiT forward once
→ Action DiT batch sample K=4 chunks
→ effect predictor batch predict 4 outcomes
→ deployable scorer
→ select one full 8-step chunk
→ execute 8 steps
→ reobserve
```

第一版 scorer：

```text
score =
  predicted goal-distance improvement
  + task-specific interaction bonus
  - invalid penalty
  - action-bound/safety penalty
```

Push：

- 主要看 predicted object-to-goal distance。

Pick：

- goal distance；
- grasp/contact 的阶段一致性；
- 禁止只通过 EEF 上抬得分而 object 不动。

第一版只从 Action DiT 产生的候选中选择，不对 action 做梯度优化，降低 out-of-
distribution exploitation 风险。

---

## 4. Counterfactual 数据设计

## 4.1 为什么必须独立 sidecar

现有 v3 数据是成功 demonstration。COE 需要：

```text
同一个 base state
+ 多个不同 action chunk
→ 多个真实 simulator outcomes
```

这不应写回现有 HDF5/NPZ/LeRobot，避免：

- 覆盖 canonical 数据；
- 把失败 branch 混入 successful-demo policy dataset；
- future effect 误进入 standard policy state；
- 破坏现有 tracker/policy 的可复现性。

建议新增：

```text
<collection_root>/coe_counterfactual_v1/
  push_cube/{train,val,test}/
  pick_cube/{train,val,test}/
```

## 4.2 Base-state split

所有 branch 必须继承 base trajectory 的 split：

```text
train base state → train branches
val base state   → val branches
test base state  → test branches
```

同一 base state 的所有 branches 永远放在同一 split。禁止按 branch 随机切分，否则
模型会在 val/test 看到相同 observation/state。

## 4.3 Base-state 采样

第一版从现有 train/val successful trajectories 选状态，按以下 strata 平衡：

- approach/no-contact；
- contact onset；
- object moving；
- near-goal；
- tracker valid；
- object 或 goal 部分不可见。

simulator phase/contact 可用于离线采样分层，但：

- 不进入 effect predictor；
- 不进入 Action DiT；
- 不作为部署 scorer 输入；
- manifest 必须标为 `label_only=true`。

仅从均匀随机 frame 采样会被大量“object 不动”状态淹没，模型可能只学零输出。

## 4.4 每个 base state 的 branch proposal

pilot 建议 `K_collect=8`：

```text
1 × original expert chunk
1 × zero/no-op chunk
3 × bounded local perturbations of expert chunk
3 × current Action DiT samples
```

每个 branch 记录：

```text
proposal_source: expert | noop | perturbation | policy
proposal_seed
policy_checkpoint
policy_noise_seed
metric action
normalized action
clipping/safety result
```

局部 perturbation 必须在现有 controller action bounds 内，并保持 chunk temporal
smoothness。不能每个 action step 独立加入大噪声，否则得到明显不真实、effect
predictor 很容易区分的 branch。

## 4.5 建议 sidecar schema

```text
schema_version                         string
task                                   string
split                                  string

base_state_id                          string
source_hdf5                            string
source_trajectory_id                   int
source_step                            int
simulator_seed                         int
env_state_hash                         string

maniskill_commit                       string
dit4dit_commit                         string
runtime_versions                       dict
control_mode                           string
sim_backend                            string
control_frequency_hz                   float

current_robot_state_raw                (8,)
current_robot_state_encoded            (16,)
current_tracker_condition              (17,)
current_tracker_valid                  (1,)
current_object_pos_base_label          (3,)   # label/QA only
current_goal_pos_base_label            (3,)   # label/QA only

candidate_action_metric                (K,8,7)
candidate_action_normalized            (K,8,7)
candidate_source                       (K,)
candidate_seed                         (K,)

object_delta_pos_base_m                (K,3,3)
effect_valid                           (K,3)
contact                                (K,3,1)
contact_valid                          (K,3,1)
grasped                                (K,3,1)
grasped_valid                          (K,3,1)

terminal_success                       (K,)
terminated_early                       (K,)
executed_steps                         (K,)
```

`current_object_pos_base_label` 和 `current_goal_pos_base_label` 是监督/QA，不得被
loader 放入 deployable input。

## 4.6 状态恢复是零号硬门槛

先做：

```text
save state
→ restore state
→ replay original expert 8-step chunk
→ repeat N times
→ compare with original and repeated outcomes
```

检查：

- TCP position/rotation；
- object position/rotation；
- object velocity；
- contact/grasp；
- success/termination；
- controller target；
- articulation/qpos/qvel；
- simulator/runtime version。

建议第一轮 acceptance：

| 场景 | h8 object position replay error | 事件一致性 |
| --- | ---: | --- |
| non-contact | ≤ 0.1 mm | termination/success 相同 |
| contact/grasp | ≤ 1 mm | contact/grasp/termination 相同 |

如果实际 PhysX 数值底噪高于阈值，先测量并解释，不可直接放宽到足以掩盖 branch
effect 的程度。若 restore 不可靠，same-state causal claim 不成立，必须停止。

## 4.7 Counterfactual QA

每个 shard 必须报告：

- base states / branches 数量；
- source split 与 seed overlap；
- restore error 分布；
- proposal source 分布；
- action clipping 比例；
- object motion magnitude 分布；
- contact/grasp positive ratio；
- h1/h4/h8 valid ratio；
- repeated-branch stochastic variance；
- zero/no-op outcome；
- expert replay outcome；
- policy-candidate outcome diversity；
- Git commits、checkpoint hash、runtime versions。

必须 fail closed：

- 缺 branch；
- restore 不通过；
- action 非有限；
- metric/normalized action 不一致；
- base state 跨 split；
- future target shape 不一致；
- branch output 路径已存在。

---

## 5. 分阶段实施计划

## Phase 0：版本冻结与 baseline contract

### 目标

保证之后任何 COE 结果都能回到当前 tracker-flat baseline。

### 工作

1. ManiSkill 当前 EORT collector、文档和规划形成 checkpoint commit；
2. DiT4DiT 当前 tracker/eval/40k baseline 相关代码形成独立 checkpoint commit；
3. 推送 Jamee11 现有远端分支；
4. 文档记录 commit、checkpoint、dataset root、test seeds 和评估协议；
5. 不提交 dataset、20GB checkpoint、48MB `eval_outputs` 或工具私有目录。

### 验收

```text
git status 中只剩明确排除的本地产物
远端分支包含本次 commit
40k checkpoint config 可读取
已有 49/50 Push、48/50 Pick summary 保留
```

## Phase 1：证明 object condition 的实际价值

### 目标

在采集新数据前，先回答当前 17D tracker condition 是否被 policy 使用。

### 最小实验

固定 40k checkpoint、相同 simulator seeds 和 policy noise：

| 条件 | 含义 |
| --- | --- |
| normal learned tracker | 当前 baseline |
| zero condition | condition dependence |
| lagged condition | 对时序错位是否敏感 |
| cross-episode shuffled condition | 对正确 object identity/geometry 是否敏感 |
| oracle condition | perception upper bound |

这一步使用同一 checkpoint，只能证明“依赖/敏感性”，不能单独证明“增加 condition
优于从头训练的 robot-only policy”。

如果 condition sensitivity 成立，再安排 matched training：

```text
robot-only 40k
vs tracker-flat 40k
```

训练预算、数据、seed、checkpoint selection 和 eval protocol 必须一致。

### 重点评估

- Pick tracker-invalid subset；
- camera/geometry/physics OOD；
- normal vs shuffled 的 paired success；
- action chunk 差异；
- goal-distance progress；
- condition invalid 后恢复能力。

### Gate

满足以下之一才继续把 tracker condition 作为 COE 主输入：

1. normal 明显优于 shuffled/lagged；
2. matched tracker-flat 优于 robot-only；
3. oracle 明显高于 learned，说明 perception 仍有可提升空间。

若三者都不成立，tracker condition 降级为诊断/teacher，不再强制进入最终方法。

## Phase 2：same-state restore smoke

### 目标

证明 ManiSkill branch 数据确实来自相同物理状态。

### 实现范围

只新增独立 restore/branch smoke 工具和 focused test；不修改 production v3 collector。

### 数据量

```text
Push  : 20 base states
Pick  : 20 base states
每个 base state 重放 expert branch 3 次
覆盖 approach/contact/moving/near-goal
```

### Gate

- restore accuracy 通过；
- repeated rollout variance 远小于计划 action perturbation 产生的 effect；
- contact/grasp 事件一致；
- 原 HDF5/sidecar 不被改写。

失败则停止 COE 数据采集，先修 restore contract。

## Phase 3：counterfactual pilot

### 目标

验证 branch outcome 有足够变化，且 proposal 分布不是明显无效动作。

### 建议规模

```text
train: 100 base states/task × 8 branches
val  :  30 base states/task × 8 branches

总计: 2,080 branch rollouts
```

这是 pilot，不进入论文最终数字。

### 必做统计

- object 不动比例；
- h1/h4/h8 displacement；
- contact/grasp balance；
- expert/no-op/perturbation/policy outcome overlap；
- policy candidates 的真实 outcome spread；
- restore error / true branch effect 比值。

### Gate

至少需要：

1. 同一 base state 的不同 action 能产生可测 outcome 差异；
2. policy candidate 之间不是全部等价；
3. contact/grasp positives 不被零 effect 淹没；
4. restore noise 显著小于 action-caused effect。

若 policy candidates 本身没有 outcome diversity，先改善 candidate proposal，不训练
复杂 effect model。

## Phase 4：离线 effect predictor

### 目标

证明模型真正读取 action，而非只预测 expert future。

### 实现方式

先冻结 40k Video DiT/Action DiT，按 base state 只缓存一次：

```text
pooled layer-17 hidden: float16 (2048,)
current 34D deployable state
```

branch action/outcome 保存在独立 sidecar。训练小 MLP，不改主 trainer。

### 训练消融

```text
A0 action only
A1 current state + action
A2 pooled Video hidden + current state + action
A3 A2 but action shuffled within batch
A4 A2 trained only on behavior/expert branch
A5 A2 trained on same-state counterfactual branches
```

### Loss

```text
L_effect  = masked Huber(object_delta_pos)
L_contact = masked BCE(contact)
L_grasp   = masked BCE(grasped)
L_total   = L_effect + λc L_contact + λg L_grasp
```

第一版不加入 action loss、video loss或 uncertainty loss。loss weight 只在 val 上按
物理指标选择，不使用 test。

### 指标

- h1/h4/h8 object position MAE，单位 cm；
- moving vs static classification；
- contact F1；
- grasp F1；
- within-base pairwise ranking accuracy；
- action-shuffle degradation；
- unseen perturbation generalization；
- expert-only vs counterfactual；
- per proposal-source error。

### Gate

必须同时满足：

1. `A5 > A4`；
2. shuffled action 显著变差；
3. within-state ranking 高于 random；
4. 在 policy proposal 上而不只是人工 perturbation 上有效；
5. physical-unit error 小于候选 outcome 间典型差异。

否则停止，不接 policy。

## Phase 5：candidate oracle upper bound

### 目标

在写在线 selector 前，确认 K 个候选中确实存在可利用的更优选择。

### 方法

对同一 observation 生成 `K=4` Action DiT candidates，并在 simulator branch 中都
执行，得到真实 outcome：

```text
candidate 0               # 公平 baseline
random candidate
oracle-best candidate     # 使用 simulator truth，仅作上界
effect-predicted candidate
```

报告：

- candidate action pairwise distance；
- true object-effect spread；
- oracle best 相对 candidate 0 的 goal-progress；
- oracle best 相对 candidate 0 的 success upper bound；
- predicted ranking regret。

### Gate

如果 oracle-best 相比 candidate 0 几乎没有收益，selector 不可能改善闭环。应先增加
Action DiT sampling diversity或转向更难任务，而不是增加 scorer 网络。

建议继续在线集成的最低信号：

```text
hard/OOD subset 的 oracle success upper bound ≥ +5 percentage points
或 goal-distance regret 有稳定、显著下降
```

`+5pp` 是工程 gate，不是统计结论；最终仍报告置信区间和 paired test。

## Phase 6：K=4 在线 sample–predict–select

### 目标

验证 effect prediction 有真实 policy value。

### 最小代码改动

1. model server 支持同一 observation 批量返回 K 个 action chunks；
2. effect head 对 K candidates 批量预测；
3. evaluator 使用同一 K candidate set：
   - baseline 取 candidate 0；
   - COE 取 predicted best；
4. 保持 `replan_every=8`、`max_steps=200`；
5. 固定 simulator seed、policy noise seeds 和 candidate ordering；
6. 保存每次 K actions、predicted effects、score、selected index 和真实 rollout
   outcome。

### 评估

先做：

```text
Push/Pick test seeds 只作回归
Pick known-failure/hard subset
camera OOD
geometry/physics OOD
tracker invalid/occlusion
```

正式对照：

| 方法 | K | selector |
| --- | ---: | --- |
| baseline | 1 | 无 |
| compute-matched random | 4 | random |
| COE | 4 | predicted effect |
| oracle upper bound | 4 | simulator truth，仅离线 |

不能只比较 `K=1 baseline` 和 `K=4 COE`，否则收益可能来自增加采样计算。

### Gate

- COE 显著优于 compute-matched random；
- 不只改善 predicted effect metric，还改善 closed-loop；
- latency/显存可接受；
- failure 不集中在 scorer exploitation；
- learned tracker 下成立，不只在 oracle condition 下成立。

## Phase 7：联合 COE-WAM

只有 Phase 6 成立后才进入。

### 7.1 修复 hidden gradient contract

现有 hook 的无条件 `detach()`必须改为可配置，并增加测试：

```text
detach=true  → effect/action loss 不回 Video hidden
detach=false → effect loss 对指定 Video block 有非零梯度
future-video flow-matching loss 仍存在且有限
```

### 7.2 联合 loss

```text
L = L_action
  + λvideo L_future_video_FM
  + λeffect L_counterfactual_effect
  + λevent L_contact/grasp
```

保留：

- Video DiT/Action DiT 全参数训练；
- text encoder/VAE 冻结；
- 无 LoRA；
- front+wrist；
- original numerical Action DiT；
- future-video loss。

### 7.3 数据组合

不要建立第二套完整 policy dataset。counterfactual sidecar 通过：

```text
(source trajectory id, source step)
```

引用原 observation/video；每次从 K branches 中采一个或多个 effect samples。标准
expert sample继续提供 action/video loss，branch sample提供 effect loss。

必须验证：

- branch target 不进入 policy state；
- standard batch 没有 future leakage；
- branch 与 standard sample 的 normalization 一致；
- effect loss 不破坏 video/action baseline；
- checkpoint 可在 effect head 缺失时加载旧 baseline；
-新 checkpoint 的 evaluator 明确读取 schema version。

### 7.4 何时增加 role query

只有出现以下情况才做：

```text
state+action effect 有效
candidate selection 有效
但 pooled Video hidden 不提供增量或 OOD localization 弱
```

再增加 EEF/object/goal query，从 spatial-temporal hidden 中抽取 role token。

暂不把 role token追加到 Cosmos prompt；先只送 effect bridge/Action DiT。

## Phase 8：不确定性、多任务与 sim2real

### 8.1 Uncertainty

先训练 3 个轻量 effect-head seeds：

```text
mean prediction
head disagreement
```

只有 disagreement 与真实 error/OOD 相关，才放入 scorer。不要第一版直接输出 learned
variance。

### 8.2 Optical flow

[FlowWAM](https://arxiv.org/html/2607.13017) 只作为相关 baseline/auxiliary：

- current/history optical flow baseline；
- motion-aware Video hidden auxiliary；
- tracker temporal cue；
-可视化 QA。

不能把 future GT optical flow 输入当前 policy；不能把主方法改成 RGB/flow 双流后再
声称主要差异。

### 8.3 任务扩展

论文级至少覆盖：

```text
push
pick-and-place
stack
insert
articulated open/close
multi-object referring
```

每类任务单独定义最小 object state，不提前做 universal padded schema。

### 8.4 Sim2Real

真实部署前必须闭合：

- front crop/resize/intrinsic；
- wrist intrinsic 和 hand-eye；
- RGB-D 对齐；
- object/goal 的真实定义；
- robot-base frame；
-控制频率与 8-step horizon 的真实时间；
- action bounds、workspace、collision、delay、急停；
- tracker invalid fallback；
- effect uncertainty/OOD fallback。

真实第一步是 offline perception/effect，不是直接闭环。

---

## 6. 两个仓库的职责和预计文件

以下是预计范围，不表示文件已创建。

## 6.1 ManiSkill

新增且不改 production v3 collector：

```text
scripts/eort/collect_coe_counterfactual.py
scripts/eort/audit_coe_counterfactual.py
tests/test_coe_counterfactual.py
```

职责：

- state restore；
- branch execution；
- effect label；
- provenance；
- fail-closed audit。

尽量复用：

- v3 robot-base transform；
- action conversion；
- HDF5 trajectory lookup；
- existing summary/audit helpers。

## 6.2 DiT4DiT

第一阶段新增：

```text
examples/RLBench_EORT/scripts/cache_coe_features.py
examples/RLBench_EORT/scripts/train_coe_effect_head.py
examples/RLBench_EORT/eval_files/eval_coe_effect.py
```

必要时最小修改：

```text
DiT4DiT/model/framework/DiT4DiT.py
examples/RLBench_EORT/eval_files/maniskill_model_client.py
examples/RLBench_EORT/eval_files/eval_maniskill_eort.py
```

只有联合训练阶段才修改：

```text
DiT4DiT/model/modules/vlm/Cosmos25.py
DiT4DiT/training/train.py
DiT4DiT/dataloader/...
```

不创建独立大框架、registry或通用插件系统；先复用当前 DiT4DiT 类、model server 和
evaluator。

---

## 7. 实验矩阵

## 7.1 Object condition

| ID | policy/input | 目的 |
| --- | --- | --- |
| C0 | normal tracker | 当前 baseline |
| C1 | zero tracker | 是否依赖 |
| C2 | lagged tracker | 时序敏感性 |
| C3 | shuffled tracker | 是否需要正确 task-object geometry |
| C4 | oracle tracker | perception upper bound |
| C5 | matched robot-only training | condition 是否带来净增益 |

## 7.2 Effect identification

| ID | 输入 | 数据 |
| --- | --- | --- |
| E0 | action | counterfactual |
| E1 | state+action | counterfactual |
| E2 | hidden+state+action | counterfactual |
| E3 | E2 | behavior/expert-only |
| E4 | E2，但 action shuffled | counterfactual |
| E5 | E2 | unseen policy candidates |

## 7.3 Candidate control

| ID | K | 选择 |
| --- | ---: | --- |
| P0 | 1 | original |
| P1 | 4 | candidate 0 |
| P2 | 4 | random |
| P3 | 4 | predicted COE |
| P4 | 4 | oracle truth，仅上界 |

## 7.4 最低报告要求

- 至少 3 个 policy/model noise seeds，或明确的 paired candidate protocol；
- simulator seeds 固定；
- success 置信区间；
- paired difference；
- goal progress；
- steps-to-success；
- tracker-valid 分层；
- proposal-source 分层；
- candidate diversity；
- effect physical-unit error；
- latency、显存和吞吐；
-失败 case 视频与 object/effect overlay。

---

## 8. 当前最重要的隐藏问题

## 8.1 科学问题

### 强 baseline ceiling

40k 已有 Push 98%、Pick 96%。新方法即使正确，在 ID 上也难显示收益。必须提前准备
hard/OOD；不能看到 1–2 个 episode 差异就得结论。

### Candidate diversity 可能不足

Action DiT 的 K 个采样可能非常相似。如果真实 outcome 几乎相同，effect selector
没有价值。Phase 5 oracle upper bound 必须先于在线集成。

### Effect predictor 可能只学 action kinematics

这是删除 `Δee_to_object` primary target 的主要原因。必须用 object displacement、
contact/grasp和 no-op/action-shuffle测试证明在学 object response。

### Counterfactual 仍可能是 simulator shortcut

如果 branch proposal 类型与 outcome 高度绑定，例如 no-op 永远零 motion、人工大
扰动永远失败，模型只需识别 proposal pattern。必须在 policy candidate 上评价，
并逐步增加分布重叠。

### “Causal”主张仍需克制

same-state intervention识别的是固定 simulator、局部 action 分布下的 causal effect。
它不自动证明真实世界、跨 embodiment 或所有 confounder 下的因果性。论文优先使用
“counterfactual object-effect”而不是宽泛的“causal world model”。

## 8.2 数据问题

### Restore 不一定恢复完整物理状态

actor/articulation pose 不代表 controller target、contact cache、solver/runtime 都
一致。这是我对整条路线最没有信心的工程点。

### Zero-effect imbalance

大量 approach frame 在 8 步内 object 不动。若不按 contact phase 平衡，模型输出零
即可得到很低 MAE。

### Early termination

失败动作可能碰撞、超界或提前 termination。必须保留 executed length 和
per-horizon valid，不能把 padding 当真实 zero effect。

### 数据泄露

所有 branches 必须按 base state split；phase/contact/GT pose只作 label。effect
target不能塞回 standard 34D state。

### Normalization

action network读取 normalized action，effect label以米报告。统计量只能从 train
branch构建，val/test不能参与 normalization。

## 8.3 模型问题

### Mean-pooled hidden 可能丢失小物体

第一版使用 mean pooling只是最低成本 gate，不代表最终最佳结构。如果 A2 不优于
A1，不能立刻得出 Video hidden 无用；还要区分“hidden无信息”和“pooling丢信息”。
只有 effect主线已成立，才值得测试 role query。

### Hidden 梯度当前断开

Cosmos hook无条件 detach。联合训练前必须修 contract，不能只改配置名。

### 多 loss 相互干扰

effect loss 可能改善 object feature，也可能破坏 future-video/action。联合阶段必须
报告每个 loss、梯度和原 baseline regression。

### Scorer exploitation

effect model在 OOD action 上可能过度乐观。第一版仅选择 Action DiT candidates，不
做 gradient optimization；保存所有 candidate prediction与真实结果做 regret分析。

## 8.4 Sim2Real 问题

### 当前 goal 来源不完整

simulator goal是已知 actor/区域，真机 goal如何观察和定义仍未完全闭合。没有真实
goal source，object-to-goal scorer无法部署。

### Wrist calibration 未闭合

front tracker有效不代表 wrist可作为metric object输入。当前 wrist只进入 Video DiT；
在 intrinsic/hand-eye完成前不能做多视角3D融合声明。

### Tracker invalid 与闭环失败有关

Pick 40k tracker-valid约91.79%。COE若依赖结构化 state，invalid时可能比原Video
hidden更脆弱。必须保留 fallback：

```text
tracker invalid
→ 不使用高置信COE选择
→ candidate 0 / baseline policy
```

---

## 9. 我最没有信心的点

按优先级：

1. **同状态 restore 是否在 contact/grasp 阶段足够确定。**
   如果不成立，counterfactual标签本身不可信。
2. **K个Action DiT候选是否有真实 outcome diversity。**
   如果oracle best都不比candidate 0好，selector没有上限。
3. **现有17D tracker condition是否在强Video DiT上提供独立信息。**
   当前高成功率不能证明模型使用了condition。
4. **两个cube任务能否支撑研究结论。**
   很可能只能完成方法gate，不能支撑代表作。
5. **mean-pooled Video hidden是否保留小object信息。**
   它是工程最小实现，不是最终结构保证。
6. **真实goal、front/wrist calibration与invalid fallback。**
   这是sim2real最危险的接口。
7. **Novelty变化速度。**
   FlowWAM、counterfactual world models和candidate ranking都在快速出现；最终贡献
   必须靠same-state paired effect protocol与严格identifiability实验，而不是名称。

---

## 10. 决策门槛与停止条件

| Gate | 通过才做什么 | 失败怎么办 |
| --- | --- | --- |
| condition sensitivity | 继续使用tracker state | 降级tracker为teacher/diagnostic |
| restore determinism | 采branch | 停止并修state contract |
| branch outcome diversity | 训练effect | 修proposal/state sampling |
| action-shuffle degradation | 接candidate ranking | 停止扩模型 |
| oracle K upper bound | 写在线selector | 增加candidate diversity或换难任务 |
| predicted ranking | 闭环COE | 修effect target/model |
| compute-matched closed-loop gain | 联合训练 | 保留离线诊断，不改主policy |
| multi-task/OOD gain | 进入sim2real | 缩小论文主张或换路线 |

以下结果出现时，不应继续堆结构：

1. normal/shuffled condition无差异；
2. restore noise接近branch effect；
3. action shuffle不影响effect；
4. counterfactual不优于expert-only；
5. oracle candidate没有上界；
6. COE不优于compute-matched random；
7. 只在oracle perception下有效；
8. 增益只在Push/Pick少量seed出现；
9. 实现逐渐变成FlowWAM式RGB/optical-flow双流动作解码。

---

## 11. Git和实验记忆策略

每个阶段使用 additive commit，不覆盖 baseline：

```text
checkpoint baseline
→ restore smoke
→ counterfactual schema + audit
→ pilot data evidence
→ effect head
→ offline ranking
→ online selector
→ joint COE
```

每次模型/训练/数据核心变更：

1. 先读 `docs/思考与隐患.md`；
2. 修改前取得用户确认；
3. 更新 `docs/MODIFICATION_LOG.md`；
4. 实验后更新 `docs/EXPERIMENT_RESULTS_AND_ANALYSIS.md`；
5. 记录两个仓库 commit；
6. 运行 focused test、`git diff --check` 和对应 smoke；
7. 推送 Jamee11 远端；
8. 不提交 checkpoint、dataset、runtime eval videos；
9. 重要结果在文档中保存 summary 和产物路径。

建议 commit 粒度：

```text
ManiSkill: Add COE counterfactual restore smoke
ManiSkill: Add COE branch schema and audit
DiT4DiT : Add offline COE effect predictor
DiT4DiT : Add COE candidate selector
DiT4DiT : Integrate COE joint training
```

---

## 12. 后续具体工作清单

### 当前立即完成

- [x] 审计现有数据、tracker、40k baseline 和 dynamics代码；
- [x] 修正 primary effect target；
- [x] 明确 FlowWAM 边界；
- [x] 写出 phased plan、gate、隐患和停止条件；
- [ ] 提交并推送当前 ManiSkill 文档/collector checkpoint；
- [ ] 审计并提交 DiT4DiT 当前 baseline 相关源码，排除 runtime artifacts。

### 下一次代码工作

必须先向用户说明并确认以下 Phase 1/2 精确变更：

1. evaluator增加 zero/lagged/shuffled/oracle condition ablation；
2. ManiSkill新增独立 state-restore smoke；
3. 不修改 production v3 collector；
4. 不开始大规模branch采集；
5. 先输出condition sensitivity与restore QA。

### 用户后续需要安排的训练

按先后顺序：

1. 如condition sensitivity有信号，安排matched robot-only 40k；
2. counterfactual pilot通过后，训练小effect head，单卡即可；
3. oracle K upper bound成立后，再运行K=4闭环；
4. 只有K=4有收益，才安排四卡联合COE全参训练；
5. optical-flow auxiliary和role-query都排在主gate之后。

---

## 13. 最终推荐

当前最可靠的COE-WAM不是一次性改造DiT4DiT，而是一条逐步识别失败点的路径：

```text
现有强baseline
+ 可部署object/goal state
+ same-state multi-action simulator truth
+ action-conditioned object response
+ compute-matched candidate selection
```

真正的核心贡献候选是：

> 成功示范把expert action与successful future纠缠在一起。COE-WAM通过同状态、多动作
> 仿真干预识别candidate action造成的task-object response，再用该response指导
> numerical Action DiT。

这条主张只有在以下证据全部成立后才可信：

```text
restore可信
action shuffle有效
counterfactual优于expert-only
oracle candidate存在上界
predicted selection改善compute-matched closed-loop
learned perception与OOD下仍成立
```

在此之前，COE-WAM是清晰且值得验证的研究假设，不是已经实现或证明的贡献。
