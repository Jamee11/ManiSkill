# EORT 跨机器人数据与动作契约

状态：设计契约；尚未改变采集器、导出器或训练代码。  
记录时间：2026-07-13 08:00:15 UTC。

## 1. 已验证的事实与边界

当前 `PushCubeEORT-v1` 真实 HDF5 使用 Panda `pd_joint_pos`，原始 action 为 `(T,8)`：七个未归一化关节绝对目标（rad）和一个归一化夹爪值。它只能完整回放 Panda 当前控制器，不能直接作为 Franka/Piper 共享动作空间。

已验证的 v2 object oracle 字段在原始 HDF5 中是 `T+1`，sidecar 中是 action-aligned `T`；物理接触来自真实 link force-norm，不能由距离替代。当前数据只有单臂单对象 PushCube，不能据此声明多物体、双臂或 sim2real 泛化。

以下三种量必须分开保存和命名：

1. `raw_action`：仿真器/硬件控制器的原始命令；本次为 `(T,8)` Panda joint target。
2. `eef_transition`：从 `EEF[t]` 到 `EEF[t+1]` 实际观测到的运动；它适合状态转移/世界模型监督，但不自动等于硬件命令。
3. `canonical_action_cmd`：部署策略输出的跨机器人末端控制命令；只有采集时确知 command-to-EEF 语义并通过 replay 后才能写作训练目标。

## 2. 推荐的坐标与动作契约

为避免把机器人基座、相机和任务物体坐标混用，每个 episode 固定记录标定：

```text
T_task_from_arm_base[A] : (A,4,4)
T_task_from_camera[V]   : (V,4,4)
```

`task` 是由工作台/任务基准定义的稳定坐标系；所有 object、goal、camera 与 EEF pose 都可变换到它。部署时这些矩阵来自机器人和相机标定，不能由训练集统计量隐式替代。

训练和部署的目标动作是每臂局部末端增量：

```text
canonical_action_cmd : (T,H,A,7)
[Δx, Δy, Δz, Δrotvec_x, Δrotvec_y, Δrotvec_z, gripper_open_fraction]
```

- `Δxyz` 与 `Δrotvec` 由当前 EEF 局部坐标系表达，定义为 `inverse(T_task_eef[t]) @ T_task_eef[t+1]`；因此对全局工作台平移/旋转不敏感。
- `gripper_open_fraction` 统一为 `[0,1]` 的目标开度，不使用 Panda 的 `[-1,1]` 或 Piper 的原生脉冲/位置单位。控制适配器负责范围、速度、限位与夹爪方向转换。
- 固定最多两臂：`A=2`，顺序为 `left,right`；单臂 Franka 用 `arm_present=[1,0]` 和全零的右臂 action/mask 表示。损失与执行器必须 mask 缺失臂。
- 同时保存 `eef_transition_local : (T,H,A,7)`，但不得把它改名为 command；只有 controller replay 证明两者等价时，才能作为 `canonical_action_cmd` 的来源。

这与 X-WAM 的每臂 7D `Δxyz + Δaxisangle + gripper` 接口直接同构。τ₀-WM 的预训练 action 是双臂 20D relative EEF-6D，需独立转换/适配，不能假称同构。

## 3. object-centric token 契约

不在数据层过早展平为一个 task-specific 向量。每一时刻保存具名、可 mask 的对象槽位，训练入口按模型需要展平或投影。

```text
object_pose_task              : (T,N,7)    # xyz + wxyz
object_velocity_task          : (T,N,6)    # linear xyz + angular xyz
object_extent_task            : (N,3)
object_role                   : (N,)       # categorical: manipulated / target / obstacle / tool
object_valid                  : (T,N) bool
object_visible                : (T,V,N) bool
object_visibility_fraction    : (T,V,N,1)
ee_pose_task                  : (T,A,7)
ee_to_object                  : (T,A,N,6)  # translation + rotvec
object_to_goal                : (T,N,G,6)  # translation + rotvec
contact_force_norm            : (T,A,N,1)
physical_contact              : (T,A,N) bool
interaction_phase             : (T,N) int8
future_object_delta           : (T,H,N,6)
future_object_valid           : (T,H,N) bool
arm_present                   : (A,) bool
```

`interaction_phase`、role、valid、visible、arm-present 是离散 token/mask，必须作为 embedding 或 mask 处理；不能 q99 连续归一化。连续 pose/velocity/force/future delta 只用训练集统计量归一化。当前 PushCube v2 是该契约的一个 `A=1,N=1,G=1,H=3` 子集；它尚缺 object ID 到 segmentation actor-ID 的映射，故 `object_visible` 与 visibility fraction 目前未生成。

## 4. 模型路线

1. **首个因果 gate：DiT4DiT。** 保持视觉和 action diffusion 主干不变，先比较 robot-only 与 GT object-token 条件。只在 object token 带来同数据、同训练预算、留出任务的闭环收益后，再接入预测 token。DiT4DiT 是最低风险的“对象信息是否有用”检验，不是最终跨机器人 backbone。
2. **最终主线候选：X-WAM。** 它原生接受 RGB-D、多视图、8/16D EEF proprio 与 7/14D 单/双臂 action；推荐在上述 gate 通过后，将 object token 作为独立条件 token 流加入其 action/proprio fusion，而非把类别 phase 拼进连续 proprio。开始前必须取得、校验可用 checkpoint，并保留不加 token 的同数据基线。
3. **迁移消融：τ₀-WM。** 仅比较其预训练视频/action trunk 在统一动作适配器上的收益。20D 双臂相对 EEF-6D 与本契约不同，action input/output 层可能重初始化；必须报告 mismatch、冻结/微调范围与单臂占位策略。
4. **后续辅助：DreamDojo。** 用于生成/筛选轨迹、学习对象未来状态或规划候选，不作为首个端到端 action policy；需要独立 action head 和动作执行评估才有价值。

## 5. 可执行门槛

1. 先给 ManiSkill v3 补齐 actor-ID→object/goal/robot-role 映射和 visibility QA；不得直接批量采 v2 后声称视觉监督可用。
2. 以已存的 raw 8D action 保留可回放性；新增明确来源的 EEF command/transition 数据，完成 action replay 误差检查后再生成 canonical action。
3. 用 episode 级 train/val/test 划分，统计量只由 train 计算；采集至少覆盖 object pose、尺寸、摩擦、光照、相机、干扰物和初始机器人姿态变化。
4. 在 ManiSkill 先完成四组：robot-only、GT object token、噪声/漏检 token、预测 token；逐项报告任务成功率、对象误差、接触/phase F1 与动作安全限幅触发率。
5. 完成单臂 Franka sim→real 后，再扩展双臂 Piper；双臂任务必须测主动臂选择、跨臂碰撞、同步和非活动臂 mask，不能由单臂零填充结果代替。

## 6. 当前最大不确定性

最没有把握的不是 simulator 中能否得到 GT pose/contact，而是预测的 object track 在真实相机遮挡、时延、ID switch、深度噪声和手眼标定偏差下，是否仍能改善而非破坏 action policy。这个问题只能通过“GT oracle → 注入受控误差 → 实际感知 token”的递进评测回答；在此之前，任何 object-centric sim2real 优势都只是待检验假设。
