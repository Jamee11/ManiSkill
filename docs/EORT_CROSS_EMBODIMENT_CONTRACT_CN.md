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
object_bbox_xyxy              : (T,V,N,4) # [x0,y0,x1,y1), oracle tracker supervision
object_mask_centroid_uv       : (T,V,N,2) # [u,v], oracle tracker supervision
object_segdepth_centroid_world: (T,N,3)   # oracle segmentation + metric depth 的可见表面几何 proxy
object_segdepth_valid         : (T,N) bool
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

`interaction_phase`、role、valid、visible、arm-present 是离散 token/mask，必须作为 embedding 或 mask 处理；不能 q99 连续归一化。连续 pose/velocity/force/future delta 只用训练集统计量归一化。当前 PushCube v2 是该契约的一个 `A=1,N=1,G=1,H=3` 子集；已持久化 object/goal actor ID 并导出 object/goal visibility fraction。它仍未覆盖多 object、遮挡或真实感知 ID 管理。

当前 v2 也导出 `eef_transition_local(T,7)`：当前 EEF frame 的 observed `Δxyz + Δrotvec` 加 `[0,1]` gripper open fraction。它保留了过渡语义，但 manifest 明确它不是 `canonical_action_cmd`；不可跳过 controller replay 而把它用于真机命令。

`object_segdepth_centroid_world` 由 actor-ID segmentation、毫米 depth、`intrinsic_cv` 和 `extrinsic_cv` 回投得到，不读取 object pose；它是带 oracle actor mask 的几何观测诊断，而不是部署 tracker。可见表面 centroid 与物体几何中心存在系统偏差，必须将其误差作为学习/标定目标，而不能直接替代 `object_pose_task`。

### Panda controller replay 的当前证据

已对一条真实 `pd_joint_pos(T,8)` PushCubeEORT 轨迹使用 ManiSkill 官方转换器，重放为 `pd_ee_delta_pose(T,7)`，转换轨迹的最终 success=True。该 HDF5 action 是 **Panda 的归一化 root-translation/root-aligned-rotation controller command**，与 `eef_transition_local` 的 observed local motion 有意分离。它证明官方 IK 转换可在该 Panda task 上回放，但不满足本契约的 local `canonical_action_cmd`：Piper 控制器、真实夹爪标定、task-frame transform 和跨机器人 replay 尚未验证。

**因果边界：** `future_object_delta` 是时刻 `t` 之后的真值，不能作为 `t` 的可部署 action-policy 输入；它只能作为 object dynamics 的辅助预测目标。首个 GT oracle action gate 只可使用当前时刻可得的 pose、relative geometry、velocity、visibility 与当前接触状态。若实验额外喂入 future GT，必须显式标为不可部署的预测上界，不能与真实 sim2real 条件比较。

## 4. 模型路线

1. **首个因果 gate：DiT4DiT。** 保持视觉和 action diffusion 主干不变，先比较 robot-only 与当前时刻 GT continuous object condition。现有 loader 可拼接具名 state key，但 RLBench 的旧 EORT config 会 q99-normalize `object_interaction_state`，不可复用到离散 phase；phase 必须暂时不输入，或新增离散 embedding 路径。只在 object condition 带来同数据、同训练预算、留出任务的闭环收益后，再接入预测 token。DiT4DiT 是最低风险的“对象信息是否有用”检验，不是最终跨机器人 backbone。
2. **结构最匹配的中期候选：X-WAM。** 本地源码审计确认其 `RobotDataset` 固定为 16D proprio（每臂 `xyz+wxyz+gripper`）和 14D action（每臂 `Δxyz+Δaxisangle+gripper`），缺失右臂由 zero+mask 表示；因此它与 Franka/Piper 的目标契约直接对齐。它也原生读取多视图 RGB-D。官方已发布约 117 GB 的 pretrained、RoboCasa-SFT 与 RoboTwin-SFT 权重，但当前本地 `checkpoints/` 仍不存在，且还需 Wan2.2-TI2V-5B 基座；在下载、checksum、单臂 mask、RGB-D 编码和小规模 forward 验证前，它不能称为可运行 baseline。获得权重后，将 object token 作为独立条件 token 流加入 action/proprio fusion，而非把类别 phase 拼进连续 proprio，并保留不加 token 的同数据基线。
3. **最快可切换的近期备选：τ₀-WM。** 本机已有约 21 GB τ₀-WM 权重和约 27 GB Wan2.2 基座，仓库内已有 `action_in_dim=7`、`dual_arm=false` 的 RLBench 后训练/部署入口，因此如果 DiT4DiT gate 很差，它比尚未下载权重的 X-WAM 更快进入同数据适配。它的原始预训练接口仍是双臂 20D relative EEF-6D，与本契约不同；切换时必须使用单臂 7D downstream head，并重新验证 Panda controller action 的 frame、归一化、gripper 与 closed-loop replay，不能把已有 RLBench checkpoint 当作 ManiSkill checkpoint。
4. **后续辅助：DreamDojo。** 用于生成/筛选轨迹、学习对象未来状态或规划候选，不作为首个端到端 action policy；需要独立 action head 和动作执行评估才有价值。

## 5. 可执行门槛

1. 先给 ManiSkill v3 补齐 actor-ID→object/goal/robot-role 映射和 visibility QA；不得直接批量采 v2 后声称视觉监督可用。
2. 以已存的 raw 8D action 保留可回放性；已新增明确标注的 observed EEF transition，仍须完成 action replay 误差检查后才能生成 canonical command。
3. 用 episode 级 train/val/test 划分，统计量只由 train 计算；采集至少覆盖 object pose、尺寸、摩擦、光照、相机、干扰物和初始机器人姿态变化。
4. 专用导出器必须写 DiT4DiT 当前可读的 LeRobot v2 parquet/video schema。ManiSkill 内置 v3 转换器只保留 raw action/qpos，会丢弃 EORT extras，不能直接使用。
5. 在 ManiSkill 先完成四组：robot-only、GT object token、噪声/漏检 token、预测 token；逐项报告任务成功率、对象误差、接触/phase F1 与动作安全限幅触发率。
6. 完成单臂 Franka sim→real 后，再扩展双臂 Piper；双臂任务必须测主动臂选择、跨臂碰撞、同步和非活动臂 mask，不能由单臂零填充结果代替。

## 6. 当前最大不确定性

最没有把握的不是 simulator 中能否得到 GT pose/contact，而是预测的 object track 在真实相机遮挡、时延、ID switch、深度噪声和手眼标定偏差下，是否仍能改善而非破坏 action policy。这个问题只能通过“GT oracle → 注入受控误差 → 实际感知 token”的递进评测回答；在此之前，任何 object-centric sim2real 优势都只是待检验假设。
