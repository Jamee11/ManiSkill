# Experiment Results and Analysis

## ManiSkill PushCube EORT pilot

状态：环境、schema smoke、v2 端到端轨迹与 object/goal 可见率 QA 已通过；尚未进行批量采集或训练。

计划：在经调度确认的 GPU 上采集 10 条成功的 `PushCubeEORT-v1` motion-planning 轨迹，并检查 EORT sidecar 的可见率分布。

预注册检查：环境版本、episode 成功数、HDF5 schema、`T+1` observation 与 `T` action 对齐、所有导出数值有限、RGB/depth/segmentation 流存在。

下一步：在经调度确认的 GPU 上扩大数据量；若失败，记录失败命令、根因与修复方案，且不进行训练。

### 2026-07-10 11:50:41 UTC — 导出器离线单元检查

- 设置：构造 1 条成功的 3-step PushCube 格式 HDF5 fixture，包含 `T+1=4` 的 `tcp_pose`、`obj_pose`、`goal_pos` 及 RGB/depth/segmentation 流。
- 命令：`/remote-home/jinminghao/miniconda3/envs/robotwin/bin/python -m unittest tests/test_eort_push_cube.py -v`。
- 结果：1/1 通过（0.202s）；确认导出器写入 `T=3` 标签，`goal_progress=[0,0.5,1]`，且 `object_next_delta` 对齐下一 observation。
- 分析：仅证明纯 HDF schema 与数值派生正确，尚不证明 ManiSkill 渲染、motion planner 或新环境可用；仍需目标环境 smoke 和 10 条成功轨迹 pilot。

### 2026-07-10 12:14:49 UTC — ManiSkill 环境与观测 smoke

- 环境：`maniskill-eort-v1`，Python 3.10.20、ManiSkill 3.0.1（当前工作树 editable）、Torch 2.7.1+cu128、SAPIEN 3.0.3、Gymnasium 1.2.3；补齐 conda-forge EGL/GL/Vulkan loader。
- 设置：`PushCube-v1`，`sim_backend='cpu'`，`obs_mode='state_dict+rgb+depth+segmentation'`，GPU renderer。
- 结果：成功创建、reset 与 step；`extra={tcp_pose,obj_pose,goal_pos}`，`base_camera` 输出 RGB `[1,128,128,3]`、depth `[1,128,128,1]`、segmentation `[1,128,128,1]`，`success` `[1]`。
- 反例：禁用 CUDA 触发 SAPIEN CPU-render fallback，在 `_setup_scene` exit 139；因此本机采集必须使用 GPU renderer。CPU PhysX 的选择来自单环境原生 motion-planning 路径，不是 GPU renderer 不可用。
- 下一步：采集 1 条端到端 motion-planning trajectory，验证真实 HDF 字段；成功后再采 10 条 pilot。

### 2026-07-10 12:18:20 UTC — 默认 GPU 0 的 10 条采集尝试

- 设置：`NUM_TRAJ=10 bash scripts/eort/collect_push_cube_eort.sh`，CPU PhysX、GPU renderer，未显式限制 CUDA device。
- 结果：失败；SAPIEN 报 `buffer.cpp:251: out of memory`，随后 exit 139。失败前未写入可保留的 raw 或 derived 数据目录。
- 分析：这不是 CPU 物理或 EORT label 问题，而是默认 renderer GPU 0 的显存不足。CPU renderer 在本机不可用，不能作为替代；下一步仅在空余较多的、显式选择的 GPU 上先复跑 1 条 smoke。

### 2026-07-10 12:19:37 UTC — 批量采集决策

- 决策：用户要求本机不再因 GPU 满载而重试采集；不选择或抢占其他正在训练的 GPU。
- 保留证据：1 条端到端 smoke 已成功，验证原生 HDF5、sidecar 导出、manifest 和数值完整性。
- 下一步：在另一台有可用 GPU 的机器上，先用 `MANISKILL_EORT_CUDA_VISIBLE_DEVICES=<gpu> NUM_TRAJ=1` 验证，再执行 `NUM_TRAJ=10`；准确命令见 `docs/MANISKILL_EORT_COLLECTION_CN.md`。

### 2026-07-11 03:45:45 UTC — object-centric v2 环境与标签 smoke

- 设置：新增 `PushCubeEORT-v1`，保持原 PushCube 物理、奖励和成功条件；额外观测为真实 `robot_obj_contact_force`、物体线速度、角速度。导出器使用 action-aligned horizons `[1,4,8]`，并以 `q_target ⊗ inverse(q_reference)` 计算旋转向量。
- 离线结果：3/3 单元测试通过。v2 fixture 证明 force 为零/非零时 `physical_contact` 分别为 false/true，接触且物体在动时 phase=push，未来 mask 不把末帧补零误当真值，π/2 旋转关系正确。
- 运行时结果：CPU PhysX + GPU renderer reset/step smoke 成功，三个新增字段均为有限 `(1,3)`。该 smoke 没有写 trajectory，也没有进行批量采集。
- 分析与下一步：真实物理 force 已进入观测，但尚未验证它在 motion-planning HDF5 中和 `T` action 持久化对齐。另一台空闲 GPU 机器必须先执行 v2 的 `NUM_TRAJ=1`，检查 raw HDF5 的 `T+1` force/velocity 和 derived `T` 标签，再扩大数据量。

### 2026-07-13 06:50:18 UTC — PushCube object-centric v2 真实 HDF5 验证

- 设置：用户授权后在 GPU 4 运行 `MANISKILL_EORT_CUDA_VISIBLE_DEVICES=4 NUM_TRAJ=1 bash scripts/eort/collect_push_cube_objectcentric_v2.sh`；输出为新目录 `/remote-home/jinminghao/datasets/maniskill_push_cube_objectcentric_v2_v2_gpu4_20260713T065018Z`，未覆盖已有数据。
- 采集结果：motion planner 1/1 成功，成功率 1，失败规划率 0，轨迹长度 `T=71`；派生器写入 `maniskill_push_cube_objectcentric_oracle_v2` manifest，horizons 为 `[1,4,8]`。
- 对齐检查：raw `obj_linear_velocity`、`obj_angular_velocity`、`robot_obj_contact_force`、`robot_obj_contact_force_norm` 分别为有限的 `(72,3)/(72,3)/(72,3)/(72,1)`；derived 对应字段为前 71 帧。逐元素断言 raw `[:T]` 等于 derived，`physical_contact` 逐帧等于 raw force-norm `>1e-6`。
- 标签结果：23/71 帧为测得接触；phase 计数为 approach=48、measured-contact-while-object-moves=23。未来位移、旋转和有效性 mask 均为 `(71,3,3)` 且有限（mask 为 bool）。
- 分析与下一步：这解除 v2 在真实成功 HDF5 上的物理接触和时间对齐疑虑，但只覆盖单条 PushCube oracle 轨迹。批量采集前仍需增加 RGB/depth/segmentation 的对象可见率与 mask 覆盖率 QA；此数据也不能替代真实感知输入或证明 sim2real 效果。

### 2026-07-13 07:57:44 UTC — v2 segmentation 可见性审计

- 设置：只读检查上述 v2 成功 HDF5 的 `base_camera` RGB、depth 与 segmentation dataset，以及 episode metadata。
- 结果：三种视觉流均为 `T+1=72` 帧，shape 分别为 `(72,128,128,3)/(72,128,128,1)/(72,128,128,1)`；segmentation 为 `int32`，包含 15 个 actor label。
- 限制：episode metadata 没有记录 `obj` 或 `goal_region` 对应的 segmentation actor ID。仅从像素 label 不能可靠判断哪一个是目标方块，故无法计算 object-visible rate 或目标 mask 面积。
- 下一步：批量采集前，新增可审计的 actor-ID→语义角色映射（至少 object、goal、robot links）并基于该映射写入每轨迹可见率/遮挡 QA；在此之前不得把“保存了 segmentation”解释为“对象视觉质量已通过”。

### 2026-07-13 08:11:30 UTC — segmentation actor-ID 映射可行性 smoke

- 设置：GPU 4 上只读创建 `PushCubeEORT-v1`，以 `state_dict+segmentation` reset，比较 runtime actor `per_scene_id` 和 `base_camera` segmentation 像素。
- 结果：`obj.per_scene_id=18`，对应 24 个 object mask 像素；`goal_region.per_scene_id=19`，对应 524 个 goal mask 像素。两个 ID 都存在于同一帧 segmentation。
- 分析：ManiSkill runtime API 能正确提供角色→actor-ID；当前缺口仅为采集器未把 ID 保存到 HDF5。补写两个静态 ID 字段即可让导出器计算目标可见率，无需猜测像素 label 或改变物理。

### 2026-07-13 10:00:29 UTC — v2 segmentation 持久化与可见率 QA 验证

- 设置：在 GPU 4 上以新输出目录运行 `NUM_TRAJ=1` 的 v2 motion-planning 采集与派生；对 raw HDF5 与 NPZ 执行逐元素断言。
- 采集结果：1/1 成功、`T=71`，raw 新增 `obj_segmentation_id` 和 `goal_segmentation_id` 均为静态 `(72,1)` int32。
- 对齐结果：object ID=18、goal ID=19；derived 的 mask pixel count、visibility fraction、visible bool 分别逐帧匹配 raw segmentation 前 `T` 帧。object 在 71/71 帧可见，面积为 22–30 px；goal 在 71/71 帧可见，面积为 510–524 px。
- 分析：v2 已具备 object/goal 角色到 segmentation mask 的可审计链路。该结果只证明一条无完全遮挡轨迹的 schema 与对齐正确；批量阶段仍须统计可见率分布，并通过遮挡、漏检和 ID-switch 注入测试验证策略鲁棒性。

### 2026-07-13 12:15:20 UTC — 跨机器人 EEF transition 可行性审计

- 设置：只读计算同一条成功 HDF5 的相邻 TCP pose 相对变化，平移在当前 EEF 局部坐标系表达；不写入数据集。
- 结果：以验证过的逆旋转公式重算，得到有限的 `(71,3)` local translation transition，逐维最大绝对值为 `[0.008741, 0.000735, 0.011376]` m。raw 仍为 `(71,8)` Panda joint target；夹爪维恒为 `-1`，对应统一 open fraction `0`。
- 分析：可以从 `T+1` TCP pose 稳定导出 observation transition，但它是实际状态转移，不是 planner 的原始控制命令。任何将它用于 DiT4DiT/X-WAM action supervision 的实验必须显式命名为 transition-target，并另做 controller replay 评估。

### 2026-07-13 12:22:41 UTC — v2 EEF transition 导出验证

- 设置：新增 `eef_transition_local` 后，在 GPU 4 以独立输出目录运行 `NUM_TRAJ=1` 完整采集、派生与 manifest 校验。
- 结果：1/1 成功、`T=71`；NPZ 的 `eef_transition_local` 为有限 `(71,7)`，前三维为当前 EEF frame 的平移，后三个姿态维为 local rotation-vector，末维逐元素等于 `(raw_action[:,7]+1)/2`。六个运动维最大绝对值为 `[0.008741,0.000735,0.011376,0.000220,0.002282,0.000484]`。
- 语义：manifest 明确 `controller_command=false`。这条标签可用于 observed-transition/dynamics 或受限的 action-target 研究；它不是 Panda joint command 的跨机器人替代品，未通过 controller replay 前不得用于真机执行结论。

### 2026-07-13 12:37:00 UTC — 真实 v2 → DiT4DiT LeRobot v2 出口验证

- 设置：在 GPU 4 显存约余 18 GB 时，以新的独立目录采集 `NUM_TRAJ=1`；CPU PhysX + GPU renderer。将 raw `push_cube_objectcentric_v2.h5` 与 matching derived v2 manifest 用 DiT4DiT 的 `convert_maniskill_eort_to_lerobot.py` 导出。
- 采集结果：1/1 motion-planning 成功，`T=71`；raw RGB/segmentation 为 `T+1=72` 帧、`(72,128,128,3)/(72,128,128,1)`。object/goal static IDs 为 `18/19`，object mask 面积 22–30 px、goal 510–524 px，二者均 71/71 帧可见。
- 导出结果：LeRobot v2 写入一个 `(71,8)` state Parquet、一个 `(71,17)` current continuous object-condition Parquet 字段、一个 `(71,7)` observed-local-EEF-transition action 字段和一条 128×128、71 帧 H.264 MP4。
- 逐元素验证：Parquet action 等于 NPZ `eef_transition_local`；17D condition 等于 `ee_to_object(3)+ee_to_object_rotvec(3)+object_to_goal(3)+object_linear_velocity(3)+object_angular_velocity(3)+robot_obj_contact_force_norm(1)+object_visibility_fraction(1)`；raw segmentation 重算的两个 pixel count 等于 NPZ QA 字段。`meta/modality.json` 不含 future 或 phase 字段，state gripper 保持零占位而非 action 回填。
- 分析：ManiSkill 采集、object-centric NPZ 和 DiT4DiT v2 loader-format 之间的实际数据契约已贯通。此处的 object condition 是 simulator GT oracle，而非从 RGB/depth 估计的 object track；只覆盖一个无遮挡 PushCube episode，不能作为训练规模、遮挡鲁棒性或 sim2real 成功的证据。
