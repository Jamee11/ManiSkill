# Modification Log

## 2026-07-10 11:41:19 UTC — ManiSkill PushCube EORT pilot data construction

- 原因：在进入训练或多任务扩展前，需要在 ManiSkill3 中建立可复现、可审计的 object-centric oracle 数据采集 gate，并避免复制 RLBench 中的固定接触区/四元数相减等不可靠标签。
- 计划的精确变更：创建隔离 conda 环境 `maniskill-eort-v1`；复用 ManiSkill 原生 `PushCube-v1` motion-planning 与 `RecordEpisode` HDF5；新增一个只派生 TCP-to-object、object-to-goal、距离、归一化当前进度、下一帧物体平移、原始 action 与 success 的 sidecar 导出器和 schema validator。
- 明确不做：不修改 ManiSkill 模型架构、不改变任务物理、不训练、不转换 LeRobot、不产生伪物理 contact/固定 normal 标签、不删除或覆盖已有数据。
- 风险约束：详见 `docs/思考与隐患.md`；当前输出是 simulator oracle，不能解释为真实部署感知。

## 2026-07-10 11:44:15 UTC — 环境路径调整

- 原因：`maniskill-eort` 目录由一次未完成的创建留下，但没有 `conda-meta/history`，不是可用环境。
- 精确变更：保留该目录不作删除或覆盖，改用新隔离环境 `maniskill-eort-v1`。

## 2026-07-10 11:50:41 UTC — PushCube EORT sidecar 实现

- 原因：原生 ManiSkill HDF5 已能保存 `state_dict+rgb+depth+segmentation`，因此只需增加可审计的标签 sidecar，而不应再实现一套重复的 RGB/mask 采集器。
- 精确变更：新增 `scripts/eort/derive_push_cube_eort.py`，从每条成功 trajectory 的 `T+1` observation 和 `T` action 导出 `ee_to_object`、`object_to_goal`、两种距离、当前 goal progress、下一帧物体平移、pose、action、success；新增不可覆盖的采集入口 `scripts/eort/collect_push_cube_eort.sh` 和 `tests/test_eort_push_cube.py`。
- 验证：2026-07-10 11:50:41 UTC，现有 Python 3.10/HDF5 兼容环境执行 `python -m unittest tests/test_eort_push_cube.py -v`，1/1 通过，验证 T+1/T 对齐、progress 和 next-delta。

## 2026-07-10 12:14:49 UTC — ManiSkill EORT 环境运行时修复

- 原因：新环境中同版本 SAPIEN 的 `sapien.Device('cpu')` 直接发生 exit 139；与已有可用环境比较后，发现其缺少 EGL/GL/Vulkan loader 运行库。
- 精确变更：在 `maniskill-eort-v1` 安装 `libegl`、`libgl`、`libglvnd`、`libglx`、`libvulkan-loader`；不改动项目运行逻辑。
- 证据：安装后 `sapien.Device('cpu')` 输出 `cpu`；`PushCube-v1` 以 CPU PhysX + GPU renderer 成功 reset/step，返回 `tcp_pose`、`obj_pose`、`goal_pos` 和 128×128 RGB/depth/segmentation。禁用 CUDA 后的 CPU renderer 仍在 `_setup_scene` exit 139，因此不得作为采集配置。

## 2026-07-10 12:18:20 UTC — 显式 GPU renderer 选择

- 原因：在默认 CUDA device（GPU 0）采集 10 条时，SAPIEN Vulkan buffer 报 out-of-memory 并 exit 139；不应自动扫描并抢占其他训练 GPU。
- 精确变更：`collect_push_cube_eort.sh` 仅在调用方提供 `MANISKILL_EORT_CUDA_VISIBLE_DEVICES` 时导出为 `CUDA_VISIBLE_DEVICES`，让用户/调度器显式选择经 smoke 验证的 renderer GPU。
- 保留行为：未指定时完全继承现有 CUDA 可见性；不删除失败目录或已有数据。

## 2026-07-11 03:45:45 UTC — PushCube object-centric oracle v2

- 原因：v1 只能记录当前位置关系和一步平移，无法为 object pose、真实接触交互阶段及多尺度未来轨迹提供完整、可审计的监督。
- 精确变更：新增不改变物理/奖励/成功条件的 `PushCubeEORT-v1` 环境，仅额外记录物体线速度、角速度和 Panda hand/finger 对物体的真实接触力；保留 `PushCube-v1` 与 v1 导出不变。`derive_push_cube_eort.py --schema objectcentric_v2` 在真实接触力基础上导出 `physical_contact`、task-specific `push_interaction_phase`、正确的 EE→object rotation-vector、`[1,4,8]` action-step 未来平移/旋转及有效性 mask。新增不可覆盖的 v2 采集脚本。
- 明确不做：不写距离阈值 contact、固定接触点/法线或伪接触面积；不训练、不修改动作模型、不批量采集。
- 验证：离线单元测试覆盖 `wxyz` 相对旋转、真实 force→contact 阶段和未来 horizon 对齐；CPU PhysX + GPU renderer reset/step smoke 输出三个新增 `(1,3)` 有限字段。

## 2026-07-11 09:11:56 UTC — Contact aggregation correction

- 原因：多个真实接触 link 的三维力向量直接求和可能相互抵消，使接触判定出现假阴性。
- 精确变更：`PushCubeEORT-v1` 限定 Panda，并额外记录 `robot_obj_contact_force_norm`：Panda hand、左右 finger 三个 link 的 force norm 之和；v2 的 `physical_contact` 仅使用该标量阈值，仍保留 net force vector 供诊断。
- 验证：更新离线 HDF5 fixture 和 manifest 断言；真实 HDF5 持久化验证仍待空闲 GPU 机器执行。

## 2026-07-13 06:50:18 UTC — v2 真实轨迹采集验证

- 原因：必须在不覆盖既有数据的前提下，确认 v2 的新增真实物理字段能由 ManiSkill motion-planning HDF5 持久化，并与 action-aligned sidecar 严格对应。
- 精确变更：未修改代码或配置；使用调用方显式指定的 GPU 4，写入新的单轨迹数据目录并运行已有 v2 采集、派生和逐元素对齐检查。
- 证据：1/1 成功轨迹有 `T=71` action 与 `T+1=72` 原始观测；速度、net force 和 force-norm 全部有限，derived 前 `T` 帧与 raw 完全一致，接触标签逐帧匹配 force-norm 阈值。
- 明确不做：不把单条 oracle 轨迹用于训练结论，不开始批量采集，不将 simulator GT 视为部署时感知。

## 2026-07-13 08:00:15 UTC — 跨 Franka/Piper EORT 契约

- 原因：当前 raw `(T,8)` Panda joint action 与跨机器人末端控制不等价；若不先明确 frame、mask、object slot 和 command/transition 的区别，后续会把 simulator 的状态变化误当作真实机器人命令。
- 精确变更：新增 `docs/EORT_CROSS_EMBODIMENT_CONTRACT_CN.md`，规定以具名 object token、task-frame 标定和最多双臂的局部 EEF 7D canonical command 为目标契约，并记录 DiT4DiT、X-WAM、τ₀-WM、DreamDojo 的分工和验收门槛。
- 明确不做：不修改现有 v2 schema、采集逻辑、模型结构或训练配置；该文档不是已实现功能，也不使当前 raw joint action 自动成为 canonical command。

## 2026-07-13 10:00:29 UTC — v2 segmentation ID 与可见率 QA

- 原因：raw segmentation 虽保存 actor label，但原轨迹不含 object/goal 的 ID 映射，导致无法审计目标对象是否被相机看到。
- 精确变更：`PushCubeEORT-v1` 在每帧额外写入 object/goal 的 static `per_scene_id`；v2 派生器验证 ID 为正整数且轨迹内恒定，并输出 object/goal 的 mask pixel count、visibility fraction、visible bool 及 manifest 语义。
- 保留行为：不改变物理、奖励、成功条件、控制模式、原始 `(T,8)` action 或既有 v1 schema；完全遮挡是有效数据，标为 `visible=false` 而非报错。
- 兼容性：旧 v2 raw 没有两个 ID 时仍导出既有字段，并在 summary 标为 `segmentation_visibility=false`；只出现一个 ID 则拒绝导出，避免不完整角色映射。
- 验证：单元测试覆盖可见、完全遮挡的 synthetic mask；GPU 4 上 1/1 成功 HDF5 保存 `(72,1)` int32 ID，derived 前 `T=71` 帧 pixel count、fraction、visible 与 raw segmentation 逐帧一致。

## 2026-07-13 12:16:30 UTC — Oracle condition 因果边界

- 原因：object future trajectory 是行动之后的真值；若直接输入 action policy，会产生未来泄露。另经 source audit 确认 DiT4DiT 现有 RLBench EORT config 将 interaction state 放入 q99 连续归一化路径。
- 精确变更：更新跨机器人契约和风险台账，规定首个 action gate 仅使用当前可得 continuous object state；future trajectory 仅作辅助预测目标，离散 phase/visibility/role 需独立 embedding/mask 或暂不作为输入。
- 明确不做：不修改 DiT4DiT、ManiSkill 训练/采集代码，也不宣称现有 RLBench proxy condition 可直接迁移。

## 2026-07-13 12:22:41 UTC — v2 observed EEF transition

- 原因：当前 raw `(T,8)` Panda joint target 无法作为跨 Franka/Piper 的统一 action，但 object-centric dynamics/action gate 仍需一个与 TCP 几何直接对齐的可审计目标。
- 精确变更：v2 导出器新增 `eef_transition_local(T,7)`，从 `tcp_pose[t:t+1]` 用当前 EEF frame 的逆旋转计算 local `Δxyz + Δrotvec`，并将 raw gripper `[-1,1]` 转为 `[0,1]` open fraction。manifest 明确 `controller_command=false`。
- 保留行为：不改变 raw `(T,8)` action 或把 transition 重命名为 command；v1 不受影响。
- 验证：单元测试覆盖非平凡局部 frame 旋转和夹爪归一化；GPU 4 的完整 1/1 采集/派生结果有有限 `(71,7)` transition，且末维逐元素匹配 raw gripper 映射。

## 2026-07-13 12:25:08 UTC — DiT4DiT 训练出口约束

- 原因：source audit 发现 ManiSkill 内置 LeRobot converter 只读取 raw action、qpos 与 RGB，输出 v3 schema；它会丢弃 EORT extras，且当前 DiT4DiT loader 的配置使用 v2。
- 精确变更：更新跨机器人契约和风险台账，要求后续复用 DiT4DiT EORT parquet/video 模式实现专用 v2 exporter，并把版本/字段完整性设为训练前 gate。
- 明确不做：不安装依赖、不修改 converter 或启动训练；本次只记录接口不兼容性。

## 2026-07-13 12:37:00 UTC — 真实 v2 到 DiT4DiT LeRobot v2 出口验证

- 原因：专用 exporter 已在 DiT4DiT 实现，仍必须用新采集的真实 `PushCubeEORT-v1` HDF5 验证 raw、derived 与 LeRobot v2 三层对齐，而不是只依赖 synthetic fixture。
- 精确变更：未修改 ManiSkill 代码或物理配置；显式选择 GPU 4，以独立时间戳目录采集 `NUM_TRAJ=1`，并将同一 raw HDF5 与 derived NPZ 导出为 DiT4DiT LeRobot v2 Parquet/MP4/meta。
- 证据：1/1 成功、`T=71`；raw RGB/segmentation 为 `(72,128,128,3)/(72,128,128,1)`，derived `eef_transition_local` 为 `(71,7)`，LeRobot Parquet state/current-object-condition/action 分别为 `(71,8)/(71,17)/(71,7)`。导出 action 与 derived transition、17D condition 与 derived 当前连续字段均逐元素一致；H.264 MP4 可解码为 71 帧。
- 因果边界：LeRobot policy fields 不含 future delta/mask、phase、segmentation ID 或 bool contact；夹爪 state 不从 action target 回填，避免监督泄漏。该数据仍是 simulator oracle，不能用于真实部署结论。

## 2026-07-13 12:50:00 UTC — baseline readiness 审计

- 原因：需要按本地可运行的接口而非论文印象决定 DiT4DiT、τ₀-WM、DreamDojo 与 X-WAM 的分工。
- 精确变更：未改模型或数据；审计 X-WAM 数据加载器、模型配置和本地权重目录，并把 16D/14D 双臂 mask 接口、无本地 checkpoint 的事实和后续门槛写入跨机器人契约与风险台账。
- 决策：保持 DiT4DiT 为首个 ManiSkill oracle 因果 gate；X-WAM 为在 checkpoint 验证后的主线候选；τ₀-WM 仅作预训练迁移消融，DreamDojo 仅作未来状态/轨迹辅助。

## 2026-07-13 13:05:00 UTC — v2 segmentation-depth object localization proxy

- 原因：真实感知路线需要先验证现有 RGB-D、相机标定与 object mask 能否提供可审计的几何观测，而不是直接把 simulator object pose 当作视觉定位结果。
- 精确变更：v2 派生器新增 `object_segdepth_centroid_world(T,3)`、`object_segdepth_valid(T,1)` 和 `object_segdepth_centroid_error(T,1)`。它以 object actor-ID 的 segmentation 像素、毫米 depth、CV intrinsics/extrinsics 回投可见点云中心；object pose 只用于误差计算。没有把这些字段接入 action policy。
- 验证：合成 2×2 calibrated fixture 覆盖可见/不可见和精确坐标；完整 EORT 单测 4/4 通过。真实 `T=71` sidecar 的 71 帧均有效，error 均值 1.97 cm。
- 明确边界：actor ID 仍是 simulator oracle；surface centroid 不等于 object center。该字段只作为 learned track 前的诊断/监督目标，不能宣称真实部署感知已完成。

## 2026-07-13 13:10:00 UTC — Panda joint-to-EEF command replay audit

- 原因：必须证明至少一种 EEF controller command 能从现有 Panda joint action 转换并闭环重放，不能把 observed TCP transition 误称作 command。
- 精确变更：未改代码；调用 ManiSkill 官方 `replay_trajectory` 将现有 raw `pd_joint_pos` trajectory 转换保存为独立的 `pd_ee_delta_pose` HDF5。
- 验证：converted action `(71,7)`、observation 72 帧、最终 success=True，metadata target control mode 为 `pd_ee_delta_pose`。
- 边界：结果是归一化 root-frame Panda command；不替代 local canonical action、Piper adapter 或真机 replay 验证。

## 2026-07-13 13:18:15 UTC — v2 十轨迹 pilot 与训练格式完整性 QA

- 原因：单条成功轨迹只能证明 schema 对齐，不能证明采集、可见率统计和专用 LeRobot 出口能跨多条 episode 保持完整。
- 精确变更：未修改 ManiSkill 物理、环境、控制或模型代码；在用户指定可用空间的 GPU 4 上，以新的不可覆盖目录采集 10 条原生 `PushCubeEORT-v1` motion-planning 轨迹，派生 v2 sidecar，并导出既有 DiT4DiT LeRobot v2 格式。
- 证据：10/10 最终 success，合计 686 action-aligned frames；10 个 Parquet、10 个 H.264 128×128/20 FPS MP4 与 `meta/info.json` 的 10 episodes/686 frames 一致。逐轨迹长度为 `[71,72,64,74,66,77,62,69,63,68]`。
- 质量结论：object/goal visibility 和 segmentation-depth valid 均为 1.0；object mask fraction 为 0.00098–0.00189，goal 为 0.02185–0.04120；segmentation-depth surface-centroid error 的均值为 1.80 cm、p95 为 2.09 cm。真实接触帧 236；phase 计数为 approach=450、contact=7、moving-in-contact=229、goal=0。
- 边界：全量可见和没有 goal phase 说明这只是“链路通畅”pilot，不是遮挡/终态分布覆盖，更不足以启动训练或形成 sim2real 结论。数据路径记录在实验分析中，二进制数据不提交 Git。
