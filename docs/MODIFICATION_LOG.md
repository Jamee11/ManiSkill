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
