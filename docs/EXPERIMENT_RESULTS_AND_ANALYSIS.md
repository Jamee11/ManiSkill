# Experiment Results and Analysis

## ManiSkill PushCube EORT pilot

状态：环境、schema smoke 与 1 条 v2 端到端轨迹验证已通过；尚未进行批量采集或训练。

计划：完成 `maniskill-eort` 环境 smoke 后，采集 10 条成功的 `PushCube-v1` motion-planning 轨迹，并导出/验证 EORT sidecar。

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
