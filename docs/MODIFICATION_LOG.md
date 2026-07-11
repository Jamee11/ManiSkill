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
