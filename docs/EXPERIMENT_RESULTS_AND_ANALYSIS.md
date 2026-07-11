# Experiment Results and Analysis

## ManiSkill PushCube EORT pilot

状态：环境与 schema smoke 已通过；尚未完成端到端轨迹采集。

计划：完成 `maniskill-eort` 环境 smoke 后，采集 10 条成功的 `PushCube-v1` motion-planning 轨迹，并导出/验证 EORT sidecar。

预注册检查：环境版本、episode 成功数、HDF5 schema、`T+1` observation 与 `T` action 对齐、所有导出数值有限、RGB/depth/segmentation 流存在。

下一步：仅在上述 smoke 全部通过后扩大数据量；若失败，记录失败命令、根因与修复方案，且不进行训练。

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
