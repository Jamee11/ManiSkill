# ManiSkill PushCube EORT 采集

当前实现是 **simulator-oracle** 数据流程，不是现实感知或训练流程：

```text
原生 PushCube motion-planning HDF5 (RGB/depth/segmentation/state_dict)
  → scripts/eort/derive_push_cube_eort.py
  → 每 trajectory 一个 EORT .npz + manifest.jsonl + summary.json
```

## 标签

每个 `.npz` 都与原始轨迹同名，并包含 `T` 个 action-aligned 行：

- `ee_to_object` / `ee_to_object_dist`
- `object_to_goal` / `object_to_goal_dist`
- `goal_progress`：相对首帧 object-goal 距离的当前进度，不保证成功时为 1（PushCube 成功条件有容差）
- `object_next_delta`：`t+1` 物体 xyz 减 `t` 物体 xyz
- `tcp_pose_wxyz`、`object_pose_wxyz`、`goal_pos`、`action`、`success`

不包含伪物理 contact、固定 normal 或四元数相减标签。HDF5 observation 为 `T+1`，action 和所有 sidecar 标签为 `T`。

## object-centric v2

v2 不替换 v1，而是使用独立的 `PushCubeEORT-v1` 和输出目录。它保持原 PushCube 物理、奖励与成功条件，只额外记录真实 robot-object contact force、物体线速度/角速度。导出器的 contact 来自力范数而不是距离；阶段仅适用于 PushCube：`0=approach`、`1=measured contact`、`2=contact while object moves`、`3=goal reached after contact`。

新采集的 v2 每条 `.npz` 额外包含：

- `robot_obj_contact_force` `(T,3)`、`robot_obj_contact_force_norm` `(T,1)`、`physical_contact` `(T,1)`、`push_interaction_phase` `(T,1)`；`physical_contact` 使用 Panda hand/左右 finger 的 force norm 之和，避免向量抵消；
- `object_linear_velocity` / `object_angular_velocity` `(T,3)`；
- `ee_to_object_rotvec` `(T,3)`，由正确的 wxyz 相对旋转得到；
- `object_future_delta_pos` / `object_future_delta_rotvec` `(T,3,3)`，默认 horizons 为 `[1,4,8]` action steps；
- `object_future_valid` `(T,3)`，末尾 horizon 不足时为 false，数值零不代表真值。
- `object_segmentation_id` / `goal_segmentation_id` `(1,)`，分别映射当前 task object/goal 到 raw segmentation actor label；`object_mask_pixels` / `goal_mask_pixels`、`object_visibility_fraction` / `goal_visibility_fraction`、`object_visible` / `goal_visible` 均为 `(T,1)`，只使用 action 前 observation。完全遮挡是有效 `visible=false`，不是导出失败。旧 v2 raw 不含 ID 时仍可导出原标签，但 manifest 标记 `segmentation_visibility=false`。

```bash
MANISKILL_EORT_PYTHON=/path/to/conda/env/bin/python \
MANISKILL_EORT_CUDA_VISIBLE_DEVICES=0 \
MANISKILL_EORT_DATA_ROOT=/path/to/datasets/maniskill_push_cube_objectcentric_v2 \
NUM_TRAJ=1 \
bash scripts/eort/collect_push_cube_objectcentric_v2.sh
```

v2 的首条真实 HDF5 已确认物理字段与 `obj_segmentation_id`/`goal_segmentation_id` 均为 `T+1`，而 derived labels 是 `T`；通过前不得将 v2 接入训练。批量阶段仍须检查 visibility fraction 分布，不能仅凭一条全可见轨迹声明感知鲁棒。

## 在有可用 GPU 的机器采集

CPU PhysX 配合 GPU renderer 是首版的目标配置。不要设置空的 `CUDA_VISIBLE_DEVICES`；本机的 SAPIEN CPU renderer 会崩溃。先以 1 条 smoke 验证指定 GPU，再扩大到 10 条。

```bash
cd /path/to/ManiSkill

MANISKILL_EORT_PYTHON=/path/to/conda/env/bin/python \
MANISKILL_EORT_CUDA_VISIBLE_DEVICES=0 \
MANISKILL_EORT_DATA_ROOT=/path/to/datasets/maniskill_eort_push_cube \
NUM_TRAJ=1 \
bash scripts/eort/collect_push_cube_eort.sh
```

Smoke 通过后，将 `NUM_TRAJ=10`。脚本拒绝覆盖已有 raw HDF5 或 `derived/` 输出。`MANISKILL_EORT_CUDA_VISIBLE_DEVICES` 是显式 renderer 选择，不会自行探测或抢占 GPU。

## 本地验证

```bash
/path/to/conda/env/bin/python -m unittest tests/test_eort_push_cube.py -v
```

该检查验证标签的 `T+1`/`T` 对齐、progress 和下一帧物体平移；批量采集前仍须核对 `docs/思考与隐患.md`。
