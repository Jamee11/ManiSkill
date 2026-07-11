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
