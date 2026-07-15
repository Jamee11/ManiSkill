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

v2 不替换 v1，而是使用独立的 `PushCubeEORT-v1` 和输出目录。它保持原 PushCube 物理、奖励与成功条件，只额外记录真实 robot-object contact force、物体线速度/角速度。导出器的 contact 来自力范数而不是距离；阶段仅适用于 PushCube：`0=approach`、`1=measured contact`、`2=contact while object moves`、`3=native task success`。

新采集的 v2 每条 `.npz` 额外包含：

- `robot_obj_contact_force` `(T,3)`、`robot_obj_contact_force_norm` `(T,1)`、`physical_contact` `(T,1)`、`push_interaction_phase` `(T,1)`；`physical_contact` 使用 Panda hand/左右 finger 的 force norm 之和，避免向量抵消；
- `object_linear_velocity` / `object_angular_velocity` `(T,3)`；
- `ee_to_object_rotvec` `(T,3)`，由正确的 wxyz 相对旋转得到；
- `object_future_delta_pos` / `object_future_delta_rotvec` `(T,3,3)`，默认 horizons 为 `[1,4,8]` action steps；
- `object_future_valid` `(T,3)`，末尾 horizon 不足时为 false，数值零不代表真值。
- `object_segmentation_id` / `goal_segmentation_id` `(1,)`，分别映射当前 task object/goal 到 raw segmentation actor label；`object_mask_pixels` / `goal_mask_pixels`、`object_visibility_fraction` / `goal_visibility_fraction`、`object_visible` / `goal_visible` 均为 `(T,1)`，只使用 action 前 observation。`object_bbox_xyxy` / `goal_bbox_xyxy` `(T,4)` 是 `[x_min,y_min,x_max_exclusive,y_max_exclusive]` 像素框，`object_mask_centroid_uv` / `goal_mask_centroid_uv` `(T,2)` 是 mask 像素均值；完全遮挡时二者全零、`visible=false`，不是导出失败。raw segmentation 仍是唯一 mask source，sidecar 不重复写 mask。旧 v2 raw 不含 ID 时仍可导出原标签，但 manifest 标记 `segmentation_visibility=false`。
- `object_segdepth_centroid_world` / `goal_segdepth_centroid_world` `(T,3)`：各自 actor-mask 像素由毫米 depth 与 CV intrinsic/extrinsic 回投的可见表面 centroid；对应 `*_segdepth_valid(T,1)` 和相对于 simulator center/goal 的 `*_segdepth_centroid_error(T,1)` 只作 oracle tracker supervision 与 QA。它们不是 object/goal pose，也不进入当前 action policy。
- `PushCubeEORTOccluded-v1` 额外导出 `occluder_active(T,1)` bool，表示 visual-only occluder 是否存在；它仅用于可见率 QA/split，manifest 标为 `policy_input=false`，绝不能作为 tracker 或 action policy 输入。
- `eef_transition_local` `(T,7)`：由 action 前后 TCP pose 的实际变化导出 `[Δxyz_local, Δrotvec_local, gripper_open_fraction]`。它是 observed transition，manifest 明确标为 `controller_command=false`，并非 raw Panda joint command 的替代。

```bash
MANISKILL_EORT_PYTHON=/path/to/conda/env/bin/python \
MANISKILL_EORT_CUDA_VISIBLE_DEVICES=0 \
MANISKILL_EORT_DATA_ROOT=/path/to/datasets/maniskill_push_cube_objectcentric_v2 \
NUM_TRAJ=1 \
bash scripts/eort/collect_push_cube_objectcentric_v2.sh
```

若需要跨 episode 的外部相机外参随机化，额外设置 `MANISKILL_EORT_ENV_ID=PushCubeEORTCameraRand-v1`。该环境在 reset 时随机一次相机 pose、episode 内保持不动；raw `sensor_param/base_camera/{intrinsic_cv,extrinsic_cv}` 是唯一的标定记录来源。固定相机与随机相机数据必须分开记录 split，不能将其混为同一泛化结论。

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

## 大规模 object-centric v2 采集、可视化与 DiT4DiT 导出

使用 `scripts/eort/collect_large_objectcentric_v2.sh`。它不改变已经验证的格式，而是为每个独立 shard 串联：

```text
raw HDF5 (T+1, RGB-D-segmentation-state) + seed_manifest
  -> derived_goal_segdepth/*.npz + manifest.jsonl + summary.json
  -> qa/*_preview.mp4 + JSON（bbox/质心/visibility/phase/contact overlay）
  -> LeRobot oracle / segdepth_proxy / track_corrupt 三个训练视图
```

HDF5+NPZ 是可审计的 object-centric 真值来源；LeRobot 只是 DiT4DiT 当前 loader 的训练视图。tracker 读取前者的 RGB-D/标定，DiT4DiT action policy 读取后者的 RGB、state、7D observed local EEF transition。任何一边都不以 LeRobot 覆盖 raw 或 sidecar。

先在目标机器上只检查计划；这不会创建目录、轨迹、视频或 LeRobot 数据：

```bash
cd /remote-home/jinminghao/WAMs/ManiSkill

MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v2 \
MANISKILL_EORT_TASK=push_cube \
MANISKILL_EORT_SPLIT=train \
NUM_TRAJ=500 \
MANISKILL_EORT_CUDA_VISIBLE_DEVICES=4 \
DRY_RUN=1 \
bash scripts/eort/collect_large_objectcentric_v2.sh
```

确认后删除 `DRY_RUN=1` 才会采集。每次调用对每个 variant 采 `NUM_TRAJ` 条**独立**成功物理轨迹；例如三个 variant 加 `NUM_TRAJ=500` 是 1,500 条独立物理轨迹，不是同一条轨迹三次重渲染。

```bash
MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v2 \
MANISKILL_EORT_TASK=push_cube \
MANISKILL_EORT_SPLIT=train \
MANISKILL_EORT_VARIANTS=fixed,camera_rand,occluded \
NUM_TRAJ=500 \
MANISKILL_EORT_CUDA_VISIBLE_DEVICES=4 \
bash scripts/eort/collect_large_objectcentric_v2.sh
```

默认 seed 区间为 train=`0`、val=`1000000`、test=`2000000`；同一命令内的 variant 再以 100,000 的 block 分开。raw 旁的 `*.seed_manifest.json` 记录每条保存 trajectory 对应的实际成功 task seed。不要手工复用 `MANISKILL_EORT_START_SEED` 或减小 `MANISKILL_EORT_SEED_BLOCK_SIZE`；否则 split 独立性失效。每个输出目录均拒绝覆盖，重复任务应使用新的 collection root。

建议第一批将 train/val/test 分别调用三次。先用 PushCube `500/100/200` 每 variant 跑完整链路，再根据 oracle gate 决定是否扩大到每 variant `1000+`；PickCube 只能在其 visual GPU smoke 后按同样协议采集。当前脚本默认 CPU PhysX + 指定 GPU renderer，单进程采集；不要为增速启用多 renderer 进程。

### 训练根目录对应关系

完成三个 variant 后，现有 DiT4DiT named mixture 无需改动。只覆盖其数据根：

```text
.../maniskill_eort_large_v2/lerobot/train/oracle
.../maniskill_eort_large_v2/lerobot/train/segdepth_proxy
.../maniskill_eort_large_v2/lerobot/train/track_corrupt
```

它们分别包含既有名称的 `maniskill_eort_{push,pick}_cube_256_{fixed,camera_rand,occluded}_..._lerobot` 数据集。DiT4DiT 仍使用 `maniskill_eort_*_256_*_splits_lerobot` mixture；不需要新 dataset loader 或修改 policy schema。val/test 根用相同 mixture 名称做数据/离线评估，不可与 train root 混用。

tracker 不读 LeRobot 视频，因为它需要 raw metric depth 与 camera calibration。将三条 train raw/sidecar 对传给既有 tracker launcher的 `SOURCES` 环境变量，并只用同一 physical trajectory 分组切分；不得把 train raw 与 val/test raw 拼在一起训练。

### 预览视频

每个 shard 自动写 `qa/*_preview.mp4`，默认拼接前三条 trajectory 的 raw RGB 并叠加 derived labels。已有 pilot 的实际示例是：

`/remote-home/jinminghao/datasets/maniskill_eort_previews/push_cube_fixed_objectcentric_preview.mp4`

预览用于检查格式和时间对齐，不是 policy rollout 视频，也不能替代逐 split 的 visibility、seed、成功率和 LeRobot loader QA。
