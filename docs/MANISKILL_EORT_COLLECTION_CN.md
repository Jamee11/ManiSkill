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
- 每个 v2 shard 的 `summary.json.visibility_qa` 自动汇总 object/goal/joint-visible、joint segdepth-valid、两类 mask pixel 的 min/median/max，以及存在时的 occluder-active 步数；大规模采集无需再手工扫描 NPZ 才能发现“object 可见但 goal 全不可见”的无效关系样本。
- `eef_transition_local` `(T,7)`：由 action 前后 TCP pose 的实际变化导出 `[Δxyz_local, Δrotvec_local, gripper_open_fraction]`。它是 observed transition，manifest 明确标为 `controller_command=false`，并非 raw Panda joint command 的替代。
- 当 source metadata 的 control mode 明确为 `pd_ee_delta_pose` 时，额外保存 `panda_pd_ee_delta_pose_command (T,7)`，逐元素等于 HDF5 action，并在 manifest 标为 Panda、normalized、root-translation/root-aligned-rotation controller command。该字段用于 Panda policy 主目标；它仍不是 Piper/真机 command。
- 同时保存可逆的 `metric_task_delta_pose_command (T,7)`：`Δxyz` 为米，旋转为 task/world 轴 rotvec（弧度），夹爪为 `[0,1]` open fraction。当前 Push/Pick Panda root 只有平移、orientation identity，因此 root 与 task/world 轴一致；它由 normalized controller action 的真实 `0.1m/-0.1rad` 缩放及 XYZ-Euler→rotvec 精确解码。它消除 Panda 数值归一化，但真机仍需 task-from-base 标定、频率/限幅和 gripper adapter。

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

该正式 wrapper 已在 GPU 7 对 PushCube 和 PickCube 的 fixed/camera-random/occluded 各 1 条完成端到端 smoke，输出分别位于 `/remote-home/jinminghao/datasets/maniskill_eort_large_v2_smoke_20260716` 与 `/remote-home/jinminghao/datasets/maniskill_eort_pick_large_v2_smoke_20260716`。这只证明链路可运行；正式采集仍按新的 collection root 执行，不能复用或覆盖 smoke 目录。Pick occluded smoke 的 goal 为 0/72 可见，说明批量 QA 必须同时报告 object、goal 与 relational-valid 比例。

```text
raw pd_joint_pos HDF5 + seed_manifest
  -> ManiSkill 官方 replay -> successful pd_ee_delta_pose HDF5
  -> derived_controller_goal_segdepth/*.npz + manifest.jsonl + summary.json
  -> qa/*_preview.mp4 + JSON（bbox/质心/visibility/phase/contact overlay）
  -> LeRobot oracle / segdepth_proxy / track_corrupt 三个训练视图
```

HDF5+NPZ 是可审计的 object-centric 真值来源；LeRobot 只是 DiT4DiT 当前 loader 的训练视图。tracker 读取 controller-replay HDF5 的 RGB-D/标定。默认 action 仍是同一轨迹的 7D Panda controller command；设置 `MANISKILL_EORT_ACTION_SOURCE=metric_task_delta_pose` 后，导出可逆解码的 metric task command，并使用独立的 `*_metric` 目录，避免覆盖默认数据。observed local EEF transition 只保留为 dynamics/辅助目标。任何一边都不以 LeRobot 覆盖 raw 或 sidecar。

先在目标机器上只检查计划；这不会创建目录、轨迹、视频或 LeRobot 数据：

`DRY_RUN=1` 可以不指定 GPU（输出中的 `gpu` 会显示为 `unset`）。任何真实采集都必须显式设置非空的 `MANISKILL_EORT_CUDA_VISIBLE_DEVICES`；否则 wrapper 会在创建 shard 前退出，避免静默落到曾 OOM 的默认 GPU 0。

```bash
cd /remote-home/jinminghao/WAMs/ManiSkill

MANISKILL_EORT_COLLECTION_ROOT=/remote-home/jinminghao/datasets/maniskill_eort_large_v2 \
MANISKILL_EORT_TASK=push_cube \
MANISKILL_EORT_SPLIT=train \
NUM_TRAJ=500 \
MANISKILL_EORT_CONTROLLER_REPLAY_ENVS=1 \
MANISKILL_EORT_CUDA_VISIBLE_DEVICES=4 \
DRY_RUN=1 \
bash scripts/eort/collect_large_objectcentric_v2.sh
```

若第一批数据明确用于 Franka/Piper 共享 action gate，建议在采集命令中加入：

```bash
MANISKILL_EORT_ACTION_SOURCE=metric_task_delta_pose
```

raw HDF5、sidecar 和预览不变；LeRobot 输出自动写到 `oracle_metric`、`segdepth_proxy_metric`、`track_corrupt_metric`。每步 action 是 `[task_dx_m,task_dy_m,task_dz_m,task_drx_rad,task_dry_rad,task_drz_rad,gripper_open]`。它是 task/world 轴增量，不是当前设计文档中更强的 EEF-local 双臂最终契约；真机 adapter 仍必须做 task-from-base 标定和限幅。

若还要排“预测未来 object motion→Action DiT”的后续消融，在同一次新 collection 中同时设置：

```bash
MANISKILL_EORT_ACTION_SOURCE=metric_task_delta_pose \
MANISKILL_EORT_INCLUDE_FUTURE_TARGETS=1
```

这会写独立的 `oracle_metric_dynamics`、`segdepth_proxy_metric_dynamics`、`track_corrupt_metric_dynamics` 数据根。默认值为 `0`，默认数据仍不包含任何 future modality。dynamics view 额外保存 h=`1/4/8` 的 18D `object Δxyz+Δrotvec` target 与 3D valid mask；普通 `_metric` mixture 会忽略这些字段。

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

建议第一批将 train/val/test 分别调用三次。脚本在未显式设置 `NUM_TRAJ` 时按 split 默认使用 PushCube `500/100/200` 每 variant；显式 `NUM_TRAJ` 仍会覆盖。先按该规模跑完整链路，再根据 oracle gate 决定是否扩大到每 variant `1000+`；PickCube 只能在其 visual GPU smoke 后按同样协议采集。当前脚本默认 CPU PhysX + 指定 GPU renderer，单进程采集；不要为增速启用多 renderer 进程。

### 训练根目录对应关系

完成三个 variant 后，只覆盖 DiT4DiT 数据根，并选择 controller mixture：

```text
.../maniskill_eort_large_v2/lerobot/train/oracle
.../maniskill_eort_large_v2/lerobot/train/segdepth_proxy
.../maniskill_eort_large_v2/lerobot/train/track_corrupt
```

它们分别包含既有名称的 `maniskill_eort_{push,pick}_cube_256_{fixed,camera_rand,occluded}_..._lerobot` 数据集。正式训练使用原 mixture 名追加 `_controller`，并设置 `MANISKILL_EORT_ACTION_SOURCE=panda_pd_ee_delta_pose`；旧 mixture 仅兼容 observed-transition pilot。val/test 根使用相同 controller mixture 做数据/离线评估，不可与 train root 混用。

metric action 的目录是上述路径追加 `_metric`，mixture 名追加 `_metric`，训练设置 `MANISKILL_EORT_ACTION_SOURCE=metric_task_delta_pose`。两套 action 数据不得写入同一 LeRobot 根目录或在一个 run 中混用。

### 训练前 collection 硬检查

train/val/test 全部采完后，在排 tracker 或 policy 训练前运行：

```bash
/remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python \
  scripts/eort/audit_large_collection.py \
  --root /remote-home/jinminghao/datasets/maniskill_eort_large_v2 \
  --task push_cube \
  --action-source metric_task_delta_pose \
  --future-targets
```

该命令只读，不创建或修改数据。它要求三个 split × 三个视觉 variant 全部存在，并检查 controller JSON 中的 ManiSkill commit、环境 ID、obs/control/sim backend、成功 seed 与 seed manifest 一致且全局无重复，同时检查 seed/sidecar trajectory、action provenance、visibility QA，以及 oracle/proxy/corrupt LeRobot episode 数、condition、action 与 future-target 标记。所有 shard 必须来自同一 ManiSkill commit。普通 metric 数据去掉 `--future-targets`。任何一项不一致都会非零退出；不要绕过后继续训练。只检查单个 smoke split 时可显式设置 `--splits train --exports oracle`。

tracker 不读 LeRobot 视频，因为它需要 raw metric depth 与 camera calibration。自动 launcher 选择 `*.pd_ee_delta_pose.physx_cpu.h5` 与 `derived_controller_goal_segdepth`；只按同一 physical trajectory 分组切分，不得把 train raw 与 val/test raw 拼在一起训练。

### 预览视频

每个 shard 自动写 `qa/*_preview.mp4`，默认拼接前三条 trajectory 的 raw RGB 并叠加 derived labels。已有 pilot 的实际示例是：

`/remote-home/jinminghao/datasets/maniskill_eort_previews/push_cube_fixed_objectcentric_preview.mp4`

预览用于检查格式和时间对齐，不是 policy rollout 视频，也不能替代逐 split 的 visibility、seed、成功率和 LeRobot loader QA。
