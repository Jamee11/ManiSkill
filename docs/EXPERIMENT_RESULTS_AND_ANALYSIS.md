# Experiment Results and Analysis

## 2026-07-17 02:34 UTC - Actual PhysX object-mass provenance smoke

- Setup: GPU 4, ManiSkill commit `7a0029f`, isolated geometry-rand metric smokes: Push seed 5100 and Pick seed 6100. Retained roots: `/remote-home/jinminghao/datasets/maniskill_eort_mass_smoke_20260717_{push,pick}` (73/87 MB).
- Result: Push and Pick source motion planning plus controller replay were each 1/1 successful over 63/76 steps. Raw `obj_mass(T+1,1)` was episode-static at `0.086086/0.072059 kg`, matched exactly between source/controller HDF5, and matched recorded extent volume × SAPIEN default density `1000 kg/m³`.
- QA: Both sidecars stored `object_mass(1,1)` with explicit raw provenance; both production audits and metric LeRobot exports passed. Mass remains sidecar-only and does not change DiT4DiT input or architecture.
- Boundary: Geometry currently couples volume and mass through fixed density. Independent density/mass randomization should wait for a measured real-object range; these two smokes are provenance evidence, not dynamics-generalization evidence.

## 2026-07-17 02:25 UTC - PickCube seeded geometry/friction source-to-controller smoke

- Setup: GPU 4, `PickCubeEORTGeometryRand-v1`, seeds 4100/4101/4102, metric action export, isolated `geometry_rand` test shard. Output: `/remote-home/jinminghao/datasets/maniskill_eort_pick_geometry_rand_smoke_20260717` (238 MB).
- Result: Source motion planning and Panda controller replay were 3/3 successful with lengths 65/76/67. Full side lengths were `3.781/3.862/4.309 cm`; static/dynamic friction was `0.434/0.333/0.451`. Both raw HDF5 files matched exactly for extent/friction on every frame and trajectory.
- QA: Production audit passed for 3 unique seeds and 208 steps. Phase counts were `122/2/73/11`; object visible 208/208, goal/relational visible and segdepth-valid 166/208. The isolated metric DiT4DiT mixture loaded one 208-step component with state `(1,64)`, action `(8,7)`, image `(3,224,224)`.
- Boundary: This verifies deterministic grasp-domain replay and plumbing only. It is not enough to estimate the success distribution or sim2real benefit, and simulator `is_grasped` remains label-only.

## 2026-07-16 19:41 UTC - Seeded geometry/friction source-to-controller smoke

- Setup: GPU 7, `PushCubeEORTGeometryRand-v1`, seeds 3100/3101/3102, metric action export, one isolated `geometry_rand` shard. Output: `/remote-home/jinminghao/datasets/maniskill_eort_geometry_rand_smoke_20260716` (232 MB).
- Result: source motion planning and converted Panda controller replay were 3/3 successful with trajectory lengths 70/65/70. Full cube side lengths were `4.542/3.575/4.211 cm`; static/dynamic friction values were `0.178/0.216/0.387`. Source and controller HDF5 parameter arrays matched exactly for all frames and trajectories.
- QA: 205/205 object and goal frames were visible and segdepth-valid; phase counts were `125/0/57/23`. Production audit passed with three unique seeds and source commit `d6ff886`. The isolated metric DiT4DiT mixture loaded one 205-step component and returned state `(1,64)`, action `(8,7)`, image `(3,224,224)`.
- Boundary: This validates deterministic asset reconstruction and data plumbing only. Three successful seeds do not estimate large-run success, optimal randomization range, or sim2real gain; mass, appearance, lighting and sensor noise remain outside this split.

## 2026-07-16 19:18 UTC - Domain-randomization prerequisite audit

- Current Push/Pick already randomize object/goal poses; EORT additionally covers camera pose and visual occlusion. It does not yet vary cube geometry, mass/friction, appearance, lighting, sensor noise or robot calibration.
- ManiSkill's existing SO100 digital-twin task demonstrates seeded scene-load randomization for geometry, friction, color and lighting. Geometry changes are not ordinary saved environment state, so reusing that pattern safely requires proving that controller replay reconstructs the identical asset from the recorded episode seed.
- Implemented only the missing prerequisite: explicit raw/sidecar object extent with production provenance enforcement. Synthetic derivation/audit checks pass; the retained real 71-step Push controller HDF5, which predates the raw field, re-derived the exact fixed `[0.04,0.04,0.04] m` extent and was marked `legacy_fixed_panda_cube_0.04m`.
- Runtime smoke: GPU 7, fixed Push, seed 0, metric action, one production-wrapper episode. Source motion planning and controller replay were 1/1 successful with 71 actions. Raw extent was `(72,3)`, sidecar extent `(1,3)`, both approximately 0.04 m; oracle/segdepth/corrupt LeRobot exports and the read-only collection audit passed. Output: `/remote-home/jinminghao/datasets/maniskill_eort_extent_smoke_20260716_retry` (81 MB).
- Size/friction/lighting variants remain deferred until a randomized source→controller replay proves identical seeded asset reconstruction; the fixed-size smoke is no generalization evidence.

## 2026-07-16 19:10 UTC - Franka/Piper deployment-interface audit

- Local evidence: StarVLA documents the desired Franka 7D delta action but leaves `env.step(action)` as a user implementation. UniVLA contains a working-style ROS path, but it is coupled to `/mk1000` joint state, a project-specific FK/IK stack, front-camera hand-eye calibration and gripper topic. No Piper SDK or Piper control node was found in the selected project workspaces.
- Result: Added only an offline calibration/limit audit around the already verified EORT metric action. Synthetic checks pass for axis rotation, native gripper mapping, and violation counting; this is not a hardware experiment or proof of safe execution.
- Next gate: On the target robot, supply measured calibration and controller limits, implement the actual SDK/ROS adapter in the robot-owning repository, then run workspace/collision/latency/estop checks and a low-speed no-object replay before any manipulation trial.

## 2026-07-16 19:05 UTC - Reproducible external collection runtime

- The production wrapper now records its actual interpreter/package versions only after a shard completes, and the training audit rejects incomplete or mixed runtimes. This reuses stdlib package metadata and adds no dependency or model/data field.
- Local static verification resolves Python `3.10.20`, ManiSkill `3.0.1`, Torch `2.7.1+cu128`, SAPIEN `3.0.3`, NumPy `1.26.4`, h5py `3.16.0`. These are not evidence that another machine's driver/GPU capacity is adequate; its first job remains a one-trajectory smoke.

## 2026-07-16 18:57 UTC - Interaction-phase distribution QA

- Added shard-level counts for the existing task-specific phases. Push class 2 means measured contact while the object moves; Pick class 2 means native grasp detected, so the counts are reported per task and are not treated as one shared semantic class.
- Collection audit now rejects a shard with no class-2 interaction or no class-3 success. This is a data-coverage gate, not a phase predictor or policy input; a discrete model head remains deferred until the robot-only/oracle object gate establishes value and cross-task phase semantics are fixed.
- Fresh CPU-only derivation of the retained fixed controller trajectories reported Push counts `0/1/2/3 = 48/0/14/9` over 71 steps and Pick `36/1/33/4` over 74 steps. This validates aggregation on real HDF5, not large-scale phase balance.

## 2026-07-16 18:45 UTC - Collection source-provenance audit

- The read-only training gate now consumes metadata already written beside controller HDF5 rather than creating another manifest. It rejects unexpected task/visual environment IDs, observation/control/simulation backends, seed disagreement, missing source commit, and mixed commits across shards.
- A read-only inspection of the retained Push fixed/camera-random/occluded controller metadata found one source commit (`5d369b5`), the expected three environment IDs/backends and 3 matching unique seeds. Its older retained sidecar summaries predate the current visibility-QA schema, so the whole legacy root does not pass today's complete audit and must not substitute for a newly collected production root. Python/Torch/SAPIEN versions remain an external environment-smoke requirement.

## 2026-07-16 18:40 UTC - Production renderer selection guard

- Static/negative-path verification only: the large-collection wrapper accepts an unset GPU during `DRY_RUN=1`, reports an explicit selector when supplied, and rejects a real invocation without `MANISKILL_EORT_CUDA_VISIBLE_DEVICES` before creating its collection root.
- This prevents accidental fallback to the previously OOMing default GPU 0. It does not measure GPU capacity, renderer peak memory, collection throughput or large-run stability; the selected GPU still requires a one-trajectory smoke on the target machine.

## 2026-07-16 20:00 UTC - 独立 future-object supervision 数据视图

- 大规模 wrapper 新增显式 `MANISKILL_EORT_INCLUDE_FUTURE_TARGETS=1`，只允许与 metric action 同用，并写入独立 `*_metric_dynamics` LeRobot 根；默认数据格式和无 future policy modality 约束不变。
- dynamics view 保存 h=1/4/8 的 18D object `Δxyz+Δrotvec` 及 3D valid mask。collection audit 增加 `--future-targets` 合同检查，防止普通数据和 auxiliary-target 数据混淆。
- 真实 Push 三视觉 dynamics export/audit 通过：3 unique seeds、3 episodes、202 steps。该结果没有训练 predictor，也不能证明未来状态预测准确。

## 2026-07-16 17:45:00 UTC - Reversible metric task action and Panda replay

- Source audit: Panda `pd_ee_delta_pose` maps normalized translation to `[-0.1,0.1] m`, maps the normalized rotation vector through unit-ball clipping then multiplies by `-0.1 rad` as root-aligned XYZ Euler, and maps gripper to the physical `[-0.01,0.04] m` target. All six Push/Pick production-smoke trajectories had rotation norm below one (`max=0.9423`), so no source command was clipped.
- Implementation: Added a retained metric command `[delta_xyz_m, delta_rotvec_rad, gripper_open_fraction]` plus exact Panda encode/decode and calibration-axis rotation. Raw normalized Panda action and non-command observed transition remain separate.
- Verification: Six EORT tests passed. On the real fixed Push trajectory, all 71 metric commands encoded back with maximum absolute error `1.49e-8`; the rewritten HDF5 replayed on GPU7 with 1/1 final success. Output: `/remote-home/jinminghao/datasets/maniskill_eort_metric_action_smoke_20260716`.
- Boundary: This validates simulator Panda conversion only. It does not validate a Franka/Piper driver, task calibration, control frequency, gripper direction/range, latency, collision limits, or real-world safety.

## 2026-07-16 16:42:08 UTC - PickCube three-variant production smoke

- Setup: Reused the unchanged production wrapper on GPU 7 for one fixed, camera-randomized, and visually occluded PickCube trajectory with disjoint seeds `0/100000/200000`, official Panda controller replay, all three export views, and 256px previews. Output: `/remote-home/jinminghao/datasets/maniskill_eort_pick_large_v2_smoke_20260716` (257 MB).
- Result: All source and replay trajectories succeeded with `74/81/72` controller steps. HDF5 actions exactly matched sidecar and all nine Parquet action arrays. Raw grasped-frame counts were `37/41/21`; derived phase counts were `[36,1,33,4]`, `[39,1,34,7]`, and `[50,1,18,3]`, and phase 2 exactly matched raw grasped-and-not-success rows. All continuous fields were finite. The three controller mixtures loaded `[74,81,72]` and emitted state `(1,64)`, action `(8,7)`, image `(3,224,224)` with status 0.
- Visibility finding: Fixed and camera-randomized object/goal were visible throughout. In the occluded seed the object remained visible 72/72, but the elevated goal was invisible 0/72; therefore every segdepth relational condition in that component is invalid/zero. This is a valid hard occlusion sample, not a deployable signal, and must be reported separately rather than hidden in aggregate object visibility.
- QA follow-up: The existing deriver summary now emits this check automatically. Fresh real-data derivation reported Push occluded relational visible/depth-valid `66/66` and Pick occluded `0/72`, including mask-pixel min/median/max; the five-test deriver suite passed.
- Interpretation: The earlier PickCube runtime-smoke gap is closed. The one-seed-per-variant result is not sufficient for a large PickCube collection decision, tracker generalization, grasp estimation, or sim2real claims.

## 2026-07-16 16:25:33 UTC - Three-variant large-collection harness smoke

- Setup: Ran `scripts/eort/collect_large_objectcentric_v2.sh` on GPU 7 with CPU PhysX, one successful PushCube trajectory per fixed/camera-random/occluded variant, start seeds `0/100000/200000`, official replay to `pd_ee_delta_pose`, all three export views, and 256px QA previews. Output: `/remote-home/jinminghao/datasets/maniskill_eort_large_v2_smoke_20260716` (229 MB).
- Result: All three source and controller-replayed trajectories succeeded; controller lengths were `71/65/66`. HDF5 actions exactly matched the sidecar Panda controller command and all nine exported Parquet action arrays. All continuous object/dynamics/future arrays were finite. Object visibility was `71/71`, `65/65`, and `66/66`; mask-pixel ranges were `90–119`, `99–115`, and `72–100`. The occluder was active for all 66 occluded frames but did not fully hide the object in this seed. All previews were 256x256 with one frame per action.
- Loader gate: Oracle, segdepth-proxy, and delayed/ID-switch controller mixtures each initialized component lengths `[71,65,66]` and emitted state `(1,64)`, action `(8,7)`, image `(3,224,224)` with status 0.
- Interpretation: The exact production wrapper is now verified end to end and did not OOM on this one-per-variant smoke. This is not a large dataset, does not estimate throughput/peak GPU memory, and the single occluded seed is not evidence of occlusion robustness. Full train/val/test collection remains an external scheduled job.

## 2026-07-16 15:58:07 UTC - DiT4DiT Panda evaluator replay smoke

- Setup: GPU 7 renderer, CPU PhysX, `PushCubeEORT-v1`, first recorded state, and exact replay-generated `pd_ee_delta_pose` actions. The DiT4DiT evaluator constructed the current oracle object condition and applied its learned-action safety path; no policy/tracker training was run.
- Result: Success at step 63, zero clipped actions, 63/63 valid condition steps, and a 64-frame 256x256 MP4. Output is `/remote-home/jinminghao/datasets/dit4dit_maniskill_closed_loop_oracle_smoke_20260716`.
- Interpretation: Simulator reset, current observation extraction, controller command, safety, video, and result logging are connected. This is replay verification, not evidence of learned-policy quality or sim2real transfer.
- Next: Train matched Panda-controller `robot_only` and `oracle` policies externally, then evaluate on identical held-out seeds before scheduling the learned-tracker policy.

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

### 2026-07-13 — X-WAM/Tau/DreamDojo baseline readiness audit

- X-WAM 本地源码明确支持每臂 8D proprio `xyz+wxyz+gripper`、每臂 7D local EEF action `Δxyz+Δaxisangle+gripper`，以 mask 表达缺失右臂，并且读取 RGB-D 多视图。这是当前唯一与 Franka 单臂、Piper 双臂目标契约直接同构的候选。
- 但 X-WAM 本机没有 `checkpoints/`；README 的公开权重仍需实际下载、加载和 forward 验证。因此当前只能将它列为 DiT4DiT oracle gate 之后的主线候选，不可报告为可用 pretrained baseline。
- τ₀-WM 当前正在以双臂预训练接口运行；其 action/proprio 语义与 14D/16D 契约不一致，适合迁移消融而不是首个统一策略。DreamDojo 是 world-model/trajectory 候选，未提供可直接比较的 action policy 接口。
- 决策不变：ManiSkill 是新的 object-centric 数据与 GT 评测主场，DiT4DiT 先回答“当前 object information 是否改善 action”；仅在此 gate 有闭环收益后，再投入 X-WAM 数据转换、权重加载和 token fusion。

### 2026-07-16 — backup backbone readiness re-audit

- X-WAM 官方 checkpoint 已发布，仓库总量约 117 GB，包含 pretrained、RoboCasa-SFT 与 RoboTwin-SFT；但本机仍没有 X-WAM checkpoint，且推理还依赖 Wan2.2-TI2V-5B。其 16D proprio、14D per-arm 7D delta action、单臂 zero/mask 与多视图 RGB-D 接口仍是 Franka/Piper 结构上最匹配的候选，但当前不能执行 forward。
- τ₀-WM 本机已有约 21 GB 基础权重和约 27 GB Wan2.2 基座；本地仓库已有 `action_in_dim=7`、`dual_arm=false` 的 RLBench downstream head、训练配置和部署入口。它与 canonical 双臂契约不直接同构，但如果 DiT4DiT 失败，它是当前最快能复用 7D action 数据进入后训练的备选。
- DreamDojo 已发布 2B/14B world-model 权重与 post-training code，但当前接口重点是 action-conditioned future generation，没有与现有 Panda 7D controller command 直接对齐的 action-policy 输出；保留为 dynamics/trajectory 辅助，不进入第一轮 policy 切换。
- 更新后的两种排序：当前主线仍为 DiT4DiT；按“最短切换时间”是 τ₀-WM 第二，按“长期 Franka/Piper 结构同构性”是 X-WAM 第二。未新增适配代码，因为 X-WAM 无本地权重无法留下可运行检查，而 τ₀-WM 已有现成训练/部署框架；只有 DiT4DiT 闭环 gate 失败或 X-WAM 权重下载完成时才启动对应适配。

### 2026-07-13 13:05:00 UTC — segmentation-depth object localization proxy

- 设置：v2 派生器从 raw `segmentation`、毫米 `depth`、逐帧 `intrinsic_cv (3,3)` 与 `extrinsic_cv (3,4)`回投 object actor-ID 像素到 world，输出 `object_segdepth_centroid_world (T,3)`、`object_segdepth_valid (T,1)` 与相对于 simulator object center 的误差。object pose 只用于误差标签，不参与回投。
- 合成验证：2×2、已知 1 m depth/单位内外参 fixture 的回投 centroid 精确为预期坐标；无 object mask 帧保持 zero centroid 且 `valid=false`。四项 EORT 单测全通过。
- 真实验证：新 `derived_segdepth` sidecar 在已采集 `T=71` 轨迹上 71/71 帧有效；中心误差为 0.0150–0.0207 m，均值 0.0197 m。首帧可见表面 centroid `[0.002364,0.047884,0.039639]`，simulator object center `[-0.000749,0.053644,0.020000]`。
- 分析：约 2 cm 偏差符合相机只能看到 cube 表面而非几何中心的预期。这是一个带 oracle actor mask 的 RGB-D 几何诊断上界，不是可部署的 pose tracker；后续 object-state model 必须学习/标定该 surface-to-center bias，并承受检测漏失、ID switch、深度噪声和相机标定误差。

### 2026-07-13 13:10:00 UTC — Panda joint-to-EEF controller replay

- 设置：使用 ManiSkill 官方 `replay_trajectory` 将真实 successful `PushCubeEORT-v1` raw `pd_joint_pos` trajectory 转换为 `pd_ee_delta_pose`，CPU PhysX + 显式 GPU 4 renderer，`count=1`，保存独立 converted HDF5。
- 结果：converted trajectory action 为 `(71,7)`，observation 为 72 帧，最终 `success=True`。动作每维绝对最大值为 `[0.182812,0.027179,0.288201,0.008141,0.243892,0.017678,1.0]`；metadata 明确 target control mode 为 `pd_ee_delta_pose`。
- 解释：这 7D 是 Panda controller 的归一化 root-frame delta command，不是从 TCP observation 差分得到的 `eef_transition_local`。官方 replay 成功证明 Panda source-action 到 EEF-controller 的 IK 转换可执行，但没有证明该 action 可直接发送给 Franka/Piper 真机，也没有证明 local-frame canonical action 的等价性。
- 下一步：保留 converted HDF5 为 Panda controller-command reference；在 Piper 控制器/标定接口到位后，建立显式 task-frame adapter 和同样的 command replay check，禁止使用 observed transition 替代硬件命令。

### 2026-07-13 13:18:15 UTC — PushCubeEORT-v1 v2 十轨迹 pilot 与 LeRobot 完整性 QA

- 设置：用户确认 GPU 4 有余量后，运行现有原生 motion-planning 收集脚本，CPU PhysX + GPU renderer；输出使用新目录 `/remote-home/jinminghao/datasets/maniskill_push_cube_objectcentric_v2_pilot10_20260713T130957Z`。对同一 raw HDF5 与 derived directory 运行专用 DiT4DiT exporter。
- 采集结果：10/10 最终 success，0 个失败规划；action 长度为 `[71,72,64,74,66,77,62,69,63,68]`，共 686 steps。GPU renderer 在 GPU 4 上完成，未发生 OOM；这只说明本次时段该设备足够，不改变“需显式选择 renderer GPU”的运行约束。
- oracle/视觉 QA：每一帧 object 与 goal 均可见（686/686）；object visibility fraction 范围 0.0009766–0.0018921，goal 为 0.0218506–0.0411987。object segmentation-depth centroid 在 686/686 帧有效；相对于 simulator object center 的 error 为 1.17–2.23 cm，均值 1.80 cm，p95 2.09 cm。它依然是 oracle actor mask 加 RGB-D 回投的可见表面中心，不是视觉 tracker 的成功率。
- interaction/未来标签 QA：physical-contact 为 236 帧；phase counts 为 approach=450、contact=7、moving-in-contact=229、goal=0。未来有效数为 horizon 1/4/8 分别 686/656/616。`goal=0` 是当前 rollout 未达到基于当前定义的 phase 终态，不是 final success 的否定，二者应保持分开解释。
- LeRobot 出口 QA：`meta/info.json` 报告 10 episodes、686 frames；磁盘上有 10 个 episode Parquet 与 10 个 H.264 MP4，行数总和 686，与 metadata 一致。Parquet 仍是每步 `state(8)`、`oracle_current(17)`、`observation_valid(1)`、observed-local `action(7)`；视频 feature 为 `observation.images.front`、128×128、20 FPS。该验证证明数据文件的数目与时序总长完整，不表示 DiT4DiT 已训练或闭环成功。
- 分析：这完成了“ManiSkill raw → v2 sidecar → LeRobot v2”多轨迹数据链路 gate。当前分布几乎没有遮挡且 object 像素非常小，故下一次采集必须先定义 camera/遮挡、漏检、ID-switch、深度噪声及 goal-phase 的覆盖计划；不能仅增加相同的无遮挡轨迹数量。另一次在本会话的 DiT4DiT import 在模块初始化后无 traceback 提前退出，尚未产生 dataset sample，因而 loader 复验仍是待办，不能用它否定已通过的 exporter 文件 QA。

### 2026-07-13 13:24:00 UTC — phase terminal-label 语义修正

- 发现：上述 pilot 的 raw `success[-1]=true`，但 `push_interaction_phase=3` 计数为零。根因是 sidecar 以 `goal_progress>=0.999` 近似终态，严格于原生 PushCube 成功容差。
- 修正：phase 3 直接采用原生 action-aligned `success(T)`，且覆盖同帧的接触阶段。这保留连续 `goal_progress` 作诊断，但不再拿它定义任务成功。
- 验证：更新了可构造的 v2 fixture，使末步 success 必须导出 phase 3。旧 pilot NPZ 保持不可变；同一 raw HDF5 已重新派生至 `derived_success_phase`。10 条、686 帧逐元素满足 `phase==3` 等于 raw `success`；phase counts 变为 approach=450、contact=7、moving-in-contact=144、goal=85。85 是 environment success 维持为 true 的帧数，不应误读为只有 10 个 terminal frame。

### 2026-07-13 13:31:00 UTC — 后续 EORT 采集分辨率决策

- 决策：后续 `PushCubeEORT-v1` 原始 RGB、depth 和 segmentation 从 128×128 改为 256×256。DiT4DiT 的 224×224 input resize 现在是下采样而非 128 的上采样；现有 pilot 保留为低分辨率 schema/出口回归数据，不混入未来 tracker 质量结论。
- 未做：不重采已有 pilot、不修改 raw HDF5、不在本次繁忙 GPU 上启动 smoke 或批量收集。下一次显式 GPU smoke 需检查 256×256 streams、object mask pixel 数、sidecar/LeRobot shape、渲染显存和导出吞吐。

### 2026-07-13 13:33:54 UTC — 256×256 raw-camera smoke

- 设置：GPU 4、CPU PhysX + GPU renderer、原生 Panda motion planner，`NUM_TRAJ=1`；独立输出目录为 `/remote-home/jinminghao/datasets/maniskill_push_cube_objectcentric_v2_256_smoke_20260713T133354Z`。
- 结果：1/1 最终 success，`T=71`。raw RGB/depth/segmentation 为 `(72,256,256,3)/(72,256,256,1)/(72,256,256,1)`；derived object mask 为 90–117 pixels、均值 95.8，71/71 object-visible 且 71/71 segmentation-depth valid。phase count 为 approach=48、contact=0、moving-in-contact=14、goal=9，符合 raw success 持续帧语义。
- 分析：相同视角下 object mask 像素约为旧 128×128 smoke 的四倍，证明采集端不再依赖上采样。GPU 4 采前/后报告显存均为 63,782 MiB 且未 OOM，但没有采样峰值；批量前需保留峰值与吞吐 QA，也仍需遮挡和视觉域随机化覆盖。

### 2026-07-13 13:36:00 UTC — 256×256 LeRobot/DiT4DiT input 验证

- 设置：对上述高分辨率 smoke 运行既有 `convert_maniskill_eort_to_lerobot.py`，再以 `maniskill_eort_push_cube_oracle_lerobot` mixture 和既有 data config 读取一个 DiT4DiT sample。
- 结果：LeRobot `observation.images.front` metadata 为 `[256,256,3]`、20 FPS；有 71 行 Parquet 和 1 个 H.264 MP4。loader 成功初始化 dataset（长度 71），sample 的实际 `image` list 含一个 `(3,224,224)` tensor；state/action masks 同时生成。
- 分析：高分辨率信息在数据层保留，训练 transform 在 encoder 前下采样为 224，不存在从 128 到 224 的伪信息增加。这个验证仅覆盖单条高分辨率 smoke；批量采集仍需峰值显存、视频 decode 吞吐和 object-pixel 分布 QA。

### 2026-07-13 13:40:00 UTC — 2D tracker-label 导出验证

- 设置：在 v2 sidecar 中从已有 raw segmentation actor-ID 计算 object/goal mask 的 bbox 与像素 centroid；不复制 mask，不改环境、物理或 raw HDF5。bbox 约定 `[x_min,y_min,x_max_exclusive,y_max_exclusive]`。
- 合成验证：2×2 fixture 中多像素 object 得到 `[0,0,2,2]`、centroid `[0.5,0.5]`；完全遮挡帧保持全零 bbox/centroid 且 `visible=false`；单像素和 goal 的独立坐标也通过。
- 真实验证：256×256 smoke 重派生输出 `object_bbox_xyxy(71,4)`、`object_mask_centroid_uv(71,2)`；bbox 宽 9–10 px、高 11–12 px，71/71 visible centroid 均在框内。
- 分析：这些标签可训练 object detection、ROI/crop 或视觉 token 对齐，但依赖 simulator actor ID，不能作为部署期输入；真实 tracker 必须只读取 RGB-D 与标定，且在遮挡/ID-switch 上单独评估。

### 2026-07-13 13:49:42 UTC — camera-randomized EORT smoke

- 设置：新增 `PushCubeEORTCameraRand-v1`；相机使用 kinematic mount，在 reset 时对默认 eye `[0.3,0,0.6]` 作最大 `[±0.04,±0.04,±0.025]` m 扰动、target 作最大 `[±0.015,±0.015,±0.01]` m 扰动。motion planner、CPU PhysX、GPU renderer、256×256 observation 与其他 EORT 标签不变。
- 结果：GPU 4 上 `NUM_TRAJ=2`，2/2 final success；trajectory 长度 71/72 steps、RGB `(72/73,256,256,3)`。raw `extrinsic_cv` 在每条 trajectory 的所有 observation 完全相等，而两条的首帧 extrinsic 不相等；v2 sidecar 成功导出。
- 分析：这提供真实相机固定、跨 episode viewpoint shift 的最小 sim2real camera split。它不覆盖遮挡、材质/光照、相机内参或真实噪声，不能宣称视觉鲁棒；这些应作为独立因素加入，而不是把相机每帧随机化。

### 2026-07-13 13:54:45 UTC — visual-only occlusion smoke

- 设置：新增 `PushCubeEORTOccluded-v1`，以 0.5 probability 放置无 collision、kinematic 的不透明薄板；薄板只改变 render/segmentation/depth，不改变 contact、动力学、planner 或 success。采集 4 条 256×256 CPU-PhysX/GPU-render trajectories。
- 结果：4/4 final success。raw `occluder_active(T+1,1)` 在每条内恒定，sidecar action-aligned `occluder_active(T,1)` 逐元素相等；两类都出现。`active=true` 的三条 object-visible rate 为 22/71=31.0%、72/72=100%、7/74=9.5%；`active=false` 的一条为 64/64=100%。
- 分析：occluder presence 不是 object invisibility 的代理：object 可能从板旁/前方移动出来。因此训练与评测应以 `object_visible`/bbox-valid 分层，`occluder_active` 只用于诊断 sim condition，必须从模型输入中排除。该简单几何遮挡不代表真实手、杂物或 detector failure，后续仍需 ID-switch 与深度/光照噪声实验。

### 2026-07-13 14:15:43 UTC — 256×256 三分支 LeRobot 导出与训练-loader gate

- 设置：GPU 4、CPU PhysX + GPU renderer、原生 Panda motion planner；固定、camera-random 和 visual-occlusion EORT 环境各采 10 条成功轨迹，再以共享 exporter 写入 `/remote-home/jinminghao/datasets/maniskill_push_cube_eort_256_splits_lerobot_20260713T140202Z`。DiT4DiT 以新 `maniskill_eort_push_cube_256_splits_lerobot` equal-weight mixture 读取该根目录。
- 采集/导出结果：三个 split 均为 10/10 final success、686 action-aligned frames、10 Parquet、10 MP4；LeRobot metadata 的 source video feature 均为 `[256,256,3]`。fixed 和 camera-random split 均为 0 invalid frame。occlusion split 为 163/686 invalid object frames，所有这 163 行的 17D `oracle_current` 已验证为全零；因此不可见时没有连续 simulator geometry 泄漏。
- 训练接口结果：真实 loader 初始化的 component lengths 为 `[686,686,686]`，总 mixture epoch 长度 2,058；采样输出保持 `state (1,64)`、`action (8,32)`、`image[0] (3,224,224)`。最后一个形状是从 retained 256px raw image 下采样而来，非 128px 上采样。
- 分析与下一步：这闭合了三分支的 raw collection → v2 sidecar → LeRobot → training loader 验证，但样本仅 30 条单任务 rollout，不能支持训练规模、object tracker 或 sim2real 结论。下一步应在更大规模分布上加入真实/合成 detection dropout、ID-switch、深度与标定误差，并用 predicted RGB-D track、robot-only 和 oracle 三个闭环条件进行同预算比较。

### 2026-07-13 14:21:32 UTC — goal segmentation-depth geometry contract

- 设置：为避免未来 track-to-action 接口在 goal 一侧偷用 simulator position，v2 deriver 对已有 goal actor ID、raw segmentation、毫米 depth、`intrinsic_cv`、`extrinsic_cv` 使用与 object 完全相同的回投流程。
- 结果：新增 `goal_segdepth_centroid_world(T,3)`、`goal_segdepth_valid(T,1)` 和只供 QA 的 center error。2×2 fixture 验证单/双像素 goal centroid；保留的 256px occlusion raw 10 条、686 帧重派生后 goal 全部可见且有效，所有 invalid centroid 保持零。
- 分析：object 与 goal 均已有“2D mask/bbox/centroid + depth back-projected surface geometry”的同构 oracle supervision，可作为 learned detector/mask 输出的替换接口。它不应当被误称为 pose，且当前 DiT4DiT policy 不读取这些新字段；下一步仍须以 RGB-D predictor 取代 actor ID。

### 2026-07-15 UTC — object-centric overlay preview QA

- 设置：未启动仿真或 GPU，使用现有 fixed-camera 256px pilot 的 raw HDF5 和 `derived_goal_segdepth`，渲染前两条 trajectory 共 143 action-aligned RGB 帧。
- 结果：生成 `/remote-home/jinminghao/datasets/maniskill_eort_previews/push_cube_fixed_objectcentric_preview.mp4` 及 JSON summary。每帧叠加 derived object/goal bbox、mask centroid、可见性、当前 phase 与 force-contact，不显示 raw segmentation ID 或 simulator pose。
- 分析：视频确认了预览工具读取的是同一 T 对齐 sidecar，且 object 小框、goal 大框、phase/contact 文本均可辨认。这是格式/时间对齐 QA，不是新数据、模型评测或 policy rollout；大规模目标机器仍须在每个 shard 采集后检查其对应 preview。

### 2026-07-16 15:30:36 UTC — observed transition replay failure and controller target

- 设置：从一条既有 successful PushCubeEORT 轨迹的首个 env state reset，以每步当前 EEF pose 将 `eef_transition_local` 转到 Panda 原生 normalized `pd_ee_delta_pose` frame，并执行全部 71 步；CPU PhysX、GPU 6 renderer，不训练模型。
- 结果：最终 `success=false`；相对 source TCP 的 position RMSE 为 `0.08648 m`、最大 `0.11084 m`，quaternion `1-|dot|` 最大 `3.86e-5`，command 最大绝对值 `0.1144`。失败主要是位置跟踪/控制语义，不是姿态数值爆炸。
- 对照：同一 joint demonstration 经 ManiSkill 官方 conversion 产生的 71-step `pd_ee_delta_pose` 轨迹最终 success=true。新 deriver 对其输出 `panda_pd_ee_delta_pose_command(71,7)`，逐元素等于 HDF5 action，同时继续输出非 command 的 `eef_transition_local(71,7)`。
- 决策：正式 Panda policy 只使用官方 replay controller command；observed transition 留作 dynamics/auxiliary target。该结果不解决 Piper/真机 Franka 的坐标、幅值、频率、夹爪和 safety adapter。
# 2026-07-16 18:15 UTC - 大规模采集器可选择 metric task action

- `collect_large_objectcentric_v2.sh` 新增显式 `MANISKILL_EORT_ACTION_SOURCE=metric_task_delta_pose`；默认仍为已验证的 Panda normalized command。metric LeRobot 视图写入独立的 `*_metric` 根目录，不覆盖 raw、sidecar或 controller 视图。
- 对既有 Push fixed/camera-random/occluded 正式 smoke 重新派生 sidecar并导出 metric action 后，DiT4DiT 三组件 loader 长度为 `[71,65,66]`，sample shape 为 state `(1,64)`、action `(8,7)`、image `(3,224,224)`。
- 这证明数据构造与模型入口连通，不是训练结果，也不证明 Franka/Piper 可执行；真机频率、task-from-base 标定、限幅和夹爪 adapter 仍是阻塞项。
# 2026-07-16 18:45 UTC - 大规模 collection 训练前审计 gate

- 新增只读 `audit_large_collection.py`，统一检查 train/val/test × fixed/camera-random/occluded 的成功 seed、全局 seed 泄漏、sidecar 数量/trajectory ID、action provenance、visibility QA 与 LeRobot conversion contract。
- unittest 通过；真实 Push metric 三视觉 smoke 通过，汇总 3 unique simulator seeds、3 episodes、202 steps，逐分支 relational-valid 为 71/71、65/65、66/66。
- 该结果只验证 audit 与当前小样本链路；正式 500/100/200 collection 仍必须在外机完成后重新运行，不以本次 smoke 代替规模验证。

# 2026-07-16 18:30 UTC - 大规模采集默认规模复核

- 只运行 production wrapper 的 dry-run，没有启动仿真或采集。train/val/test 在未设置 `NUM_TRAJ` 时分别解析为每视觉分支 500/100/200 条，显式 `NUM_TRAJ=7` 仍覆盖默认。
- 该修改只避免误采规模；成功率、存储、visibility 分布和长时稳定性仍须由外机正式 collection 与 audit 证明。
