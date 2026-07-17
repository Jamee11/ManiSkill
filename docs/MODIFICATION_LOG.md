# Modification Log

## 2026-07-17 02:25 UTC - Persist actual object mass

- Reason: The geometry split already changes cube mass because SAPIEN uses fixed density with variable volume, but mass was absent from raw and sidecar data. Future-motion supervision would therefore be paired with an unaudited dynamics parameter.
- Change: Push/Pick EORT observations now read the actual PhysX actor mass into `obj_mass(T+1,1)`; the sidecar validates it as positive and episode-static, stores `object_mass(1,1)`, and the production audit requires explicit raw provenance. Legacy fixed/geometry recordings remain derivable from their exact default density `1000 kg/m³` and recorded extent, but cannot pass the current production audit. Mass remains QA-only and does not enter the 17D policy condition.
- Verification: Focused derivation/audit tests pass. GPU 4 production smokes used Push seed 5100 (`0.086086 kg`, 63 steps) and Pick seed 6100 (`0.072059 kg`, 76 steps): source motion planning and controller replay were each 1/1 successful, raw mass arrays matched exactly source→controller and matched volume × default density, both production audits and metric exports passed. No training was started.

## 2026-07-17 02:10 UTC - Extend the isolated geometry/friction split to PickCube

- Reason: Push-only physical variation does not exercise the grasp-sensitive PickCube path, where object size and contact friction directly affect acquisition and transport.
- Change: Added optional `PickCubeEORTGeometryRand-v1` with the same seeded half-size `[0.017,0.023]m`, shared static/dynamic friction `[0.15,0.60]`, one-environment and per-episode scene-reconfiguration constraints as the verified Push split. Registered the existing Pick motion planner, sidecar schema, production collector and audit. Default fixed/camera/occluded collection and DiT4DiT policy inputs remain unchanged.
- Verification: Python/shell syntax and focused registration/audit tests pass. GPU 4 collected seeds 4100–4102; source motion planning and controller replay were 3/3 successful over 208 steps, with exact source/controller extent+friction equality. Production audit and the isolated DiT4DiT metric loader passed with `(1,64)/(8,7)/(3,224,224)` samples. No training was started.

## 2026-07-16 19:33 UTC - Add an isolated seeded geometry/friction split

- Reason: Fixed 4 cm/0.3-friction cubes cannot test whether object-centric dynamics generalize across basic physical variation. Scene-load geometry is absent from ordinary env state, so the split must reconfigure from the recorded episode seed and persist its actual parameters.
- Change: Added optional `PushCubeEORTGeometryRand-v1`, limited to one environment, with `reconfiguration_freq=1`, half-size `[0.017,0.023]m` and shared static/dynamic friction `[0.15,0.60]`. All Push/Pick EORT raw observations now persist friction; sidecars and production audit require explicit `(1,2)` provenance. The existing three variants and current 17D DiT4DiT condition are unchanged.
- Verification: Focused derivation/audit tests and syntax checks passed. GPU 7 collected three seeds; source motion planning and controller replay were 3/3 successful over 205 steps. Source/controller extent and friction arrays matched exactly per trajectory, the production audit passed, and DiT4DiT loaded the metric output as one 205-step component with `(1,64)/(8,7)/(3,224,224)` sample shapes. No training was started.

## 2026-07-16 19:18 UTC - Persist object extent before domain randomization

- Reason: The cross-embodiment schema declared object extent, but current raw/sidecar data omitted it. Adding size randomization first would therefore create an unobservable data-construction change and make replay/QA ambiguous.
- Change: Push/Pick EORT observations now record full cube side lengths as `obj_extent(T+1,3)`. The v2 sidecar requires positive episode-static values and writes `object_extent(1,3)` plus provenance; the production audit requires the explicit raw source. Historical Panda EORT raw remains derivable through its exact fixed 4 cm task constant, but that fallback cannot pass the production audit. The current DiT4DiT policy condition is unchanged.
- Verification: Focused v2 derivation and collection-audit tests pass. A retained real 71-step Push controller HDF5 without the new raw field re-derived `object_extent(1,3)=[0.04,0.04,0.04]` with explicit legacy provenance. Then GPU 7 ran the production wrapper on one new fixed Push episode: 1/1 source and controller replay success, raw `(72,3)` and sidecar `(1,3)` extents aligned, all three metric LeRobot exports and the collection audit passed. No training was started.

## 2026-07-16 19:10 UTC - Add an offline real-robot calibration gate

- Reason: The verified Panda metric action still lacked an auditable boundary before a Franka/Piper driver. Local source inspection found only a StarVLA interface placeholder and a UniVLA deployment node tied to a specific ROS/IK/eye-hand setup; no reusable Piper SDK was present.
- Change: Added a read-only CLI that requires measured robot calibration, rotates task-axis metric commands into robot-base axes, maps normalized gripper openness to declared native values, and rejects per-step translation/rotation or gripper-range violations. It imports no robot SDK, sends no command, refuses to overwrite a report, and always reports that hardware execution is unvalidated.
- Verification: Three focused unit tests cover a valid rotated command, all three violation classes, and rejection of an invalid calibration rotation. No robot, simulation, data collection, or training was started.

## 2026-07-16 19:05 UTC - Persist and audit the collection runtime

- Reason: Controller metadata recorded the source commit and environment configuration but not the target machine's Python/Torch/SAPIEN versions, leaving an external collection impossible to reproduce exactly.
- Change: After a shard completes all exports, the existing wrapper writes one `runtime_manifest.json` with Python, ManiSkill, Torch, SAPIEN, NumPy, h5py and the explicit renderer GPU. The read-only audit requires complete manifests and one shared Python/package runtime across requested shards. Raw HDF5, simulator behavior and training inputs are unchanged.
- Verification: Shell/Python syntax, wrapper dry-run and focused audit regression pass. The local environment resolves Python 3.10.20, ManiSkill 3.0.1, Torch 2.7.1+cu128, SAPIEN 3.0.3, NumPy 1.26.4 and h5py 3.16.0. No collection or training was launched.

## 2026-07-16 18:57 UTC - Gate interaction-phase coverage before training

- Reason: Push/Pick phase labels were persisted per frame but absent from shard summaries, so a large collection could pass visibility/action checks while containing no useful push-motion or grasp interval.
- Change: The existing v2 summary now reports task-specific phase 0/1/2/3 counts plus interaction/progress/success totals. The read-only collection audit requires counts to cover every step and each shard to contain phase 2 task interaction and phase 3 success. Labels remain simulator-only QA targets and are not added to policy input.
- Verification: Focused derivation/audit tests pass; fresh read-only derivations of retained Push and Pick controller trajectories report phase histograms without simulation or training.

## 2026-07-16 18:45 UTC - Audit simulator source and environment provenance

- Reason: Controller records already contained commit/environment metadata, but the training preflight did not reject a collection assembled from different ManiSkill revisions or an unexpected environment/control backend.
- Change: The existing read-only audit now verifies each controller JSON's environment ID, RGB-D-segmentation observation mode, `pd_ee_delta_pose`, `physx_cpu`, successful episode seeds, and nonempty source commit; all requested shards must share one commit. No new manifest or collection output was added.
- Verification: The focused unittest accepts a consistent fixture and rejects mixed commits; a read-only inspection of the retained three-variant Push controller metadata reports one commit (`5d369b5`), the three expected environment IDs/backends and matching seeds. The older retained sidecar summaries predate the current visibility-QA schema, so they do not pass the entire current audit and are not presented as production data. No data or training was launched.

## 2026-07-16 18:40 UTC - Require an explicit renderer GPU for production collection

- Reason: The host CPU renderer crashes and default GPU 0 previously OOMed, but the production wrapper still silently inherited the ambient CUDA device when its explicit selector was omitted.
- Change: Real collection now exits before creating a shard unless `MANISKILL_EORT_CUDA_VISIBLE_DEVICES` is nonempty. `DRY_RUN=1` remains side-effect-free and may report `gpu=unset`; physics, renderer, data schema, split sizes and exporters are unchanged.
- Verification: Shell syntax passed; dry-runs with unset and explicit GPU selectors passed, while a non-dry invocation without the selector exited with status 2 and created no collection root. No simulation, data collection or training was launched.

## 2026-07-16 18:30 UTC - Align collection defaults with the approved split plan

- Reason: The documented first production pass is 500/100/200 successful episodes per visual variant for train/val/test, but the wrapper defaulted every split to 1000. Omitting one environment variable could therefore launch 9,000 episodes instead of the planned 2,400.
- Change: The existing wrapper now chooses `NUM_TRAJ=500/100/200` from the selected split when the variable is unset. An explicit `NUM_TRAJ` still overrides the default; seed blocks, output layout, data schema, physics, renderer, exporter and audit are unchanged.
- Verification: Shell syntax passed; dry-runs for train/val/test reported 500/100/200, and an explicit `NUM_TRAJ=7` override remained effective. No data was collected.

## 2026-07-16 20:00 UTC - Add an isolated future-object supervision export

- Reason: h=1/4/8 future object labels existed only in sidecars. DiT4DiT needs an explicit auxiliary-target view, while default policy datasets must continue to exclude future truth.
- Change: Added `MANISKILL_EORT_INCLUDE_FUTURE_TARGETS=1`, valid only with metric actions. It passes the opt-in exporter flag and writes collision-free `*_metric_dynamics` roots. The read-only collection audit now verifies the future-target manifest bit. Default collection behavior is unchanged.
- Verification: Shell syntax/dry-run and audit unittest passed. A real three-variant Push dynamics view passed export, loader and audit with 3 episodes/202 steps/3 unique seeds. No training was launched.

## 2026-07-16 17:45:00 UTC — 可逆 metric task action 契约

- 原因：现有 policy target 是 Panda normalized root-frame command，无法直接用于 Franka/Piper；observed EEF transition 又已被真实 replay 证明不可执行。
- 精确变更：新增独立 `action_contract.py`，按 ManiSkill Panda 控制器真实 scale/frame 将 normalized action 可逆转换为 `delta_xyz_m + delta_rotvec_rad + gripper_open_fraction`，并支持由标定旋转变换 task axes。v2 sidecar/manifest 在保留原 Panda command 的同时新增 `metric_task_delta_pose_command`。未改变物理、原 action、环境、future/phase 标签或已有训练默认值。
- 验证：六项 EORT 测试通过；Push/Pick 六条 smoke 的 rotation norm 均小于 1。固定 Push 71 步反编码最大误差 `1.49e-8`，GPU7 真实 replay 1/1 final success。未启动训练，未宣称真机适配完成。

## 2026-07-16 16:42:08 UTC - Persist shard-level relational visibility QA

- Reason: The PickCube occluded smoke kept the object visible but hid the goal for every frame, making the deployable object-to-goal proxy invalid while object-only visibility looked healthy. The production wrapper previously required manual NPZ inspection to detect this failure mode.
- Change: Reused the existing v2 deriver `summary.json` and existing sidecar arrays to add total, object-visible, goal-visible, jointly visible, jointly depth-valid, object/goal mask-pixel min/median/max, and optional occluder-active step counts. No new script, schema field, dependency, or policy input was added.
- Scope: Additive QA metadata only. Raw HDF5, per-trajectory NPZ, controller action, physics, environment, LeRobot fields, model, and training behavior are unchanged.
- Verification: The existing deriver suite passed 5/5 with exact synthetic QA assertions. Fresh derivation of real occluded controller HDF5 reported Push relational-visible/depth-valid `66/66` and Pick `0/72`, matching direct NPZ/Parquet inspection. No training was launched.

## 2026-07-16 15:58:07 UTC - Enable the isolated environment for DiT4DiT websocket evaluation

- Reason: The Panda closed-loop evaluator must run inside `maniskill-eort-v1`, while its DiT4DiT policy client uses the existing websocket/msgpack protocol. Those two runtime packages were absent from the isolated simulator environment.
- Change: Installed and verified `websockets==16.0` and `msgpack==1.1.2` in `maniskill-eort-v1`; recorded the versions in `docs/思考与隐患.md`.
- Scope: Environment dependencies only. No ManiSkill task, physics, rendering, collection schema, controller, or source code changed.
- Verification: Both packages and `websockets.sync.client` import successfully. The same environment completed the new 63-step PushCube closed-loop replay smoke with 256px video.

## 2026-07-16 15:30:36 UTC — Build replay-verified Panda controller-command data path

- 原因：真实 71-step replay 证明 `eef_transition_local` 经换帧后仍无法复现成功轨迹，TCP 位置 RMSE 8.65 cm、最大 11.08 cm；它不能继续作为正式 action-policy 主目标。
- 精确变更：object-centric v2 deriver 现在读取 source metadata control mode；对官方成功 replay 的 `pd_ee_delta_pose` HDF5，额外持久化逐元素等于 source action 的 `panda_pd_ee_delta_pose_command(T,7)` 和明确 manifest。大规模 wrapper 先采 joint demo，再调用 ManiSkill 官方 replay，拒绝任何缺失/部分成功的 controller shard，复制同一 seed manifest 供 tracker 按物理 seed 分组，随后从 controller HDF5 派生、预览并导出 DiT4DiT。旧 `pd_joint_pos` 和 `eef_transition_local` 路径保留。
- 风险约束：该 command 仅对 ManiSkill Panda 成立；Piper 与真机 Franka 尚无 adapter/replay 证据。PickCube visual/controller replay 仍需单独 GPU smoke。没有启动训练。
- 验证：合成 controller sidecar 单测通过；现有真实成功 controller HDF5 派生出 71 行 transition 与 71 行 controller command，manifest 明确区分二者；DiT4DiT 端已完成 exact export 和三组件 loader gate。

## 2026-07-16 15:18:24 UTC — Fix PushCube EORT state-only runtime

- 原因：闭环 action replay 使用 `obs_mode=state_dict` 构造 `PushCubeEORT-v1` 时，`_get_obs_extra()` 调用 `torch.stack`，但任务模块遗漏了 `torch` 导入，环境在 reset 阶段报 `NameError`。
- 精确变更：仅在 `push_cube_eort.py` 补充现有实现所需的 `import torch`；不改变 observation schema、physics、reward、success、controller 或数据。
- 验证：环境可完成 reset 与 71-step replay；该 replay 最终失败，证明后续 blocker 是 action 语义而非此 import bug。详细结果记入实验分析。

## 2026-07-15 UTC - Add non-overwriting large-scale EORT collection pipeline

- Reason: The prior 30 rendered PushCube pilot trajectories only smoke-tested the object-centric data path.  A reproducible external-GPU collection command is needed before tracker or DiT4DiT policy training can be scheduled, while preserving the raw simulator record as the source of truth.
- Change: Added an opt-in collection wrapper that produces disjoint fixed/camera-random/occluded PushCube (or, after its required smoke, PickCube) shards, then derives the existing object-centric v2 sidecar, emits an annotated 256px RGB QA preview, and exports separate LeRobot oracle/proxy/corrupted-track training views using the existing DiT4DiT converter.  Extended the official Panda motion-planning runner with explicit start seeds and an adjacent successful-trajectory seed manifest.
- Scope: Additive collection, provenance, and visualization support only.  Raw HDF5 plus v2 sidecar remains authoritative; no environment observation, label schema, policy input, model architecture, controller command, existing dataset, or default training configuration was replaced.
- Safety: Outputs reject pre-existing shard, derived, preview, and converter destinations.  Split/variant seed blocks are disjoint.  The wrapper keeps CPU physics with the existing GPU renderer selection mechanism; it does not attempt CPU rendering or automatic GPU selection.
- Verification: Shell syntax and dry-run planning checks passed; the preview unit test passed; an existing 256px PushCube pilot rendered a 143-frame annotated MP4.  This is not a target-GPU batch collection smoke and does not clear the PickCube visual-split prerequisite.

## 2026-07-15 UTC — Shared visual generalization variants for PickCube EORT

- 原因：仅固定相机的 PickCube 无法作为与 PushCube 对称的跨任务泛化/感知鲁棒性评测；camera shift 和 visual-only occlusion 必须保持不影响原生物理与 motion planner。
- 精确变更：将已验证的 PushCube per-episode camera-random 和 non-colliding occluder 抽为共享 mixin；PushCube 的注册 ID 和默认参数不变。新增 `PickCubeEORTCameraRand-v1` 与 `PickCubeEORTOccluded-v1`，映射到原生 `solvePickCube`，并注册 Pick task-specific schema/phase 导出支持。
- 风险约束：新增 Pick variants 尚未 GPU smoke；`occluder_active` 继续只作 QA、不进入 policy，camera random 固定于 episode 内；不采集、不训练、不修改 physics/reward/success。

## 2026-07-15 UTC — PickCube EORT task-specific collection scaffold

- 原因：PushCube 只覆盖接触推动，无法验证 object-centric 表示在“接近—双指抓取—搬运—放置”的接触阶段是否仍有用；PickCube 是保留 Panda 原生 motion planner 的最小第二任务。
- 精确变更：新增 `PickCubeEORT-v1`，不改变 PickCube 物理、reward、success 或 controller；只以 256×256 sensor 保存 cube/goal actor ID、cube 线/角速度、Panda hand/finger 实测接触力及 `(T+1,1)` native `is_grasped` 标签。goal marker 仅在这个 EORT variant 的 sensor 中可见以支持 goal mask/depth QA。派生器按 task 导出独立 `pick_interaction_phase`（approach/contact/native grasp/native success），并为 PickCube 使用独立 schema version；未来 object delta 仍仅作辅助监督。采集入口新增可覆写的 trajectory name，默认 PushCube 行为保持不变。
- 风险约束：`is_grasped`、actor ID、接触与 future delta 均为 simulator oracle，绝不进入 action policy；`eef_transition_local` 仍是 observed transition 而非真机 command；不在当前繁忙 GPU 采集或训练。

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

## 2026-07-16 — backup backbone readiness 更新

- 原因：X-WAM 官方已在初次审计后发布 checkpoint，需要区分“长期结构匹配”与“当前最快可切换”两种 baseline 排序。
- 精确变更：未改模型、训练、数据或部署代码；更新跨机器人契约、实验分析和风险台账，记录 X-WAM checkpoint 总量约 117 GB 但本地未下载、τ₀-WM 本地已有约 21 GB 权重与 7D 单臂 downstream 入口、DreamDojo 仍无直接 Panda 7D action-policy 输出。
- 决策：DiT4DiT 保持主线；若其闭环 gate 失败，近期优先切 τ₀-WM；X-WAM 在权重下载及 forward gate 通过后作为 Franka/Piper 结构最匹配的中期候选。没有新增无法验证的适配器。

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

## 2026-07-13 13:24:00 UTC — PushCube interaction phase 与原生 success 对齐

- 原因：10 条最终成功轨迹的 `phase=3` 为零；原实现以 `goal_progress>=0.999` 判终态，比 PushCube 原生成功容差严格，造成语义不一致。
- 精确变更：v2 导出器仅将 terminal phase 改为使用已有的 action-aligned raw `success(T)`；phase 3 优先于接触/运动阶段。未改变物理、成功条件、进度字段、future 标签或 policy 输入字段。
- 验证：离线 v2 fixture 末步 native success 现在导出 phase 3，前两步仍为 approach/moving-contact；10-trajectory pilot 已重新派生至独立 `derived_success_phase`，逐帧断言 `phase==3` 等于 raw `success`，通过。

## 2026-07-13 13:31:00 UTC — EORT 原始视觉分辨率提升至 256×256

- 原因：128×128 raw RGB-D/segmentation 经过 DiT4DiT 的 224×224 resize 只能插值，不能恢复 object 边界、纹理或深度细节；这不适合后续 object tracker。
- 精确变更：仅 `PushCubeEORT-v1` 覆盖 inherited `base_camera` 的 width/height 为 256。保留相机位姿、FOV、物理、奖励、原始 PushCube-v1 和既有 128×128 pilot 不变；训练端仍可按 encoder 要求下采样。
- 验证门槛：下次 GPU renderer smoke 必须确认 raw RGB/depth/segmentation 为 `(T+1,256,256,3/1/1)`，并重新记录显存与 object-pixel QA；本次不在繁忙 GPU 上启动新采集。

## 2026-07-13 13:33:54 UTC — 256×256 EORT 真实采集 smoke

- 原因：需实际确认分辨率改动经 GPU renderer、原生 motion planner、HDF5 与 v2 sidecar 后仍完整，而不是只通过语法检查。
- 精确变更：未再修改代码；GPU 4 上使用新的不可覆盖目录采集 `NUM_TRAJ=1`，CPU PhysX + GPU renderer，随后运行既有 v2 派生器。
- 证据：1/1 最终 success、`T=71`；raw RGB/depth/segmentation 均为 `(72,256,256,3/1/1)`；object mask 90–117 pixels（均值 95.8）、71/71 visible、71/71 segmentation-depth valid。GPU 4 采集前后 `nvidia-smi` 均报告 63,782 MiB used，且无 OOM；该数值不是峰值显存。

## 2026-07-13 13:36:00 UTC — 256×256 到 DiT4DiT 训练输入链路验证

- 原因：仅 raw HDF5 为高分辨率不足以证明训练没有重新上采样或错误读取视频，必须检查 exporter metadata 和实际 loader sample。
- 精确变更：未修改 DiT4DiT 代码；对 256×256 smoke 运行已有专用 exporter，并用既有 `ManiSkillEORTPushCubeOracleDataConfig` 读取一个 sample。
- 证据：LeRobot video feature 为 `[256,256,3]`、20 FPS、71 Parquet rows 和 1 个 MP4；DiT4DiT dataset 初始化成功，实际 `image` tensor 为 `(3,224,224)`。这是 encoder 前的下采样，而非把 128×128 放大。

## 2026-07-13 13:40:00 UTC — EORT 2D tracker supervision

- 原因：actor ID、可见率与 3D oracle pose 足以审计数据，但没有直接给视觉 tracker 的 2D detection/crop supervision；将整张 segmentation mask 再写入 NPZ 会重复 raw HDF5。
- 精确变更：v2 sidecar 新增 object/goal 的 `bbox_xyxy(T,4)` 与 `mask_centroid_uv(T,2)`。bbox 采用 `[x_min,y_min,x_max_exclusive,y_max_exclusive]`；不可见时 bbox/centroid 为零，已有 `visible(T,1)` 是唯一有效性判据。raw segmentation 继续是 mask 的唯一来源。manifest phase-3 文案同步为 `native_task_success`。
- 验证：2×2 synthetic fixture 覆盖多像素、单像素和完全遮挡 bbox/centroid；256×256 real smoke 重派生后，object bbox 为 9–10 px 宽、11–12 px 高，所有 visible centroid 均位于对应 bbox 内。

## 2026-07-13 13:49:42 UTC — 固定 episode 的相机外参随机化

- 原因：固定相机的无障碍 PushCube 数据无法检验 tracker 是否依赖单一外参；逐控制步相机抖动又不符合真实静态外部相机。
- 精确变更：新增独立 `PushCubeEORTCameraRand-v1`。它复用 PushCubeEORT 的物理、奖励、256×256 sensor 与 EORT fields，仅把 `base_camera` 挂到 kinematic mount，并在 reset 时对 eye/target 作有限均匀扰动；同一 episode 内 pose 不变。Panda motion-planning runner 和 collection script 接受该 env ID；v2 deriver 显式允许两个 EORT env ID。
- 验证：GPU 4 上 2/2 原生 motion-planning 成功、raw RGB 均为 256×256；逐 trajectory 断言 extrinsic 在 episode 内恒定且两条之间不同。未增加物理遮挡物、纹理或逐帧 camera motion。

## 2026-07-13 13:54:45 UTC — visual-only occlusion EORT split

- 原因：camera shift 不能产生 object 的漏失/部分可见状态，无法评估 tracker 在遮挡下的 valid-mask 行为。
- 精确变更：新增独立 `PushCubeEORTOccluded-v1`。它以 0.5 概率放置无碰撞、kinematic、不透明薄板于相机与工作区之间；原物理、motion planner 和 task success 不变。raw/sidecar 新增 `occluder_active(T+1/T,1)`，manifest 显式 `policy_input=false`。Panda runner 与 v2 deriver 允许该环境。
- 验证：GPU 4 4/4 success；active/inactive 均出现。active episode 的 object visible fraction 为 9.5%、31.0%、100%，inactive 为 100%，表明该 split 提供实际部分/完全遮挡候选但必须按最终 `object_visible` 而非 active flag 分层。synthetic test 覆盖 flag 的 T+1→T 对齐和 manifest 禁入 policy。

## 2026-07-13 14:15:43 UTC — 256×256 EORT 三分支数据集与 DiT4DiT mixture 验证

- 原因：单条 high-resolution smoke 及单一无障碍相机视角不足以验证 object-centric 链路在 viewpoint shift 和自然不可见帧下的导出契约；训练端也需要显式、可复现地同时读取三种条件。
- 精确变更：未修改 ManiSkill 环境、物理、控制或标签代码；在 GPU 4 上以 CPU PhysX + GPU renderer 分别采集固定相机、reset-static 相机随机化和 visual-only occlusion 三个 `NUM_TRAJ=10` raw/derived split。使用已有 LeRobot exporter 导出至共享根目录，并在 DiT4DiT 添加等权 named mixture `maniskill_eort_push_cube_256_splits_lerobot`。保留全部旧输出，不覆盖 128px pilot。
- 数据与 QA：三个 split 各为 10/10 native-success episodes、686 action-aligned frames、10 个 Parquet 和 10 个 MP4；stored front video feature 都是 `[256,256,3]`。occlusion split 有 163/686 `object_visible=false` 帧，逐帧检验这些帧的 17D continuous oracle condition 均为零；fixed/camera-random split 无不可见帧。camera split 的外参在 episode 内静态、跨 episode 变化；occluder flag 仍只作诊断而不作模型输入。
- 训练接口验证：真实 DiT4DiT loader 初始化三组长度 `[686,686,686]`，mixture epoch 为 2,058 frames，并输出 `state (1,64)`、`action (8,32)` 和由 256px 下采样的 `image (3,224,224)`。
- 边界：这只是 30 条 PushCube 的数据/加载器 gate，visible-frame geometry 仍是 simulator oracle upper bound，未训练或闭环评测；不得据此声称 tracker、sim2real 或遮挡鲁棒性。

## 2026-07-13 14:21:32 UTC — goal RGB-D 几何 tracker supervision

- 原因：现有 sidecar 只提供 object 的 segmentation-depth visible-surface centroid；后续非-GT track 要计算 object-to-goal relation 时，goal 也必须有同一观测几何契约，不能退回 simulator goal position。
- 精确变更：v2 deriver 现在对 goal actor 复用既有 mask/depth/intrinsic/extrinsic back-projection，新增 `goal_segdepth_centroid_world(T,3)`、`goal_segdepth_valid(T,1)` 与相对于现有 `goal_pos` 的 `goal_segdepth_centroid_error(T,1)`。未修改环境、raw HDF5、policy export 或任何 action input；字段仍显式为 oracle actor-mask supervision。
- 验证：2×2 synthetic test 覆盖可见 goal 的 centroid/valid；在保留的 256px occlusion raw HDF5 上重派生到独立目录，10 条/686 帧均得到有限 `(T,3)` goal centroid 和 valid=true，无效 centroid 非零数为零。旧 derived 目录保持不变。
- 边界：depth centroid 是可见表面，不是物体/目标几何中心，更不是 learned detection；它只建立日后 detector/mask 输出可替换的输入契约。
# 2026-07-16 18:15 UTC - 大规模采集入口支持独立 metric action 视图

- 原因：Panda normalized command 可逆解码已验证，但正式大规模 wrapper 仍只能生成 robot-specific LeRobot action，无法直接排跨机器人 action gate。
- 改动：为大规模采集脚本增加显式 `MANISKILL_EORT_ACTION_SOURCE`，支持默认 Panda command 或 metric task delta pose；metric 数据自动写入独立的 `*_metric` LeRobot 根目录，raw、sidecar、视频与已有 controller 数据均不替换。
- 验证：shell 语法和 dry-run 通过；真实三视觉分支 metric 数据已由 DiT4DiT loader 读为 `[71,65,66]`，sample shape 为 `(1,64)/(8,7)/(3,224,224)`。未启动训练。
# 2026-07-16 18:45 UTC - Add a fail-fast collection audit before training

- Reason: Collection shards already carried seed, sidecar, visibility and LeRobot metadata, but no single command proved that all train/val/test inputs were complete, disjoint and action-compatible before an expensive training run.
- Change: Added one read-only stdlib audit that checks successful unique simulator seeds, seed-to-sidecar identity/count, verified action provenance, relational visibility QA, and LeRobot episode/condition/action contracts across requested splits and variants. It does not inspect or change model architecture, raw data or training code.
- Verification: The focused unittest passed, including intentional train/val seed leakage rejection. The real three-variant Push metric smoke passed with 3 unique seeds, 3 episodes and 202 steps. No training was launched.
