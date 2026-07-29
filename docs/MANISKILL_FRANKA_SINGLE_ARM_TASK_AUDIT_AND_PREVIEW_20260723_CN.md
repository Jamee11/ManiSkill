# ManiSkill3 Franka 单臂任务审计与三案例预览

日期：2026-07-23
代码仓库：`/remote-home/jinminghao/WAMs/ManiSkill`
预览输出：`/remote-home/jinminghao/datasets/maniskill_task_preview_20260723`

## 1. 目的与结论

本次工作回答两个不同问题：

1. 当前 ManiSkill 注册表里有多少 Franka/Panda 相关环境？
2. 其中多少任务已经有可直接运行的成功 expert rollout，可作为后续 EORT/object-centric 数据采集候选？

核心结论：

- 当前本地 ManiSkill 共注册 `84` 个 environment ID。
- 普通 `panda`/`panda_wristcam` 相关 ID 有 `36` 个，另外有 `4` 个 `panda_stick` ID。
- `36` 个普通 Panda ID 不等于 36 种独立任务：其中包含 `10` 个 Push/Pick EORT 变体、`9` 个 SceneManipulation 场景注册项和 `17` 个明确命名的普通夹爪任务。
- 17 个明确任务中，当前官方 Panda motion-planning runner 已覆盖并实际成功生成视频的有 `10` 个。
- 这 10 个任务各保留 3 条成功 trajectory 和 3 个 MP4，共 `30` 条成功预览。
- 其余环境能注册或 reset，不代表已有成功 demonstration；不能用随机动作视频替代 expert capability 证据。

## 2. 为什么“30 多个 Franka ID”不等于“30 多种可采集任务”

### 2.1 十个 EORT ID 只对应两种任务

```text
PushCubeEORT-v1
PushCubeEORTCameraRand-v1
PushCubeEORTGeometryRand-v1
PushCubeEORTOccluded-v1
PushCubeEORTSim2Real-v1

PickCubeEORT-v1
PickCubeEORTCameraRand-v1
PickCubeEORTGeometryRand-v1
PickCubeEORTOccluded-v1
PickCubeEORTSim2Real-v1
```

它们分别是 PushCube/PickCube 的相机、遮挡、几何或 Sim2Real 配置变体，不是十种新的操作语义。本次任务预览按语义去重，没有重复生成每个 EORT 变体。

### 2.2 九个 SceneManipulation ID 是场景配置

```text
SceneManipulation-v1
ArchitecTHOR_SceneManipulation-v1
ReplicaCAD_SceneManipulation-v1
ReplicaCADPrepareGroceriesTrain_SceneManipulation-v1
ReplicaCADPrepareGroceriesVal_SceneManipulation-v1
ReplicaCADSetTableTrain_SceneManipulation-v1
ReplicaCADSetTableVal_SceneManipulation-v1
ReplicaCADTidyHouseTrain_SceneManipulation-v1
ReplicaCADTidyHouseVal_SceneManipulation-v1
```

这些 ID 主要选择大型场景或数据 split，不等价于九个已经带确定 object/goal、语言指令和 expert solver 的任务。

## 3. 十七个普通 Franka 夹爪任务

| 任务 | 当前官方 Panda solver | 本次 3 条成功视频 | 对当前 object-centric 表示的主要要求 |
| --- | --- | --- | --- |
| `AssemblingKits-v1` | 无 | 未生成 | 多物体、装配槽位、阶段切换 |
| `FMBAssembly1Easy-v1` | 无 | 未生成 | 多物体、多阶段、精密装配 |
| `LiftPegUpright-v1` | 有 | 3/3 | 位置 + orientation/upright |
| `PegInsertionSide-v1` | 有 | 3/3 | peg/hole 位置与方向、精密接触 |
| `PickClutterYCB-v1` | 无 | 未生成 | 多物体身份、遮挡、目标选择 |
| `PickCube-v1` | 有 | 3/3 | 单 object→goal，已进入 EORT 主线 |
| `PickSingleYCB-v1` | 无 | 未生成 | 类别/实例泛化、复杂几何 |
| `PlaceSphere-v1` | 有 | 3/3 | 单 object→container/goal |
| `PlugCharger-v1` | 有 | 3/3 | 插头/插座位置与 6D orientation |
| `PokeCube-v1` | 无 | 未生成 | 接触点、推动方向 |
| `PullCube-v1` | 有 | 3/3 | 单 object→goal，和 Push 互补 |
| `PullCubeTool-v1` | 有 | 3/3 | tool→object→goal，多阶段接触 |
| `PushCube-v1` | 有 | 3/3 | 单 object→goal，已进入 EORT 主线 |
| `RollBall-v1` | 无 | 未生成 | 接触动力学、滚动方向 |
| `StackCube-v1` | 有 | 3/3 | manipuland + target cube |
| `StackPyramid-v1` | 有 | 3/3 | 多物体、目标顺序、阶段切换 |
| `TurnFaucet-v1` | 无 | 未生成 | articulated joint、交互区域、角度进度 |

“无 solver”表示当前
`mani_skill/examples/motionplanning/panda/run.py::MP_SOLUTIONS`
没有对应入口，不表示环境一定不能 reset，也不表示任务在理论上无法由 Franka 完成。

## 4. PandaStick 任务

| 任务 | 状态 |
| --- | --- |
| `PushT-v1` | 当前 runner 无 solver |
| `DrawSVG-v1` | 有 solver，但当前环境缺少 `svgpathtools`，无法实例化 |
| `DrawTriangle-v1` | 有 solver，但调用当前 `BaseMotionPlanningSolver` 时参数签名失配 |
| `TableTopFreeDraw-v1` | 无明确 success condition，不适合作为成功 demonstration 任务 |

这些任务使用 Franka 本体但末端是 stick，不属于当前双指夹爪 sim-to-real 主线。

## 5. 三案例实验协议

### 5.1 通用命令

```bash
cd /remote-home/jinminghao/WAMs/ManiSkill

CUDA_VISIBLE_DEVICES=<gpu> \
PYTHONNOUSERSITE=1 \
/remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python \
  -m mani_skill.examples.motionplanning.panda.run \
  -e <env_id> \
  -n 3 \
  --only-count-success \
  --max-attempts <budget> \
  --start-seed <seed> \
  --save-video \
  --save-seed-manifest \
  --obs-mode none \
  --sim-backend cpu \
  --record-dir /remote-home/jinminghao/datasets/maniskill_task_preview_20260723 \
  --traj-name preview_3cases
```

协议含义：

- CPU PhysX + 可见 GPU renderer；没有强制 CPU renderer。
- `--only-count-success`：失败 attempt 不写入最终三条 demonstration。
- `--max-attempts`：防止 solver 无限重试。
- 每个任务使用独立 seed block，避免案例重叠。
- 输出使用新目录；没有删除或覆盖已有 EORT 数据。

### 5.2 Seed 与结果

| 任务 | start seed | attempted | retained success | failed motion plan |
| --- | ---: | ---: | ---: | ---: |
| `PickCube-v1` | 9,100,000 | 3 | 3 | 0 |
| `StackCube-v1` | 9,110,000 | 3 | 3 | 0 |
| `PlaceSphere-v1` | 9,120,000 | 5 | 3 | 0 |
| `PullCube-v1` | 9,130,000 | 3 | 3 | 0 |
| `LiftPegUpright-v1` | 9,140,000 | 3 | 3 | 0 |
| `PegInsertionSide-v1` | 9,150,000 | 3 | 3 | 0 |
| `PlugCharger-v1` | 9,160,000 | 5 | 3 | 0 |
| `PullCubeTool-v1` | 9,170,000 | 4 | 3 | 1 |
| `StackPyramid-v1` | 9,180,000 | 3 | 3 | 0 |
| `PushCube-v1` | 9,190,000 | 3 | 3 | 0 |

本次 attempt 数很少，只能证明“当前链路能生成三个成功案例”，不能用于估计稳定成功率。

## 6. 输出格式与视频

每个成功任务目录为：

```text
/remote-home/jinminghao/datasets/maniskill_task_preview_20260723/
  <EnvID>/motionplanning/
    0.mp4
    1.mp4
    2.mp4
    preview_3cases.h5
    preview_3cases.json
    preview_3cases.seed_manifest.json
```

视频审计结果：

```text
count      = 30
codec      = H.264
pixel fmt  = yuv420p
resolution = 512×512
fps        = 30
```

任务目录：

- [PickCube](/remote-home/jinminghao/datasets/maniskill_task_preview_20260723/PickCube-v1/motionplanning)
- [PushCube](/remote-home/jinminghao/datasets/maniskill_task_preview_20260723/PushCube-v1/motionplanning)
- [StackCube](/remote-home/jinminghao/datasets/maniskill_task_preview_20260723/StackCube-v1/motionplanning)
- [PlaceSphere](/remote-home/jinminghao/datasets/maniskill_task_preview_20260723/PlaceSphere-v1/motionplanning)
- [PullCube](/remote-home/jinminghao/datasets/maniskill_task_preview_20260723/PullCube-v1/motionplanning)
- [PullCubeTool](/remote-home/jinminghao/datasets/maniskill_task_preview_20260723/PullCubeTool-v1/motionplanning)
- [LiftPegUpright](/remote-home/jinminghao/datasets/maniskill_task_preview_20260723/LiftPegUpright-v1/motionplanning)
- [PegInsertionSide](/remote-home/jinminghao/datasets/maniskill_task_preview_20260723/PegInsertionSide-v1/motionplanning)
- [PlugCharger](/remote-home/jinminghao/datasets/maniskill_task_preview_20260723/PlugCharger-v1/motionplanning)
- [StackPyramid](/remote-home/jinminghao/datasets/maniskill_task_preview_20260723/StackPyramid-v1/motionplanning)

## 7. 两个失败的现成 solver

### 7.1 DrawSVG

环境实例化阶段报错：

```text
ModuleNotFoundError: No module named 'svgpathtools'
```

本次没有为一个低优先级绘图任务改变当前 EORT 环境依赖。

### 7.2 DrawTriangle

30 个 attempt 均在 solver 构造阶段失败：

```text
BaseMotionPlanningSolver.__init__() takes from 2 to 8 positional arguments
but 9 were given
```

这是 solver 与当前 planner API 的兼容性问题，不是任务难度或 motion planning 成功率。未在没有单独确认的情况下修改 solver 核心逻辑。

## 8. 对 object-centric sim-to-real 的初步筛选

### 第一优先级：最小改动扩展

```text
StackCube-v1
PlaceSphere-v1
PullCube-v1
```

原因：

- 已有成功 expert solver；
- object/goal 定义清楚；
- 和当前 Push/Pick 的单物体关系表示最接近；
- 可以优先复用相机、robot-base action、visibility 和 tracker 数据框架。

### 第二优先级：需要 orientation 表示

```text
LiftPegUpright-v1
PegInsertionSide-v1
PlugCharger-v1
```

当前 tracker 主要输出 object/goal 位置关系，orientation 槽仍为零。直接采集并训练可能把关键方向信息留给 Video DiT 隐式恢复，无法形成完整 object-centric 条件。

### 第三优先级：需要多物体或阶段状态

```text
PullCubeTool-v1
StackPyramid-v1
AssemblingKits-v1
FMBAssembly1Easy-v1
PickClutterYCB-v1
```

这些任务需要多 role object association、active target 切换或 tool state，不能把当前单 object/goal tracker 原样复制过去。

### 单独建模

```text
TurnFaucet-v1
PokeCube-v1
RollBall-v1
```

它们分别依赖 articulated progress、接触区域或滚动动力学，需要不同于单纯 object→goal translation 的结构化 message。

## 9. Piper 与双臂边界

当前 ManiSkill checkout 没有 Piper agent、URDF、controller、SDK 或注册任务。现有双臂任务：

```text
TwoRobotPickCube-v1
TwoRobotStackCube-v1
```

均固定为两个 `panda_wristcam`，且没有 checked-in motion-planning solver。因此本报告只证明单臂仿真 Panda 的任务可运行性，不证明双臂 Piper 或真实 Franka 的执行能力。

后续若进入 Piper，需要先补齐 robot asset/controller、双臂 base frame、每臂 7D metric action adapter、夹爪标定、同步/碰撞约束和成功 replay，再讨论规模化数据采集。
