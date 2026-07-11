# RLBench EORT 逻辑审查

审查时间：2026-07-10 UTC。依据：`DiT4DiT/docs/EORT_Implementation_And_Test_Path_CN.md`、`examples/RLBench_EORT` 的采集/派生/转换代码、本地 RLBench 任务源码和已有数据样本。

## 结论

现有实现可以作为 **PushButton 的 GT oracle gate**：它验证“给 action policy 额外的 simulator privileged condition 是否可能提高闭环成功率”。它还不是通用 EORT 数据集，更不能代表真实部署的 object-centric 感知输入。

最重要的边界是：当前 `gt_area` 不是从 RGB/mask 推出的接触几何，而是 simulator state 加手工规则组成的 controller hint。将它称为 oracle 是合理的；将其直接视作可部署 EORT 或真实 contact area 则不合理。

## 合理部分

- raw 数据保留 RGB、mask、机器人状态、语义 object/joint state、progress、相机元数据和 task semantics，诊断信息充分。
- 先做 direct baseline、GT replay、GT-EORT teacher forcing，再决定是否训练 object predictor 的 gate 顺序正确；它把模型、动作执行和感知问题分开。
- `push_buttons` 的 `variation_id % 3 == 0` 会筛出单按钮 variation：RLBench 源码定义 `buttons_to_push = 1 + index % 3`。因此作为单目标子集是有效的。
- EORT-to-LeRobot converter 会拒绝缺失的 `gt_area` 字段，至少不会把不完整样本静默送入训练。

## P0：标签数学与语义不成立

### 四元数直接相减

`derive_eort_labels.py` 的 `compute_eort_rel_pose` 和 `action_delta_from_absolute_ee_pose` 都直接相减四元数分量。四元数不是线性坐标：同一旋转可由 `q` 或 `-q` 表示，分量差既不等于相对旋转也不具备连续性。

修复应为 `q_rel = q_object ⊗ inverse(q_ee)`，或只输出经过 wrap 的 RPY/rotation-vector。转换器的 RPY delta 路径较接近正确做法，但派生 `.npy` 仍不应保留错误的 quaternion delta。

### “contact”实际是物体原点 proximity

`contact_state` 使用 EE 到 primary object 原点的 0.08m/0.04m 阈值；`contact_point` 在 `contact_state > 0`（包括 near）时写 EE 位置。因此它既不是物理接触，也不是接触点。

应该重命名为 `proximity_state`，或采集引擎真实 contact force/contact pair 后再输出 contact。`contact_area_center=object xyz`、固定 normal `[1,0,0]` 和固定 radius `0.025` 同样只能叫 proxy，不能叫接触面。

## P0：任务泛化声明过强

任务 JSON 中有 `phase_rule_type`，但派生器无条件调用 `derive_push_button_phase`。`object_goal_rel_pose` 也只将剩余 progress 写入第 0 维。对 drawer、grasp、place 或 tool-use，这不是物体到目标的相对位姿，phase 也没有任务语义。

结论：当前脚本只应接受显式支持的 task schema（首版为 `push_button`）；其他任务需要各自的 goal、progress、phase 和接触定义，不能只靠通用 JSON 字段名称。

## P0：帧对齐没有被证明

RLBench 同时会在 demo 初始、路径规划和 `_demo_record_step` 调用 callback；collector 通过“保留最后 N 个 callback frame”与 observations 对齐。这个启发式没有时间戳、pose hash 或长度/内容逐帧断言。

训练前必须验证 callback object pose 与同一 observation/frame 的 scene pose 一致。否则 object label 有可能与 RGB/action 错位，且任何后续模型结果都不可解释。

## P1：多按钮 active target 与离散状态

`push_buttons` 的 primary object/progress 固定为 button0；多按钮任务实际按 `button0 → button1 → button2` 变化。未切换 active target 时，后续动作会收到错误目标标签。虽然单按钮 variation 可以使用，但多按钮数据在实现 `active_button_index`、active contact area 和 remaining buttons 前不得进入训练。

`object_interaction_state`、target id、phase 等是离散变量；把它们与世界坐标、距离一起 q99 归一化为连续数值，会掩盖其语义。应保留 raw category 或用 embedding，并明确每个字段的 normalization。

## P1：oracle 与部署差距

当前 13D 同时包含重复几何（center、EE→center、距离）和 simulator-only progress/phase。它可能成为强 controller hint，从而提高 teacher-forcing 指标，但不能证明视觉端学会对象关系。

真实部署缺少内部 joint progress、完美 object ID 和无抖动 6D pose。只有 GT oracle 显著超过 baseline 后，才值得做 predicted EORT；届时必须做 pose/mask jitter、dropout、ID switch、遮挡和 no-object fallback ablation。

## P1：动作、成功与 QA

- canonical action 是相邻 observation 的下一 EE pose，最后一帧复制自身；要在 metadata 中保留 control rate、frame interval 与末帧处理规则。
- `episode_meta.success` 写为 `None`，尽管 live demo 通常成功；应持久化最终 success 和 collector/action mode。
- 采集前需要检查 RGB/mask/object/progress 的长度、有限性、mask 可见率、目标覆盖率和 success；仅检查帧数不足以保证样本可训。
- 现有 GT-area eval client 尚未完整接入 live 13D 输入，闭环 oracle success 尚不能与 baseline 公平比较。

## 对 ManiSkill 流程的设计影响

本项目的新 `PushCube-v1` 流程刻意不复制上述 proxy：原始 ManiSkill HDF5 保存 `state_dict+rgb+depth+segmentation`，sidecar 仅输出当前可验证的 TCP→object、object→goal、距离、当前 progress、下一帧 object xyz delta、action 和 success。Pose 约定显式为 `wxyz`，observation `T+1` 对 action/label `T` 有硬校验。

因此 ManiSkill pilot 同样是 oracle gate，但其标签语义更窄、更可审计。后续新增 contact、orientation、预测 EORT 或多任务前，必须逐项解决 `docs/思考与隐患.md` 中的对应风险。

## 推荐优先级

1. 修正 RLBench 四元数/接触命名和 callback 对齐验证。
2. 打通 GT-area 评估输入，完成 baseline vs GT oracle 的闭环 gate。
3. 仅在 gate 有明显收益后，补多按钮 active target 与 task-specific schema。
4. 之后才做视觉 predictor、噪声感知训练和 sim2real。
