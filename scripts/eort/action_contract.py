"""Reversible metric action contract for ManiSkill Panda EORT commands."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


PANDA_POSITION_SCALE_M = 0.1
PANDA_ROTATION_SCALE_RAD = -0.1


def _actions(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 7 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite (T,7), got {value.shape}")
    return value


def panda_normalized_to_metric_task(action: np.ndarray) -> np.ndarray:
    """Decode Panda normalized root command to task-axis metric rotvec command.

    PushCube/PickCube load the Panda with identity root orientation, so root and
    task/world axes coincide.  Translation origin does not affect a delta.
    """
    action = _actions(action, "Panda action")
    if np.any(np.abs(action) > 1.0 + 1e-6) or np.any(np.linalg.norm(action[:, 3:6], axis=1) > 1.0 + 1e-6):
        raise ValueError("Panda normalized action exceeds controller bounds")
    command = np.empty_like(action)
    command[:, :3] = action[:, :3] * PANDA_POSITION_SCALE_M
    command[:, 3:6] = Rotation.from_euler("XYZ", action[:, 3:6] * PANDA_ROTATION_SCALE_RAD).as_rotvec()
    command[:, 6] = (action[:, 6] + 1.0) / 2.0
    return command.astype(np.float32)


def metric_task_to_panda_normalized(command: np.ndarray) -> np.ndarray:
    """Encode task-axis metric rotvec command for the current Panda controller."""
    command = _actions(command, "Metric task command")
    action = np.empty_like(command)
    action[:, :3] = command[:, :3] / PANDA_POSITION_SCALE_M
    action[:, 3:6] = Rotation.from_rotvec(command[:, 3:6]).as_euler("XYZ") / PANDA_ROTATION_SCALE_RAD
    action[:, 6] = command[:, 6] * 2.0 - 1.0
    if np.any(np.abs(action) > 1.0 + 1e-5) or np.any(np.linalg.norm(action[:, 3:6], axis=1) > 1.0 + 1e-5):
        raise ValueError("Metric task command exceeds Panda controller bounds")
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def rotate_metric_task_command(command: np.ndarray, target_from_source_rotation: np.ndarray) -> np.ndarray:
    """Rotate metric translation and rotvec axes; preserve gripper openness."""
    command = _actions(command, "Metric task command")
    rotation = np.asarray(target_from_source_rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("target_from_source_rotation must be a proper 3x3 rotation")
    transformed = command.copy()
    transformed[:, :3] = command[:, :3] @ rotation.T
    transformed[:, 3:6] = command[:, 3:6] @ rotation.T
    return transformed.astype(np.float32)
