#!/usr/bin/env python3
"""Derive aligned, simulator-oracle EORT labels from PushCube trajectory HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    from scripts.eort.action_contract import panda_normalized_to_metric_task
except ModuleNotFoundError:  # Direct `python scripts/eort/derive_push_cube_eort.py` entrypoint.
    from action_contract import panda_normalized_to_metric_task


SCHEMA_VERSION = "maniskill_push_cube_eort_oracle_v1"
OBJECTCENTRIC_V2_SCHEMA_VERSION = "maniskill_push_cube_objectcentric_oracle_v2"
PICK_CUBE_OBJECTCENTRIC_V2_SCHEMA_VERSION = "maniskill_pick_cube_objectcentric_oracle_v2"
OBJECTCENTRIC_V2_ENV_ID = "PushCubeEORT-v1"
OBJECTCENTRIC_V2_ENV_IDS = (
    OBJECTCENTRIC_V2_ENV_ID,
    "PushCubeEORTCameraRand-v1",
    "PushCubeEORTOccluded-v1",
    "PushCubeEORTGeometryRand-v1",
    "PickCubeEORT-v1",
    "PickCubeEORTCameraRand-v1",
    "PickCubeEORTOccluded-v1",
    "PickCubeEORTGeometryRand-v1",
)
DEFAULT_FUTURE_HORIZONS = (1, 4, 8)

PUSH_CUBE_TASK_SPEC = {
    "schema_version": OBJECTCENTRIC_V2_SCHEMA_VERSION,
    "phase_key": "push_interaction_phase",
    "phase_labels": {
        "0": "approach",
        "1": "measured_contact",
        "2": "measured_contact_while_object_moves",
        "3": "native_task_success",
    },
}
PICK_CUBE_TASK_SPEC = {
    "schema_version": PICK_CUBE_OBJECTCENTRIC_V2_SCHEMA_VERSION,
    "phase_key": "pick_interaction_phase",
    "phase_labels": {
        "0": "approach",
        "1": "measured_contact",
        "2": "native_grasp_detected",
        "3": "native_task_success",
    },
}
OBJECTCENTRIC_V2_TASK_SPECS = {
    "PushCubeEORT-v1": PUSH_CUBE_TASK_SPEC,
    "PushCubeEORTCameraRand-v1": PUSH_CUBE_TASK_SPEC,
    "PushCubeEORTOccluded-v1": PUSH_CUBE_TASK_SPEC,
    "PushCubeEORTGeometryRand-v1": PUSH_CUBE_TASK_SPEC,
    "PickCubeEORT-v1": PICK_CUBE_TASK_SPEC,
    "PickCubeEORTCameraRand-v1": PICK_CUBE_TASK_SPEC,
    "PickCubeEORTOccluded-v1": PICK_CUBE_TASK_SPEC,
    "PickCubeEORTGeometryRand-v1": PICK_CUBE_TASK_SPEC,
}


def _dataset(group: h5py.Group, path: str) -> h5py.Dataset:
    node: h5py.Group | h5py.Dataset = group
    for part in path.split("/"):
        if not isinstance(node, h5py.Group) or part not in node:
            raise ValueError(f"Missing required dataset: {group.name}/{path}")
        node = node[part]
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"Expected dataset at {group.name}/{path}")
    return node


def _finite_array(group: h5py.Group, path: str, steps: int, width: int) -> np.ndarray:
    array = np.asarray(_dataset(group, path), dtype=np.float32)
    if array.ndim != 2 or array.shape != (steps + 1, width):
        raise ValueError(
            f"{group.name}/{path} must have shape {(steps + 1, width)}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{group.name}/{path} contains non-finite values")
    return array


def _static_segmentation_id(group: h5py.Group, path: str, steps: int) -> int:
    array = np.asarray(_dataset(group, path))
    if array.shape != (steps + 1, 1) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"{group.name}/{path} must be an integer array with shape {(steps + 1, 1)}, got {array.shape} {array.dtype}"
        )
    if np.any(array <= 0) or not np.all(array == array[0, 0]):
        raise ValueError(f"{group.name}/{path} must contain one positive static actor ID")
    return int(array[0, 0])


def _optional_bool_observation(group: h5py.Group, path: str, steps: int) -> np.ndarray | None:
    if path not in group:
        return None
    array = np.asarray(group[path], dtype=bool)
    if array.shape != (steps + 1, 1):
        raise ValueError(f"{group.name}/{path} must have shape {(steps + 1, 1)}, got {array.shape}")
    return array[:-1]


def _required_bool_observation(group: h5py.Group, path: str, steps: int) -> np.ndarray:
    array = _optional_bool_observation(group, path, steps)
    if array is None:
        raise ValueError(f"Missing required dataset: {group.name}/{path}")
    return array


def _segmentation_visibility(
    group: h5py.Group, camera: str, steps: int, actor_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    segmentation = np.asarray(_dataset(group, f"obs/sensor_data/{camera}/segmentation"))
    if (
        segmentation.ndim != 4
        or segmentation.shape[0] != steps + 1
        or segmentation.shape[-1] != 1
    ):
        raise ValueError(
            f"{group.name}/obs/sensor_data/{camera}/segmentation must have shape (T+1,H,W,1), got {segmentation.shape}"
        )
    mask = segmentation[:steps, ..., 0] == actor_id
    pixels = mask.sum(axis=(1, 2), dtype=np.int32)
    bboxes = np.zeros((steps, 4), dtype=np.int32)
    centroids = np.zeros((steps, 2), dtype=np.float32)
    for step in range(steps):
        rows, cols = np.nonzero(mask[step])
        if len(rows):
            bboxes[step] = [cols.min(), rows.min(), cols.max() + 1, rows.max() + 1]
            centroids[step] = [cols.mean(), rows.mean()]
    return (
        pixels[:, None],
        (pixels / mask[0].size).astype(np.float32)[:, None],
        bboxes,
        centroids,
    )


def _segdepth_centroid_world(
    group: h5py.Group, camera: str, steps: int, actor_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project an actor's segmentation pixels using metric depth and CV calibration."""
    segmentation = np.asarray(_dataset(group, f"obs/sensor_data/{camera}/segmentation"))
    depth = np.asarray(_dataset(group, f"obs/sensor_data/{camera}/depth"), dtype=np.float32)
    intrinsic = np.asarray(_dataset(group, f"obs/sensor_param/{camera}/intrinsic_cv"), dtype=np.float32)
    extrinsic = np.asarray(_dataset(group, f"obs/sensor_param/{camera}/extrinsic_cv"), dtype=np.float32)
    expected_image_shape = (steps + 1, *segmentation.shape[1:3], 1)
    if segmentation.shape != expected_image_shape or depth.shape != expected_image_shape:
        raise ValueError(f"{group.name}/{camera} depth and segmentation must share (T+1,H,W,1)")
    if intrinsic.shape != (steps + 1, 3, 3) or extrinsic.shape != (steps + 1, 3, 4):
        raise ValueError(f"{group.name}/{camera} requires intrinsic (T+1,3,3) and extrinsic (T+1,3,4)")
    centroids = np.zeros((steps, 3), dtype=np.float32)
    valid = np.zeros((steps, 1), dtype=bool)
    for step in range(steps):
        rows, cols = np.nonzero(segmentation[step, ..., 0] == actor_id)
        if len(rows) == 0:
            continue
        z = depth[step, rows, cols, 0] / 1000.0  # ManiSkill depth is millimeters.
        keep = z > 0
        if not np.any(keep):
            continue
        rows, cols, z = rows[keep], cols[keep], z[keep]
        k = intrinsic[step]
        if k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError(f"{group.name}/{camera} has invalid focal length")
        points_camera = np.stack(
            ((cols - k[0, 2]) * z / k[0, 0], (rows - k[1, 2]) * z / k[1, 1], z), axis=1
        )
        rotation, translation = extrinsic[step, :, :3], extrinsic[step, :, 3]
        centroids[step] = ((points_camera - translation) @ rotation).mean(axis=0)
        valid[step, 0] = True
    return centroids, valid


def _normalized_quaternion_wxyz(quaternion: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm <= 1e-6):
        raise ValueError(f"{name} contains zero-norm quaternions")
    return quaternion / norm


def _relative_rotation_rotvec(
    reference_wxyz: np.ndarray, target_wxyz: np.ndarray
) -> np.ndarray:
    """Return the rotation vector for ``target * inverse(reference)``."""
    reference = _normalized_quaternion_wxyz(reference_wxyz, "reference quaternion")
    target = _normalized_quaternion_wxyz(target_wxyz, "target quaternion")
    rw, rx, ry, rz = reference.T
    tw, tx, ty, tz = target.T
    # target * conjugate(reference), in ManiSkill's documented wxyz convention.
    w = tw * rw + tx * rx + ty * ry + tz * rz
    x = -tw * rx + tx * rw - ty * rz + tz * ry
    y = -tw * ry + tx * rz + ty * rw - tz * rx
    z = -tw * rz - tx * ry + ty * rx + tz * rw
    relative = np.stack((w, x, y, z), axis=1)
    relative = _normalized_quaternion_wxyz(relative, "relative quaternion")
    relative[relative[:, 0] < 0] *= -1.0
    sin_half = np.linalg.norm(relative[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(sin_half, np.clip(relative[:, 0], -1.0, 1.0))
    axis = np.zeros_like(relative[:, 1:])
    nonzero = sin_half > 1e-6
    axis[nonzero] = relative[nonzero, 1:] / sin_half[nonzero, None]
    return (axis * angle[:, None]).astype(np.float32)


def _local_eef_transition(tcp_pose_wxyz: np.ndarray, gripper_action: np.ndarray) -> np.ndarray:
    """Return observed next-EEF motion in the current EEF frame, not a command."""
    reference_quat = _normalized_quaternion_wxyz(
        tcp_pose_wxyz[:-1, 3:], "TCP reference quaternion"
    )
    target_quat = _normalized_quaternion_wxyz(
        tcp_pose_wxyz[1:, 3:], "TCP target quaternion"
    )
    world_delta = tcp_pose_wxyz[1:, :3] - tcp_pose_wxyz[:-1, :3]
    w = reference_quat[:, :1]
    xyz = reference_quat[:, 1:]
    local_delta = (
        world_delta
        + 2 * np.cross(xyz, np.cross(xyz, world_delta))
        - 2 * w * np.cross(xyz, world_delta)
    )
    rw, rx, ry, rz = reference_quat.T
    tw, tx, ty, tz = target_quat.T
    relative_quat = np.stack(
        (
            rw * tw + rx * tx + ry * ty + rz * tz,
            rw * tx - rx * tw - ry * tz + rz * ty,
            rw * ty + rx * tz - ry * tw - rz * tx,
            rw * tz - rx * ty + ry * tx - rz * tw,
        ),
        axis=1,
    )
    local_rotvec = _relative_rotation_rotvec(
        np.tile([1.0, 0.0, 0.0, 0.0], (len(relative_quat), 1)), relative_quat
    )
    gripper_open_fraction = (gripper_action[:, None] + 1.0) / 2.0
    if not np.isfinite(local_delta).all() or not np.isfinite(gripper_open_fraction).all():
        raise ValueError("EEF transition contains non-finite values")
    return np.concatenate(
        (local_delta, local_rotvec, gripper_open_fraction), axis=1
    ).astype(np.float32)


def _future_object_transitions(
    object_pose_wxyz: np.ndarray, steps: int, horizons: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError(f"Future horizons must be positive, got {horizons}")
    current_pos = object_pose_wxyz[:steps, :3]
    current_quat = object_pose_wxyz[:steps, 3:]
    frame_ids = np.arange(steps)[:, None]
    horizon_array = np.asarray(horizons, dtype=np.int64)[None, :]
    valid = frame_ids + horizon_array <= steps
    target_ids = np.minimum(frame_ids + horizon_array, steps)
    target_pose = object_pose_wxyz[target_ids]
    pos_delta = target_pose[:, :, :3] - current_pos[:, None, :]
    rotvec = np.stack(
        [
            _relative_rotation_rotvec(current_quat, target_pose[:, horizon, 3:])
            for horizon in range(len(horizons))
        ],
        axis=1,
    )
    pos_delta[~valid] = 0.0
    rotvec[~valid] = 0.0
    return pos_delta.astype(np.float32), rotvec.astype(np.float32), valid


def _validate_camera_streams(group: h5py.Group, camera: str, steps: int) -> None:
    for name in ("rgb", "depth", "segmentation"):
        array = _dataset(group, f"obs/sensor_data/{camera}/{name}")
        if array.ndim < 3 or array.shape[0] != steps + 1:
            raise ValueError(
                f"{group.name}/obs/sensor_data/{camera}/{name} must have {steps + 1} frames, "
                f"got {array.shape}"
            )
    if _dataset(group, f"obs/sensor_data/{camera}/rgb").shape[-1] != 3:
        raise ValueError(f"{group.name}/obs/sensor_data/{camera}/rgb must have RGB channels")


def _trajectory_keys(file: h5py.File) -> list[str]:
    keys = [key for key in file if key.startswith("traj_")]
    if not keys:
        raise ValueError(f"No trajectory groups found in {file.filename}")
    return sorted(keys, key=lambda key: int(key.removeprefix("traj_")))


def _load_source_metadata(trajectory_path: Path) -> dict[str, Any]:
    metadata_path = trajectory_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing ManiSkill trajectory metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    env_id = metadata.get("env_info", {}).get("env_id")
    if not isinstance(env_id, str):
        raise ValueError(f"Missing env_id in {metadata_path}")
    return metadata


def derive_trajectory(group: h5py.Group, camera: str) -> dict[str, np.ndarray]:
    actions = np.asarray(_dataset(group, "actions"), dtype=np.float32)
    if actions.ndim != 2 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError(f"{group.name}/actions must be a non-empty finite 2D array")
    steps = len(actions)

    success = np.asarray(_dataset(group, "success"), dtype=bool).reshape(-1)
    if len(success) != steps or not bool(success[-1]):
        raise ValueError(f"{group.name} is not a successful {steps}-step trajectory")

    tcp_pose = _finite_array(group, "obs/extra/tcp_pose", steps, 7)
    object_pose = _finite_array(group, "obs/extra/obj_pose", steps, 7)
    goal_pos = _finite_array(group, "obs/extra/goal_pos", steps, 3)
    _validate_camera_streams(group, camera, steps)

    tcp_pos = tcp_pose[:-1, :3]
    object_pos = object_pose[:-1, :3]
    current_goal_pos = goal_pos[:-1]
    ee_to_object = object_pos - tcp_pos
    object_to_goal = current_goal_pos - object_pos
    object_to_goal_dist = np.linalg.norm(object_to_goal, axis=1, keepdims=True)
    initial_distance = float(object_to_goal_dist[0, 0])
    if initial_distance <= 1e-6:
        raise ValueError(f"{group.name} starts at the goal; progress is undefined")

    return {
        "action": actions,
        "success": success,
        "tcp_pose_wxyz": tcp_pose[:-1],
        "object_pose_wxyz": object_pose[:-1],
        "goal_pos": current_goal_pos,
        "ee_to_object": ee_to_object,
        "ee_to_object_dist": np.linalg.norm(ee_to_object, axis=1, keepdims=True),
        "object_to_goal": object_to_goal,
        "object_to_goal_dist": object_to_goal_dist,
        "goal_progress": np.clip(1.0 - object_to_goal_dist / initial_distance, 0.0, 1.0),
        "object_next_delta": object_pose[1:, :3] - object_pose[:-1, :3],
    }


def derive_trajectory_objectcentric_v2(
    group: h5py.Group,
    camera: str,
    *,
    task: str,
    control_mode: str,
    future_horizons: tuple[int, ...],
    contact_force_threshold: float,
) -> dict[str, np.ndarray]:
    arrays = derive_trajectory(group, camera)
    steps = len(arrays["action"])
    expected_width = {"pd_joint_pos": 8, "pd_ee_delta_pose": 7}[control_mode]
    if arrays["action"].shape[1] != expected_width or np.any(np.abs(arrays["action"][:, -1]) > 1):
        raise ValueError(
            f"{group.name} requires Panda {control_mode} (T,{expected_width}) actions with gripper in [-1,1]"
        )
    tcp_pose = _finite_array(group, "obs/extra/tcp_pose", steps, 7)
    object_pose = _finite_array(group, "obs/extra/obj_pose", steps, 7)
    if "obj_extent" in group["obs/extra"]:
        object_extent = _finite_array(group, "obs/extra/obj_extent", steps, 3)
    else:
        # Legacy Panda EORT recordings predate this field; both tasks used a fixed 4 cm cube.
        object_extent = np.full((steps + 1, 3), 0.04, dtype=np.float32)
    if np.any(object_extent <= 0) or not np.allclose(object_extent, object_extent[:1]):
        raise ValueError(f"{group.name} object extent must be positive and fixed within an episode")
    if "obj_mass" in group["obs/extra"]:
        object_mass = _finite_array(group, "obs/extra/obj_mass", steps, 1)
    else:
        object_mass = np.prod(object_extent, axis=1, keepdims=True) * 1000.0
    if np.any(object_mass <= 0) or not np.allclose(object_mass, object_mass[:1]):
        raise ValueError(f"{group.name} object mass must be positive and fixed within an episode")
    if "obj_friction" in group["obs/extra"]:
        object_friction = _finite_array(group, "obs/extra/obj_friction", steps, 2)
    else:
        object_friction = np.full((steps + 1, 2), 0.3, dtype=np.float32)
    if np.any(object_friction <= 0) or not np.allclose(object_friction, object_friction[:1]):
        raise ValueError(f"{group.name} object friction must be positive and fixed within an episode")
    object_linear_velocity = _finite_array(
        group, "obs/extra/obj_linear_velocity", steps, 3
    )
    object_angular_velocity = _finite_array(
        group, "obs/extra/obj_angular_velocity", steps, 3
    )
    robot_obj_contact_force = _finite_array(
        group, "obs/extra/robot_obj_contact_force", steps, 3
    )
    robot_obj_contact_force_norm = _finite_array(
        group, "obs/extra/robot_obj_contact_force_norm", steps, 1
    )
    extra = group["obs/extra"]
    occluder_active = _optional_bool_observation(group, "obs/extra/occluder_active", steps)
    has_object_id = "obj_segmentation_id" in extra
    has_goal_id = "goal_segmentation_id" in extra
    if has_object_id != has_goal_id:
        raise ValueError(f"{group.name} must contain both object and goal segmentation IDs")
    visibility_arrays: dict[str, np.ndarray] = {}
    if has_object_id:
        object_segmentation_id = _static_segmentation_id(
            group, "obs/extra/obj_segmentation_id", steps
        )
        goal_segmentation_id = _static_segmentation_id(
            group, "obs/extra/goal_segmentation_id", steps
        )
        object_mask_pixels, object_visibility_fraction, object_bbox_xyxy, object_mask_centroid_uv = _segmentation_visibility(
            group, camera, steps, object_segmentation_id
        )
        object_segdepth_centroid_world, object_segdepth_valid = _segdepth_centroid_world(
            group, camera, steps, object_segmentation_id
        )
        goal_mask_pixels, goal_visibility_fraction, goal_bbox_xyxy, goal_mask_centroid_uv = _segmentation_visibility(
            group, camera, steps, goal_segmentation_id
        )
        goal_segdepth_centroid_world, goal_segdepth_valid = _segdepth_centroid_world(
            group, camera, steps, goal_segmentation_id
        )
        visibility_arrays = {
            "object_segmentation_id": np.asarray([object_segmentation_id], dtype=np.int32),
            "goal_segmentation_id": np.asarray([goal_segmentation_id], dtype=np.int32),
            "object_mask_pixels": object_mask_pixels,
            "object_visibility_fraction": object_visibility_fraction,
            "object_visible": object_mask_pixels > 0,
            "object_bbox_xyxy": object_bbox_xyxy,
            "object_mask_centroid_uv": object_mask_centroid_uv,
            "object_segdepth_centroid_world": object_segdepth_centroid_world,
            "object_segdepth_valid": object_segdepth_valid,
            "object_segdepth_centroid_error": np.where(
                object_segdepth_valid,
                np.linalg.norm(object_segdepth_centroid_world - object_pose[:steps, :3], axis=1, keepdims=True),
                0.0,
            ).astype(np.float32),
            "goal_mask_pixels": goal_mask_pixels,
            "goal_visibility_fraction": goal_visibility_fraction,
            "goal_visible": goal_mask_pixels > 0,
            "goal_bbox_xyxy": goal_bbox_xyxy,
            "goal_mask_centroid_uv": goal_mask_centroid_uv,
            "goal_segdepth_centroid_world": goal_segdepth_centroid_world,
            "goal_segdepth_valid": goal_segdepth_valid,
            "goal_segdepth_centroid_error": np.where(
                goal_segdepth_valid,
                np.linalg.norm(goal_segdepth_centroid_world - arrays["goal_pos"], axis=1, keepdims=True),
                0.0,
            ).astype(np.float32),
        }
    future_pos_delta, future_rotvec, future_valid = _future_object_transitions(
        object_pose, steps, future_horizons
    )
    contact_force = robot_obj_contact_force[:-1]
    physical_contact = robot_obj_contact_force_norm[:-1, 0] > contact_force_threshold
    task_spec = OBJECTCENTRIC_V2_TASK_SPECS[task]
    interaction_phase = np.zeros(steps, dtype=np.int8)
    interaction_phase[physical_contact] = 1
    if task.startswith("PushCube"):
        object_speed = np.linalg.norm(object_linear_velocity[:-1], axis=1)
        interaction_phase[physical_contact & (object_speed > 1e-4)] = 2
    else:
        # Native Panda grasp detection is an oracle label, not a policy input.
        interaction_phase[_required_bool_observation(group, "obs/extra/is_grasped", steps)[:, 0]] = 2
    # Native task success is authoritative; geometric progress is only diagnostic.
    interaction_phase[arrays["success"]] = 3
    arrays.update(
        {
            "object_extent": object_extent[:1],
            "object_mass": object_mass[:1],
            "object_friction": object_friction[:1],
            "object_linear_velocity": object_linear_velocity[:-1],
            "object_angular_velocity": object_angular_velocity[:-1],
            "eef_transition_local": _local_eef_transition(
                tcp_pose, arrays["action"][:, -1]
            ),
            "robot_obj_contact_force": contact_force,
            "robot_obj_contact_force_norm": robot_obj_contact_force_norm[:-1],
            "physical_contact": physical_contact[:, None],
            task_spec["phase_key"]: interaction_phase[:, None],
            "ee_to_object_rotvec": _relative_rotation_rotvec(
                tcp_pose[:-1, 3:], object_pose[:-1, 3:]
            ),
            "future_horizons_steps": np.asarray(future_horizons, dtype=np.int32),
            "object_future_delta_pos": future_pos_delta,
            "object_future_delta_rotvec": future_rotvec,
            "object_future_valid": future_valid,
        }
    )
    if control_mode == "pd_ee_delta_pose":
        arrays["panda_pd_ee_delta_pose_command"] = arrays["action"].copy()
        arrays["metric_task_delta_pose_command"] = panda_normalized_to_metric_task(arrays["action"])
    arrays.update(visibility_arrays)
    if occluder_active is not None:
        arrays["occluder_active"] = occluder_active
    return arrays


def derive_dataset(
    trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    camera: str = "base_camera",
    schema: str = "v1",
    future_horizons: tuple[int, ...] = DEFAULT_FUTURE_HORIZONS,
    contact_force_threshold: float = 1e-6,
) -> dict[str, Any]:
    trajectory_path = Path(trajectory_path).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    metadata = _load_source_metadata(trajectory_path)
    task = metadata["env_info"]["env_id"]
    if schema == "v1":
        if task != "PushCube-v1":
            raise ValueError(f"v1 requires PushCube-v1 metadata, got {task!r}")
        schema_version = SCHEMA_VERSION
        derive = lambda group: derive_trajectory(group, camera)
    elif schema == "objectcentric_v2":
        if task not in OBJECTCENTRIC_V2_ENV_IDS:
            raise ValueError(
                f"objectcentric_v2 requires one of {OBJECTCENTRIC_V2_ENV_IDS} metadata, got {task!r}"
            )
        if contact_force_threshold < 0:
            raise ValueError("contact_force_threshold must be non-negative")
        task_spec = OBJECTCENTRIC_V2_TASK_SPECS[task]
        control_mode = metadata.get("env_info", {}).get("env_kwargs", {}).get("control_mode")
        if control_mode not in ("pd_joint_pos", "pd_ee_delta_pose"):
            raise ValueError(
                "objectcentric_v2 requires metadata control_mode pd_joint_pos or pd_ee_delta_pose, "
                f"got {control_mode!r}"
            )
        schema_version = task_spec["schema_version"]
        derive = lambda group: derive_trajectory_objectcentric_v2(
            group,
            camera,
            task=task,
            control_mode=control_mode,
            future_horizons=future_horizons,
            contact_force_threshold=contact_force_threshold,
        )
    else:
        raise ValueError(f"Unsupported schema: {schema!r}")

    records: list[dict[str, Any]] = []
    with h5py.File(trajectory_path, "r") as file:
        extent_sources = {
            key: "obs/extra/obj_extent"
            if "obj_extent" in file[key]["obs/extra"]
            else "legacy_fixed_panda_cube_0.04m"
            for key in _trajectory_keys(file)
        }
        mass_sources = {
            key: "obs/extra/obj_mass"
            if "obj_mass" in file[key]["obs/extra"]
            else "legacy_default_density_1000kg_m3_from_extent"
            for key in _trajectory_keys(file)
        }
        friction_sources = {
            key: "obs/extra/obj_friction"
            if "obj_friction" in file[key]["obs/extra"]
            else "legacy_default_material_0.3"
            for key in _trajectory_keys(file)
        }
        outputs = [(key, derive(file[key])) for key in _trajectory_keys(file)]

    output_dir.mkdir(parents=True)
    for trajectory_id, arrays in outputs:
        npz_path = output_dir / f"{trajectory_id}.npz"
        np.savez_compressed(npz_path, **arrays)
        records.append(
            {
                "schema_version": schema_version,
                "oracle": True,
                "source_h5": str(trajectory_path),
                "source_trajectory": trajectory_id,
                "camera": camera,
                "source_control_mode": control_mode if schema == "objectcentric_v2" else None,
                "quaternion_convention": "wxyz",
                "num_steps": int(len(arrays["action"])),
                "fields": {name: list(value.shape) for name, value in arrays.items()},
                "output": npz_path.name,
            }
        )
        if schema == "objectcentric_v2":
            records[-1].update(
                {
                    "physical_contact": {
                        "source": "robot_obj_contact_force_norm",
                        "force_norm_threshold": contact_force_threshold,
                    },
                    f"{task_spec['phase_key']}_labels": task_spec["phase_labels"],
                    "future_horizons_steps": list(future_horizons),
                    "eef_transition_local": {
                        "source": "observed tcp_pose[t:t+1] plus trajectory gripper action",
                        "frame": "current end-effector local frame",
                        "controller_command": False,
                    },
                }
            )
            if "panda_pd_ee_delta_pose_command" in arrays:
                records[-1]["panda_pd_ee_delta_pose_command"] = {
                    "source": "recorded ManiSkill trajectory action",
                    "frame": "root translation and root-aligned body rotation",
                    "normalized": True,
                    "controller_command": True,
                    "robot": "Panda",
                }
                records[-1]["metric_task_delta_pose_command"] = {
                    "source": "exact decode of recorded Panda pd_ee_delta_pose action",
                    "layout": "delta_xyz_m + delta_rotvec_rad + gripper_open_fraction",
                    "frame": "task/world axes; current PushCube/PickCube Panda root orientation is identity",
                    "normalized": False,
                    "controller_command": True,
                    "robot_specific_scaling": False,
                }
            records[-1]["object_extent"] = {
                "source": extent_sources[trajectory_id],
                "layout": "full xyz side lengths in meters",
                "static_within_episode": True,
                "policy_input": False,
            }
            records[-1]["object_mass"] = {
                "source": mass_sources[trajectory_id],
                "layout": "mass in kilograms",
                "static_within_episode": True,
                "policy_input": False,
            }
            records[-1]["object_friction"] = {
                "source": friction_sources[trajectory_id],
                "layout": "static_friction + dynamic_friction",
                "static_within_episode": True,
                "policy_input": False,
            }
            if "object_visible" in arrays:
                records[-1]["segmentation_visibility"] = {
                    "object_id_source": "obs/extra/obj_segmentation_id",
                    "goal_id_source": "obs/extra/goal_segmentation_id",
                    "fraction": "matching pixels divided by camera image pixels",
                    "bbox_xyxy": "[x_min,y_min,x_max_exclusive,y_max_exclusive] pixels; all zeros when invisible",
                    "mask_centroid_uv": "mean [u,v] mask pixel coordinate; all zeros when invisible",
                    "segdepth_centroid": "object and goal actor pixels back-projected from millimeter depth with intrinsic_cv/extrinsic_cv; oracle actor ID only",
                }
            if "occluder_active" in arrays:
                records[-1]["occluder_active"] = {
                    "source": "obs/extra/occluder_active",
                    "policy_input": False,
                    "meaning": "visual-only occluder is present for this observation",
                }

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": schema_version,
        "oracle": True,
        "task": task,
        "source_h5": str(trajectory_path),
        "source_metadata": str(trajectory_path.with_suffix(".json")),
        "camera": camera,
        "trajectories": len(records),
        "steps": sum(record["num_steps"] for record in records),
    }
    if schema == "objectcentric_v2":
        summary.update(
            {
                "physical_contact_source": "robot_obj_contact_force_norm",
                "source_control_mode": control_mode,
                "contact_force_threshold": contact_force_threshold,
                "future_horizons_steps": list(future_horizons),
                "segmentation_visibility": all(
                    "object_visible" in arrays for _, arrays in outputs
                ),
            }
        )
        phase = np.concatenate([arrays[task_spec["phase_key"]][:, 0] for _, arrays in outputs])
        phase_counts = np.bincount(phase, minlength=len(task_spec["phase_labels"]))
        summary["phase_qa"] = {
            "phase_key": task_spec["phase_key"],
            "labels": task_spec["phase_labels"],
            "counts": {str(index): int(count) for index, count in enumerate(phase_counts)},
            "interaction_steps": int(phase_counts[1] + phase_counts[2]),
            "task_progress_steps": int(phase_counts[2]),
            "success_steps": int(phase_counts[3]),
        }
        if summary["segmentation_visibility"]:
            object_visible = np.concatenate([arrays["object_visible"][:, 0] for _, arrays in outputs])
            goal_visible = np.concatenate([arrays["goal_visible"][:, 0] for _, arrays in outputs])
            object_depth_valid = np.concatenate([arrays["object_segdepth_valid"][:, 0] for _, arrays in outputs])
            goal_depth_valid = np.concatenate([arrays["goal_segdepth_valid"][:, 0] for _, arrays in outputs])
            visibility_qa = {
                "total_steps": summary["steps"],
                "object_visible_steps": int(object_visible.sum()),
                "goal_visible_steps": int(goal_visible.sum()),
                "relational_visible_steps": int((object_visible & goal_visible).sum()),
                "relational_segdepth_valid_steps": int((object_depth_valid & goal_depth_valid).sum()),
            }
            for role in ("object", "goal"):
                pixels = np.concatenate([arrays[f"{role}_mask_pixels"][:, 0] for _, arrays in outputs])
                visibility_qa[f"{role}_mask_pixels_min_median_max"] = [
                    int(pixels.min()), float(np.median(pixels)), int(pixels.max())
                ]
            if all("occluder_active" in arrays for _, arrays in outputs):
                visibility_qa["occluder_active_steps"] = int(
                    sum(arrays["occluder_active"].sum() for _, arrays in outputs)
                )
            summary["visibility_qa"] = visibility_qa
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--schema", choices=("v1", "objectcentric_v2"), default="v1")
    parser.add_argument(
        "--future-horizons",
        type=int,
        nargs="+",
        default=DEFAULT_FUTURE_HORIZONS,
        help="Positive action-aligned horizons for objectcentric_v2.",
    )
    parser.add_argument("--contact-force-threshold", type=float, default=1e-6)
    args = parser.parse_args()
    print(
        json.dumps(
            derive_dataset(
                args.traj_path,
                args.output_dir,
                camera=args.camera,
                schema=args.schema,
                future_horizons=tuple(args.future_horizons),
                contact_force_threshold=args.contact_force_threshold,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
