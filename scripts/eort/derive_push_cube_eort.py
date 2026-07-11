#!/usr/bin/env python3
"""Derive aligned, simulator-oracle EORT labels from PushCube trajectory HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA_VERSION = "maniskill_push_cube_eort_oracle_v1"
OBJECTCENTRIC_V2_SCHEMA_VERSION = "maniskill_push_cube_objectcentric_oracle_v2"
OBJECTCENTRIC_V2_ENV_ID = "PushCubeEORT-v1"
DEFAULT_FUTURE_HORIZONS = (1, 4, 8)


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
    future_horizons: tuple[int, ...],
    contact_force_threshold: float,
) -> dict[str, np.ndarray]:
    arrays = derive_trajectory(group, camera)
    steps = len(arrays["action"])
    tcp_pose = _finite_array(group, "obs/extra/tcp_pose", steps, 7)
    object_pose = _finite_array(group, "obs/extra/obj_pose", steps, 7)
    object_linear_velocity = _finite_array(
        group, "obs/extra/obj_linear_velocity", steps, 3
    )
    object_angular_velocity = _finite_array(
        group, "obs/extra/obj_angular_velocity", steps, 3
    )
    robot_obj_contact_force = _finite_array(
        group, "obs/extra/robot_obj_contact_force", steps, 3
    )
    future_pos_delta, future_rotvec, future_valid = _future_object_transitions(
        object_pose, steps, future_horizons
    )
    contact_force = robot_obj_contact_force[:-1]
    physical_contact = np.linalg.norm(contact_force, axis=1) > contact_force_threshold
    object_speed = np.linalg.norm(object_linear_velocity[:-1], axis=1)
    interaction_phase = np.zeros(steps, dtype=np.int8)
    interaction_phase[physical_contact] = 1
    interaction_phase[physical_contact & (object_speed > 1e-4)] = 2
    interaction_phase[
        ~physical_contact & (arrays["goal_progress"][:, 0] >= 0.999)
    ] = 3
    arrays.update(
        {
            "object_linear_velocity": object_linear_velocity[:-1],
            "object_angular_velocity": object_angular_velocity[:-1],
            "robot_obj_contact_force": contact_force,
            "physical_contact": physical_contact[:, None],
            "push_interaction_phase": interaction_phase[:, None],
            "ee_to_object_rotvec": _relative_rotation_rotvec(
                tcp_pose[:-1, 3:], object_pose[:-1, 3:]
            ),
            "future_horizons_steps": np.asarray(future_horizons, dtype=np.int32),
            "object_future_delta_pos": future_pos_delta,
            "object_future_delta_rotvec": future_rotvec,
            "object_future_valid": future_valid,
        }
    )
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
        if task != OBJECTCENTRIC_V2_ENV_ID:
            raise ValueError(
                f"objectcentric_v2 requires {OBJECTCENTRIC_V2_ENV_ID} metadata, got {task!r}"
            )
        if contact_force_threshold < 0:
            raise ValueError("contact_force_threshold must be non-negative")
        schema_version = OBJECTCENTRIC_V2_SCHEMA_VERSION
        derive = lambda group: derive_trajectory_objectcentric_v2(
            group,
            camera,
            future_horizons=future_horizons,
            contact_force_threshold=contact_force_threshold,
        )
    else:
        raise ValueError(f"Unsupported schema: {schema!r}")

    records: list[dict[str, Any]] = []
    with h5py.File(trajectory_path, "r") as file:
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
                        "source": "robot_obj_contact_force",
                        "force_norm_threshold": contact_force_threshold,
                    },
                    "push_interaction_phase_labels": {
                        "0": "approach",
                        "1": "measured_contact",
                        "2": "measured_contact_while_object_moves",
                        "3": "goal_reached_after_contact",
                    },
                    "future_horizons_steps": list(future_horizons),
                }
            )

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
                "physical_contact_source": "robot_obj_contact_force",
                "contact_force_threshold": contact_force_threshold,
                "future_horizons_steps": list(future_horizons),
            }
        )
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
