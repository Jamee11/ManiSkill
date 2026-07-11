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
    if env_id != "PushCube-v1":
        raise ValueError(f"Expected PushCube-v1 metadata, got {env_id!r}")
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


def derive_dataset(
    trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    camera: str = "base_camera",
) -> dict[str, Any]:
    trajectory_path = Path(trajectory_path).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    metadata = _load_source_metadata(trajectory_path)

    records: list[dict[str, Any]] = []
    with h5py.File(trajectory_path, "r") as file:
        outputs = [(key, derive_trajectory(file[key], camera)) for key in _trajectory_keys(file)]

    output_dir.mkdir(parents=True)
    for trajectory_id, arrays in outputs:
        npz_path = output_dir / f"{trajectory_id}.npz"
        np.savez_compressed(npz_path, **arrays)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
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

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "oracle": True,
        "task": metadata["env_info"]["env_id"],
        "source_h5": str(trajectory_path),
        "source_metadata": str(trajectory_path.with_suffix(".json")),
        "camera": camera,
        "trajectories": len(records),
        "steps": sum(record["num_steps"] for record in records),
    }
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
    args = parser.parse_args()
    print(json.dumps(derive_dataset(args.traj_path, args.output_dir, camera=args.camera), indent=2))


if __name__ == "__main__":
    main()
