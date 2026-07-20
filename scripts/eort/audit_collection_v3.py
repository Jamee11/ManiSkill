#!/usr/bin/env python3
"""Fail closed unless a v3 Sim2Real shard matches the raw/sidecar/LeRobot contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


CAMERAS = ("front_camera", "right_shoulder_camera", "hand_camera")
VIDEO_KEYS = {
    "observation.images.front",
    "observation.images.right_shoulder",
    "observation.images.wrist",
}


def _one(paths, label):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f"Expected one {label}, found {len(paths)}")
    return paths[0]


def audit(root: Path, task: str, splits: tuple[str, ...], expected: dict[str, int]) -> dict:
    prefix = f"maniskill_eort_{task}_256_sim2real_robot_base_oracle_lerobot"
    report = {"root": str(root), "task": task, "canonical_frame": "robot_base", "shards": []}
    for split in splits:
        shard = root / task / split / "sim2real"
        raw_path = _one(shard.glob("raw/*/motionplanning/*.pd_ee_delta_pose.physx_cpu.h5"), f"{split} HDF5")
        derived = shard / "derived_objectcentric_v3"
        records = [json.loads(line) for line in (derived / "manifest.jsonl").read_text().splitlines() if line]
        if split in expected and len(records) != expected[split]:
            raise ValueError(f"{split} has {len(records)} episodes, expected {expected[split]}")
        required = {
            "robot_base_pose_wxyz", "tcp_pose_base_wxyz", "object_pose_base_wxyz",
            "goal_pos_base", "ee_to_object_base", "object_to_goal_base",
            "metric_robot_base_delta_pose_command", "object_future_delta_pos_base",
        }
        if not records or not all(
            row.get("canonical_frame") == "robot_base"
            and tuple(row.get("cameras") or ()) == CAMERAS
            and row.get("metric_robot_base_delta_pose_command", {}).get("controller_command") is True
            and required <= set(row.get("fields") or {})
            for row in records
        ):
            raise ValueError(f"{split} sidecar manifest violates v3 frame/camera/action fields")
        with h5py.File(raw_path) as source:
            if set(source) != {row["source_trajectory"] for row in records}:
                raise ValueError(f"{split} raw and sidecar trajectory IDs differ")
            steps = 0
            for trajectory in source.values():
                count = len(trajectory["actions"])
                steps += count
                for camera in CAMERAS:
                    sensor = trajectory[f"obs/sensor_data/{camera}"]
                    if sensor["rgb"].shape != (count + 1, 256, 256, 3):
                        raise ValueError(f"{trajectory.name}/{camera}/rgb is not (T+1,256,256,3)")
                    for modality in ("depth", "segmentation"):
                        if sensor[modality].shape != (count + 1, 256, 256, 1):
                            raise ValueError(f"{trajectory.name}/{camera}/{modality} has invalid shape")
        summary = json.loads((derived / "summary.json").read_text())
        visibility = summary.get("visibility_qa") or {}
        if (
            summary.get("canonical_frame") != "robot_base"
            or summary.get("steps") != steps
            or set(visibility) != set(CAMERAS)
            or any(view.get("total_steps") != steps for view in visibility.values())
        ):
            raise ValueError(f"{split} visibility QA is incomplete")
        dataset = root / "lerobot" / split / "oracle" / prefix
        conversion = json.loads((dataset / "meta/conversion_summary.json").read_text())
        info = json.loads((dataset / "meta/info.json").read_text())
        videos = {key for key, value in info["features"].items() if value.get("dtype") == "video"}
        if (
            conversion.get("episodes") != len(records)
            or conversion.get("coordinate_frame") != "robot_base"
            or tuple(conversion.get("cameras") or ()) != CAMERAS
            or videos != VIDEO_KEYS
            or len(list((dataset / "videos").rglob("*.mp4"))) != 3 * len(records)
        ):
            raise ValueError(f"{split} LeRobot three-view export is incomplete")
        report["shards"].append({"split": split, "episodes": len(records), "steps": steps, "visibility_qa": visibility})
    report["episodes"] = sum(row["episodes"] for row in report["shards"])
    report["steps"] = sum(row["steps"] for row in report["shards"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=("push_cube", "pick_cube"), default="push_cube")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--expected-counts", default="train=500,val=100,test=200")
    args = parser.parse_args()
    expected = dict(item.split("=", 1) for item in args.expected_counts.split(",") if item)
    print(json.dumps(audit(args.root, args.task, tuple(args.splits.split(",")), {k: int(v) for k, v in expected.items()}), indent=2))


if __name__ == "__main__":
    main()
