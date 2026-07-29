"""Offline calibration and limit audit for EORT metric task actions.

This tool transforms axes and checks declared per-command limits.  It never
imports a robot SDK or sends a hardware command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.eort.action_contract import rotate_metric_task_command
except ModuleNotFoundError:  # Direct `python scripts/eort/...` entrypoint.
    from action_contract import rotate_metric_task_command


SCHEMA_VERSION = "eort_hardware_calibration_v1"


def load_calibration(path: Path) -> dict:
    calibration = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "robot_id",
        "arm_id",
        "command_hz",
        "robot_base_from_task_rotation",
        "max_translation_step_m",
        "max_rotation_step_rad",
        "gripper_native_closed",
        "gripper_native_open",
    }
    missing = sorted(required - calibration.keys())
    if missing:
        raise ValueError(f"calibration is missing: {', '.join(missing)}")
    if calibration["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("robot_id", "arm_id"):
        if not isinstance(calibration[key], str) or not calibration[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    for key in ("command_hz", "max_translation_step_m", "max_rotation_step_rad"):
        value = float(calibration[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("gripper_native_closed", "gripper_native_open"):
        if not np.isfinite(float(calibration[key])):
            raise ValueError(f"{key} must be finite")
    if np.isclose(calibration["gripper_native_closed"], calibration["gripper_native_open"]):
        raise ValueError("native gripper open and closed values must differ")
    # Reuse the production transform's proper-rotation validation.
    rotate_metric_task_command(
        np.zeros((1, 7), dtype=np.float32),
        np.asarray(calibration["robot_base_from_task_rotation"], dtype=np.float64),
    )
    return calibration


def load_actions(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise ValueError(f"{path} does not contain {key!r}")
            return archive[key]
    raise ValueError("actions must be a .npy file or an EORT .npz sidecar")


def audit_actions(actions: np.ndarray, calibration: dict) -> dict:
    if np.asarray(actions).shape[0] == 0:
        raise ValueError("actions must contain at least one command")
    transformed = rotate_metric_task_command(
        actions, np.asarray(calibration["robot_base_from_task_rotation"], dtype=np.float64)
    )
    translation_norm = np.linalg.norm(transformed[:, :3], axis=1)
    rotation_norm = np.linalg.norm(transformed[:, 3:6], axis=1)
    gripper = transformed[:, 6]
    translation_violations = int(
        np.count_nonzero(translation_norm > float(calibration["max_translation_step_m"]) + 1e-9)
    )
    rotation_violations = int(
        np.count_nonzero(rotation_norm > float(calibration["max_rotation_step_rad"]) + 1e-9)
    )
    gripper_violations = int(np.count_nonzero((gripper < 0.0) | (gripper > 1.0)))
    native_gripper = float(calibration["gripper_native_closed"]) + gripper * (
        float(calibration["gripper_native_open"]) - float(calibration["gripper_native_closed"])
    )
    passed = translation_violations == rotation_violations == gripper_violations == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "robot_id": calibration["robot_id"],
        "arm_id": calibration["arm_id"],
        "command_frame": "robot_base",
        "command_hz": float(calibration["command_hz"]),
        "action_count": int(transformed.shape[0]),
        "max_observed_translation_step_m": float(translation_norm.max()),
        "max_observed_rotation_step_rad": float(rotation_norm.max()),
        "native_gripper_range_observed": [
            float(native_gripper.min()),
            float(native_gripper.max()),
        ],
        "translation_limit_violations": translation_violations,
        "rotation_limit_violations": rotation_violations,
        "gripper_range_violations": gripper_violations,
        "passed": passed,
        "hardware_execution_validated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--action-key", default="metric_task_delta_pose_command")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = audit_actions(
        load_actions(args.actions, args.action_key), load_calibration(args.calibration)
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as file:
            file.write(rendered + "\n")
    print(rendered)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
