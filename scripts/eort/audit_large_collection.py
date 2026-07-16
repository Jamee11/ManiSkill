"""Fail-fast QA for one completed ManiSkill EORT collection root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPORTS = {
    "oracle": ("oracle", "_lerobot"),
    "segdepth_proxy": ("segdepth_track_proxy", "_segdepth_track_proxy_lerobot"),
    "track_corrupt": ("segdepth_track_proxy", "_segdepth_track_corrupt_lerobot"),
}


def _one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {label}, found {len(paths)}")
    return paths[0]


def _json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing {path}")
    return json.loads(path.read_text())


def audit_collection(
    root: Path,
    task: str,
    splits: list[str],
    variants: list[str],
    action_source: str,
    exports: list[str],
) -> dict:
    unknown_exports = set(exports) - EXPORTS.keys()
    if unknown_exports:
        raise ValueError(f"Unknown exports: {sorted(unknown_exports)}")
    action_key = {
        "panda_pd_ee_delta_pose": "panda_pd_ee_delta_pose_command",
        "metric_task_delta_pose": "metric_task_delta_pose_command",
    }[action_source]
    action_suffix = "_metric" if action_source == "metric_task_delta_pose" else ""
    prefix = f"maniskill_eort_{task}_256"
    owners: dict[int, str] = {}
    report = {"root": str(root), "task": task, "action_source": action_source, "shards": []}

    for split in splits:
        for variant in variants:
            name = f"{task}/{split}/{variant}"
            shard = root / task / split / variant
            seed_path = _one(
                list(shard.glob("raw/*/motionplanning/*.pd_ee_delta_pose.physx_cpu.seed_manifest.json")),
                f"controller seed manifest for {name}",
            )
            seed_rows = _json(seed_path).get("saved_trajectory_seeds", [])
            if not seed_rows or not all(row.get("success") is True for row in seed_rows):
                raise ValueError(f"{name} has missing or unsuccessful seeds")
            for row in seed_rows:
                seed = int(row["seed"])
                if seed in owners:
                    raise ValueError(f"Simulator seed {seed} appears in both {owners[seed]} and {name}")
                owners[seed] = name

            derived = shard / "derived_controller_goal_segdepth"
            records = [json.loads(line) for line in (derived / "manifest.jsonl").read_text().splitlines() if line]
            expected_trajectories = {row["trajectory"] for row in seed_rows}
            if {row.get("source_trajectory") for row in records} != expected_trajectories:
                raise ValueError(f"{name} seed and sidecar trajectory IDs differ")
            if len(list(derived.glob("traj_*.npz"))) != len(seed_rows):
                raise ValueError(f"{name} sidecar count differs from successful seed count")
            if not all(
                row.get("source_control_mode") == "pd_ee_delta_pose"
                and row.get(action_key, {}).get("controller_command") is True
                for row in records
            ):
                raise ValueError(f"{name} lacks verified {action_source} provenance")

            summary = _json(derived / "summary.json")
            qa = summary.get("visibility_qa")
            if (
                summary.get("oracle") is not True
                or summary.get("source_control_mode") != "pd_ee_delta_pose"
                or summary.get("trajectories") != len(seed_rows)
                or not qa
                or qa.get("total_steps") != summary.get("steps")
                or not 0 <= qa.get("relational_visible_steps", -1) <= qa["total_steps"]
                or not 0 <= qa.get("relational_segdepth_valid_steps", -1) <= qa["total_steps"]
            ):
                raise ValueError(f"{name} summary/provenance/visibility QA is incomplete")

            for export in exports:
                condition, dataset_tail = EXPORTS[export]
                dataset = root / "lerobot" / split / f"{export}{action_suffix}" / f"{prefix}_{variant}{dataset_tail}"
                converted = _json(dataset / "meta" / "conversion_summary.json")
                if (
                    converted.get("episodes") != len(seed_rows)
                    or converted.get("action_representation") != action_source
                    or converted.get("condition_source") != condition
                ):
                    raise ValueError(f"{name} {export} LeRobot contract differs from source shard")

            report["shards"].append(
                {"name": name, "episodes": len(seed_rows), "steps": summary["steps"], **qa}
            )

    report["episodes"] = sum(row["episodes"] for row in report["shards"])
    report["steps"] = sum(row["steps"] for row in report["shards"])
    report["unique_simulator_seeds"] = len(owners)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=("push_cube", "pick_cube"), default="push_cube")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--variants", default="fixed,camera_rand,occluded")
    parser.add_argument("--action-source", choices=("panda_pd_ee_delta_pose", "metric_task_delta_pose"), default="panda_pd_ee_delta_pose")
    parser.add_argument("--exports", default="oracle,segdepth_proxy,track_corrupt")
    args = parser.parse_args()
    print(json.dumps(audit_collection(
        args.root,
        args.task,
        args.splits.split(","),
        args.variants.split(","),
        args.action_source,
        args.exports.split(","),
    ), indent=2))


if __name__ == "__main__":
    main()
