"""Fail-fast QA for one completed ManiSkill EORT collection root."""

from __future__ import annotations

import argparse
import json
import math
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


def _valid_positive_range(value: object, width: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(row, list) and len(row) == width for row in value)
        and all(isinstance(item, (int, float)) and math.isfinite(item) for row in value for item in row)
        and all(0 < value[0][i] <= value[1][i] <= value[2][i] for i in range(width))
    )


def _parse_expected_counts(value: str) -> dict[str, int]:
    if not value:
        return {}
    result: dict[str, int] = {}
    for item in value.split(","):
        split, separator, count = item.partition("=")
        if separator != "=" or split not in {"train", "val", "test"} or split in result:
            raise ValueError(f"Invalid expected count: {item!r}")
        try:
            result[split] = int(count)
        except ValueError as error:
            raise ValueError(f"Invalid expected count: {item!r}") from error
        if result[split] <= 0:
            raise ValueError(f"Expected count must be positive: {item!r}")
    return result


def audit_collection(
    root: Path,
    task: str,
    splits: list[str],
    variants: list[str],
    action_source: str,
    exports: list[str],
    future_targets: bool = False,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    unknown_exports = set(exports) - EXPORTS.keys()
    if unknown_exports:
        raise ValueError(f"Unknown exports: {sorted(unknown_exports)}")
    action_key = {
        "panda_pd_ee_delta_pose": "panda_pd_ee_delta_pose_command",
        "metric_task_delta_pose": "metric_task_delta_pose_command",
    }[action_source]
    if future_targets and action_source != "metric_task_delta_pose":
        raise ValueError("Future targets require metric_task_delta_pose")
    action_suffix = "_metric_dynamics" if future_targets else ("_metric" if action_source == "metric_task_delta_pose" else "")
    prefix = f"maniskill_eort_{task}_256"
    owners: dict[int, str] = {}
    source_commits: set[str] = set()
    runtimes: dict[str, dict] = {}
    expected_counts = expected_counts or {}
    report = {"root": str(root), "task": task, "action_source": action_source, "future_targets": future_targets, "expected_episodes_per_variant": expected_counts, "shards": []}

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
            if split in expected_counts and len(seed_rows) != expected_counts[split]:
                raise ValueError(
                    f"{name} has {len(seed_rows)} episodes, expected {expected_counts[split]}"
                )
            metadata_path = seed_path.with_name(seed_path.name.removesuffix(".seed_manifest.json") + ".json")
            metadata = _json(metadata_path)
            runtime = _json(shard / "runtime_manifest.json")
            packages = runtime.get("packages") or {}
            if (
                runtime.get("schema") != "maniskill_eort_runtime_v1"
                or not runtime.get("python")
                or not runtime.get("cuda_visible_devices")
                or set(packages) != {"mani_skill", "torch", "sapien", "numpy", "h5py"}
                or not all(packages.values())
            ):
                raise ValueError(f"{name} runtime manifest is incomplete")
            signature = {"python": runtime["python"], "packages": packages}
            runtimes[json.dumps(signature, sort_keys=True)] = signature
            env_info, commit_info = metadata.get("env_info", {}), metadata.get("commit_info", {})
            suffix = {"fixed": "", "camera_rand": "CameraRand", "occluded": "Occluded", "geometry_rand": "GeometryRand"}.get(variant)
            if suffix is None:
                raise ValueError(f"Unsupported variant {variant!r} for {task}")
            expected_env = f"{'PushCube' if task == 'push_cube' else 'PickCube'}EORT{suffix}-v1"
            env_kwargs = env_info.get("env_kwargs", {})
            if (
                env_info.get("env_id") != expected_env
                or env_kwargs.get("obs_mode") != "state_dict+rgb+depth+segmentation"
                or env_kwargs.get("control_mode") != "pd_ee_delta_pose"
                or env_kwargs.get("sim_backend") != "physx_cpu"
            ):
                raise ValueError(f"{name} controller environment provenance is invalid")
            commit = commit_info.get("commit_id")
            if not commit:
                raise ValueError(f"{name} lacks a ManiSkill source commit")
            source_commits.add(commit)
            metadata_seeds = sorted(int(row["episode_seed"]) for row in metadata.get("episodes", []) if row.get("success") is True)
            if metadata_seeds != sorted(int(row["seed"]) for row in seed_rows):
                raise ValueError(f"{name} controller metadata and seed manifest differ")
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
                and row.get("fields", {}).get("object_extent") == [1, 3]
                and row.get("object_extent", {}).get("policy_input") is False
                and row.get("object_extent", {}).get("source") == "obs/extra/obj_extent"
                and row.get("fields", {}).get("object_mass") == [1, 1]
                and row.get("object_mass", {}).get("source") == "obs/extra/obj_mass"
                and row.get("object_mass", {}).get("policy_input") is False
                and row.get("fields", {}).get("object_friction") == [1, 2]
                and row.get("object_friction", {}).get("source") == "obs/extra/obj_friction"
                and row.get("object_friction", {}).get("policy_input") is False
                for row in records
            ):
                raise ValueError(f"{name} lacks verified {action_source} or object-physics provenance")

            summary = _json(derived / "summary.json")
            qa = summary.get("visibility_qa")
            physics_qa = summary.get("object_physics_qa") or {}
            phase_qa = summary.get("phase_qa") or {}
            phase_counts = phase_qa.get("counts") or {}
            expected_phase_key = f"{'push' if task == 'push_cube' else 'pick'}_interaction_phase"
            if (
                summary.get("oracle") is not True
                or summary.get("source_control_mode") != "pd_ee_delta_pose"
                or summary.get("trajectories") != len(seed_rows)
                or not qa
                or qa.get("total_steps") != summary.get("steps")
                or not 0 <= qa.get("relational_visible_steps", -1) <= qa["total_steps"]
                or not 0 <= qa.get("relational_segdepth_valid_steps", -1) <= qa["total_steps"]
                or phase_qa.get("phase_key") != expected_phase_key
                or set(phase_counts) != {"0", "1", "2", "3"}
                or sum(phase_counts.values()) != summary.get("steps")
                or phase_counts["2"] <= 0
                or phase_counts["3"] <= 0
                or physics_qa.get("episodes") != len(seed_rows)
                or not _valid_positive_range(physics_qa.get("object_extent_min_median_max"), 3)
                or not _valid_positive_range(physics_qa.get("object_mass_min_median_max"), 1)
                or not _valid_positive_range(physics_qa.get("object_friction_min_median_max"), 2)
            ):
                raise ValueError(f"{name} summary/provenance/visibility phase QA or physics QA is incomplete")

            for export in exports:
                condition, dataset_tail = EXPORTS[export]
                dataset = root / "lerobot" / split / f"{export}{action_suffix}" / f"{prefix}_{variant}{dataset_tail}"
                converted = _json(dataset / "meta" / "conversion_summary.json")
                if (
                    converted.get("episodes") != len(seed_rows)
                    or converted.get("action_representation") != action_source
                    or converted.get("condition_source") != condition
                    or bool(converted.get("object_future_targets", False)) != future_targets
                ):
                    raise ValueError(f"{name} {export} LeRobot contract differs from source shard")

            report["shards"].append(
                {"name": name, "episodes": len(seed_rows), "steps": summary["steps"], "phase_qa": phase_qa, "object_physics_qa": physics_qa, **qa}
            )

    report["episodes"] = sum(row["episodes"] for row in report["shards"])
    report["steps"] = sum(row["steps"] for row in report["shards"])
    report["unique_simulator_seeds"] = len(owners)
    if len(source_commits) != 1:
        raise ValueError(f"Collection mixes ManiSkill commits: {sorted(source_commits)}")
    if len(runtimes) != 1:
        raise ValueError("Collection mixes Python/package runtimes")
    report["maniskill_commit"] = source_commits.pop()
    report["runtime"] = next(iter(runtimes.values()))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=("push_cube", "pick_cube"), default="push_cube")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--variants", default="fixed,camera_rand,occluded")
    parser.add_argument("--action-source", choices=("panda_pd_ee_delta_pose", "metric_task_delta_pose"), default="panda_pd_ee_delta_pose")
    parser.add_argument("--exports", default="oracle,segdepth_proxy,track_corrupt")
    parser.add_argument("--future-targets", action="store_true")
    parser.add_argument("--expected-counts", default="", help="Optional per-variant contract, e.g. train=500,val=100,test=200")
    args = parser.parse_args()
    print(json.dumps(audit_collection(
        args.root,
        args.task,
        args.splits.split(","),
        args.variants.split(","),
        args.action_source,
        args.exports.split(","),
        args.future_targets,
        _parse_expected_counts(args.expected_counts),
    ), indent=2))


if __name__ == "__main__":
    main()
