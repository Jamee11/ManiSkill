import json
import tempfile
import unittest
from pathlib import Path

from scripts.eort.audit_large_collection import _parse_expected_counts, audit_collection


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _collection(root: Path, split: str, seed: int, commit: str = "abc123", torch_version: str = "2.7.1") -> None:
    shard = root / "push_cube" / split / "fixed"
    _write(shard / "raw/Env/motionplanning/data.pd_ee_delta_pose.physx_cpu.seed_manifest.json", {
        "only_count_success": True,
        "max_attempts": 5,
        "attempted_episodes": 1,
        "successful_episodes": 1,
        "failed_motion_plans": 0,
        "attempt_budget_exhausted": False,
        "saved_trajectory_seeds": [{"trajectory": "traj_0", "seed": seed, "success": True}]
    })
    _write(shard / "raw/Env/motionplanning/data.pd_ee_delta_pose.physx_cpu.json", {
        "env_info": {"env_id": "PushCubeEORT-v1", "env_kwargs": {
            "obs_mode": "state_dict+rgb+depth+segmentation", "control_mode": "pd_ee_delta_pose", "sim_backend": "physx_cpu",
        }},
        "commit_info": {"commit_id": commit, "branch": "test"},
        "episodes": [{"episode_seed": seed, "success": True}],
    })
    _write(shard / "runtime_manifest.json", {
        "schema": "maniskill_eort_runtime_v1", "python": "3.10.20", "cuda_visible_devices": "7",
        "packages": {"mani_skill": "3.0.1", "torch": torch_version, "sapien": "3.0.3", "numpy": "1.26.4", "h5py": "3.16.0"},
    })
    derived = shard / "derived_controller_goal_segdepth"
    _write(derived / "manifest.jsonl", {
        "source_trajectory": "traj_0", "source_control_mode": "pd_ee_delta_pose",
        "metric_task_delta_pose_command": {"controller_command": True},
        "fields": {"object_extent": [1, 3], "object_mass": [1, 1], "object_friction": [1, 2]},
        "object_extent": {"source": "obs/extra/obj_extent", "policy_input": False},
        "object_mass": {"source": "obs/extra/obj_mass", "policy_input": False},
        "object_friction": {"source": "obs/extra/obj_friction", "policy_input": False},
    })
    (derived / "traj_0.npz").write_bytes(b"fixture")
    _write(derived / "summary.json", {
        "oracle": True, "source_control_mode": "pd_ee_delta_pose", "trajectories": 1, "steps": 3,
        "visibility_qa": {"total_steps": 3, "relational_visible_steps": 2, "relational_segdepth_valid_steps": 1},
        "phase_qa": {"phase_key": "push_interaction_phase", "counts": {"0": 1, "1": 0, "2": 1, "3": 1}},
        "object_physics_qa": {
            "episodes": 1,
            "object_extent_min_median_max": [[0.04, 0.04, 0.04]] * 3,
            "object_mass_min_median_max": [[0.064]] * 3,
            "object_friction_min_median_max": [[0.3, 0.3]] * 3,
        },
    })
    dataset = root / "lerobot" / split / "oracle_metric/maniskill_eort_push_cube_256_fixed_lerobot/meta/conversion_summary.json"
    _write(dataset, {"episodes": 1, "action_representation": "metric_task_delta_pose", "condition_source": "oracle"})


class CollectionAuditTest(unittest.TestCase):
    def test_expected_episode_contract_rejects_partial_shards(self):
        self.assertEqual(_parse_expected_counts("train=500,val=100,test=200"), {"train": 500, "val": 100, "test": 200})
        with self.assertRaisesRegex(ValueError, "Invalid expected count"):
            _parse_expected_counts("train=1,train=2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _collection(root, "train", 10)
            with self.assertRaisesRegex(ValueError, "has 1 episodes, expected 500"):
                audit_collection(
                    root, "push_cube", ["train"], ["fixed"], "metric_task_delta_pose", ["oracle"],
                    expected_counts={"train": 500},
                )

    def test_accepts_complete_collection_and_rejects_seed_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _collection(root, "train", 10)
            _collection(root, "val", 20)
            report = audit_collection(root, "push_cube", ["train", "val"], ["fixed"], "metric_task_delta_pose", ["oracle"])
            self.assertEqual(report["episodes"], report["unique_simulator_seeds"])
            self.assertEqual(report["episodes"], 2)
            self.assertEqual(report["maniskill_commit"], "abc123")
            self.assertEqual(report["runtime"]["packages"]["sapien"], "3.0.3")
            self.assertEqual(report["shards"][0]["phase_qa"]["counts"]["2"], 1)
            self.assertEqual(report["shards"][0]["attempted_episodes"], 1)

            seed_manifest_path = root / "push_cube/val/fixed/raw/Env/motionplanning/data.pd_ee_delta_pose.physx_cpu.seed_manifest.json"
            seed_manifest = json.loads(seed_manifest_path.read_text())
            seed_manifest["attempt_budget_exhausted"] = True
            _write(seed_manifest_path, seed_manifest)
            with self.assertRaisesRegex(ValueError, "exhausted seed attempts"):
                audit_collection(root, "push_cube", ["train", "val"], ["fixed"], "metric_task_delta_pose", ["oracle"])

            _collection(root, "val", 20, commit="different")
            with self.assertRaisesRegex(ValueError, "mixes ManiSkill commits"):
                audit_collection(root, "push_cube", ["train", "val"], ["fixed"], "metric_task_delta_pose", ["oracle"])

            _collection(root, "val", 20, torch_version="different")
            with self.assertRaisesRegex(ValueError, "mixes Python/package runtimes"):
                audit_collection(root, "push_cube", ["train", "val"], ["fixed"], "metric_task_delta_pose", ["oracle"])

            _collection(root, "val", 10)
            with self.assertRaisesRegex(ValueError, "appears in both"):
                audit_collection(root, "push_cube", ["train", "val"], ["fixed"], "metric_task_delta_pose", ["oracle"])

            _collection(root, "val", 20)
            summary_path = root / "push_cube/val/fixed/derived_controller_goal_segdepth/summary.json"
            summary = json.loads(summary_path.read_text())
            summary["phase_qa"]["counts"] = {"0": 2, "1": 0, "2": 0, "3": 1}
            _write(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "phase QA"):
                audit_collection(root, "push_cube", ["train", "val"], ["fixed"], "metric_task_delta_pose", ["oracle"])

            summary["phase_qa"]["counts"] = {"0": 1, "1": 0, "2": 1, "3": 1}
            summary["object_physics_qa"]["object_mass_min_median_max"][2][0] = float("nan")
            _write(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "physics QA"):
                audit_collection(root, "push_cube", ["train", "val"], ["fixed"], "metric_task_delta_pose", ["oracle"])


if __name__ == "__main__":
    unittest.main()
