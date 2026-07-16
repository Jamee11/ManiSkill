import json
import tempfile
import unittest
from pathlib import Path

from scripts.eort.audit_large_collection import audit_collection


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _collection(root: Path, split: str, seed: int, commit: str = "abc123", torch_version: str = "2.7.1") -> None:
    shard = root / "push_cube" / split / "fixed"
    _write(shard / "raw/Env/motionplanning/data.pd_ee_delta_pose.physx_cpu.seed_manifest.json", {
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
    })
    (derived / "traj_0.npz").write_bytes(b"fixture")
    _write(derived / "summary.json", {
        "oracle": True, "source_control_mode": "pd_ee_delta_pose", "trajectories": 1, "steps": 3,
        "visibility_qa": {"total_steps": 3, "relational_visible_steps": 2, "relational_segdepth_valid_steps": 1},
        "phase_qa": {"phase_key": "push_interaction_phase", "counts": {"0": 1, "1": 0, "2": 1, "3": 1}},
    })
    dataset = root / "lerobot" / split / "oracle_metric/maniskill_eort_push_cube_256_fixed_lerobot/meta/conversion_summary.json"
    _write(dataset, {"episodes": 1, "action_representation": "metric_task_delta_pose", "condition_source": "oracle"})


class CollectionAuditTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
