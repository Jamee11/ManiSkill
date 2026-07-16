import json
import tempfile
import unittest
from pathlib import Path

from scripts.eort.audit_large_collection import audit_collection


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _collection(root: Path, split: str, seed: int) -> None:
    shard = root / "push_cube" / split / "fixed"
    _write(shard / "raw/Env/motionplanning/data.pd_ee_delta_pose.physx_cpu.seed_manifest.json", {
        "saved_trajectory_seeds": [{"trajectory": "traj_0", "seed": seed, "success": True}]
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

            _collection(root, "val", 10)
            with self.assertRaisesRegex(ValueError, "appears in both"):
                audit_collection(root, "push_cube", ["train", "val"], ["fixed"], "metric_task_delta_pose", ["oracle"])


if __name__ == "__main__":
    unittest.main()
