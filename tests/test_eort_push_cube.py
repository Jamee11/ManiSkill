import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eort" / "derive_push_cube_eort.py"


def load_module():
    spec = importlib.util.spec_from_file_location("derive_push_cube_eort", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PushCubeEORTTest(unittest.TestCase):
    def test_derivation_preserves_transition_alignment(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "push_cube.h5"
            trajectory.with_suffix(".json").write_text(
                json.dumps({"env_info": {"env_id": "PushCube-v1"}}), encoding="utf-8"
            )
            with h5py.File(trajectory, "w") as file:
                group = file.create_group("traj_0")
                group.create_dataset("actions", data=np.arange(6, dtype=np.float32).reshape(3, 2))
                group.create_dataset("success", data=np.array([False, False, True]))
                obs = group.create_group("obs")
                extra = obs.create_group("extra")
                zeros = np.zeros((4, 7), dtype=np.float32)
                zeros[:, 3] = 1.0
                extra.create_dataset("tcp_pose", data=zeros)
                object_pose = zeros.copy()
                object_pose[:, 0] = [0.0, 0.5, 1.0, 1.0]
                extra.create_dataset("obj_pose", data=object_pose)
                extra.create_dataset("goal_pos", data=np.tile([1.0, 0.0, 0.0], (4, 1)))
                camera = obs.create_group("sensor_data").create_group("base_camera")
                camera.create_dataset("rgb", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
                camera.create_dataset("depth", data=np.zeros((4, 2, 2, 1), dtype=np.uint16))
                camera.create_dataset("segmentation", data=np.zeros((4, 2, 2, 1), dtype=np.uint16))

            summary = module.derive_dataset(trajectory, root / "derived")
            self.assertEqual(summary["trajectories"], 1)
            self.assertEqual(summary["steps"], 3)
            with np.load(root / "derived" / "traj_0.npz") as labels:
                np.testing.assert_allclose(labels["ee_to_object"][:, 0], [0.0, 0.5, 1.0])
                np.testing.assert_allclose(labels["object_to_goal_dist"][:, 0], [1.0, 0.5, 0.0])
                np.testing.assert_allclose(labels["goal_progress"][:, 0], [0.0, 0.5, 1.0])
                np.testing.assert_allclose(labels["object_next_delta"][:, 0], [0.5, 0.5, 0.0])
                self.assertEqual(labels["action"].shape, (3, 2))

    def test_objectcentric_v2_uses_measured_contact_and_future_pose(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "push_cube_eort.h5"
            trajectory.with_suffix(".json").write_text(
                json.dumps({"env_info": {"env_id": "PushCubeEORT-v1"}}),
                encoding="utf-8",
            )
            with h5py.File(trajectory, "w") as file:
                group = file.create_group("traj_0")
                group.create_dataset("actions", data=np.arange(6, dtype=np.float32).reshape(3, 2))
                group.create_dataset("success", data=np.array([False, False, True]))
                obs = group.create_group("obs")
                extra = obs.create_group("extra")
                tcp_pose = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (4, 1)).astype(
                    np.float32
                )
                object_pose = tcp_pose.copy()
                object_pose[:, 0] = [0.0, 0.5, 1.0, 1.0]
                quarter_turn = np.float32(np.sqrt(0.5))
                object_pose[1:, 3:] = [quarter_turn, 0.0, 0.0, quarter_turn]
                extra.create_dataset("tcp_pose", data=tcp_pose)
                extra.create_dataset("obj_pose", data=object_pose)
                extra.create_dataset("goal_pos", data=np.tile([1.0, 0.0, 0.0], (4, 1)))
                extra.create_dataset(
                    "obj_linear_velocity",
                    data=np.array([[0, 0, 0], [0.2, 0, 0], [0.1, 0, 0], [0, 0, 0]], dtype=np.float32),
                )
                extra.create_dataset("obj_angular_velocity", data=np.zeros((4, 3), dtype=np.float32))
                extra.create_dataset(
                    "robot_obj_contact_force",
                    data=np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32),
                )
                extra.create_dataset(
                    "robot_obj_contact_force_norm",
                    data=np.array([[0], [2], [1], [0]], dtype=np.float32),
                )
                extra.create_dataset("obj_segmentation_id", data=np.full((4, 1), 7, dtype=np.int32))
                extra.create_dataset("goal_segmentation_id", data=np.full((4, 1), 9, dtype=np.int32))
                camera = obs.create_group("sensor_data").create_group("base_camera")
                camera.create_dataset("rgb", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
                camera.create_dataset("depth", data=np.zeros((4, 2, 2, 1), dtype=np.uint16))
                camera.create_dataset(
                    "segmentation",
                    data=np.array(
                        [
                            [[[7], [9]], [[0], [7]]],
                            [[[9], [9]], [[0], [0]]],
                            [[[7], [0]], [[0], [9]]],
                            [[[7], [0]], [[0], [9]]],
                        ],
                        dtype=np.int32,
                    ),
                )

            summary = module.derive_dataset(
                trajectory,
                root / "derived",
                schema="objectcentric_v2",
            )
            self.assertEqual(summary["schema_version"], module.OBJECTCENTRIC_V2_SCHEMA_VERSION)
            self.assertEqual(summary["future_horizons_steps"], [1, 4, 8])
            manifest = json.loads((root / "derived" / "manifest.jsonl").read_text())
            self.assertEqual(manifest["physical_contact"]["source"], "robot_obj_contact_force_norm")
            self.assertEqual(manifest["push_interaction_phase_labels"]["2"], "measured_contact_while_object_moves")
            self.assertEqual(manifest["segmentation_visibility"]["object_id_source"], "obs/extra/obj_segmentation_id")
            with np.load(root / "derived" / "traj_0.npz") as labels:
                self.assertEqual(labels["physical_contact"].tolist(), [[False], [True], [True]])
                self.assertEqual(labels["push_interaction_phase"].tolist(), [[0], [2], [2]])
                self.assertEqual(labels["object_future_delta_pos"].shape, (3, 3, 3))
                self.assertEqual(labels["object_future_valid"].tolist(), [[True, False, False], [True, False, False], [True, False, False]])
                np.testing.assert_allclose(labels["object_future_delta_pos"][:, 0, 0], [0.5, 0.5, 0.0])
                np.testing.assert_allclose(labels["ee_to_object_rotvec"][0], [0.0, 0.0, 0.0])
                np.testing.assert_allclose(
                    labels["ee_to_object_rotvec"][1], [0.0, 0.0, np.pi / 2], atol=1e-6
                )
                self.assertEqual(labels["object_segmentation_id"].tolist(), [7])
                self.assertEqual(labels["goal_segmentation_id"].tolist(), [9])
                self.assertEqual(labels["object_mask_pixels"].tolist(), [[2], [0], [1]])
                self.assertEqual(labels["object_visible"].tolist(), [[True], [False], [True]])
                np.testing.assert_allclose(labels["object_visibility_fraction"][:, 0], [0.5, 0.0, 0.25])
                self.assertEqual(labels["goal_mask_pixels"].tolist(), [[1], [2], [1]])

    def test_objectcentric_task_is_registered_without_changing_pushcube(self):
        import gymnasium as gym
        import mani_skill.envs.tasks  # noqa: F401
        from mani_skill.envs.tasks.tabletop.push_cube_eort import PushCubeEORTEnv

        self.assertEqual(gym.spec("PushCubeEORT-v1").id, "PushCubeEORT-v1")
        self.assertEqual(PushCubeEORTEnv.SUPPORTED_ROBOTS, ["panda"])


if __name__ == "__main__":
    unittest.main()
