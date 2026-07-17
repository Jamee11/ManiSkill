import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.eort.action_contract import (
    metric_task_to_panda_normalized,
    panda_normalized_to_metric_task,
    rotate_metric_task_command,
)


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
                camera.create_dataset("depth", data=np.full((4, 2, 2, 1), 1000, dtype=np.uint16))
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
                json.dumps(
                    {"env_info": {"env_id": "PushCubeEORT-v1", "env_kwargs": {"control_mode": "pd_joint_pos"}}}
                ),
                encoding="utf-8",
            )
            with h5py.File(trajectory, "w") as file:
                group = file.create_group("traj_0")
                actions = np.zeros((3, 8), dtype=np.float32)
                actions[:, 7] = [-1, 0, 1]
                group.create_dataset("actions", data=actions)
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
                extra.create_dataset("obj_extent", data=np.tile([0.04, 0.04, 0.04], (4, 1)))
                extra.create_dataset("obj_friction", data=np.tile([0.3, 0.3], (4, 1)))
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
                extra.create_dataset("occluder_active", data=np.array([[False], [True], [True], [True]]))
                extra.create_dataset("is_grasped", data=np.array([[False], [True], [True], [True]]))
                camera = obs.create_group("sensor_data").create_group("base_camera")
                camera.create_dataset("rgb", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
                camera.create_dataset("depth", data=np.full((4, 2, 2, 1), 1000, dtype=np.uint16))
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
                params = obs.create_group("sensor_param").create_group("base_camera")
                params.create_dataset("intrinsic_cv", data=np.tile(np.eye(3, dtype=np.float32), (4, 1, 1)))
                params.create_dataset(
                    "extrinsic_cv",
                    data=np.tile(
                        np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1).astype(np.float32),
                        (4, 1, 1),
                    ),
                )

            summary = module.derive_dataset(
                trajectory,
                root / "derived",
                schema="objectcentric_v2",
            )
            self.assertEqual(summary["schema_version"], module.OBJECTCENTRIC_V2_SCHEMA_VERSION)
            self.assertEqual(summary["future_horizons_steps"], [1, 4, 8])
            self.assertEqual(
                summary["visibility_qa"],
                {
                    "total_steps": 3,
                    "object_visible_steps": 2,
                    "goal_visible_steps": 3,
                    "relational_visible_steps": 2,
                    "relational_segdepth_valid_steps": 2,
                    "object_mask_pixels_min_median_max": [0, 1.0, 2],
                    "goal_mask_pixels_min_median_max": [1, 1.0, 2],
                    "occluder_active_steps": 2,
                },
            )
            self.assertEqual(summary["phase_qa"]["counts"], {"0": 1, "1": 0, "2": 1, "3": 1})
            self.assertEqual(summary["phase_qa"]["task_progress_steps"], 1)
            manifest = json.loads((root / "derived" / "manifest.jsonl").read_text())
            self.assertEqual(manifest["physical_contact"]["source"], "robot_obj_contact_force_norm")
            self.assertEqual(manifest["push_interaction_phase_labels"]["2"], "measured_contact_while_object_moves")
            self.assertEqual(manifest["push_interaction_phase_labels"]["3"], "native_task_success")
            self.assertEqual(manifest["segmentation_visibility"]["object_id_source"], "obs/extra/obj_segmentation_id")
            self.assertIn("x_max_exclusive", manifest["segmentation_visibility"]["bbox_xyxy"])
            self.assertFalse(manifest["occluder_active"]["policy_input"])
            self.assertFalse(manifest["eef_transition_local"]["controller_command"])
            self.assertFalse(manifest["object_extent"]["policy_input"])
            self.assertFalse(manifest["object_friction"]["policy_input"])
            with np.load(root / "derived" / "traj_0.npz") as labels:
                np.testing.assert_allclose(labels["object_extent"], [[0.04, 0.04, 0.04]])
                np.testing.assert_allclose(labels["object_friction"], [[0.3, 0.3]])
                self.assertEqual(labels["physical_contact"].tolist(), [[False], [True], [True]])
                self.assertEqual(labels["push_interaction_phase"].tolist(), [[0], [2], [3]])
                self.assertEqual(labels["object_future_delta_pos"].shape, (3, 3, 3))
                self.assertEqual(labels["object_future_valid"].tolist(), [[True, False, False], [True, False, False], [True, False, False]])
                np.testing.assert_allclose(labels["object_future_delta_pos"][:, 0, 0], [0.5, 0.5, 0.0])
                np.testing.assert_allclose(labels["ee_to_object_rotvec"][0], [0.0, 0.0, 0.0])
                np.testing.assert_allclose(
                    labels["ee_to_object_rotvec"][1], [0.0, 0.0, np.pi / 2], atol=1e-6
                )
                self.assertEqual(labels["object_segmentation_id"].tolist(), [7])
                self.assertEqual(labels["goal_segmentation_id"].tolist(), [9])
                self.assertEqual(labels["occluder_active"].tolist(), [[False], [True], [True]])
                self.assertEqual(labels["object_mask_pixels"].tolist(), [[2], [0], [1]])
                self.assertEqual(labels["object_visible"].tolist(), [[True], [False], [True]])
                np.testing.assert_allclose(labels["object_visibility_fraction"][:, 0], [0.5, 0.0, 0.25])
                self.assertEqual(labels["object_bbox_xyxy"].tolist(), [[0, 0, 2, 2], [0, 0, 0, 0], [0, 0, 1, 1]])
                np.testing.assert_allclose(labels["object_mask_centroid_uv"], [[0.5, 0.5], [0, 0], [0, 0]])
                self.assertEqual(labels["object_segdepth_valid"].tolist(), [[True], [False], [True]])
                np.testing.assert_allclose(labels["object_segdepth_centroid_world"], [[0.5, 0.5, 1.0], [0, 0, 0], [0, 0, 1.0]])
                self.assertEqual(labels["goal_mask_pixels"].tolist(), [[1], [2], [1]])
                self.assertEqual(labels["goal_bbox_xyxy"].tolist(), [[1, 0, 2, 1], [0, 0, 2, 1], [1, 1, 2, 2]])
                np.testing.assert_allclose(labels["goal_mask_centroid_uv"], [[1, 0], [0.5, 0], [1, 1]])
                self.assertEqual(labels["goal_segdepth_valid"].tolist(), [[True], [True], [True]])
                np.testing.assert_allclose(
                    labels["goal_segdepth_centroid_world"],
                    [[1, 0, 1], [0.5, 0, 1], [1, 1, 1]],
                )
                np.testing.assert_allclose(labels["eef_transition_local"][:, :6], 0.0)
                np.testing.assert_allclose(labels["eef_transition_local"][:, 6], [0.0, 0.5, 1.0])

            with h5py.File(trajectory, "r+") as file:
                del file["traj_0/obs/extra/obj_extent"]
                del file["traj_0/obs/extra/obj_friction"]
            trajectory.with_suffix(".json").write_text(
                json.dumps(
                    {"env_info": {"env_id": "PickCubeEORT-v1", "env_kwargs": {"control_mode": "pd_joint_pos"}}}
                ),
                encoding="utf-8",
            )
            pick_summary = module.derive_dataset(
                trajectory, root / "derived_pick", schema="objectcentric_v2"
            )
            self.assertEqual(pick_summary["schema_version"], module.PICK_CUBE_OBJECTCENTRIC_V2_SCHEMA_VERSION)
            self.assertEqual(pick_summary["phase_qa"]["phase_key"], "pick_interaction_phase")
            pick_manifest = json.loads((root / "derived_pick" / "manifest.jsonl").read_text())
            self.assertEqual(pick_manifest["pick_interaction_phase_labels"]["2"], "native_grasp_detected")
            self.assertEqual(pick_manifest["object_extent"]["source"], "legacy_fixed_panda_cube_0.04m")
            self.assertEqual(pick_manifest["object_friction"]["source"], "legacy_default_material_0.3")
            with np.load(root / "derived_pick" / "traj_0.npz") as labels:
                self.assertEqual(labels["pick_interaction_phase"].tolist(), [[0], [2], [3]])
                np.testing.assert_allclose(labels["object_extent"], [[0.04, 0.04, 0.04]])

            controller_actions = np.zeros((3, 7), dtype=np.float32)
            controller_actions[:, :3] = [[0.2, -0.3, 0.4], [0, 0, 0], [-0.5, 0.1, 0.2]]
            controller_actions[:, 3:6] = [[0.1, 0.2, -0.3], [0, 0, 0], [0.4, -0.2, 0.1]]
            controller_actions[:, -1] = [-1, 0, 1]
            with h5py.File(trajectory, "r+") as file:
                del file["traj_0/actions"]
                file["traj_0"].create_dataset("actions", data=controller_actions)
            trajectory.with_suffix(".json").write_text(
                json.dumps(
                    {"env_info": {"env_id": "PushCubeEORT-v1", "env_kwargs": {"control_mode": "pd_ee_delta_pose"}}}
                ),
                encoding="utf-8",
            )
            module.derive_dataset(trajectory, root / "derived_controller", schema="objectcentric_v2")
            controller_manifest = json.loads(
                (root / "derived_controller" / "manifest.jsonl").read_text()
            )
            self.assertTrue(controller_manifest["panda_pd_ee_delta_pose_command"]["controller_command"])
            self.assertTrue(controller_manifest["metric_task_delta_pose_command"]["controller_command"])
            with np.load(root / "derived_controller" / "traj_0.npz") as labels:
                np.testing.assert_array_equal(
                    labels["panda_pd_ee_delta_pose_command"], controller_actions
                )
                metric = labels["metric_task_delta_pose_command"]
                np.testing.assert_allclose(metric[:, :3], controller_actions[:, :3] * 0.1)
                np.testing.assert_allclose(metric[:, 6], [0.0, 0.5, 1.0])
                np.testing.assert_allclose(metric_task_to_panda_normalized(metric), controller_actions, atol=1e-6)

    def test_metric_task_action_rotation_is_calibration_reversible(self):
        action = np.array([[0.2, -0.3, 0.4, 0.1, 0.2, -0.3, 0.0]], dtype=np.float32)
        metric = panda_normalized_to_metric_task(action)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        rotated = rotate_metric_task_command(metric, rotation)
        recovered = rotate_metric_task_command(rotated, rotation.T)
        np.testing.assert_allclose(recovered, metric, atol=1e-7)

    def test_local_eef_transition_uses_current_eef_frame(self):
        module = load_module()
        quarter_turn = np.sqrt(0.5)
        tcp_pose = np.array(
            [
                [0, 0, 0, quarter_turn, 0, 0, quarter_turn],
                [0, 1, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        transition = module._local_eef_transition(tcp_pose, np.array([-1], dtype=np.float32))
        np.testing.assert_allclose(transition[0, :3], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(transition[0, 3:6], [0, 0, np.pi / 2], atol=1e-6)
        self.assertEqual(transition[0, 6], 0.0)

    def test_objectcentric_task_is_registered_without_changing_pushcube(self):
        import gymnasium as gym
        import mani_skill.envs.tasks  # noqa: F401
        from mani_skill.envs.tasks.tabletop.push_cube_eort import (
            PushCubeEORTCameraRandEnv,
            PushCubeEORTEnv,
            PushCubeEORTOccludedEnv,
            PushCubeEORTGeometryRandEnv,
        )

        self.assertEqual(gym.spec("PushCubeEORT-v1").id, "PushCubeEORT-v1")
        self.assertEqual(gym.spec("PushCubeEORTCameraRand-v1").id, "PushCubeEORTCameraRand-v1")
        self.assertEqual(gym.spec("PushCubeEORTOccluded-v1").id, "PushCubeEORTOccluded-v1")
        self.assertEqual(gym.spec("PushCubeEORTGeometryRand-v1").id, "PushCubeEORTGeometryRand-v1")
        self.assertEqual(PushCubeEORTEnv.SUPPORTED_ROBOTS, ["panda"])
        self.assertEqual(PushCubeEORTEnv.EORT_CAMERA_RESOLUTION, 256)
        self.assertEqual(PushCubeEORTCameraRandEnv.CAMERA_EYE_JITTER, (0.08, 0.08, 0.05))
        self.assertEqual(PushCubeEORTOccludedEnv.OCCLUDER_PROBABILITY, 0.5)
        self.assertEqual(PushCubeEORTGeometryRandEnv.CUBE_HALF_SIZE_RANGE, (0.017, 0.023))

    def test_pickcube_eort_is_registered_with_visual_goal_marker(self):
        import gymnasium as gym
        import mani_skill.envs.tasks  # noqa: F401
        from mani_skill.envs.tasks.tabletop.pick_cube_eort import (
            PickCubeEORTCameraRandEnv,
            PickCubeEORTEnv,
            PickCubeEORTGeometryRandEnv,
            PickCubeEORTOccludedEnv,
        )

        self.assertEqual(gym.spec("PickCubeEORT-v1").id, "PickCubeEORT-v1")
        self.assertEqual(gym.spec("PickCubeEORTCameraRand-v1").id, "PickCubeEORTCameraRand-v1")
        self.assertEqual(gym.spec("PickCubeEORTOccluded-v1").id, "PickCubeEORTOccluded-v1")
        self.assertEqual(gym.spec("PickCubeEORTGeometryRand-v1").id, "PickCubeEORTGeometryRand-v1")
        self.assertEqual(PickCubeEORTEnv.SUPPORTED_ROBOTS, ["panda"])
        self.assertEqual(PickCubeEORTEnv.EORT_CAMERA_RESOLUTION, 256)
        self.assertEqual(PickCubeEORTCameraRandEnv.CAMERA_EYE_JITTER, (0.08, 0.08, 0.05))
        self.assertEqual(PickCubeEORTOccludedEnv.OCCLUDER_PROBABILITY, 0.5)
        self.assertEqual(PickCubeEORTGeometryRandEnv.CUBE_HALF_SIZE_RANGE, (0.017, 0.023))


if __name__ == "__main__":
    unittest.main()
