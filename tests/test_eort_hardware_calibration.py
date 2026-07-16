import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.eort.audit_metric_action_calibration import audit_actions, load_calibration


class TestEORTHardwareCalibration(unittest.TestCase):
    def setUp(self):
        self.calibration = {
            "schema_version": "eort_hardware_calibration_v1",
            "robot_id": "offline_test_robot",
            "arm_id": "left",
            "command_hz": 10.0,
            "robot_base_from_task_rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            "max_translation_step_m": 0.02,
            "max_rotation_step_rad": 0.1,
            "gripper_native_closed": 0.0,
            "gripper_native_open": 0.08,
        }

    def test_axis_transform_and_limits_pass(self):
        actions = np.array([[0.01, 0, 0, 0, 0, 0.05, 0.5]], dtype=np.float32)
        report = audit_actions(actions, self.calibration)
        self.assertTrue(report["passed"])
        self.assertFalse(report["hardware_execution_validated"])
        self.assertAlmostEqual(report["native_gripper_range_observed"][0], 0.04)

    def test_limit_and_gripper_violations_fail(self):
        actions = np.array([[0.03, 0, 0, 0, 0, 0.2, 1.1]], dtype=np.float32)
        report = audit_actions(actions, self.calibration)
        self.assertFalse(report["passed"])
        self.assertEqual(report["translation_limit_violations"], 1)
        self.assertEqual(report["rotation_limit_violations"], 1)
        self.assertEqual(report["gripper_range_violations"], 1)

    def test_calibration_rejects_improper_rotation(self):
        self.calibration["robot_base_from_task_rotation"] = np.eye(3).tolist()
        self.calibration["robot_base_from_task_rotation"][0][0] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(self.calibration), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "proper 3x3 rotation"):
                load_calibration(path)


if __name__ == "__main__":
    unittest.main()
