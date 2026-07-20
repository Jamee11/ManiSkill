import importlib.util
import unittest
from pathlib import Path

import numpy as np


DERIVER = Path(__file__).resolve().parents[1] / "scripts/eort/derive_push_cube_eort.py"


def load_deriver():
    spec = importlib.util.spec_from_file_location("derive_eort_v3", DERIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EORTSim2RealV3Test(unittest.TestCase):
    def test_world_to_base_uses_translation_and_nonidentity_rotation(self):
        module = load_deriver()
        half = np.sqrt(0.5)
        base = np.array([[1.0, 2.0, 0.0, half, 0.0, 0.0, half]], dtype=np.float32)
        world = np.array([[1.0, 3.0, 0.0, half, 0.0, 0.0, half]], dtype=np.float32)
        pose_base = module._world_pose_to_base(world, base)
        np.testing.assert_allclose(pose_base[0, :3], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(pose_base[0, 3:], [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_real_intrinsic_crop_resize_contract(self):
        from mani_skill.envs.tasks.tabletop.eort_visual_variants import (
            EORTSim2RealV3Mixin,
            center_crop_resize_intrinsic,
        )

        intrinsic = center_crop_resize_intrinsic(EORTSim2RealV3Mixin.FRONT_INTRINSIC_1280X720)
        np.testing.assert_allclose(
            intrinsic,
            [[229.73863, 0.0, 128.59645], [0.0, 229.62198, 126.62317], [0.0, 0.0, 1.0]],
            atol=1e-4,
        )

    def test_v3_environments_are_additive_registrations(self):
        import gymnasium as gym
        import mani_skill.envs.tasks  # noqa: F401

        self.assertEqual(gym.spec("PushCubeEORTSim2Real-v1").id, "PushCubeEORTSim2Real-v1")
        self.assertEqual(gym.spec("PickCubeEORTSim2Real-v1").id, "PickCubeEORTSim2Real-v1")
        self.assertEqual(gym.spec("PushCubeEORTOccluded-v1").id, "PushCubeEORTOccluded-v1")


if __name__ == "__main__":
    unittest.main()
