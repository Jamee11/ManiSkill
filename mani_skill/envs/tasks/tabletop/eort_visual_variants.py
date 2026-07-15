"""Reusable visual-only domain shifts for EORT tabletop tasks."""

import torch
import sapien

from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose


class EORTCameraRandomizationMixin:
    """Fix a randomized external camera pose for each episode."""

    CAMERA_EYE = (0.3, 0.0, 0.6)
    CAMERA_TARGET = (-0.1, 0.0, 0.1)
    CAMERA_EYE_JITTER = (0.08, 0.08, 0.05)
    CAMERA_TARGET_JITTER = (0.03, 0.03, 0.02)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.camera_mount = builder.build_kinematic("eort_camera_mount")

    @property
    def _default_sensor_configs(self):
        configs = super()._default_sensor_configs
        base_camera = next(config for config in configs if config.uid == "base_camera")
        base_camera.pose = sapien.Pose()
        base_camera.mount = self.camera_mount
        return configs

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            count = len(env_idx)
            eye = torch.tensor(self.CAMERA_EYE, device=self.device) + (
                torch.rand((count, 3), device=self.device) - 0.5
            ) * torch.tensor(self.CAMERA_EYE_JITTER, device=self.device)
            target = torch.tensor(self.CAMERA_TARGET, device=self.device) + (
                torch.rand((count, 3), device=self.device) - 0.5
            ) * torch.tensor(self.CAMERA_TARGET_JITTER, device=self.device)
            self.camera_mount.set_pose(sapien_utils.look_at(eye=eye, target=target))


class EORTVisualOcclusionMixin:
    """Add a static, non-colliding occluder to half of EORT episodes."""

    OCCLUDER_PROBABILITY = 0.5
    OCCLUDER_POSE = (0.1, 0.0, 0.35)
    HIDDEN_POSE = (0.0, 0.0, -1.0)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.occluder = actors.build_box(
            self.scene,
            half_sizes=(0.02, 0.18, 0.14),
            color=(0.15, 0.15, 0.15, 1.0),
            name="eort_visual_occluder",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=self.HIDDEN_POSE),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            count = len(env_idx)
            active = torch.rand((count,), device=self.device) < self.OCCLUDER_PROBABILITY
            positions = torch.tensor(self.OCCLUDER_POSE, device=self.device).repeat(count, 1)
            positions[~active] = torch.tensor(self.HIDDEN_POSE, device=self.device)
            self.occluder.set_pose(Pose.create_from_pq(p=positions))
            if not hasattr(self, "_occluder_active"):
                self._occluder_active = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self._occluder_active[env_idx] = active

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs["occluder_active"] = self._occluder_active[:, None]
        return obs
