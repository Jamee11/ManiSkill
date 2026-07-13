"""PushCube variant that records simulator-true object interaction signals.

This task changes observations only. Physics, rewards, and success conditions
remain those of :class:`PushCubeEnv`.
"""

from __future__ import annotations

from typing import Any

import torch
import sapien

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PushCubeEORT-v1", max_episode_steps=50)
class PushCubeEORTEnv(PushCubeEnv):
    """PushCube with raw object velocity and measured robot-object contact force."""

    SUPPORTED_ROBOTS = ["panda"]
    EORT_CAMERA_RESOLUTION = 256

    @property
    def _default_sensor_configs(self):
        configs = super()._default_sensor_configs
        base_camera = next(config for config in configs if config.uid == "base_camera")
        base_camera.width = self.EORT_CAMERA_RESOLUTION
        base_camera.height = self.EORT_CAMERA_RESOLUTION
        return configs

    def _get_obs_extra(self, info: dict) -> dict[str, Any]:
        obs = super()._get_obs_extra(info)
        contact_forces = torch.stack(
            [
                self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map[link_name], self.obj
                )
                for link_name in ("panda_hand", "panda_leftfinger", "panda_rightfinger")
            ],
            dim=1,
        )
        obs.update(
            obj_segmentation_id=self.obj.per_scene_id[:, None],
            goal_segmentation_id=self.goal_region.per_scene_id[:, None],
            obj_linear_velocity=self.obj.linear_velocity,
            obj_angular_velocity=self.obj.angular_velocity,
            robot_obj_contact_force=contact_forces.sum(dim=1),
            robot_obj_contact_force_norm=torch.linalg.norm(contact_forces, dim=2).sum(
                dim=1, keepdim=True
            ),
        )
        return obs


@register_env("PushCubeEORTCameraRand-v1", max_episode_steps=50)
class PushCubeEORTCameraRandEnv(PushCubeEORTEnv):
    """EORT PushCube with a fixed-within-episode randomized external camera."""

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


@register_env("PushCubeEORTOccluded-v1", max_episode_steps=50)
class PushCubeEORTOccludedEnv(PushCubeEORTEnv):
    """EORT PushCube with a visual-only static occluder in half of episodes."""

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
                self._occluder_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._occluder_active[env_idx] = active

    def _get_obs_extra(self, info: dict) -> dict[str, Any]:
        obs = super()._get_obs_extra(info)
        obs["occluder_active"] = self._occluder_active[:, None]
        return obs
