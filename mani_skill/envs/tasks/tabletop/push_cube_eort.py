"""PushCube variant that records simulator-true object interaction signals.

This task changes observations only. Physics, rewards, and success conditions
remain those of :class:`PushCubeEnv`.
"""

from __future__ import annotations

from typing import Any

import torch

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env


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
