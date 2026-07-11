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

    def _get_obs_extra(self, info: dict) -> dict[str, Any]:
        obs = super()._get_obs_extra(info)
        contact_force = torch.zeros_like(self.obj.linear_velocity)
        for link_name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
            contact_force += self.scene.get_pairwise_contact_forces(
                self.agent.robot.links_map[link_name], self.obj
            )
        obs.update(
            obj_linear_velocity=self.obj.linear_velocity,
            obj_angular_velocity=self.obj.angular_velocity,
            robot_obj_contact_force=contact_force,
        )
        return obs
