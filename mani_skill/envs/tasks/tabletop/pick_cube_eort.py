"""PickCube variant that records simulator-true object interaction signals."""

from __future__ import annotations

from typing import Any

import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.eort_visual_variants import (
    EORTCameraRandomizationMixin,
    EORTVisualOcclusionMixin,
)
from mani_skill.utils.registration import register_env


@register_env("PickCubeEORT-v1", max_episode_steps=50)
class PickCubeEORTEnv(PickCubeEnv):
    """Panda PickCube with visual object/goal IDs and physical grasp labels."""

    SUPPORTED_ROBOTS = ["panda"]
    EORT_CAMERA_RESOLUTION = 256

    @property
    def _default_sensor_configs(self):
        configs = super()._default_sensor_configs
        base_camera = next(config for config in configs if config.uid == "base_camera")
        base_camera.width = self.EORT_CAMERA_RESOLUTION
        base_camera.height = self.EORT_CAMERA_RESOLUTION
        return configs

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # PickCube hides this marker from ordinary agent observations.  EORT needs
        # its actor ID in the recorded sensor stream for goal-localization QA.
        self._hidden_objects.remove(self.goal_site)

    def _get_obs_extra(self, info: dict) -> dict[str, Any]:
        obs = super()._get_obs_extra(info)
        contact_forces = torch.stack(
            [
                self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map[link_name], self.cube
                )
                for link_name in ("panda_hand", "panda_leftfinger", "panda_rightfinger")
            ],
            dim=1,
        )
        obs.update(
            # Keep this label explicitly (T+1,1) in HDF5.  It is supervision
            # only; the policy exporter must not treat it as an observation.
            is_grasped=info["is_grasped"][:, None],
            obj_segmentation_id=self.cube.per_scene_id[:, None],
            goal_segmentation_id=self.goal_site.per_scene_id[:, None],
            obj_linear_velocity=self.cube.linear_velocity,
            obj_angular_velocity=self.cube.angular_velocity,
            robot_obj_contact_force=contact_forces.sum(dim=1),
            robot_obj_contact_force_norm=torch.linalg.norm(contact_forces, dim=2).sum(
                dim=1, keepdim=True
            ),
        )
        return obs


@register_env("PickCubeEORTCameraRand-v1", max_episode_steps=50)
class PickCubeEORTCameraRandEnv(EORTCameraRandomizationMixin, PickCubeEORTEnv):
    """PickCube EORT with a fixed-within-episode randomized external camera."""


@register_env("PickCubeEORTOccluded-v1", max_episode_steps=50)
class PickCubeEORTOccludedEnv(EORTVisualOcclusionMixin, PickCubeEORTEnv):
    """PickCube EORT with a visual-only static occluder in half of episodes."""
