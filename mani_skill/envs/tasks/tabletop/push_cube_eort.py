"""PushCube variant that records simulator-true object interaction signals.

This task changes observations only. Physics, rewards, and success conditions
remain those of :class:`PushCubeEnv`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from sapien.physx import PhysxMaterial

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.envs.tasks.tabletop.eort_visual_variants import (
    EORTCameraRandomizationMixin,
    EORTVisualOcclusionMixin,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder


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
            obj_extent=torch.full_like(self.obj.pose.p, 2 * self.cube_half_size),
            obj_mass=self.obj.mass[:, None].to(self.device),
            obj_friction=torch.tensor(
                getattr(self, "eort_obj_friction", (0.3, 0.3)),
                dtype=self.obj.pose.p.dtype,
                device=self.device,
            ).repeat(self.num_envs, 1),
            obj_linear_velocity=self.obj.linear_velocity,
            obj_angular_velocity=self.obj.angular_velocity,
            robot_obj_contact_force=contact_forces.sum(dim=1),
            robot_obj_contact_force_norm=torch.linalg.norm(contact_forces, dim=2).sum(
                dim=1, keepdim=True
            ),
        )
        return obs


@register_env("PushCubeEORTCameraRand-v1", max_episode_steps=50)
class PushCubeEORTCameraRandEnv(EORTCameraRandomizationMixin, PushCubeEORTEnv):
    """EORT PushCube with a fixed-within-episode randomized external camera."""

    pass


@register_env("PushCubeEORTOccluded-v1", max_episode_steps=50)
class PushCubeEORTOccludedEnv(EORTVisualOcclusionMixin, PushCubeEORTEnv):
    """EORT PushCube with a visual-only static occluder in half of episodes."""

    pass


@register_env("PushCubeEORTGeometryRand-v1", max_episode_steps=50)
class PushCubeEORTGeometryRandEnv(PushCubeEORTEnv):
    """Seeded per-episode cube size and friction split for replay QA."""

    CUBE_HALF_SIZE_RANGE = (0.017, 0.023)
    CUBE_FRICTION_RANGE = (0.15, 0.60)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("reconfiguration_freq", 1)
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        if self.num_envs != 1:
            raise ValueError("PushCubeEORTGeometryRand-v1 currently requires num_envs=1")
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube_half_size = float(
            self._batched_episode_rng.uniform(*self.CUBE_HALF_SIZE_RANGE)[0]
        )
        friction = float(self._batched_episode_rng.uniform(*self.CUBE_FRICTION_RANGE)[0])
        self.eort_obj_friction = (friction, friction)
        builder = self.scene.create_actor_builder()
        material = PhysxMaterial(
            static_friction=friction, dynamic_friction=friction, restitution=0.0
        )
        builder.add_box_collision(
            half_size=[self.cube_half_size] * 3, material=material
        )
        builder.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=sapien.render.RenderMaterial(
                base_color=np.array([12, 42, 160, 255]) / 255
            ),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
        self.obj = builder.build(name="cube")
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )
