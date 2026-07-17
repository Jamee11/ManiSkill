"""PickCube variant that records simulator-true object interaction signals."""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from sapien.physx import PhysxMaterial

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.eort_visual_variants import (
    EORTCameraRandomizationMixin,
    EORTVisualOcclusionMixin,
)
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder


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
            obj_extent=torch.full_like(self.cube.pose.p, 2 * self.cube_half_size),
            obj_friction=torch.tensor(
                getattr(self, "eort_obj_friction", (0.3, 0.3)),
                dtype=self.cube.pose.p.dtype,
                device=self.device,
            ).repeat(self.num_envs, 1),
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


@register_env("PickCubeEORTGeometryRand-v1", max_episode_steps=50)
class PickCubeEORTGeometryRandEnv(PickCubeEORTEnv):
    """Seeded per-episode cube size and friction split for replay QA."""

    CUBE_HALF_SIZE_RANGE = (0.017, 0.023)
    CUBE_FRICTION_RANGE = (0.15, 0.60)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("reconfiguration_freq", 1)
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        if self.num_envs != 1:
            raise ValueError("PickCubeEORTGeometryRand-v1 currently requires num_envs=1")
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube_half_size = float(
            self._batched_episode_rng.uniform(*self.CUBE_HALF_SIZE_RANGE)[0]
        )
        friction = float(self._batched_episode_rng.uniform(*self.CUBE_FRICTION_RANGE)[0])
        self.eort_obj_friction = (friction, friction)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[self.cube_half_size] * 3,
            material=PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0.0,
            ),
        )
        builder.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=sapien.render.RenderMaterial(
                base_color=np.array([255, 0, 0, 255]) / 255
            ),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
        self.cube = builder.build(name="cube")
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
