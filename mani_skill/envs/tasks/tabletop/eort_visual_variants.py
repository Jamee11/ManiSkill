"""Reusable visual-only domain shifts for EORT tabletop tasks."""

import numpy as np
import torch
import sapien
from transforms3d.euler import euler2quat

from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose


def opencv_cam2base_to_sapien_pose(cam2base: np.ndarray) -> sapien.Pose:
    """Convert an OpenCV optical-camera pose into SAPIEN's camera convention.

    OpenCV axes are right/down/forward.  SAPIEN cameras use forward/left/up.
    Both inputs and outputs express the camera pose in the robot-base frame.
    """
    cam2base = np.asarray(cam2base, dtype=np.float64)
    if cam2base.shape != (4, 4):
        raise ValueError(f"cam2base must have shape (4, 4), got {cam2base.shape}")
    transform = np.eye(4, dtype=np.float64)
    rotation_cv = cam2base[:3, :3]
    transform[:3, :3] = np.column_stack(
        (rotation_cv[:, 2], -rotation_cv[:, 0], -rotation_cv[:, 1])
    )
    transform[:3, 3] = cam2base[:3, 3]
    return sapien.Pose(transform)


def center_crop_resize_intrinsic(
    intrinsic: np.ndarray, source_width: int = 1280, source_height: int = 720, output_size: int = 256
) -> np.ndarray:
    """Map K through a centered square crop followed by square resize."""
    intrinsic = np.asarray(intrinsic, dtype=np.float64).copy()
    crop_size = min(source_width, source_height)
    intrinsic[0, 2] -= (source_width - crop_size) / 2
    intrinsic[1, 2] -= (source_height - crop_size) / 2
    intrinsic[:2] *= output_size / crop_size
    return intrinsic.astype(np.float32)


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


class EORTSim2RealV3Mixin:
    """Three-view, robot-base-frame collection setup for the EORT v3 dataset.

    The two static-camera nominal poses and intrinsics are copied from the
    checked-in experiment contract below, avoiding a runtime dependency on the
    calibration machine.  Lens distortion is intentionally not simulated;
    real frames must be undistorted before applying the same crop and resize.
    """

    EORT_CAMERA_RESOLUTION = 256
    CAMERA_UIDS = ("front_camera", "right_shoulder_camera", "hand_camera")
    CAMERA_TRANSLATION_JITTER_M = 0.01
    CAMERA_ROTATION_JITTER_RAD = np.deg2rad(2.0)

    FRONT_INTRINSIC_1280X720 = np.array(
        [[646.13992278, 0.0, 641.6775164], [0.0, 645.81182312, 356.1276761], [0.0, 0.0, 1.0]]
    )
    FRONT_CAM2BASE_CV = np.array(
        [
            [-0.00641402697, 0.503546341, -0.863944409, 1.00028611],
            [0.999893753, -0.00807976703, -0.0121325892, 0.000941980844],
            [-0.0130897905, -0.863930437, -0.503441017, 0.379842197],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    RIGHT_SHOULDER_INTRINSIC_1280X720 = np.array(
        [[645.54395684, 0.0, 648.41203679], [0.0, 645.40870529, 365.90684133], [0.0, 0.0, 1.0]]
    )
    RIGHT_SHOULDER_CAM2BASE_CV = np.array(
        [
            [0.9477211, -0.26464183, 0.17829585, 0.30905803],
            [-0.31883032, -0.80827794, 0.4950091, -0.343116],
            [0.01311249, -0.52597669, -0.8503979, 0.54295386],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    TABLE_COLORS = np.array(
        [[0.34, 0.30, 0.25, 1.0], [0.48, 0.48, 0.46, 1.0], [0.22, 0.27, 0.31, 1.0], [0.52, 0.43, 0.33, 1.0]],
        dtype=np.float32,
    )

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", "panda_wristcam")
        kwargs.setdefault("reconfiguration_freq", 1)
        sensor_configs = dict(kwargs.get("sensor_configs") or {})
        sensor_configs.setdefault("width", self.EORT_CAMERA_RESOLUTION)
        sensor_configs.setdefault("height", self.EORT_CAMERA_RESOLUTION)
        kwargs["sensor_configs"] = sensor_configs
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        if self.num_envs != 1:
            raise ValueError("EORT Sim2Real v3 currently requires num_envs=1")
        super()._load_scene(options)
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.front_camera_mount = builder.build_kinematic("eort_front_camera_mount")
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.right_shoulder_camera_mount = builder.build_kinematic(
            "eort_right_shoulder_camera_mount"
        )
        color_index = int(self._batched_episode_rng.randint(len(self.TABLE_COLORS))[0])
        self._eort_table_color = self.TABLE_COLORS[color_index]
        self.eort_tabletop_visual = actors.build_box(
            self.scene,
            half_sizes=(0.60, 0.60, 0.00005),
            color=self._eort_table_color,
            name="eort_randomized_tabletop_visual",
            body_type="kinematic",
            add_collision=False,
            # Top surface is 0.1 mm above the GLB tabletop: enough to avoid
            # z-fighting, still well below the goal marker at z=1 mm.
            initial_pose=sapien.Pose(p=[-0.12, 0.0, 0.00005]),
        )

    def _load_lighting(self, options: dict):
        intensity = float(self._batched_episode_rng.uniform(0.65, 1.25)[0])
        tint = np.asarray(
            self._batched_episode_rng.uniform(0.92, 1.08, size=3)[0], dtype=np.float32
        )
        self._eort_light_rgb = np.clip(intensity * tint, 0.4, 1.4).astype(np.float32)
        self.scene.set_ambient_light((0.28 * self._eort_light_rgb).tolist())
        self.scene.add_directional_light(
            [1, 1, -1], self._eort_light_rgb.tolist(), shadow=self.enable_shadow,
            shadow_scale=5, shadow_map_size=2048,
        )
        self.scene.add_directional_light([0, 0, -1], (0.75 * self._eort_light_rgb).tolist())

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                "front_camera", sapien.Pose(), self.EORT_CAMERA_RESOLUTION,
                self.EORT_CAMERA_RESOLUTION, intrinsic=center_crop_resize_intrinsic(self.FRONT_INTRINSIC_1280X720),
                near=0.01, far=100, mount=self.front_camera_mount,
            ),
            CameraConfig(
                "right_shoulder_camera", sapien.Pose(), self.EORT_CAMERA_RESOLUTION,
                self.EORT_CAMERA_RESOLUTION, intrinsic=center_crop_resize_intrinsic(self.RIGHT_SHOULDER_INTRINSIC_1280X720),
                near=0.01, far=100, mount=self.right_shoulder_camera_mount,
            ),
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        count = len(env_idx)
        translation = (
            torch.rand((count, 2, 3), device=self.device) * 2.0 - 1.0
        ) * self.CAMERA_TRANSLATION_JITTER_M
        euler = (
            torch.rand((count, 2, 3), device=self.device) * 2.0 - 1.0
        ) * self.CAMERA_ROTATION_JITTER_RAD
        quaternions = torch.tensor(
            np.stack([euler2quat(*angles) for angles in euler.detach().cpu().numpy().reshape(-1, 3)]),
            dtype=torch.float32, device=self.device,
        ).reshape(count, 2, 4)
        self._eort_camera_translation_jitter = torch.zeros(
            (self.num_envs, 2, 3), dtype=torch.float32, device=self.device
        )
        self._eort_camera_rotation_jitter = torch.zeros(
            (self.num_envs, 2, 3), dtype=torch.float32, device=self.device
        )
        self._eort_camera_translation_jitter[env_idx] = translation
        self._eort_camera_rotation_jitter[env_idx] = euler
        base_pose = self.agent.robot.pose[env_idx]
        nominals = (self.FRONT_CAM2BASE_CV, self.RIGHT_SHOULDER_CAM2BASE_CV)
        mounts = (self.front_camera_mount, self.right_shoulder_camera_mount)
        for camera_index, (cam2base, mount) in enumerate(zip(nominals, mounts)):
            nominal = Pose.create(opencv_cam2base_to_sapien_pose(cam2base), device=self.device)
            jitter = Pose.create_from_pq(
                translation[:, camera_index], quaternions[:, camera_index]
            )
            mount.set_pose(base_pose * nominal * jitter)

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs.update(
            robot_base_pose=self.agent.robot.pose.raw_pose,
            eort_camera_translation_jitter=self._eort_camera_translation_jitter,
            eort_camera_rotation_jitter=self._eort_camera_rotation_jitter,
            eort_table_color=torch.as_tensor(self._eort_table_color, device=self.device)[None],
            eort_light_rgb=torch.as_tensor(self._eort_light_rgb, device=self.device)[None],
        )
        return obs
