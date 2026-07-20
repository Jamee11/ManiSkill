#!/usr/bin/env python3
"""Render a compact RGB preview with object-centric EORT labels overlaid."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np


OBJECT_COLOR = (48, 220, 48)  # BGR
GOAL_COLOR = (32, 220, 255)


def _phase_key(labels: np.lib.npyio.NpzFile) -> str | None:
    return next((key for key in labels.files if key.endswith("interaction_phase")), None)


def annotate(frame_rgb: np.ndarray, labels: np.lib.npyio.NpzFile, step: int, *, prefix: str = "", title: str = "") -> np.ndarray:
    """Draw only derived labels; raw segmentation IDs are intentionally absent."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(f"RGB frame must be (H,W,3), got {frame_rgb.shape}")
    output = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for role, color in (("object", OBJECT_COLOR), ("goal", GOAL_COLOR)):
        bbox = np.asarray(labels[f"{prefix}{role}_bbox_xyxy"][step], dtype=np.int32)
        visible = bool(labels[f"{prefix}{role}_visible"][step, 0])
        if visible:
            x0, y0, x1, y1 = bbox.tolist()
            cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), color, 1)
            centroid = np.rint(labels[f"{prefix}{role}_mask_centroid_uv"][step]).astype(int)
            cv2.circle(output, tuple(centroid), 2, color, -1)
        cv2.putText(output, f"{role}:{int(visible)}", (4, 18 if role == "object" else 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    phase_key = _phase_key(labels)
    phase = int(labels[phase_key][step, 0]) if phase_key else -1
    contact = int(labels["physical_contact"][step, 0]) if "physical_contact" in labels.files else -1
    cv2.putText(output, f"t={step} phase={phase} contact={contact}", (4, output.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(output, title, (output.shape[1] - 116, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def render_preview(trajectory_path: Path, derived_dir: Path, output_path: Path, *, cameras: tuple[str, ...], max_episodes: int, fps: int) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite preview: {output_path}")
    records = [json.loads(line) for line in (derived_dir / "manifest.jsonl").read_text().splitlines() if line]
    if max_episodes <= 0 or fps <= 0:
        raise ValueError("max_episodes and fps must be positive")
    records = records[:max_episodes]
    if not records:
        raise ValueError("No derived trajectories selected")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode H.264 previews")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer, frames, selected = None, 0, []
    with tempfile.TemporaryDirectory(prefix=".eort_preview_", dir=output_path.parent) as temporary_dir:
        intermediate_path = Path(temporary_dir) / "preview_mp4v.mp4"
        try:
            with h5py.File(trajectory_path, "r") as raw:
                for record in records:
                    trajectory = record["source_trajectory"]
                    with np.load(derived_dir / record["output"]) as labels:
                        videos = {
                            camera: raw[trajectory][f"obs/sensor_data/{camera}/rgb"][: len(labels["action"])]
                            for camera in cameras
                        }
                        for step in range(len(labels["action"])):
                            views = []
                            for camera, rgb in videos.items():
                                prefix = f"{camera}_" if f"{camera}_object_visible" in labels else ""
                                views.append(annotate(rgb[step], labels, step, prefix=prefix, title=camera))
                            annotated = np.concatenate(views, axis=1)
                            if writer is None:
                                height, width = annotated.shape[:2]
                                writer = cv2.VideoWriter(str(intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                                if not writer.isOpened():
                                    raise RuntimeError(f"Cannot open MP4 writer: {intermediate_path}")
                            writer.write(annotated)
                            frames += 1
                    selected.append(trajectory)
        finally:
            if writer is not None:
                writer.release()
        if writer is None:
            raise RuntimeError("No preview frames were written")
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-i", str(intermediate_path), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)],
            check=True,
        )
    summary = {"preview": str(output_path), "trajectory_path": str(trajectory_path), "derived_dir": str(derived_dir), "cameras": list(cameras), "episodes": selected, "frames": frames, "fps": fps, "codec": "h264", "pixel_format": "yuv420p", "overlay": "tiled object/goal bbox+centroid+visibility, phase and force-contact labels"}
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-path", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            render_preview(
                args.trajectory_path,
                args.derived_dir,
                args.output,
                cameras=tuple(args.cameras or (args.camera,)),
                max_episodes=args.max_episodes,
                fps=args.fps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
