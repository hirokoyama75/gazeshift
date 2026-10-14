"""WangWilly/Hsu geometric offset, without integer truncation.

Copyright 2019 Chih-Fan Hsu. BSD-3-Clause; see LICENSE.
"""
from dataclasses import dataclass, replace, asdict
import json
import os
from pathlib import Path
import tempfile
import numpy as np


@dataclass(frozen=True)
class CameraGeometry:
    focal_length: float = 650.0
    ipd: float = 6.3
    camera_pos: tuple = (0.0, -21.0, -1.0)
    provider: str | None = None
    reference_width: int = 1280

    def validate(self):
        if len(self.camera_pos) != 3 or not np.all(np.isfinite([self.focal_length, self.ipd, self.reference_width, *self.camera_pos])):
            raise ValueError("Non-finite Camera Geometry")
        if self.focal_length <= 0 or self.ipd <= 0 or self.reference_width <= 0:
            raise ValueError("Focal length and IPD must be positive")
        return self


def load_config(path):
    path = Path(path)
    if not path.exists():
        return CameraGeometry()
    data = json.loads(path.read_text())
    data["camera_pos"] = tuple(data["camera_pos"])
    return CameraGeometry(**data).validate()


def save_config(path, geometry):
    geometry.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(asdict(geometry), stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def gaze_offset(centers, frame_size, geometry):
    geometry.validate()
    centers = np.asarray(centers, dtype=float)
    if centers.shape != (2, 2) or not np.all(np.isfinite(centers)):
        raise ValueError("Invalid Eye Centers")
    separation = np.linalg.norm(centers[0] - centers[1])
    if separation <= 0:
        raise ValueError("Coincident Eye Centers")
    depth = geometry.focal_length * geometry.ipd / separation
    camera = np.asarray(geometry.camera_pos)
    if depth + camera[2] <= 0:
        raise ValueError("Invalid eye depth")
    pixel = centers.mean(axis=0) - np.asarray(frame_size) / 2
    eye = np.array([-depth * pixel[0] / geometry.focal_length + camera[0],
                    depth * pixel[1] / geometry.focal_length + camera[1], -depth])
    angles_xy = np.degrees(np.arctan(-eye[:2] / depth) + np.arctan((eye[:2] - camera[:2]) / (camera[2] + depth)))
    return angles_xy[::-1].copy(), eye


def rotate_offset(angles, roll):
    r = np.radians(roll)
    x, y = np.asarray(angles)[::-1]
    return np.array([-np.sin(r) * x + np.cos(r) * y, np.cos(r) * x + np.sin(r) * y])


def pose_strength(facing, eye):
    facing, screen = np.asarray(facing, float), -np.asarray(eye, float)
    den = np.linalg.norm(facing) * np.linalg.norm(screen)
    if not np.isfinite(den) or den <= 0:
        return 0.0
    theta = np.degrees(np.arccos(np.clip(np.dot(facing, screen) / den, -1, 1)))
    return float(np.clip((18 - theta) / 6, 0, 1))


def nudge(geometry, key):
    x, y, z = geometry.camera_pos
    f = geometry.focal_length
    # OpenCV waitKeyEx codes on Windows and macOS.
    if key in (2424832, 63234): x -= 0.5
    elif key in (2555904, 63235): x += 0.5
    elif key in (2490368, 63232): y -= 0.5
    elif key in (2621440, 63233): y += 0.5
    elif key in (ord('+'), ord('=')): z += 0.5
    elif key == ord('-'): z -= 0.5
    elif key == ord('['): f -= 10
    elif key == ord(']'): f += 10
    elif key == ord('r'):
        return replace(geometry, focal_length=650. * geometry.reference_width / 1280, camera_pos=(0., -21., -1.))
    return replace(geometry, focal_length=max(10., f), camera_pos=(x, y, z)).validate()
