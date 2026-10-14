"""Shared two-look Calibration state machine; all persistence follows validation."""
from dataclasses import dataclass, replace
import time
import numpy as np
from engine.geometry import gaze_offset, rotate_offset, save_config


@dataclass
class Observation:
    centers: np.ndarray
    irises: np.ndarray
    pose: np.ndarray
    frame_size: tuple
    blink: float = 1.0
    face: bool = True


def iris_direction(observation, geometry):
    centers, irises = np.asarray(observation.centers), np.asarray(observation.irises)
    if centers.shape != (2, 2) or irises.shape != (2, 2) or np.asarray(observation.pose).shape != (3,):
        raise ValueError("Malformed Calibration observation")
    if not np.all(np.isfinite([*centers.ravel(), *irises.ravel(), *observation.pose])):
        raise ValueError("Non-finite Calibration observation")
    if len(observation.frame_size) != 2 or not np.all(np.isfinite(observation.frame_size)) or min(observation.frame_size) <= 0:
        raise ValueError("Invalid frame dimensions")
    separation = np.linalg.norm(centers[0] - centers[1])
    if separation <= 0:
        raise ValueError("Invalid depth")
    geometry.validate()
    depth = geometry.focal_length * geometry.ipd / separation
    if depth + geometry.camera_pos[2] <= 0:
        raise ValueError("Invalid depth")
    r = np.radians(observation.pose[2])
    rot = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
    q = (irises - centers) @ rot.T * depth / geometry.focal_length
    if np.any(np.linalg.norm(q, axis=1) >= 1.2):
        raise ValueError("Iris outside eye model")
    return np.degrees(q[:, ::-1] / 1.2).mean(axis=0)


def fit_xy(geometry, centers, size, measured):
    _, eye = gaze_offset(centers, size, geometry)
    depth = -eye[2]
    camera = np.asarray(geometry.camera_pos)
    relative = eye[:2] - camera[:2]
    xy = -depth * np.tan(np.radians(measured[::-1]) - np.arctan(relative / (depth + camera[2]))) - relative
    candidate = replace(geometry, camera_pos=(*xy, camera[2])).validate()
    predicted, _ = gaze_offset(centers, size, candidate)
    residual = np.abs(predicted - measured)
    if not np.all(np.isfinite(residual)) or np.any(residual > .1):
        raise ValueError("Calibration angular residual exceeds 0.1 degrees")
    return candidate, residual


class Calibration:
    def __init__(self, geometry, config_path, clock=time.monotonic):
        self.geometry, self.path, self.clock = geometry, config_path, clock
        self.state = "idle"
        self.message = ""
        self.lens, self.screen = [], []
        self.reference = None
        self.result = None

    @property
    def active(self):
        return self.state in ("lens", "screen")

    def start(self):
        self.state, self.started = "lens", self.clock()
        self.message = "Look at the webcam lens."
        self.lens, self.screen, self.reference, self.result = [], [], None, None

    def cancel(self):
        if self.active:
            self.state, self.message = "cancelled", "Calibration cancelled; settings preserved."

    def update(self, observation):
        if not self.active:
            return None
        if self.clock() - self.started >= 8:
            self.state, self.message = "timeout", "Calibration timed out; retry or nudge."
            return None
        if observation is None or not observation.face or observation.blink != 1:
            return None
        try:
            direction = iris_direction(observation, self.geometry)
            pose = np.asarray(observation.pose)
            if self.reference is None:
                self.reference = pose.copy()
            deviation = (pose - self.reference + 180) % 360 - 180
            if np.any(np.abs(deviation) > 3):
                raise ValueError("Head pose moved more than 3 degrees")
            bucket = self.lens if self.state == "lens" else self.screen
            bucket.append((direction, observation))
            if len(bucket) != 15:
                return None
            if self.state == "lens":
                self.state, self.message = "screen", "Look at the crosshair."
                return None
            difference = np.mean([v[0] for v in self.screen], axis=0) - np.mean([v[0] for v in self.lens], axis=0)
            rolls = np.radians([v[1].pose[2] for v in self.lens + self.screen])
            roll = np.degrees(np.arctan2(np.sin(rolls).mean(), np.cos(rolls).mean()))
            measured = rotate_offset(difference, -roll)
            centers = np.mean([v[1].centers for v in self.lens + self.screen], axis=0)
            candidate, residual = fit_xy(self.geometry, centers, observation.frame_size, measured)
            save_config(self.path, candidate)
            self.result, self.residual = candidate, residual
            self.state, self.message = "success", "Calibration saved."
            return candidate
        except (ValueError, OSError) as error:
            self.state, self.message = "failed", str(error) + "; retry or nudge."
            return None


def monitor_center(bounds):
    x, y, width, height = bounds
    return (x + width / 2, y + height / 2)
