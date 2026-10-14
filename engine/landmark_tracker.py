"""MediaPipe Tasks VIDEO adapter; raw iris is accessed only during Calibration."""
from dataclasses import dataclass
import numpy as np
import cv2
from engine.geometry import rotate_offset
from utils.filters import Blink, OneEuro
from utils.calibration import Observation
from scripts.assets import ROOT, fetch

LEFT = [362, 385, 387, 263, 373, 380]
RIGHT = [33, 160, 158, 133, 153, 144]
RIGID = [1, 152, 263, 33, 61, 291]
# Camera projection uses x right, y down, z away: the nose is nearest.
MODEL = np.array([[0, 0, 0], [0, 330, 65], [225, -170, 135], [-225, -170, 135], [-150, 150, 125], [150, 150, 125]], np.float64)


@dataclass
class TrackedFace:
    anchors: np.ndarray
    centers: np.ndarray
    pose: np.ndarray
    facing: np.ndarray
    blink: float
    observation: Observation | None = None


def head_pose(points, frame_size, focal):
    width, height = frame_size
    camera = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], np.float64)
    ok, rotation, _ = cv2.solvePnP(MODEL, np.asarray(points, np.float64), camera, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Head pose solve failed")
    matrix, _ = cv2.Rodrigues(rotation)
    # Frontal rotation is identity in the camera's projection coordinates.
    pitch, yaw, roll = cv2.RQDecomp3x3(matrix)[0]
    return np.array([pitch, yaw, roll], float)


def facing_from_pose(pose):
    pitch, yaw, roll = np.radians(pose)
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    # The head faces -z in camera coordinates. WangWilly geometry reverses
    # camera x/z, so convert that ray to the screen coordinate system.
    return np.diag([-1., 1., -1.]) @ (rz @ ry @ rx @ np.array([0., 0., -1.]))


class LandmarkTracker:
    def __init__(self, geometry, detector=None):
        self.geometry = geometry
        self.mp = None
        if detector is None:
            import mediapipe as mp
            self.mp = mp
            task = ROOT / "weights/face_landmarker.task"
            fetch("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", task)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
                output_face_blendshapes=True, output_facial_transformation_matrixes=True)
            detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.detector = detector
        self.blink, self.points_filter, self.pose_filter = Blink(), OneEuro(), OneEuro()
        self.last_ms = -1
        self.calibrating = False
        self.previous_pose = None

    def track(self, frame, timestamp_ms):
        if timestamp_ms <= self.last_ms:
            raise ValueError("MediaPipe timestamps must increase")
        self.last_ms = timestamp_ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb) if self.mp else rgb
        result = self.detector.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            self.points_filter, self.pose_filter, self.previous_pose = OneEuro(), OneEuro(), None
            return None
        width, height = frame.shape[1], frame.shape[0]
        points = np.array([[p.x * width, p.y * height] for p in result.face_landmarks[0]])
        anchors = points[[LEFT, RIGHT]]
        shapes = {b.category_name: b.score for b in result.face_blendshapes[0]} if result.face_blendshapes else None
        timestamp = timestamp_ms / 1000
        blink = self.blink.update(timestamp, shapes, anchors)
        raw_pose = head_pose(points[RIGID], (width, height), self.geometry.focal_length)
        unwrapped = raw_pose.copy()
        if self.previous_pose is not None:
            unwrapped = self.previous_pose + (raw_pose - self.previous_pose + 180) % 360 - 180
        self.previous_pose = unwrapped
        filtered = self.points_filter(anchors, timestamp)
        pose = self.pose_filter(unwrapped, timestamp)
        centers = filtered[:, [0, 3]].mean(axis=1)
        observation = None
        if self.calibrating:
            observation = Observation(anchors[:, [0, 3]].mean(axis=1), points[[473, 468]], raw_pose,
                                      (width, height), blink)
        return TrackedFace(filtered, centers, pose, facing_from_pose(pose), blink, observation)

    def close(self):
        self.detector.close()
