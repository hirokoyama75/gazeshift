"""Finite synthetic replay and bounded file replay with explicit provenance labels."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from engine.geometry import CameraGeometry
from engine.landmark_tracker import TrackedFace, LandmarkTracker
from engine.onnx_warper import OnnxWarper
from engine.pipeline import Pipeline
from scripts.assets import ROOT

JITTER = np.random.default_rng(0).normal(0, .15, (90, 2, 2))


def synthetic_sample(index):
    # Labeled scenarios: still, blink, yaw 30, keyboard, depth, side light.
    frame = np.full((720, 1280, 3), (75, 100, 140), np.uint8)
    centers = np.array([[675., 320.], [605., 320.]])
    centers += JITTER[index] if index < 15 else 0
    if 60 <= index < 75:
        centers = (centers - [640, 320]) * (1 + .2 * np.sin(index)) + [640, 320]
    outline = np.array([[-16, 0], [-8, -5], [8, -5], [16, 0], [8, 5], [-8, 5]])
    anchors = centers[:, None, :] + outline
    for center in centers:
        cv2.ellipse(frame, tuple(np.rint(center).astype(int)), (16, 6), 0, 0, 360, (210, 210, 210), -1)
        cv2.circle(frame, tuple(np.rint(center).astype(int)), 5, (35, 35, 35), -1)
    blink = 0. if 15 <= index < 20 else min(1., (index-20)/3) if 20 <= index < 23 else 1.
    yaw = 30 * (index-30)/14 if 30 <= index < 45 else 0
    pitch = 30 if 45 <= index < 60 else 0
    facing = np.array([np.sin(np.radians(yaw)), np.sin(np.radians(pitch)), np.cos(np.radians(yaw))*np.cos(np.radians(pitch))])
    # The stationary synthetic subject faces screen center, not the camera axis.
    from engine.geometry import gaze_offset
    _, eye = gaze_offset(centers, (1280,720), CameraGeometry())
    screen = -eye / np.linalg.norm(eye)
    facing = screen * facing[2] + np.array([facing[0], facing[1], 0.])
    facing /= np.linalg.norm(facing)
    if index >= 75:
        frame[:, 640:] = np.clip(frame[:, 640:].astype(float)*.6, 0, 255).astype(np.uint8)
    tracked = TrackedFace(anchors, centers, np.array([pitch, yaw, 0.]), facing, blink)
    return frame, tracked


def synthetic_replay(output_dir=ROOT / "artifacts/replay"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = OnnxWarper()
    pipeline = Pipeline(engine, CameraGeometry())
    paths = [output_dir / name for name in ("synthetic_raw.avi", "synthetic_corrected.avi", "synthetic_raw_corrected.avi")]
    writers = [cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size) for path, size in zip(paths, ((1280, 720), (1280, 720), (2560, 720)))]
    if not all(w.isOpened() for w in writers):
        for w in writers: w.release()
        raise RuntimeError("Offline replay video encoder unavailable")
    iris_centers = []
    try:
        for i in range(90):
            frame, tracked = synthetic_sample(i)
            corrected = pipeline.process(frame, tracked)
            if i < 15:
                # Diagnostic centroid of the synthetic dark iris in its fixed ROI.
                roi = cv2.cvtColor(corrected[305:335, 650:700], cv2.COLOR_BGR2GRAY)
                weights = np.maximum(0, 100-roi.astype(float))
                yy, xx = np.indices(weights.shape)
                if weights.sum():
                    iris_centers.append([float((xx*weights).sum()/weights.sum()), float((yy*weights).sum()/weights.sum())])
            raw = frame.copy()
            corrected = corrected.copy()
            for image, label in ((raw, "SYNTHETIC RAW"), (corrected, "SYNTHETIC CORRECTED")):
                cv2.putText(image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            for writer, image in zip(writers, (raw, corrected, np.concatenate([raw, corrected], axis=1))):
                writer.write(image)
    finally:
        for writer in writers: writer.release()
    centers = np.asarray(iris_centers)
    if len(centers) != 15:
        raise ValueError("Synthetic iris-centroid measurements missing")
    result = {"label": "SYNTHETIC", "seed": 0, "frames": 90, "inference_calls": engine.calls,
              "scenarios": ["stationary", "blinks", "yaw_30", "keyboard", "lean", "side_light"],
              "synthetic_iris_center_rms_px": float(np.sqrt(np.mean(np.sum((centers-centers.mean(axis=0))**2, axis=1)))),
              "artifacts": [p.relative_to(ROOT).as_posix() for p in paths],
              "live_visual_quality": "manual_pending"}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def replay_file(source, destination, max_frames=900):
    capture = cv2.VideoCapture(str(source))
    tracker = LandmarkTracker(CameraGeometry())
    engine = OnnxWarper()
    pipeline = Pipeline(engine, tracker.geometry)
    writer = None
    count = 0
    try:
        if not capture.isOpened(): raise ValueError("Cannot decode replay file")
        for index in range(max_frames):
            ok, frame = capture.read()
            if not ok: break
            frame = cv2.resize(frame, (1280, 720))
            if writer is None:
                writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"MJPG"), 30, (2560, 720))
                if not writer.isOpened(): raise RuntimeError("Cannot encode replay")
            tracked = tracker.track(frame, index*34)
            writer.write(np.concatenate((frame, pipeline.process(frame, tracked)), axis=1))
            count += 1
    finally:
        capture.release()
        tracker.close()
        if writer: writer.release()
    if not count: raise ValueError("Empty replay")
    return {"frames": count, "source": str(source), "inference_calls": engine.calls}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/replay.avi")
    parser.add_argument("--max-frames", type=int, default=900)
    args = parser.parse_args()
    print(json.dumps(replay_file(args.input, args.output, args.max_frames) if args.input else synthetic_replay(), indent=2))
