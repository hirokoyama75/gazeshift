"""The same real engine/compositor path is used by CLI and injected replay."""
from dataclasses import dataclass
import numpy as np
from engine.compositor import EyeCrop, compose
from engine.geometry import gaze_offset, rotate_offset, pose_strength


@dataclass(frozen=True)
class Correction:
    amount: float = 1.0
    horizontal: float = 0.0
    vertical: float = 0.0

    def __post_init__(self):
        if not np.all(np.isfinite([self.amount, self.horizontal, self.vertical])):
            raise ValueError("Correction controls must be finite")
        if not 0 <= self.amount <= 2.5 or max(abs(self.horizontal), abs(self.vertical)) > 10:
            raise ValueError("Correction amount must be 0–250%; aim must be within ±10 degrees")


class Pipeline:
    def __init__(self, engine, geometry, correction=None):
        self.engine, self.geometry = engine, geometry
        self.correction = correction or Correction()
        self.enabled = True

    def process(self, frame, tracked):
        if tracked is None or not self.enabled:
            return frame
        angle, eye = gaze_offset(tracked.centers, frame.shape[1::-1], self.geometry)
        controls = self.correction
        strength = tracked.blink * pose_strength(tracked.facing, eye) * min(1., controls.amount)
        if strength == 0:
            return frame
        # Above 100%, boost the requested angle, never the compositor opacity.
        angle = (angle + [controls.vertical, controls.horizontal]) * max(1., controls.amount)
        redirection = np.clip(rotate_offset(angle, tracked.pose[2]), -15, 15).astype(np.float32)[None]
        crops, inputs = {}, {}
        for side, points in zip(("l", "r"), tracked.anchors):
            crop = EyeCrop.from_points(points, tracked.pose[2])
            crops[side] = crop
            inputs[f"img_{side}"], inputs[f"fp_{side}"] = crop.model_inputs(frame, side)
            inputs[f"ang_{side}"] = redirection
        outputs = self.engine.run(inputs)
        result = frame
        for side in ("l", "r"):
            result = compose(result, crops[side], outputs[f"flow_{side}"][0], outputs[f"lcm_{side}"][0], strength)
        return result


def run_stream(capture, tracker, pipeline, virtual=None, preview=None, on_frame=None):
    count = 0
    try:
        while True:
            sample = capture.read()
            if sample is None:
                break
            frame, timestamp = sample
            tracked = tracker.track(frame, timestamp)
            output = pipeline.process(frame, tracked)
            if virtual:
                virtual.send(output)
            if on_frame:
                on_frame(frame, output, tracked)
            if preview and preview.show(output, f"gazeshift | frame {count}") == ord('q'):
                break
            count += 1
        return count
    finally:
        errors = []
        for resource in (capture, tracker, virtual, preview):
            if resource:
                try:
                    resource.close()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise errors[0]
