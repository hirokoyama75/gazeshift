"""Timestamped stabilization; blink closure deliberately uses raw observations."""
import numpy as np


class OneEuro:
    def __init__(self, f_min=1.5, beta=0.01, d_cutoff=1.0):
        self.f_min, self.beta, self.d_cutoff = f_min, beta, d_cutoff
        self.t = self.raw = self.value = None
        self.derivative = 0.0

    def __call__(self, value, timestamp):
        value = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(value)) or not np.isfinite(timestamp):
            raise ValueError("Non-finite filter input")
        if self.t is None:
            self.t, self.raw, self.value = timestamp, value.copy(), value.copy()
            return self.value.copy()
        dt = timestamp - self.t
        if dt <= 0:
            raise ValueError("Filter timestamps must increase")
        def a(cutoff):
            return 1 / (1 + 1 / (2 * np.pi * cutoff * dt))
        self.derivative += a(self.d_cutoff) * ((value - self.raw) / dt - self.derivative)
        self.value += a(self.f_min + self.beta * np.abs(self.derivative)) * (value - self.value)
        self.raw, self.t = value.copy(), timestamp
        return self.value.copy()


def ear(points):
    p = np.asarray(points)
    width = np.linalg.norm(p[0] - p[3])
    return 0.0 if width <= 0 else (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * width)


class Blink:
    def __init__(self):
        self.strength, self.t = 1.0, None

    def update(self, timestamp, blendshapes=None, anchors=None):
        if blendshapes is not None and all(k in blendshapes for k in ("eyeBlinkLeft", "eyeBlinkRight")):
            k = np.clip((0.55 - max(blendshapes["eyeBlinkLeft"], blendshapes["eyeBlinkRight"])) / 0.35, 0, 1)
        elif anchors is not None:
            k = min(np.clip((ear(p) - 0.18) / 0.07, 0, 1) for p in anchors)
        else:
            k = 0.0
        if not np.isfinite(k) or (self.t is not None and timestamp <= self.t):
            raise ValueError("Invalid blink observation/time")
        dt = 0 if self.t is None else timestamp - self.t
        self.strength = float(min(k, self.strength + dt / 0.100))
        self.t = timestamp
        return self.strength
