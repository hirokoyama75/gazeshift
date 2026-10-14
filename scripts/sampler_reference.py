"""Faithful NumPy reference of Hsu's clipped-neighbor TF bilinear sampler.

Copyright 2019 Chih-Fan Hsu. Derived work, BSD-3-Clause; see LICENSE. Unlike runtime
replicated borders, collapsed TF neighbors cancel the interpolation weights.
"""
import numpy as np


def coordinates(flow):
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32), np.linspace(-1, 1, h, dtype=np.float32))
    return .5 * (x + flow[..., 0] + 1) * w, .5 * (y + flow[..., 1] + 1) * h


def tf_sampler(image, flow):
    h, w = image.shape[:2]
    x, y = coordinates(flow)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, w-1), np.clip(y0 + 1, 0, h-1)
    x0, y0 = np.clip(x0, 0, w-1), np.clip(y0, 0, h-1)
    return sum(weight[..., None] * image[iy, ix] for weight, iy, ix in (
        ((x1-x)*(y1-y), y0, x0), ((x1-x)*(y-y0), y1, x0),
        ((x-x0)*(y1-y), y0, x1), ((x-x0)*(y-y0), y1, x1)))
