"""Full coordinate fading and one original-frame sample, including rolled crops."""
from dataclasses import dataclass
import cv2
import numpy as np


def full_coordinates(flow, width=None, height=None):
    h, w = flow.shape[:2]
    width, height = width or w, height or h
    f = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    x, y = np.meshgrid(np.linspace(-1, 1, width, dtype=np.float32), np.linspace(-1, 1, height, dtype=np.float32))
    return np.stack((0.5 * (x + f[..., 0] + 1) * width,
                     0.5 * (y + f[..., 1] + 1) * height), axis=-1)


def remap_patch(image, flow):
    s = full_coordinates(flow)
    return remap_exact(image, s[..., 0], s[..., 1])


def remap_exact(image, map_x, map_y):
    """Correct OpenCV's 1/32-pixel table quantization to meet TF parity.

    One frame remap plus a local bilinear weight residual; there is no second
    image warp, image cross-fade, or rotate/unrotate sampling round trip.
    """
    h, w = image.shape[:2]
    x = np.clip(map_x, 0, w-1).astype(np.float32)
    y = np.clip(map_y, 0, h-1).astype(np.float32)
    source = image.astype(np.float32)
    sampled = cv2.remap(source, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    fx, fy = x-x0, y-y0
    qx, qy = np.rint(x*32)/32-x0, np.rint(y*32)/32-y0
    for ix, iy, exact, table in (
        (x0,y0,(1-fx)*(1-fy),(1-qx)*(1-qy)),
        (x1,y0,fx*(1-fy),qx*(1-qy)),
        (x0,y1,(1-fx)*fy,(1-qx)*qy),
        (x1,y1,fx*fy,qx*qy)):
        residual = exact-table
        if source.ndim == 3:
            residual = residual[..., None]
        sampled += (residual * source[iy,ix]).astype(np.float32)
    return sampled


@dataclass
class EyeCrop:
    to_frame: np.ndarray
    size: tuple
    points: np.ndarray

    @classmethod
    def from_points(cls, points, roll):
        points = np.asarray(points, dtype=float)
        center = points.mean(axis=0)
        width = max(8, int(np.ceil(np.linalg.norm(points[3] - points[0]) * 1.5)))
        height = max(6, int(np.ceil(width * .75)))
        r = np.radians(roll)
        rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        # Upstream eye box has 7/12 of its height above the anchor mean.
        origin = center - rot @ np.array([width / 2, height * 7 / 12])
        matrix = np.column_stack((rot, origin))
        local = (points - origin) @ rot
        return cls(matrix, (width, height), local)

    def model_inputs(self, frame, side):
        width, height = self.size
        patch = cv2.warpAffine(frame, self.to_frame, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
        image = cv2.resize(patch, (64, 48))[..., ::-1].astype(np.float32) / 255
        points = self.points * (64 / width, 48 / height)
        order = (3, 2, 1, 0, 5, 4) if side == "l" else range(6)
        x, y = np.meshgrid(np.arange(64), np.arange(48))
        anchors = np.stack([channel for i in order for channel in (x - points[i, 0], y - points[i, 1])], axis=-1)
        return image[None].copy(), anchors[None].astype(np.float32)


def eyelid_mask(crop):
    width, height = crop.size
    mask = np.zeros((height, width), np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(crop.points).astype(np.int32)), 1.)
    radius = max(1, int(round(.05 * width)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    return np.clip(cv2.GaussianBlur(mask, (0, 0), .04 * width), 0, 1)


def compose(frame, crop, flow, lighting, strength, mask=None):
    if strength == 0:
        return frame
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Invalid Blend Strength")
    width, height = crop.size
    mask = eyelid_mask(crop) if mask is None else np.asarray(mask, np.float32)
    full = full_coordinates(flow, width, height)
    light = cv2.resize(lighting, (width, height), interpolation=cv2.INTER_LINEAR)
    # Evaluate crop fields at each destination FRAME pixel, then compose transforms.
    corners = np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
    corners = corners @ crop.to_frame[:, :2].T + crop.to_frame[:, 2]
    x0, y0 = np.maximum(0, np.floor(corners.min(axis=0)).astype(int))
    x1, y1 = np.minimum(frame.shape[1::-1], np.ceil(corners.max(axis=0)).astype(int) + 1)
    if x1 <= x0 or y1 <= y0:
        return frame
    xx, yy = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    inverse = cv2.invertAffineTransform(crop.to_frame)
    cx = inverse[0, 0]*xx + inverse[0, 1]*yy + inverse[0, 2]
    cy = inverse[1, 0]*xx + inverse[1, 1]*yy + inverse[1, 2]
    cx, cy = cx.astype(np.float32), cy.astype(np.float32)
    def field(data):
        return remap_exact(data, cx, cy)
    m = field(mask) * ((cx >= 0) & (cx <= width-1) & (cy >= 0) & (cy <= height-1))
    weight = strength * m
    full_at = field(full)
    sx = np.clip(cx + weight * (full_at[..., 0] - cx), 0, width-1)
    sy = np.clip(cy + weight * (full_at[..., 1] - cy), 0, height-1)
    matrix = crop.to_frame
    map_x = (matrix[0, 0]*sx + matrix[0, 1]*sy + matrix[0, 2]).astype(np.float32)
    map_y = (matrix[1, 0]*sx + matrix[1, 1]*sy + matrix[1, 2]).astype(np.float32)
    # Exact zero-weight identity even outside the rotated box.
    map_x = np.where(weight == 0, xx, map_x)
    map_y = np.where(weight == 0, yy, map_y)
    warped = remap_exact(frame, map_x, map_y)
    l = field(light)
    corrected = warped * (1 + weight[..., None] * (l[..., :1] - 1)) + 255 * weight[..., None] * l[..., 1:2]
    original = frame[y0:y1, x0:x1].astype(np.float32)
    result = frame.copy()
    result[y0:y1, x0:x1] = np.clip(np.rint(original + m[..., None] * (corrected-original)), 0, 255).astype(np.uint8)
    return result
