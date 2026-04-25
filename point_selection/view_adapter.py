from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from point_selection.core import normalize


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraFrame:
    origin: Vector3
    right: Vector3
    up: Vector3
    forward: Vector3

    @classmethod
    def look_at(cls, origin: Vector3, target: Vector3, up_hint: Vector3) -> "CameraFrame":
        forward = normalize(subtract(target, origin))
        right = normalize(cross(up_hint, forward))
        up = normalize(cross(forward, right))
        return cls(origin=origin, right=right, up=up, forward=forward)


def screen_click_to_ray(
    screen_x: float,
    screen_y: float,
    intrinsics: CameraIntrinsics,
    frame: CameraFrame,
) -> Tuple[Vector3, Vector3]:
    x_cam = (screen_x - intrinsics.cx) / intrinsics.fx
    y_cam = (screen_y - intrinsics.cy) / intrinsics.fy

    direction = normalize(
        add(
            scale(frame.forward, 1.0),
            add(scale(frame.right, x_cam), scale(frame.up, -y_cam)),
        )
    )
    return frame.origin, direction


def project_point_to_screen(
    point_xyz: Vector3,
    intrinsics: CameraIntrinsics,
    frame: CameraFrame,
) -> Optional[Tuple[float, float, float]]:
    relative = subtract(point_xyz, frame.origin)
    depth = dot(relative, frame.forward)
    if depth <= 0.0:
        return None

    x_cam = dot(relative, frame.right)
    y_up = dot(relative, frame.up)
    screen_x = intrinsics.cx + intrinsics.fx * (x_cam / depth)
    screen_y = intrinsics.cy - intrinsics.fy * (y_up / depth)
    return (screen_x, screen_y, depth)


def add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(vector: Vector3, factor: float) -> Vector3:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
