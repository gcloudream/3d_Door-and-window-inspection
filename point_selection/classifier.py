from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from point_selection.core import PointRecord


Vector3 = Tuple[float, float, float]
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
HORIZONTAL_AXES = ("x", "z")


@dataclass(frozen=True)
class BoundaryPlane:
    axis: str
    side: str
    coordinate: float
    support_ratio: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "axis": self.axis,
            "side": self.side,
            "coordinate": self.coordinate,
            "support_ratio": round(self.support_ratio, 3),
        }


@dataclass(frozen=True)
class SceneStructure:
    min_xyz: Vector3
    max_xyz: Vector3
    floor_y: float
    ceiling_y: float
    wall_planes: List[BoundaryPlane]

    def to_dict(self) -> Dict[str, object]:
        return {
            "min_xyz": list(self.min_xyz),
            "max_xyz": list(self.max_xyz),
            "floor_y": self.floor_y,
            "ceiling_y": self.ceiling_y,
            "wall_planes": [plane.to_dict() for plane in self.wall_planes],
        }


@dataclass(frozen=True)
class CandidateClassification:
    label: str
    confidence: float
    attached_to_wall: bool
    width: float
    height: float
    thickness: float
    bottom_clearance: float
    top_clearance: float
    wall_distance: float
    wall_axis: str
    wall_side: str
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "attached_to_wall": self.attached_to_wall,
            "width": round(self.width, 3),
            "height": round(self.height, 3),
            "thickness": round(self.thickness, 3),
            "bottom_clearance": round(self.bottom_clearance, 3),
            "top_clearance": round(self.top_clearance, 3),
            "wall_distance": round(self.wall_distance, 3) if self.wall_distance != float("inf") else None,
            "wall_axis": self.wall_axis,
            "wall_side": self.wall_side,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CandidateBox:
    label: str
    min_xyz: Vector3
    max_xyz: Vector3
    center: Vector3
    size: Vector3
    wall_axis: str
    wall_side: str
    shape: str = "aabb"
    anchor_mode: str = "bounds"
    corners: Tuple[Vector3, ...] = ()
    front_face: Tuple[Vector3, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "min": list(self.min_xyz),
            "max": list(self.max_xyz),
            "center": list(self.center),
            "size": list(self.size),
            "wall_axis": self.wall_axis,
            "wall_side": self.wall_side,
            "shape": self.shape,
            "anchor_mode": self.anchor_mode,
            "corners": [list(corner) for corner in self.corners],
            "front_face": [list(point) for point in self.front_face],
        }


@dataclass(frozen=True)
class OpeningCandidate:
    label: str
    confidence: float
    wall_axis: str
    wall_side: str
    min_xyz: Vector3
    max_xyz: Vector3
    size: Vector3
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "wall_axis": self.wall_axis,
            "wall_side": self.wall_side,
            "min": list(self.min_xyz),
            "max": list(self.max_xyz),
            "size": list(self.size),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _AxisPlaneCluster:
    axis: str
    coordinate: float
    support_count: int
    support_ratio: float


def analyze_scene_structure(
    points: Sequence[PointRecord],
    boundary_margin: float = 0.15,
    min_support_ratio: float = 0.05,
) -> SceneStructure:
    if not points:
        raise ValueError("Scene structure requires at least one point")

    min_xyz, max_xyz = compute_bounds(points)
    wall_planes: List[BoundaryPlane] = []
    point_count = len(points)
    min_support_count = max(4, int(math.ceil(point_count * min_support_ratio)))

    y_tolerance = max(boundary_margin, (max_xyz[1] - min_xyz[1]) * 0.03, 0.05)
    y_clusters = filter_supported_clusters(
        build_axis_plane_clusters(points, axis="y", tolerance=y_tolerance),
        min_support_count=min_support_count,
    )
    floor_y = y_clusters[0].coordinate if y_clusters else min_xyz[1]
    ceiling_y = y_clusters[-1].coordinate if y_clusters else max_xyz[1]

    for axis in HORIZONTAL_AXES:
        index = AXIS_INDEX[axis]
        span = max_xyz[index] - min_xyz[index]
        tolerance = max(boundary_margin, span * 0.03, 0.05)
        clusters = filter_supported_clusters(
            build_axis_plane_clusters(points, axis=axis, tolerance=tolerance),
            min_support_count=min_support_count,
        )
        if clusters:
            min_cluster = clusters[0]
            max_cluster = clusters[-1]
            wall_planes.append(
                BoundaryPlane(
                    axis=axis,
                    side="min",
                    coordinate=min_cluster.coordinate,
                    support_ratio=min_cluster.support_ratio,
                )
            )
            if max_cluster.coordinate - min_cluster.coordinate > tolerance * 0.5:
                wall_planes.append(
                    BoundaryPlane(
                        axis=axis,
                        side="max",
                        coordinate=max_cluster.coordinate,
                        support_ratio=max_cluster.support_ratio,
                    )
                )
            continue

        margin = max(boundary_margin, span * 0.05)
        min_support = sum(abs(point.xyz[index] - min_xyz[index]) <= margin for point in points) / point_count
        max_support = sum(abs(point.xyz[index] - max_xyz[index]) <= margin for point in points) / point_count
        if min_support >= min_support_ratio:
            wall_planes.append(BoundaryPlane(axis=axis, side="min", coordinate=min_xyz[index], support_ratio=min_support))
        if max_support >= min_support_ratio:
            wall_planes.append(BoundaryPlane(axis=axis, side="max", coordinate=max_xyz[index], support_ratio=max_support))

    if not wall_planes:
        wall_planes = [
            BoundaryPlane(axis="x", side="min", coordinate=min_xyz[0], support_ratio=0.0),
            BoundaryPlane(axis="x", side="max", coordinate=max_xyz[0], support_ratio=0.0),
            BoundaryPlane(axis="z", side="min", coordinate=min_xyz[2], support_ratio=0.0),
            BoundaryPlane(axis="z", side="max", coordinate=max_xyz[2], support_ratio=0.0),
        ]

    return SceneStructure(
        min_xyz=min_xyz,
        max_xyz=max_xyz,
        floor_y=floor_y,
        ceiling_y=ceiling_y,
        wall_planes=wall_planes,
    )


def classify_mask_points(
    points: Sequence[PointRecord],
    scene_structure: SceneStructure,
) -> CandidateClassification:
    if not points:
        return CandidateClassification(
            label="unknown",
            confidence=0.2,
            attached_to_wall=False,
            width=0.0,
            height=0.0,
            thickness=0.0,
            bottom_clearance=0.0,
            top_clearance=0.0,
            wall_distance=float("inf"),
            wall_axis="",
            wall_side="",
            reason="结果为空",
        )

    min_xyz, max_xyz = compute_bounds(points)
    x_extent = max_xyz[0] - min_xyz[0]
    y_extent = max_xyz[1] - min_xyz[1]
    z_extent = max_xyz[2] - min_xyz[2]

    width = max(x_extent, z_extent)
    thickness = min(x_extent, z_extent)
    height = y_extent
    bottom_clearance = max(0.0, min_xyz[1] - scene_structure.floor_y)
    top_clearance = max(0.0, scene_structure.ceiling_y - max_xyz[1])

    nearest_wall = find_nearest_wall(scene_structure, min_xyz, max_xyz)
    wall_axis = nearest_wall.axis if nearest_wall is not None else ""
    wall_side = nearest_wall.side if nearest_wall is not None else ""
    wall_distance = nearest_wall.distance if nearest_wall is not None else float("inf")

    wall_attach_tolerance = max(0.3, min(0.8, width * 0.8 + thickness * 0.4))
    wall_attach_score = max(0.0, 1.0 - wall_distance / wall_attach_tolerance)
    attached_to_wall = wall_attach_score >= 0.15

    opening = detect_wall_opening(points, scene_structure, nearest_wall)
    opening_score = opening.confidence if opening is not None else 0.0
    opening_bonus = 0.12 * opening_score

    if height < 0.35 or not attached_to_wall:
        return build_unknown_result(
            width=width,
            height=height,
            thickness=thickness,
            bottom_clearance=bottom_clearance,
            top_clearance=top_clearance,
            wall_distance=wall_distance,
            wall_axis=wall_axis,
            wall_side=wall_side,
            reason="高度不足或未贴墙",
        )

    door_score = (
        0.35 * wall_attach_score
        + 0.25 * band_score(bottom_clearance, 0.0, 0.25, 0.4)
        + 0.2 * band_score(height, 1.7, 2.4, 0.6)
        + 0.12 * band_score(width, 0.55, 1.5, 0.5)
        + 0.08 * band_score(thickness, 0.0, 0.25, 0.25)
        + opening_bonus * band_score(bottom_clearance, 0.0, 0.2, 0.2)
    )

    window_score = (
        0.3 * wall_attach_score
        + 0.2 * band_score(bottom_clearance, 0.5, 2.1, 0.7)
        + 0.22 * band_score(height, 0.45, 1.6, 0.5)
        + 0.14 * band_score(width, 0.4, 2.2, 0.6)
        + 0.08 * band_score(top_clearance, 0.1, 1.5, 0.7)
        + 0.06 * band_score(thickness, 0.0, 0.55, 0.3)
        + opening_bonus * band_score(bottom_clearance, 0.35, 2.2, 0.6)
    )

    if opening is not None:
        if opening.label == "door":
            door_score += 0.08 * opening_score
        elif opening.label == "window":
            window_score += 0.08 * opening_score

    if door_score >= window_score and door_score >= 0.62:
        reason = build_reason("door", width, height, thickness, bottom_clearance, wall_axis, wall_side)
        if opening is not None:
            reason = f"{reason}；{opening.reason}"
        return CandidateClassification(
            label="door",
            confidence=round(min(0.99, max(0.2, door_score)), 3),
            attached_to_wall=attached_to_wall,
            width=width,
            height=height,
            thickness=thickness,
            bottom_clearance=bottom_clearance,
            top_clearance=top_clearance,
            wall_distance=wall_distance,
            wall_axis=wall_axis,
            wall_side=wall_side,
            reason=reason,
        )

    if window_score >= 0.62:
        reason = build_reason("window", width, height, thickness, bottom_clearance, wall_axis, wall_side)
        if opening is not None:
            reason = f"{reason}；{opening.reason}"
        return CandidateClassification(
            label="window",
            confidence=round(min(0.99, max(0.2, window_score)), 3),
            attached_to_wall=attached_to_wall,
            width=width,
            height=height,
            thickness=thickness,
            bottom_clearance=bottom_clearance,
            top_clearance=top_clearance,
            wall_distance=wall_distance,
            wall_axis=wall_axis,
            wall_side=wall_side,
            reason=reason,
        )

    return build_unknown_result(
        width=width,
        height=height,
        thickness=thickness,
        bottom_clearance=bottom_clearance,
        top_clearance=top_clearance,
        wall_distance=wall_distance,
        wall_axis=wall_axis,
        wall_side=wall_side,
        reason=opening.reason if opening is not None else "几何特征未命中门窗阈值",
    )


def build_candidate_box(
    points: Sequence[PointRecord],
    classification: CandidateClassification,
    scene_structure: SceneStructure,
    default_padding: float = 0.03,
) -> CandidateBox:
    if not points:
        zero = (0.0, 0.0, 0.0)
        return CandidateBox(
            label=classification.label,
            min_xyz=zero,
            max_xyz=zero,
            center=zero,
            size=zero,
            wall_axis=classification.wall_axis,
            wall_side=classification.wall_side,
            shape="empty",
            anchor_mode="none",
        )

    wall_rect_box = build_wall_rect_candidate_box(
        points=points,
        classification=classification,
        scene_structure=scene_structure,
        default_padding=default_padding,
    )
    if wall_rect_box is not None:
        return wall_rect_box

    return build_axis_aligned_candidate_box(
        points=points,
        classification=classification,
        scene_structure=scene_structure,
        default_padding=default_padding,
    )


def detect_wall_opening(
    points: Sequence[PointRecord],
    scene_structure: SceneStructure,
    nearest_wall: Optional[_NearestWall],
) -> Optional[OpeningCandidate]:
    if nearest_wall is None or nearest_wall.axis not in AXIS_INDEX:
        return None

    wall_coordinate = resolve_wall_coordinate(scene_structure, nearest_wall.axis, nearest_wall.side)
    if wall_coordinate is None:
        return None

    wall_index = AXIS_INDEX[nearest_wall.axis]
    span_axis = "z" if nearest_wall.axis == "x" else "x"
    span_index = AXIS_INDEX[span_axis]

    wall_points = [
        point
        for point in points
        if abs(point.xyz[wall_index] - wall_coordinate) <= 0.35
    ]
    if len(wall_points) < 4:
        return None

    span_low, span_high = robust_bounds((point.xyz[span_index] for point in wall_points), trim_ratio=0.08)
    height_low, height_high = robust_bounds((point.xyz[1] for point in wall_points), trim_ratio=0.08)
    depth_low, depth_high = robust_bounds((point.xyz[wall_index] for point in wall_points), trim_ratio=0.0)

    opening_width = max(0.0, span_high - span_low)
    opening_height = max(0.0, height_high - height_low)
    opening_thickness = max(0.0, depth_high - depth_low)
    floor_gap = max(0.0, height_low - scene_structure.floor_y)
    ceiling_gap = max(0.0, scene_structure.ceiling_y - height_high)

    door_score = (
        0.34 * band_score(opening_height, 1.7, 2.5, 0.6)
        + 0.24 * band_score(floor_gap, 0.0, 0.25, 0.25)
        + 0.18 * band_score(opening_width, 0.7, 1.8, 0.6)
        + 0.12 * band_score(opening_thickness, 0.0, 0.3, 0.15)
        + 0.12 * band_score(ceiling_gap, 0.0, 1.0, 0.8)
    )
    window_score = (
        0.3 * band_score(opening_height, 0.4, 1.6, 0.5)
        + 0.24 * band_score(floor_gap, 0.45, 2.1, 0.6)
        + 0.2 * band_score(ceiling_gap, 0.0, 1.4, 0.6)
        + 0.16 * band_score(opening_width, 0.4, 2.4, 0.7)
        + 0.1 * band_score(opening_thickness, 0.0, 0.4, 0.2)
    )

    if door_score < 0.5 and window_score < 0.5:
        return None

    if door_score >= window_score:
        label = "door"
        confidence = door_score
        reason = (
            f"墙面开口更像门：宽 {opening_width:.2f}m，高 {opening_height:.2f}m，"
            f"离地 {floor_gap:.2f}m，厚 {opening_thickness:.2f}m"
        )
    else:
        label = "window"
        confidence = window_score
        reason = (
            f"墙面开口更像窗：宽 {opening_width:.2f}m，高 {opening_height:.2f}m，"
            f"离地 {floor_gap:.2f}m，顶距 {ceiling_gap:.2f}m"
        )

    if span_index == 0:
        min_xyz = (span_low, height_low, depth_low)
        max_xyz = (span_high, height_high, depth_high)
    else:
        min_xyz = (depth_low, height_low, span_low)
        max_xyz = (depth_high, height_high, span_high)

    return OpeningCandidate(
        label=label,
        confidence=round(min(0.99, max(0.2, confidence)), 3),
        wall_axis=nearest_wall.axis,
        wall_side=nearest_wall.side,
        min_xyz=min_xyz,
        max_xyz=max_xyz,
        size=(opening_width, opening_height, opening_thickness),
        reason=reason,
    )


def build_axis_aligned_candidate_box(
    points: Sequence[PointRecord],
    classification: CandidateClassification,
    scene_structure: SceneStructure,
    default_padding: float,
) -> CandidateBox:
    min_xyz, max_xyz = compute_bounds(points)
    min_values = [min_xyz[0], min_xyz[1], min_xyz[2]]
    max_values = [max_xyz[0], max_xyz[1], max_xyz[2]]

    for axis_index in range(3):
        min_values[axis_index] -= default_padding
        max_values[axis_index] += default_padding

    if classification.label == "door":
        min_values[1] = scene_structure.floor_y

    if classification.wall_axis in AXIS_INDEX and classification.wall_side in {"min", "max"}:
        wall_index = AXIS_INDEX[classification.wall_axis]
        minimum_thickness = max(classification.thickness, 0.08)
        wall_coordinate = resolve_wall_coordinate(scene_structure, classification.wall_axis, classification.wall_side)
        if wall_coordinate is not None:
            if classification.wall_side == "min":
                min_values[wall_index] = wall_coordinate
                max_values[wall_index] = max(max_values[wall_index], wall_coordinate + minimum_thickness)
            else:
                max_values[wall_index] = wall_coordinate
                min_values[wall_index] = min(min_values[wall_index], wall_coordinate - minimum_thickness)

    refined_min = tuple(min_values)
    refined_max = tuple(max_values)
    center = tuple((low + high) / 2 for low, high in zip(refined_min, refined_max))
    size = tuple(high - low for low, high in zip(refined_min, refined_max))
    corners = corners_from_bounds(refined_min, refined_max)
    return CandidateBox(
        label=classification.label,
        min_xyz=refined_min,
        max_xyz=refined_max,
        center=center,
        size=size,
        wall_axis=classification.wall_axis,
        wall_side=classification.wall_side,
        shape="aabb",
        anchor_mode="bounds",
        corners=corners,
    )


def build_wall_rect_candidate_box(
    points: Sequence[PointRecord],
    classification: CandidateClassification,
    scene_structure: SceneStructure,
    default_padding: float,
) -> Optional[CandidateBox]:
    if classification.wall_axis not in AXIS_INDEX or classification.wall_side not in {"min", "max"}:
        return None

    wall_index = AXIS_INDEX[classification.wall_axis]
    span_axis = "z" if classification.wall_axis == "x" else "x"
    span_index = AXIS_INDEX[span_axis]

    span_low, span_high = robust_bounds((point.xyz[span_index] for point in points), trim_ratio=0.08)
    height_low, height_high = robust_bounds((point.xyz[1] for point in points), trim_ratio=0.08)
    depth_low, depth_high = robust_bounds((point.xyz[wall_index] for point in points), trim_ratio=0.0)

    span_low -= default_padding
    span_high += default_padding
    height_low = height_low - default_padding
    height_high += default_padding

    if classification.label == "door":
        height_low = scene_structure.floor_y

    wall_coordinate = resolve_wall_coordinate(scene_structure, classification.wall_axis, classification.wall_side)
    should_snap_to_wall = (
        wall_coordinate is not None
        and classification.attached_to_wall
        and classification.wall_distance <= wall_snap_tolerance(classification, default_padding)
    )
    depth_min, depth_max, anchor_mode = resolve_wall_depth_bounds(
        raw_low=depth_low,
        raw_high=depth_high,
        wall_side=classification.wall_side,
        wall_coordinate=wall_coordinate,
        snap_to_wall=should_snap_to_wall,
        padding=default_padding,
    )

    min_values = [0.0, height_low, 0.0]
    max_values = [0.0, height_high, 0.0]
    min_values[span_index] = span_low
    max_values[span_index] = span_high
    min_values[wall_index] = depth_min
    max_values[wall_index] = depth_max

    # The remaining axis is already set by either span or wall depth above.
    if span_index == AXIS_INDEX["x"] and wall_index == AXIS_INDEX["z"]:
        min_values[0] = span_low
        max_values[0] = span_high
        min_values[2] = depth_min
        max_values[2] = depth_max
    elif span_index == AXIS_INDEX["z"] and wall_index == AXIS_INDEX["x"]:
        min_values[2] = span_low
        max_values[2] = span_high
        min_values[0] = depth_min
        max_values[0] = depth_max

    refined_min = tuple(min_values)
    refined_max = tuple(max_values)
    center = tuple((low + high) / 2 for low, high in zip(refined_min, refined_max))
    size = tuple(high - low for low, high in zip(refined_min, refined_max))
    corners = corners_from_bounds(refined_min, refined_max)
    front_face = front_face_from_bounds(refined_min, refined_max, classification.wall_axis, classification.wall_side)

    return CandidateBox(
        label=classification.label,
        min_xyz=refined_min,
        max_xyz=refined_max,
        center=center,
        size=size,
        wall_axis=classification.wall_axis,
        wall_side=classification.wall_side,
        shape="wall_rect",
        anchor_mode=anchor_mode,
        corners=corners,
        front_face=front_face,
    )


def robust_bounds(values, trim_ratio: float) -> Tuple[float, float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Expected at least one value")
    if len(ordered) < 8 or trim_ratio <= 0.0:
        return ordered[0], ordered[-1]

    high_fraction = 1.0 - trim_ratio
    low_index = min(len(ordered) - 1, int(math.floor((len(ordered) - 1) * trim_ratio)))
    high_index = max(low_index, min(len(ordered) - 1, int(math.ceil((len(ordered) - 1) * high_fraction))))
    return ordered[low_index], ordered[high_index]


def wall_snap_tolerance(classification: CandidateClassification, padding: float) -> float:
    return max(0.12, min(0.28, classification.thickness * 0.35 + padding * 2.0))


def resolve_wall_depth_bounds(
    *,
    raw_low: float,
    raw_high: float,
    wall_side: str,
    wall_coordinate: Optional[float],
    snap_to_wall: bool,
    padding: float,
) -> Tuple[float, float, str]:
    minimum_depth = max(0.08, padding * 2.5)
    anchor_mode = "wall_plane" if snap_to_wall and wall_coordinate is not None else "mask_face"

    if wall_side == "min":
        near_face = wall_coordinate if anchor_mode == "wall_plane" and wall_coordinate is not None else raw_low - padding
        far_face = max(raw_high + padding, near_face + minimum_depth)
        return near_face, far_face, anchor_mode

    near_face = wall_coordinate if anchor_mode == "wall_plane" and wall_coordinate is not None else raw_high + padding
    far_face = min(raw_low - padding, near_face - minimum_depth)
    return far_face, near_face, anchor_mode


def corners_from_bounds(min_xyz: Vector3, max_xyz: Vector3) -> Tuple[Vector3, ...]:
    min_x, min_y, min_z = min_xyz
    max_x, max_y, max_z = max_xyz
    return (
        (min_x, min_y, min_z),
        (max_x, min_y, min_z),
        (max_x, max_y, min_z),
        (min_x, max_y, min_z),
        (min_x, min_y, max_z),
        (max_x, min_y, max_z),
        (max_x, max_y, max_z),
        (min_x, max_y, max_z),
    )


def front_face_from_bounds(
    min_xyz: Vector3,
    max_xyz: Vector3,
    wall_axis: str,
    wall_side: str,
) -> Tuple[Vector3, ...]:
    min_x, min_y, min_z = min_xyz
    max_x, max_y, max_z = max_xyz

    if wall_axis == "x":
        face_x = min_x if wall_side == "min" else max_x
        return (
            (face_x, min_y, min_z),
            (face_x, min_y, max_z),
            (face_x, max_y, max_z),
            (face_x, max_y, min_z),
            (face_x, min_y, min_z),
        )

    face_z = min_z if wall_side == "min" else max_z
    return (
        (min_x, min_y, face_z),
        (max_x, min_y, face_z),
        (max_x, max_y, face_z),
        (min_x, max_y, face_z),
        (min_x, min_y, face_z),
    )


@dataclass(frozen=True)
class _NearestWall:
    axis: str
    side: str
    distance: float


def find_nearest_wall(
    scene_structure: SceneStructure,
    min_xyz: Vector3,
    max_xyz: Vector3,
) -> Optional[_NearestWall]:
    best_wall: Optional[_NearestWall] = None

    for plane in scene_structure.wall_planes:
        index = AXIS_INDEX[plane.axis]
        boundary_value = min_xyz[index] if plane.side == "min" else max_xyz[index]
        distance = abs(boundary_value - plane.coordinate)
        candidate = _NearestWall(axis=plane.axis, side=plane.side, distance=distance)
        if best_wall is None or candidate.distance < best_wall.distance:
            best_wall = candidate

    return best_wall


def build_unknown_result(
    *,
    width: float,
    height: float,
    thickness: float,
    bottom_clearance: float,
    top_clearance: float,
    wall_distance: float,
    wall_axis: str,
    wall_side: str,
    reason: str,
) -> CandidateClassification:
    return CandidateClassification(
        label="unknown",
        confidence=0.35,
        attached_to_wall=wall_distance != float("inf") and wall_distance < 0.8,
        width=width,
        height=height,
        thickness=thickness,
        bottom_clearance=bottom_clearance,
        top_clearance=top_clearance,
        wall_distance=wall_distance,
        wall_axis=wall_axis,
        wall_side=wall_side,
        reason=reason,
    )


def build_reason(
    label: str,
    width: float,
    height: float,
    thickness: float,
    bottom_clearance: float,
    wall_axis: str,
    wall_side: str,
) -> str:
    label_text = "门候选" if label == "door" else "窗候选"
    wall_text = f"{wall_axis}-{wall_side}" if wall_axis else "未定位墙面"
    return (
        f"{label_text}，贴近墙面 {wall_text}，"
        f"宽 {width:.2f}m，高 {height:.2f}m，厚 {thickness:.2f}m，离地 {bottom_clearance:.2f}m"
    )


def band_score(value: float, lower: float, upper: float, soft_margin: float) -> float:
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        if soft_margin <= 0.0:
            return 0.0
        return max(0.0, 1.0 - (lower - value) / soft_margin)
    if soft_margin <= 0.0:
        return 0.0
    return max(0.0, 1.0 - (value - upper) / soft_margin)


def resolve_wall_coordinate(scene_structure: SceneStructure, axis: str, side: str) -> Optional[float]:
    for plane in scene_structure.wall_planes:
        if plane.axis == axis and plane.side == side:
            return plane.coordinate
    return None


def build_axis_plane_clusters(
    points: Sequence[PointRecord],
    axis: str,
    tolerance: float,
) -> List[_AxisPlaneCluster]:
    index = AXIS_INDEX[axis]
    ordered_points = sorted(points, key=lambda point: point.xyz[index])
    if not ordered_points:
        return []

    clusters: List[List[PointRecord]] = []
    current_cluster = [ordered_points[0]]

    for point in ordered_points[1:]:
        previous = current_cluster[-1]
        if abs(point.xyz[index] - previous.xyz[index]) <= tolerance:
            current_cluster.append(point)
            continue
        clusters.append(current_cluster)
        current_cluster = [point]

    clusters.append(current_cluster)

    total_count = len(points)
    plane_clusters = []
    for cluster_points in clusters:
        coordinate = sum(point.xyz[index] for point in cluster_points) / len(cluster_points)
        plane_clusters.append(
            _AxisPlaneCluster(
                axis=axis,
                coordinate=coordinate,
                support_count=len(cluster_points),
                support_ratio=len(cluster_points) / total_count,
            )
        )
    return plane_clusters


def filter_supported_clusters(
    clusters: Sequence[_AxisPlaneCluster],
    min_support_count: int,
) -> List[_AxisPlaneCluster]:
    return [cluster for cluster in clusters if cluster.support_count >= min_support_count]


def compute_bounds(points: Sequence[PointRecord]) -> Tuple[Vector3, Vector3]:
    first = points[0]
    min_x, min_y, min_z = first.xyz
    max_x, max_y, max_z = first.xyz

    for point in points[1:]:
        x, y, z = point.xyz
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)

    return (min_x, min_y, min_z), (max_x, max_y, max_z)
