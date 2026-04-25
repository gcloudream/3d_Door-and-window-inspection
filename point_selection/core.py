from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class PointRecord:
    point_id: int
    xyz: Vector3
    rgb: Optional[Tuple[int, int, int]] = None


@dataclass(frozen=True)
class PointPickResult:
    point_id: int
    xyz: Vector3
    distance_to_ray: float
    projection_length: float


@dataclass(frozen=True)
class ROIConfig:
    radius: float
    max_points: int
    min_points: int = 3000
    max_radius: Optional[float] = None
    radius_step: float = 0.30

    def resolved_max_radius(self) -> float:
        return self.max_radius if self.max_radius is not None else self.radius


@dataclass(frozen=True)
class ROIResult:
    center_point_id: int
    center_xyz: Vector3
    radius: float
    point_ids: List[int]
    points: List[PointRecord]
    expansions: int
    truncated: bool

    def to_debug_dict(self) -> Dict[str, object]:
        return {
            "center_point_id": self.center_point_id,
            "center_xyz": list(self.center_xyz),
            "radius": self.radius,
            "point_ids": self.point_ids,
            "expansions": self.expansions,
            "truncated": self.truncated,
        }


class PointCloud:
    def __init__(self, points: Iterable[PointRecord]) -> None:
        point_list = list(points)
        if not point_list:
            raise ValueError("PointCloud requires at least one point")

        self._points = point_list
        self._points_by_id = {point.point_id: point for point in point_list}

        if len(self._points_by_id) != len(self._points):
            raise ValueError("point_id values must be unique")

        self._bounds_min = (
            min(point.xyz[0] for point in point_list),
            min(point.xyz[1] for point in point_list),
            min(point.xyz[2] for point in point_list),
        )
        self._bounds_max = (
            max(point.xyz[0] for point in point_list),
            max(point.xyz[1] for point in point_list),
            max(point.xyz[2] for point in point_list),
        )
        self._spatial_cell_size = resolve_spatial_cell_size(
            bounds_min=self._bounds_min,
            bounds_max=self._bounds_max,
            point_count=len(point_list),
        )
        self._spatial_index = build_spatial_index(
            points=point_list,
            bounds_min=self._bounds_min,
            cell_size=self._spatial_cell_size,
        )

    @property
    def points(self) -> Sequence[PointRecord]:
        return self._points

    def get_point(self, point_id: int) -> PointRecord:
        try:
            return self._points_by_id[point_id]
        except KeyError as exc:
            raise KeyError(f"Unknown point_id: {point_id}") from exc

    def box_query(self, min_xyz: Vector3, max_xyz: Vector3) -> List[PointRecord]:
        resolved_min = tuple(min(left, right) for left, right in zip(min_xyz, max_xyz))
        resolved_max = tuple(max(left, right) for left, right in zip(min_xyz, max_xyz))
        min_cell = self._cell_for_xyz(resolved_min)
        max_cell = self._cell_for_xyz(resolved_max)
        results: List[PointRecord] = []

        for cell_x in range(min_cell[0], max_cell[0] + 1):
            for cell_y in range(min_cell[1], max_cell[1] + 1):
                for cell_z in range(min_cell[2], max_cell[2] + 1):
                    for point in self._spatial_index.get((cell_x, cell_y, cell_z), ()):
                        if point_in_box(point.xyz, resolved_min, resolved_max):
                            results.append(point)

        results.sort(key=lambda point: point.point_id)
        return results

    def _cell_for_xyz(self, xyz: Vector3) -> Tuple[int, int, int]:
        return cell_for_xyz(xyz, bounds_min=self._bounds_min, cell_size=self._spatial_cell_size)

    def radius_query(self, center_xyz: Vector3, radius: float) -> List[Tuple[float, PointRecord]]:
        results: List[Tuple[float, PointRecord]] = []
        radius_sq = radius * radius
        min_xyz = tuple(coordinate - radius for coordinate in center_xyz)
        max_xyz = tuple(coordinate + radius for coordinate in center_xyz)

        for point in self.box_query(min_xyz=min_xyz, max_xyz=max_xyz):
            distance_sq = squared_distance(center_xyz, point.xyz)
            if distance_sq <= radius_sq:
                results.append((math.sqrt(distance_sq), point))

        results.sort(key=lambda item: (item[0], item[1].point_id))
        return results

    def ray_query(
        self,
        ray_origin: Vector3,
        ray_direction: Vector3,
        radius: float,
        max_projection_length: Optional[float] = None,
    ) -> List[PointRecord]:
        expanded_min = tuple(value - radius for value in self._bounds_min)
        expanded_max = tuple(value + radius for value in self._bounds_max)
        projection_bounds = ray_box_projection_bounds(
            ray_origin=ray_origin,
            ray_direction=ray_direction,
            bounds_min=expanded_min,
            bounds_max=expanded_max,
        )
        if projection_bounds is None:
            return []

        start_projection = max(0.0, projection_bounds[0])
        end_projection = projection_bounds[1]
        if max_projection_length is not None:
            end_projection = min(end_projection, max_projection_length)
        if end_projection < start_projection:
            return []

        start_xyz = add_scaled(ray_origin, ray_direction, start_projection)
        end_xyz = add_scaled(ray_origin, ray_direction, end_projection)
        min_xyz = tuple(min(left, right) - radius for left, right in zip(start_xyz, end_xyz))
        max_xyz = tuple(max(left, right) + radius for left, right in zip(start_xyz, end_xyz))
        return self.box_query(min_xyz=min_xyz, max_xyz=max_xyz)

    @classmethod
    def from_json(cls, path: Path) -> "PointCloud":
        payload = json.loads(path.read_text())
        points = [
            PointRecord(
                point_id=entry["point_id"],
                xyz=tuple(entry["xyz"]),
                rgb=tuple(entry["rgb"]) if entry.get("rgb") is not None else None,
            )
            for entry in payload["points"]
        ]
        return cls(points)


class SelectionEngine:
    def __init__(self, cloud: PointCloud) -> None:
        self._cloud = cloud

    def pick_point(
        self,
        ray_origin: Vector3,
        ray_direction: Vector3,
        max_distance_to_ray: Optional[float] = None,
        max_projection_length: Optional[float] = None,
        prefer_frontmost: bool = False,
    ) -> Optional[PointPickResult]:
        direction = normalize(ray_direction)
        best_match: Optional[PointPickResult] = None
        candidates: Iterable[PointRecord] = self._cloud.points

        if max_distance_to_ray is not None:
            candidates = self._cloud.ray_query(
                ray_origin=ray_origin,
                ray_direction=direction,
                radius=max_distance_to_ray,
                max_projection_length=max_projection_length,
            )

        for point in candidates:
            relative = subtract(point.xyz, ray_origin)
            projection = dot(relative, direction)
            if projection < 0.0:
                continue
            if max_projection_length is not None and projection > max_projection_length:
                continue

            perpendicular_sq = max(0.0, dot(relative, relative) - projection * projection)
            distance_to_ray = math.sqrt(perpendicular_sq)
            if max_distance_to_ray is not None and distance_to_ray > max_distance_to_ray:
                continue

            candidate = PointPickResult(
                point_id=point.point_id,
                xyz=point.xyz,
                distance_to_ray=distance_to_ray,
                projection_length=projection,
            )
            if is_better_pick(candidate, best_match, prefer_frontmost=prefer_frontmost):
                best_match = candidate

        return best_match

    def build_roi(self, center_point_id: int, config: ROIConfig) -> ROIResult:
        validate_roi_config(config)

        center_point = self._cloud.get_point(center_point_id)
        current_radius = config.radius
        max_radius = config.resolved_max_radius()
        expansions = 0
        results = self._cloud.radius_query(center_point.xyz, current_radius)

        while len(results) < config.min_points and current_radius < max_radius:
            current_radius = min(current_radius + config.radius_step, max_radius)
            expansions += 1
            results = self._cloud.radius_query(center_point.xyz, current_radius)

        truncated = len(results) > config.max_points
        if truncated:
            results = results[: config.max_points]

        return ROIResult(
            center_point_id=center_point.point_id,
            center_xyz=center_point.xyz,
            radius=current_radius,
            point_ids=[point.point_id for _, point in results],
            points=[point for _, point in results],
            expansions=expansions,
            truncated=truncated,
        )

    def build_wall_guided_roi(
        self,
        center_point_id: int,
        config: ROIConfig,
        wall_axis: str,
        wall_coordinate: float,
        wall_depth_margin: float = 0.20,
        color_tolerance: float = 40.0,
    ) -> ROIResult:
        validate_roi_config(config)

        center_point = self._cloud.get_point(center_point_id)
        axis_index = axis_to_index(wall_axis)
        base_depth = abs(center_point.xyz[axis_index] - wall_coordinate)
        max_radius = resolve_wall_guided_max_radius(config)
        depth_limit = max(base_depth + wall_depth_margin, max(config.radius, max_radius * 0.5) + wall_depth_margin)
        current_radius = config.radius
        expansions = 0
        results = self._wall_guided_query(center_point, wall_axis, wall_coordinate, depth_limit, current_radius)

        while current_radius < max_radius:
            if not should_expand_wall_guided_roi(
                center_point=center_point,
                results=results,
                current_radius=current_radius,
                min_points=config.min_points,
                radius_step=config.radius_step,
                color_tolerance=color_tolerance,
            ):
                break
            current_radius = min(current_radius + config.radius_step, max_radius)
            expansions += 1
            results = self._wall_guided_query(center_point, wall_axis, wall_coordinate, depth_limit, current_radius)

        truncated = len(results) > config.max_points
        if truncated:
            results = results[: config.max_points]

        return ROIResult(
            center_point_id=center_point.point_id,
            center_xyz=center_point.xyz,
            radius=current_radius,
            point_ids=[point.point_id for _, point in results],
            points=[point for _, point in results],
            expansions=expansions,
            truncated=truncated,
        )

    def _wall_guided_query(
        self,
        center_point: PointRecord,
        wall_axis: str,
        wall_coordinate: float,
        depth_limit: float,
        radius: float,
    ) -> List[Tuple[float, PointRecord]]:
        results: List[Tuple[float, PointRecord]] = []
        axis_index = axis_to_index(wall_axis)
        min_xyz = list(center_point.xyz)
        max_xyz = list(center_point.xyz)
        min_xyz[axis_index] = wall_coordinate - depth_limit
        max_xyz[axis_index] = wall_coordinate + depth_limit

        for dimension in range(3):
            if dimension == axis_index:
                continue
            min_xyz[dimension] = center_point.xyz[dimension] - radius
            max_xyz[dimension] = center_point.xyz[dimension] + radius

        for point in self._cloud.box_query(min_xyz=tuple(min_xyz), max_xyz=tuple(max_xyz)):
            if abs(point.xyz[axis_index] - wall_coordinate) > depth_limit:
                continue
            lateral_distance = wall_plane_distance(center_point.xyz, point.xyz, wall_axis)
            if lateral_distance > radius:
                continue
            results.append((lateral_distance, point))

        results.sort(key=lambda item: (item[0], item[1].point_id))
        return results


def is_better_pick(
    candidate: PointPickResult,
    best_match: Optional[PointPickResult],
    *,
    prefer_frontmost: bool = False,
) -> bool:
    if best_match is None:
        return True

    if prefer_frontmost:
        candidate_key = (
            candidate.projection_length,
            candidate.distance_to_ray,
            candidate.point_id,
        )
        best_key = (
            best_match.projection_length,
            best_match.distance_to_ray,
            best_match.point_id,
        )
    else:
        candidate_key = (
            candidate.distance_to_ray,
            candidate.projection_length,
            candidate.point_id,
        )
        best_key = (
            best_match.distance_to_ray,
            best_match.projection_length,
            best_match.point_id,
        )
    return candidate_key < best_key


def normalize(vector: Vector3) -> Vector3:
    length_sq = dot(vector, vector)
    if length_sq == 0.0:
        raise ValueError("ray_direction must be non-zero")
    length = math.sqrt(length_sq)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def squared_distance(a: Vector3, b: Vector3) -> float:
    delta = subtract(a, b)
    return dot(delta, delta)


def add_scaled(origin: Vector3, direction: Vector3, scale: float) -> Vector3:
    return (
        origin[0] + direction[0] * scale,
        origin[1] + direction[1] * scale,
        origin[2] + direction[2] * scale,
    )


def validate_roi_config(config: ROIConfig) -> None:
    if config.radius <= 0.0:
        raise ValueError("ROI radius must be positive")
    if config.max_points <= 0:
        raise ValueError("ROI max_points must be positive")
    if config.min_points <= 0:
        raise ValueError("ROI min_points must be positive")
    if config.radius_step <= 0.0:
        raise ValueError("ROI radius_step must be positive")


def resolve_wall_guided_max_radius(config: ROIConfig) -> float:
    if config.max_radius is not None:
        return config.max_radius
    return max(config.radius, config.radius + max(config.radius_step * 4, 0.9), config.radius * 2.5)


def should_expand_wall_guided_roi(
    center_point: PointRecord,
    results: Sequence[Tuple[float, PointRecord]],
    current_radius: float,
    min_points: int,
    radius_step: float,
    color_tolerance: float,
) -> bool:
    if len(results) < min_points:
        return True

    support_points = [
        (distance, point)
        for distance, point in results
        if point_color_distance(center_point, point) <= color_tolerance
    ]
    if len(support_points) < min_points:
        return True

    boundary_margin = max(radius_step * 0.75, current_radius * 0.12, 0.08)
    boundary_hits = [
        point
        for distance, point in support_points
        if distance >= current_radius - boundary_margin
    ]
    return len(boundary_hits) >= 2


def resolve_spatial_cell_size(bounds_min: Vector3, bounds_max: Vector3, point_count: int) -> float:
    spans = [max(right - left, 1e-6) for left, right in zip(bounds_min, bounds_max)]
    max_span = max(spans)
    if max_span <= 1e-6:
        return 0.05
    target_cells_per_axis = min(max(int(round(point_count ** (1.0 / 3.0))), 24), 72)
    return max(max_span / target_cells_per_axis, 0.05)


def build_spatial_index(
    *,
    points: Sequence[PointRecord],
    bounds_min: Vector3,
    cell_size: float,
) -> Dict[Tuple[int, int, int], List[PointRecord]]:
    index: Dict[Tuple[int, int, int], List[PointRecord]] = {}
    for point in points:
        cell = cell_for_xyz(point.xyz, bounds_min=bounds_min, cell_size=cell_size)
        index.setdefault(cell, []).append(point)
    return index


def cell_for_xyz(xyz: Vector3, *, bounds_min: Vector3, cell_size: float) -> Tuple[int, int, int]:
    return tuple(
        int(math.floor((coordinate - min_coordinate) / cell_size))
        for coordinate, min_coordinate in zip(xyz, bounds_min)
    )


def point_in_box(xyz: Vector3, min_xyz: Vector3, max_xyz: Vector3) -> bool:
    return all(lower <= value <= upper for value, lower, upper in zip(xyz, min_xyz, max_xyz))


def ray_box_projection_bounds(
    *,
    ray_origin: Vector3,
    ray_direction: Vector3,
    bounds_min: Vector3,
    bounds_max: Vector3,
) -> Optional[Tuple[float, float]]:
    lower_bound = -math.inf
    upper_bound = math.inf

    for axis in range(3):
        origin_coordinate = ray_origin[axis]
        direction_component = ray_direction[axis]
        axis_min = bounds_min[axis]
        axis_max = bounds_max[axis]

        if abs(direction_component) <= 1e-12:
            if origin_coordinate < axis_min or origin_coordinate > axis_max:
                return None
            continue

        inverse_direction = 1.0 / direction_component
        first_hit = (axis_min - origin_coordinate) * inverse_direction
        second_hit = (axis_max - origin_coordinate) * inverse_direction
        entry_projection = min(first_hit, second_hit)
        exit_projection = max(first_hit, second_hit)

        lower_bound = max(lower_bound, entry_projection)
        upper_bound = min(upper_bound, exit_projection)
        if lower_bound > upper_bound:
            return None

    return lower_bound, upper_bound


def wall_plane_distance(a: Vector3, b: Vector3, wall_axis: str) -> float:
    dimensions = [0, 1, 2]
    dimensions.remove(axis_to_index(wall_axis))
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in dimensions))


def axis_to_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def point_color_distance(a: PointRecord, b: PointRecord) -> float:
    if a.rgb is None or b.rgb is None:
        return 0.0
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a.rgb, b.rgb)))
