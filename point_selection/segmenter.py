from __future__ import annotations

import importlib
import math
import os
import statistics
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from point_selection.classifier import SceneStructure
from point_selection.core import PointRecord, ROIResult, squared_distance


@dataclass(frozen=True)
class SegmentConfig:
    neighbor_radius: float = 0.65
    color_tolerance: float = 36.0
    plane_distance_tolerance: float = 0.12
    plane_neighbor_count: int = 10
    plane_min_area: float = 0.01
    wall_depth_margin: float = 0.18
    wall_max_depth: float = 1.2
    wall_rectangle_occupancy_threshold: float = 0.55
    wall_rectangle_border_coverage_threshold: float = 0.72
    wall_rectangle_min_points: int = 4
    wall_rectangle_max_bridge_gap_cells: int = 1
    window_projection_min_span_cells: int = 4
    window_surface_seed_band: float = 0.18
    window_surface_min_points: int = 4
    negative_max_hops: int = 2


@dataclass(frozen=True)
class SeedPlane:
    normal: Tuple[float, float, float]
    offset: float
    support_area: float
    support_point_ids: List[int]


@dataclass(frozen=True)
class WallGuidance:
    axis: str
    side: str
    coordinate: float
    seed_depth: float
    max_depth: float


@dataclass(frozen=True)
class ProjectionRectangleSupport:
    min_first: int
    max_first: int
    min_second: int
    max_second: int
    occupancy_ratio: float
    border_coverage: float

    @property
    def width_cells(self) -> int:
        return self.max_first - self.min_first + 1

    @property
    def height_cells(self) -> int:
        return self.max_second - self.min_second + 1


@dataclass(frozen=True)
class SegmenterStatus:
    requested_backend: str
    active_backend: str
    fallback_reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "active_backend": self.active_backend,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class SegmentResult:
    seed_point_id: int
    point_ids: List[int]
    point_count: int
    confidence: float
    method: str
    positive_point_ids: List[int] = field(default_factory=list)
    negative_point_ids: List[int] = field(default_factory=list)
    removed_point_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "seed_point_id": self.seed_point_id,
            "point_ids": self.point_ids,
            "point_count": self.point_count,
            "confidence": self.confidence,
            "method": self.method,
            "positive_point_ids": self.positive_point_ids,
            "negative_point_ids": self.negative_point_ids,
            "removed_point_ids": self.removed_point_ids,
        }


@dataclass(frozen=True)
class PositiveRefinementResult:
    refined_point_ids: List[int]
    positive_point_ids: List[int]


@dataclass(frozen=True)
class NegativeRefinementResult:
    refined_point_ids: List[int]
    negative_point_ids: List[int]
    removed_point_ids: List[int]


@dataclass(frozen=True)
class PointSAMRuntimeConfig:
    checkpoint_path: Path
    repo_dir: Path
    config_name: str = "large"
    device: str = "cuda"


class HeuristicRegionSegmenter:
    def __init__(self, config: SegmentConfig | None = None) -> None:
        self._config = config or SegmentConfig()

    def segment(
        self,
        roi: ROIResult,
        seed_point_id: int,
        scene_structure: Optional[SceneStructure] = None,
        positive_point_ids: Optional[Sequence[int]] = None,
        negative_point_ids: Optional[Sequence[int]] = None,
    ) -> SegmentResult:
        points_by_id = {point.point_id: point for point in roi.points}
        try:
            seed_point = points_by_id[seed_point_id]
        except KeyError as exc:
            raise KeyError(f"Seed point {seed_point_id} not found in ROI") from exc

        cluster_ids, seed_plane, wall_guidance = segment_point_cluster(
            roi_points=roi.points,
            seed_point_id=seed_point_id,
            scene_structure=scene_structure,
            config=self._config,
            use_rectangle_prior=True,
        )
        positive_refinement = apply_positive_refinement(
            base_point_ids=cluster_ids,
            roi_points=roi.points,
            seed_point_id=seed_point_id,
            positive_point_ids=positive_point_ids,
            scene_structure=scene_structure,
            config=self._config,
        )
        refinement = apply_negative_refinement(
            base_point_ids=positive_refinement.refined_point_ids,
            roi_points=roi.points,
            seed_point_id=seed_point_id,
            negative_point_ids=negative_point_ids,
            scene_structure=scene_structure,
            config=self._config,
        )
        refined_ids = refinement.refined_point_ids
        confidence = estimate_confidence(
            seed_point=seed_point,
            mask_points=[points_by_id[point_id] for point_id in refined_ids],
            seed_plane=seed_plane,
            plane_distance_tolerance=self._config.plane_distance_tolerance,
            wall_guidance=wall_guidance,
        )
        return SegmentResult(
            seed_point_id=seed_point_id,
            point_ids=refined_ids,
            point_count=len(refined_ids),
            confidence=confidence,
            method="heuristic_region_growing",
            positive_point_ids=positive_refinement.positive_point_ids,
            negative_point_ids=refinement.negative_point_ids,
            removed_point_ids=refinement.removed_point_ids,
        )


class PointSAMSegmenter:
    def __init__(self, runtime_config: PointSAMRuntimeConfig) -> None:
        self._runtime_config = runtime_config
        self._model = None
        self._torch = None
        self._loaded_modules = None

    @classmethod
    def try_create_from_env(cls) -> Tuple[Optional["PointSAMSegmenter"], str]:
        checkpoint_raw = os.getenv("POINT_SAM_CHECKPOINT", "").strip()
        if not checkpoint_raw:
            return None, "Point-SAM fallback: POINT_SAM_CHECKPOINT is not configured"

        checkpoint_path = Path(checkpoint_raw).expanduser()
        if not checkpoint_path.exists():
            return None, f"Point-SAM fallback: checkpoint not found at {checkpoint_path}"

        module_result = probe_point_sam_modules()
        if module_result[0] is None:
            return None, module_result[1]

        pc_sam_module = module_result[0]["pc_sam"]
        repo_dir = resolve_repo_dir(pc_sam_module)
        if repo_dir is None:
            return None, "Point-SAM fallback: unable to locate repo configs; set POINT_SAM_REPO_DIR"

        if not (repo_dir / "configs").exists():
            return None, f"Point-SAM fallback: configs directory missing in {repo_dir}"

        torch_module = module_result[0]["torch"]
        device = os.getenv("POINT_SAM_DEVICE", "cuda").strip() or "cuda"
        if device.startswith("cuda") and not torch_module.cuda.is_available():
            return None, "Point-SAM fallback: CUDA is unavailable in the current environment"

        runtime_config = PointSAMRuntimeConfig(
            checkpoint_path=checkpoint_path,
            repo_dir=repo_dir,
            config_name=os.getenv("POINT_SAM_CONFIG_NAME", "large").strip() or "large",
            device=device,
        )
        segmenter = cls(runtime_config)
        segmenter._loaded_modules = module_result[0]
        return segmenter, ""

    def segment(
        self,
        roi: ROIResult,
        seed_point_id: int,
        scene_structure: Optional[SceneStructure] = None,
        positive_point_ids: Optional[Sequence[int]] = None,
        negative_point_ids: Optional[Sequence[int]] = None,
    ) -> SegmentResult:
        self._ensure_model()
        points_by_id = {point.point_id: point for point in roi.points}
        try:
            seed_point = points_by_id[seed_point_id]
        except KeyError as exc:
            raise KeyError(f"Seed point {seed_point_id} not found in ROI") from exc

        torch = self._torch
        assert torch is not None

        xyz, rgb = build_roi_arrays(roi.points)
        shift = xyz.mean(axis=0, keepdims=True)
        centered = xyz - shift
        scale = max(1e-6, float((centered**2).sum(axis=1).max() ** 0.5))
        normalized_xyz = centered / scale
        normalized_rgb = rgb / 255.0

        point_index_by_id = {point_id: index for index, point_id in enumerate(roi.point_ids)}
        applied_positive_ids = [
            point_id
            for point_id in dedupe_point_ids(positive_point_ids or [])
            if point_id != seed_point_id and point_id in point_index_by_id
        ]
        prompt_point_ids = [seed_point_id, *applied_positive_ids]
        prompt_indices = [point_index_by_id[point_id] for point_id in prompt_point_ids]
        prompt_point = normalized_xyz[prompt_indices]

        xyz_tensor = torch.from_numpy(normalized_xyz).float().to(self._runtime_config.device).unsqueeze(0)
        rgb_tensor = torch.from_numpy(normalized_rgb).float().to(self._runtime_config.device).unsqueeze(0)
        prompt_points = torch.from_numpy(prompt_point).float().to(self._runtime_config.device).unsqueeze(0)
        prompt_labels = torch.tensor([[1] * len(prompt_point_ids)], device=self._runtime_config.device)

        with torch.no_grad():
            self._model.set_pointcloud(xyz_tensor, rgb_tensor)
            masks, scores, _ = self._model.predict_masks(prompt_points, prompt_labels, None, True)

        best_index = int(torch.argmax(scores[0]).item())
        best_mask = masks[0][best_index] > 0
        selected_ids = [
            point_id
            for point_id, keep in zip(roi.point_ids, best_mask.detach().cpu().tolist())
            if keep
        ]
        refinement = apply_negative_refinement(
            base_point_ids=selected_ids,
            roi_points=roi.points,
            seed_point_id=seed_point_id,
            negative_point_ids=negative_point_ids,
            scene_structure=scene_structure,
            config=SegmentConfig(),
        )
        raw_confidence = float(scores[0][best_index].detach().cpu().item())
        confidence = normalize_confidence(raw_confidence)
        return SegmentResult(
            seed_point_id=seed_point_id,
            point_ids=refinement.refined_point_ids,
            point_count=len(refinement.refined_point_ids),
            confidence=confidence,
            method="point_sam",
            positive_point_ids=applied_positive_ids,
            negative_point_ids=refinement.negative_point_ids,
            removed_point_ids=refinement.removed_point_ids,
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        loaded = self._loaded_modules
        if loaded is None:
            loaded, reason = probe_point_sam_modules()
            if loaded is None:
                raise RuntimeError(reason)

        torch = loaded["torch"]
        hydra = loaded["hydra"]
        omegaconf = loaded["omegaconf"]
        load_model = loaded["load_model"]
        point_cloud_sam = loaded["point_cloud_sam"]
        replace_with_fused_layernorm = loaded["replace_with_fused_layernorm"]

        config_dir = str((self._runtime_config.repo_dir / "configs").resolve())
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(config_name=self._runtime_config.config_name)
        omegaconf.OmegaConf.resolve(cfg)

        model = hydra.utils.instantiate(cfg.model)
        model.apply(replace_with_fused_layernorm)
        load_model(model, str(self._runtime_config.checkpoint_path))
        model.eval()
        model.to(self._runtime_config.device)

        if not isinstance(model, point_cloud_sam.PointCloudSAM):
            raise RuntimeError("Point-SAM fallback: unexpected model type")

        self._model = model
        self._torch = torch


def build_segmenter() -> Tuple[object, SegmenterStatus]:
    requested_backend = (os.getenv("POINT_SEGMENTER_BACKEND", "heuristic").strip() or "heuristic").lower()
    if requested_backend == "point_sam":
        segmenter, reason = PointSAMSegmenter.try_create_from_env()
        if segmenter is not None:
            return segmenter, SegmenterStatus(
                requested_backend="point_sam",
                active_backend="point_sam",
            )
        return HeuristicRegionSegmenter(), SegmenterStatus(
            requested_backend="point_sam",
            active_backend="heuristic_region_growing",
            fallback_reason=reason,
        )

    return HeuristicRegionSegmenter(), SegmenterStatus(
        requested_backend=requested_backend,
        active_backend="heuristic_region_growing",
    )


def segment_point_cluster(
    roi_points: Sequence[PointRecord],
    seed_point_id: int,
    scene_structure: Optional[SceneStructure],
    config: SegmentConfig,
    *,
    use_rectangle_prior: bool,
) -> Tuple[List[int], Optional[SeedPlane], Optional[WallGuidance]]:
    points_by_id = {point.point_id: point for point in roi_points}
    try:
        seed_point = points_by_id[seed_point_id]
    except KeyError as exc:
        raise KeyError(f"Seed point {seed_point_id} not found in ROI") from exc

    ordered_points = sorted(roi_points, key=lambda point: point.point_id)
    visited = set()
    queue = deque([seed_point.point_id])
    cluster_ids: List[int] = []
    wall_guidance = estimate_wall_guidance(
        seed_point=seed_point,
        roi_points=roi_points,
        scene_structure=scene_structure,
        config=config,
    )
    seed_plane = estimate_seed_plane(
        seed_point=seed_point,
        roi_points=roi_points,
        config=config,
    ) if wall_guidance is None else None

    if wall_guidance is not None:
        cluster_ids = segment_with_wall_projection(
            roi_points=roi_points,
            seed_point=seed_point,
            wall_guidance=wall_guidance,
            config=config,
            use_rectangle_prior=use_rectangle_prior,
        )
        return cluster_ids, seed_plane, wall_guidance

    while queue:
        current_id = queue.popleft()
        if current_id in visited:
            continue
        visited.add(current_id)
        current_point = points_by_id[current_id]
        cluster_ids.append(current_id)

        for candidate in ordered_points:
            if candidate.point_id in visited:
                continue
            if not is_neighbor_match(
                current_point=current_point,
                candidate_point=candidate,
                seed_point=seed_point,
                config=config,
                seed_plane=seed_plane,
                wall_guidance=wall_guidance,
            ):
                continue
            queue.append(candidate.point_id)

    cluster_ids.sort()
    return cluster_ids, seed_plane, wall_guidance


def apply_negative_refinement(
    base_point_ids: Sequence[int],
    roi_points: Sequence[PointRecord],
    seed_point_id: int,
    negative_point_ids: Optional[Sequence[int]],
    scene_structure: Optional[SceneStructure],
    config: SegmentConfig,
) -> NegativeRefinementResult:
    if not negative_point_ids:
        return NegativeRefinementResult(
            refined_point_ids=sorted(base_point_ids),
            negative_point_ids=[],
            removed_point_ids=[],
        )

    points_by_id = {point.point_id: point for point in roi_points}
    refined_ids = set(base_point_ids)
    removed_ids = set()
    applied_negative_ids: List[int] = []

    for negative_point_id in dedupe_point_ids(negative_point_ids):
        if negative_point_id == seed_point_id:
            continue
        negative_point = points_by_id.get(negative_point_id)
        if negative_point is None:
            continue

        candidate_ids = sorted(refined_ids | {negative_point_id})
        candidate_points = [points_by_id[point_id] for point_id in candidate_ids]
        removal_cluster = collect_negative_region(
            candidate_points=candidate_points,
            seed_point_id=negative_point_id,
            scene_structure=scene_structure,
            config=config,
        )
        removable_ids = set(removal_cluster) & refined_ids
        removable_ids.discard(seed_point_id)
        if not removable_ids:
            continue

        refined_ids -= removable_ids
        removed_ids.update(removable_ids)
        applied_negative_ids.append(negative_point_id)

    if seed_point_id not in refined_ids and seed_point_id in points_by_id:
        refined_ids.add(seed_point_id)
        removed_ids.discard(seed_point_id)

    return NegativeRefinementResult(
        refined_point_ids=sorted(refined_ids),
        negative_point_ids=applied_negative_ids,
        removed_point_ids=sorted(removed_ids),
    )


def apply_positive_refinement(
    base_point_ids: Sequence[int],
    roi_points: Sequence[PointRecord],
    seed_point_id: int,
    positive_point_ids: Optional[Sequence[int]],
    scene_structure: Optional[SceneStructure],
    config: SegmentConfig,
) -> PositiveRefinementResult:
    if not positive_point_ids:
        return PositiveRefinementResult(
            refined_point_ids=sorted(base_point_ids),
            positive_point_ids=[],
        )

    points_by_id = {point.point_id: point for point in roi_points}
    refined_ids = set(base_point_ids)
    applied_positive_ids: List[int] = []

    for positive_point_id in dedupe_point_ids(positive_point_ids):
        if positive_point_id == seed_point_id:
            continue
        if positive_point_id not in points_by_id:
            continue

        positive_cluster_ids, _, _ = segment_point_cluster(
            roi_points=roi_points,
            seed_point_id=positive_point_id,
            scene_structure=scene_structure,
            config=config,
            use_rectangle_prior=True,
        )
        refined_ids.update(positive_cluster_ids)
        applied_positive_ids.append(positive_point_id)

    refined_ids.add(seed_point_id)
    return PositiveRefinementResult(
        refined_point_ids=sorted(refined_ids),
        positive_point_ids=applied_positive_ids,
    )


def dedupe_point_ids(point_ids: Sequence[int]) -> List[int]:
    ordered_ids: List[int] = []
    seen = set()
    for point_id in point_ids:
        if point_id in seen:
            continue
        seen.add(point_id)
        ordered_ids.append(point_id)
    return ordered_ids


def collect_negative_region(
    candidate_points: Sequence[PointRecord],
    seed_point_id: int,
    scene_structure: Optional[SceneStructure],
    config: SegmentConfig,
) -> List[int]:
    points_by_id = {point.point_id: point for point in candidate_points}
    try:
        seed_point = points_by_id[seed_point_id]
    except KeyError as exc:
        raise KeyError(f"Seed point {seed_point_id} not found in ROI") from exc

    ordered_points = sorted(candidate_points, key=lambda point: point.point_id)
    wall_guidance = estimate_wall_guidance(
        seed_point=seed_point,
        roi_points=candidate_points,
        scene_structure=scene_structure,
        config=config,
    )
    seed_plane = estimate_seed_plane(
        seed_point=seed_point,
        roi_points=candidate_points,
        config=config,
    ) if wall_guidance is None else None

    visited = set()
    queue = deque([(seed_point_id, 0)])
    cluster_ids: List[int] = []

    while queue:
        current_id, hop_count = queue.popleft()
        if current_id in visited:
            continue
        visited.add(current_id)
        current_point = points_by_id[current_id]
        cluster_ids.append(current_id)
        if hop_count >= config.negative_max_hops:
            continue

        for candidate in ordered_points:
            if candidate.point_id in visited:
                continue
            if not is_neighbor_match(
                current_point=current_point,
                candidate_point=candidate,
                seed_point=seed_point,
                config=config,
                seed_plane=seed_plane,
                wall_guidance=wall_guidance,
            ):
                continue
            queue.append((candidate.point_id, hop_count + 1))

    cluster_ids.sort()
    return cluster_ids


def is_neighbor_match(
    current_point: PointRecord,
    candidate_point: PointRecord,
    seed_point: PointRecord,
    config: SegmentConfig,
    seed_plane: Optional[SeedPlane] = None,
    wall_guidance: Optional[WallGuidance] = None,
) -> bool:
    if color_distance(seed_point, candidate_point) > config.color_tolerance:
        return False
    if wall_guidance is not None:
        if wall_depth(candidate_point, wall_guidance) > wall_guidance.max_depth:
            return False
        return wall_local_distance(current_point, candidate_point, wall_guidance.axis) <= config.neighbor_radius
    if squared_distance(current_point.xyz, candidate_point.xyz) > config.neighbor_radius ** 2:
        return False
    if seed_plane is None or config.plane_distance_tolerance <= 0.0:
        return True
    return point_plane_distance(candidate_point, seed_plane) <= config.plane_distance_tolerance


def color_distance(a: PointRecord, b: PointRecord) -> float:
    if a.rgb is None or b.rgb is None:
        return 0.0
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a.rgb, b.rgb)))


def estimate_confidence(
    seed_point: PointRecord,
    mask_points: Sequence[PointRecord],
    seed_plane: Optional[SeedPlane] = None,
    plane_distance_tolerance: float = 0.0,
    wall_guidance: Optional[WallGuidance] = None,
) -> float:
    if not mask_points:
        return 0.0

    average_color_distance = sum(color_distance(seed_point, point) for point in mask_points) / len(mask_points)
    max_rgb_distance = math.sqrt(3 * (255 ** 2))
    color_similarity = max(0.0, 1.0 - average_color_distance / max_rgb_distance)
    size_factor = min(1.0, len(mask_points) / 4.0)
    plane_similarity = 0.5
    if seed_plane is not None and plane_distance_tolerance > 0.0:
        average_plane_distance = sum(point_plane_distance(point, seed_plane) for point in mask_points) / len(mask_points)
        scaled_tolerance = max(plane_distance_tolerance * 1.5, 1e-6)
        plane_similarity = max(0.0, 1.0 - average_plane_distance / scaled_tolerance)
    elif wall_guidance is not None:
        average_depth = sum(wall_depth(point, wall_guidance) for point in mask_points) / len(mask_points)
        max_depth = max(wall_guidance.max_depth, 1e-6)
        plane_similarity = max(0.0, 1.0 - average_depth / max_depth)

    confidence = 0.45 * color_similarity + 0.3 * size_factor + 0.25 * plane_similarity
    return round(max(0.2, min(0.99, confidence)), 3)


def estimate_wall_guidance(
    seed_point: PointRecord,
    roi_points: Sequence[PointRecord],
    scene_structure: Optional[SceneStructure],
    config: SegmentConfig,
) -> Optional[WallGuidance]:
    if scene_structure is None or not scene_structure.wall_planes:
        return None

    nearest_plane = None
    nearest_distance = float("inf")
    for plane in scene_structure.wall_planes:
        axis_index = axis_to_index(plane.axis)
        distance = abs(seed_point.xyz[axis_index] - plane.coordinate)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_plane = plane

    if nearest_plane is None or nearest_distance > config.wall_max_depth:
        return None

    supported_depths = [
        abs(point.xyz[axis_to_index(nearest_plane.axis)] - nearest_plane.coordinate)
        for point in roi_points
        if color_distance(seed_point, point) <= config.color_tolerance
    ]
    if len(supported_depths) < 3:
        return None

    max_depth = min(max(supported_depths) + config.wall_depth_margin, config.wall_max_depth)
    return WallGuidance(
        axis=nearest_plane.axis,
        side=nearest_plane.side,
        coordinate=nearest_plane.coordinate,
        seed_depth=nearest_distance,
        max_depth=max_depth,
    )


def segment_with_wall_projection(
    roi_points: Sequence[PointRecord],
    seed_point: PointRecord,
    wall_guidance: WallGuidance,
    config: SegmentConfig,
    use_rectangle_prior: bool = True,
) -> List[int]:
    candidate_points = [
        point
        for point in roi_points
        if color_distance(seed_point, point) <= config.color_tolerance
        and wall_depth(point, wall_guidance) <= wall_guidance.max_depth
    ]
    if not candidate_points:
        return [seed_point.point_id]

    cell_size = max(config.neighbor_radius, 1e-6)
    point_cells: Dict[int, Tuple[int, int]] = {}
    occupied_cells = set()

    for point in candidate_points:
        cell = project_point_to_wall_cell(point, seed_point, wall_guidance.axis, cell_size)
        point_cells[point.point_id] = cell
        occupied_cells.add(cell)

    seed_cell = point_cells.get(seed_point.point_id)
    if seed_cell is None:
        return [seed_point.point_id]

    component_cells = grow_cell_component(seed_cell, occupied_cells)
    rectangle_support = describe_projection_rectangle(seed_cell, occupied_cells)
    selected_ids = {
        point_id
        for point_id, cell in point_cells.items()
        if cell in component_cells
    }
    if use_rectangle_prior:
        rectangle_ids = apply_wall_rectangle_prior(
            seed_cell=seed_cell,
            rectangle_support=rectangle_support,
            occupied_cells=occupied_cells,
            point_cells=point_cells,
            config=config,
        )
        selected_ids.update(rectangle_ids)
        selected_ids = refine_window_projection_surface(
            selected_ids=selected_ids,
            point_lookup={point.point_id: point for point in candidate_points},
            seed_point=seed_point,
            wall_guidance=wall_guidance,
            rectangle_support=rectangle_support,
            config=config,
        )
    selected_ids.add(seed_point.point_id)
    return sorted(selected_ids)


def project_point_to_wall_cell(
    point: PointRecord,
    seed_point: PointRecord,
    wall_axis: str,
    cell_size: float,
) -> Tuple[int, int]:
    first_axis, second_axis = wall_projection_axes(wall_axis)
    first_value = (point.xyz[first_axis] - seed_point.xyz[first_axis]) / cell_size
    second_value = (point.xyz[second_axis] - seed_point.xyz[second_axis]) / cell_size
    return (int(round(first_value)), int(round(second_value)))


def wall_projection_axes(wall_axis: str) -> Tuple[int, int]:
    if wall_axis == "x":
        return (2, 1)
    if wall_axis == "z":
        return (0, 1)
    return (0, 2)


def grow_cell_component(
    seed_cell: Tuple[int, int],
    occupied_cells: Sequence[Tuple[int, int]],
) -> set[Tuple[int, int]]:
    occupied = set(occupied_cells)
    queue = deque([seed_cell])
    visited = set()

    while queue:
        current = queue.popleft()
        if current in visited or current not in occupied:
            continue
        visited.add(current)
        for delta_u in (-1, 0, 1):
            for delta_v in (-1, 0, 1):
                if delta_u == 0 and delta_v == 0:
                    continue
                queue.append((current[0] + delta_u, current[1] + delta_v))

    return visited


def apply_wall_rectangle_prior(
    seed_cell: Tuple[int, int],
    rectangle_support: Optional[ProjectionRectangleSupport],
    occupied_cells: Sequence[Tuple[int, int]],
    point_cells: Dict[int, Tuple[int, int]],
    config: SegmentConfig,
) -> set[int]:
    if len(point_cells) < config.wall_rectangle_min_points or rectangle_support is None:
        return set()
    if (
        rectangle_support.occupancy_ratio < config.wall_rectangle_occupancy_threshold
        and rectangle_support.border_coverage < config.wall_rectangle_border_coverage_threshold
    ):
        return set()

    reachable_cells = grow_projection_cell_graph(
        seed_cell=seed_cell,
        occupied_cells=occupied_cells,
        rectangle_support=rectangle_support,
        max_bridge_gap_cells=config.wall_rectangle_max_bridge_gap_cells,
    )
    if not reachable_cells:
        return set()

    return {
        point_id
        for point_id, cell in point_cells.items()
        if (
            cell in reachable_cells
        )
    }


def describe_projection_rectangle(
    seed_cell: Tuple[int, int],
    occupied_cells: Sequence[Tuple[int, int]],
) -> Optional[ProjectionRectangleSupport]:
    unique_cells = set(occupied_cells)
    if not unique_cells:
        return None

    first_values = [cell[0] for cell in unique_cells]
    second_values = [cell[1] for cell in unique_cells]
    min_first = min(first_values)
    max_first = max(first_values)
    min_second = min(second_values)
    max_second = max(second_values)
    if min_first == max_first or min_second == max_second:
        return None
    if not (min_first <= seed_cell[0] <= max_first and min_second <= seed_cell[1] <= max_second):
        return None

    bbox_area = (max_first - min_first + 1) * (max_second - min_second + 1)
    occupancy_ratio = len(unique_cells) / max(bbox_area, 1)
    border_cells = build_rectangle_border_cells(min_first, max_first, min_second, max_second)
    border_hits = sum(cell in unique_cells for cell in border_cells)
    border_coverage = border_hits / max(len(border_cells), 1)
    return ProjectionRectangleSupport(
        min_first=min_first,
        max_first=max_first,
        min_second=min_second,
        max_second=max_second,
        occupancy_ratio=occupancy_ratio,
        border_coverage=border_coverage,
    )


def build_rectangle_border_cells(
    min_first: int,
    max_first: int,
    min_second: int,
    max_second: int,
) -> set[Tuple[int, int]]:
    border_cells = set()
    for first in range(min_first, max_first + 1):
        border_cells.add((first, min_second))
        border_cells.add((first, max_second))
    for second in range(min_second, max_second + 1):
        border_cells.add((min_first, second))
        border_cells.add((max_first, second))
    return border_cells


def grow_projection_cell_graph(
    seed_cell: Tuple[int, int],
    occupied_cells: Sequence[Tuple[int, int]],
    rectangle_support: ProjectionRectangleSupport,
    max_bridge_gap_cells: int,
) -> set[Tuple[int, int]]:
    occupied_in_rectangle = {
        cell
        for cell in occupied_cells
        if is_cell_inside_rectangle(cell, rectangle_support)
    }
    if seed_cell not in occupied_in_rectangle:
        return set()

    max_gap = max(0, int(max_bridge_gap_cells))
    queue = deque([seed_cell])
    visited = set()
    reachable = set()

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if current not in occupied_in_rectangle:
            continue
        reachable.add(current)

        for neighbor in iter_projection_graph_neighbors(
            current=current,
            occupied_cells=occupied_in_rectangle,
            rectangle_support=rectangle_support,
            max_bridge_gap_cells=max_gap,
        ):
            if neighbor not in visited:
                queue.append(neighbor)

    return reachable


def iter_projection_graph_neighbors(
    *,
    current: Tuple[int, int],
    occupied_cells: set[Tuple[int, int]],
    rectangle_support: ProjectionRectangleSupport,
    max_bridge_gap_cells: int,
):
    max_step = max(1, max_bridge_gap_cells + 1)
    for delta_first in (-1, 0, 1):
        for delta_second in (-1, 0, 1):
            if delta_first == 0 and delta_second == 0:
                continue
            for step in range(1, max_step + 1):
                candidate = (
                    current[0] + delta_first * step,
                    current[1] + delta_second * step,
                )
                if not is_cell_inside_rectangle(candidate, rectangle_support):
                    break
                if candidate in occupied_cells:
                    yield candidate
                    break


def is_cell_inside_rectangle(
    cell: Tuple[int, int],
    rectangle_support: ProjectionRectangleSupport,
) -> bool:
    return (
        rectangle_support.min_first <= cell[0] <= rectangle_support.max_first
        and rectangle_support.min_second <= cell[1] <= rectangle_support.max_second
    )


def refine_window_projection_surface(
    selected_ids: set[int],
    point_lookup: Dict[int, PointRecord],
    seed_point: PointRecord,
    wall_guidance: WallGuidance,
    rectangle_support: Optional[ProjectionRectangleSupport],
    config: SegmentConfig,
) -> set[int]:
    if rectangle_support is None:
        return selected_ids
    if rectangle_support.border_coverage < config.wall_rectangle_border_coverage_threshold:
        return selected_ids
    if rectangle_support.width_cells < config.window_projection_min_span_cells:
        return selected_ids
    if rectangle_support.height_cells < config.window_projection_min_span_cells:
        return selected_ids

    axis_index = axis_to_index(wall_guidance.axis)
    seed_coordinate = seed_point.xyz[axis_index]
    support_points = [
        point_lookup[point_id]
        for point_id in selected_ids
        if point_id in point_lookup
        and abs(point_lookup[point_id].xyz[axis_index] - seed_coordinate) <= config.window_surface_seed_band
    ]
    if len(support_points) < config.window_surface_min_points:
        return selected_ids

    support_coordinates = [point.xyz[axis_index] for point in support_points]
    surface_coordinate = statistics.median(support_coordinates)
    deviations = sorted(abs(coordinate - surface_coordinate) for coordinate in support_coordinates)
    tolerance = max(
        config.plane_distance_tolerance,
        percentile_value(deviations, 0.75) + max(config.plane_distance_tolerance * 0.5, 0.02),
    )
    refined_ids = {
        point_id
        for point_id in selected_ids
        if point_id in point_lookup
        and abs(point_lookup[point_id].xyz[axis_index] - surface_coordinate) <= tolerance
    }
    if seed_point.point_id not in refined_ids or len(refined_ids) < config.window_surface_min_points:
        return selected_ids
    return refined_ids


def percentile_value(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(float(len(ordered) - 1), quantile * float(len(ordered) - 1)))
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def estimate_seed_plane(
    seed_point: PointRecord,
    roi_points: Sequence[PointRecord],
    config: SegmentConfig,
) -> Optional[SeedPlane]:
    candidate_neighbors = [
        (
            squared_distance(seed_point.xyz, point.xyz),
            point.point_id,
            point,
        )
        for point in roi_points
        if point.point_id != seed_point.point_id
    ]
    candidate_neighbors.sort(key=lambda item: (item[0], item[1]))
    support_points = [point for _, _, point in candidate_neighbors[: config.plane_neighbor_count]]
    if len(support_points) < 2:
        return None

    best_normal = None
    best_area = 0.0
    best_support_ids: List[int] = []
    best_inlier_count = -1
    best_average_distance = float("inf")

    for left_index in range(len(support_points) - 1):
        left_vector = subtract_vectors(support_points[left_index].xyz, seed_point.xyz)
        for right_index in range(left_index + 1, len(support_points)):
            right_vector = subtract_vectors(support_points[right_index].xyz, seed_point.xyz)
            normal = cross_product(left_vector, right_vector)
            area = vector_length(normal)
            if area < config.plane_min_area:
                continue
            normalized_normal = normalize_vector(normal)
            offset = -dot_product(normalized_normal, seed_point.xyz)
            distances = [
                abs(dot_product(normalized_normal, point.xyz) + offset)
                for point in support_points
            ]
            inlier_count = sum(distance <= config.plane_distance_tolerance for distance in distances)
            average_distance = sum(distances) / len(distances)
            candidate_key = (inlier_count, -average_distance, area)
            best_key = (best_inlier_count, -best_average_distance, best_area)
            if candidate_key <= best_key:
                continue
            best_inlier_count = inlier_count
            best_average_distance = average_distance
            best_area = area
            best_normal = normalized_normal
            best_support_ids = [
                seed_point.point_id,
                support_points[left_index].point_id,
                support_points[right_index].point_id,
            ]

    if best_normal is None or best_area < config.plane_min_area:
        return None

    return SeedPlane(
        normal=best_normal,
        offset=-dot_product(best_normal, seed_point.xyz),
        support_area=best_area,
        support_point_ids=best_support_ids,
    )


def point_plane_distance(point: PointRecord, plane: SeedPlane) -> float:
    return abs(dot_product(plane.normal, point.xyz) + plane.offset)


def wall_depth(point: PointRecord, guidance: WallGuidance) -> float:
    return abs(point.xyz[axis_to_index(guidance.axis)] - guidance.coordinate)


def wall_local_distance(a: PointRecord, b: PointRecord, wall_axis: str) -> float:
    dimensions = [0, 1, 2]
    dimensions.remove(axis_to_index(wall_axis))
    left = a.xyz
    right = b.xyz
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in dimensions))


def axis_to_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def subtract_vectors(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross_product(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vector_length(vector: Sequence[float]) -> float:
    return math.sqrt(dot_product(vector, vector))


def normalize_vector(vector: Sequence[float]) -> Tuple[float, float, float]:
    length = vector_length(vector)
    if length == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def probe_point_sam_modules() -> Tuple[Optional[Dict[str, object]], str]:
    try:
        torch = importlib.import_module("torch")
        hydra = importlib.import_module("hydra")
        omegaconf = importlib.import_module("omegaconf")
        safetensors_torch = importlib.import_module("safetensors.torch")
        point_cloud_sam = importlib.import_module("pc_sam.model.pc_sam")
        torch_utils = importlib.import_module("pc_sam.utils.torch_utils")
        pc_sam = importlib.import_module("pc_sam")
    except Exception as exc:  # pragma: no cover - exercised via failure path in tests
        return None, f"Point-SAM fallback: dependencies unavailable ({exc})"

    return (
        {
            "torch": torch,
            "hydra": hydra,
            "omegaconf": omegaconf,
            "load_model": safetensors_torch.load_model,
            "point_cloud_sam": point_cloud_sam,
            "replace_with_fused_layernorm": torch_utils.replace_with_fused_layernorm,
            "pc_sam": pc_sam,
        },
        "",
    )


def resolve_repo_dir(pc_sam_module) -> Optional[Path]:
    explicit = os.getenv("POINT_SAM_REPO_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()

    module_file = getattr(pc_sam_module, "__file__", None)
    if not module_file:
        return None

    package_root = Path(module_file).resolve().parent
    for candidate in (package_root.parent, package_root.parent.parent):
        if (candidate / "configs").exists():
            return candidate
    return None


def build_roi_arrays(points: Sequence[PointRecord]):
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on external environment
        raise RuntimeError(f"Point-SAM fallback: numpy is unavailable ({exc})") from exc

    xyz = np.array([point.xyz for point in points], dtype="float32")
    rgb = np.array(
        [point.rgb if point.rgb is not None else (210, 220, 235) for point in points],
        dtype="float32",
    )
    return xyz, rgb


def normalize_confidence(raw_confidence: float) -> float:
    if 0.0 <= raw_confidence <= 1.0:
        return round(raw_confidence, 3)
    squashed = 1.0 / (1.0 + math.exp(-raw_confidence))
    return round(squashed, 3)
