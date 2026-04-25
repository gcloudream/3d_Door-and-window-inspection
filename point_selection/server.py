from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.parse import unquote, urlparse

from point_selection.classifier import analyze_scene_structure, build_candidate_box, classify_mask_points, detect_wall_opening, find_nearest_wall
from point_selection.core import ROIConfig, PointPickResult, PointRecord, SelectionEngine, dot, normalize, squared_distance, subtract
from point_selection.io import DEFAULT_MAX_POINTS, load_point_cloud_from_upload, sample_point_cloud
from point_selection.segmenter import build_segmenter
from point_selection.view_adapter import CameraFrame, CameraIntrinsics, project_point_to_screen, screen_click_to_ray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
DEFAULT_SCENE = Path("/Users/gengchen/Desktop/3d/rgb109.pcd")
DEFAULT_ANALYSIS_MAX_POINTS = int(os.getenv("POINT_CLOUD_ANALYSIS_MAX_POINTS", "1200000"))


class DemoService:
    def __init__(self, scene_path: Path) -> None:
        self._scene_path = scene_path
        self._segmenter, self._segmenter_status = build_segmenter()
        self._scene_source_name = scene_path.name
        self._scene_source_bytes = scene_path.read_bytes()
        self._scene_max_points = DEFAULT_MAX_POINTS
        self._load_scene_source(scene_path.name, self._scene_source_bytes, max_points=self._scene_max_points)

    def get_scene_payload(self) -> Dict[str, object]:
        return {
            "scene_name": self._scene_name,
            "point_count": len(self._points_payload),
            "bounds": self._bounds,
            "points": self._points_payload,
            "sampling": {
                "max_points": self._scene_max_points,
                "display_max_points": self._scene_max_points,
                "analysis_max_points": self._scene_analysis_max_points,
                "display_point_count": len(self._points_payload),
                "analysis_point_count": len(self._cloud.points),
            },
            "segmentation": self._segmenter_status.to_dict(),
        }

    def get_scene_metadata(self) -> Dict[str, object]:
        return {
            "scene_name": self._scene_name,
            "point_count": len(self._points_payload),
            "bounds": self._bounds,
            "sampling": {
                "max_points": self._scene_max_points,
                "display_max_points": self._scene_max_points,
                "analysis_max_points": self._scene_analysis_max_points,
                "display_point_count": len(self._points_payload),
                "analysis_point_count": len(self._cloud.points),
            },
            "segmentation": self._segmenter_status.to_dict(),
        }

    def get_scene_points(self) -> Dict[str, object]:
        return {"points": self._points_payload}

    def pick_roi(self, payload: Dict[str, object]) -> Dict[str, object]:
        pick_result, roi_result = self._resolve_pick_and_roi(payload)
        roi_payload = roi_result.to_debug_dict()
        roi_payload["points"] = [serialize_point(point) for point in roi_result.points]
        return {
            "matched": True,
            "pick": serialize_pick(pick_result),
            "roi": roi_payload,
        }

    def preview_pick(self, payload: Dict[str, object]) -> Dict[str, object]:
        pick_result = self._resolve_pick(payload)
        return {
            "matched": True,
            "pick": serialize_pick(pick_result),
        }

    def segment_roi(self, payload: Dict[str, object]) -> Dict[str, object]:
        pick_result = self._resolve_pick(payload)
        seed_point_id = extract_refine_seed_point_id(payload) or pick_result.point_id
        positive_point_ids = extract_positive_point_ids(payload)
        negative_point_ids = extract_negative_point_ids(payload)
        roi_result = self._build_roi(
            payload,
            center_point_id=seed_point_id,
            required_point_ids=[*positive_point_ids, *negative_point_ids],
        )
        mask_result = self._segmenter.segment(
            roi_result,
            seed_point_id=seed_point_id,
            scene_structure=self._scene_structure,
            positive_point_ids=positive_point_ids,
            negative_point_ids=negative_point_ids,
        )
        mask_points = [self._cloud.get_point(point_id) for point_id in mask_result.point_ids]
        classification = classify_mask_points(mask_points, self._scene_structure)
        candidate_box = build_candidate_box(mask_points, classification, self._scene_structure)
        nearest_wall = find_nearest_wall(
            self._scene_structure,
            candidate_box.min_xyz if mask_points else pick_result.xyz,
            candidate_box.max_xyz if mask_points else pick_result.xyz,
        )
        opening_candidate = detect_wall_opening(mask_points, self._scene_structure, nearest_wall)
        roi_payload = roi_result.to_debug_dict()
        roi_payload["points"] = [serialize_point(point) for point in roi_result.points]
        mask_payload = mask_result.to_dict()
        mask_payload["points"] = [serialize_point(point) for point in mask_points]
        response = {
            "matched": True,
            "pick": serialize_pick(pick_result),
            "roi": roi_payload,
            "mask": mask_payload,
            "classification": classification.to_dict(),
            "candidate_box": candidate_box.to_dict(),
            "segmentation": self._segmenter_status.to_dict(),
        }
        response["opening_candidate"] = opening_candidate.to_dict() if opening_candidate is not None else None
        if positive_point_ids or negative_point_ids or seed_point_id != pick_result.point_id:
            response["refinement"] = {
                "applied": bool(mask_result.positive_point_ids or mask_result.negative_point_ids),
                "seed_point_id": mask_result.seed_point_id,
                "positive_point_ids": mask_result.positive_point_ids,
                "negative_point_ids": mask_result.negative_point_ids,
                "positive_points": serialize_point_ids(self._cloud, mask_result.positive_point_ids),
                "negative_points": serialize_point_ids(self._cloud, mask_result.negative_point_ids),
                "removed_point_ids": mask_result.removed_point_ids,
                "removed_point_count": len(mask_result.removed_point_ids),
            }
        return response

    def try_segment_roi(self, payload: Dict[str, object]) -> Dict[str, object]:
        try:
            return self.segment_roi(payload)
        except LookupError as exc:
            return {"matched": False, "message": str(exc)}

    def try_preview_pick(self, payload: Dict[str, object]) -> Dict[str, object]:
        try:
            return self.preview_pick(payload)
        except LookupError as exc:
            return {"matched": False, "message": str(exc)}

    def _resolve_pick(self, payload: Dict[str, object]) -> PointPickResult:
        screen_x = float(payload["screen_x"])
        screen_y = float(payload["screen_y"])
        camera = payload["camera"]
        pick_settings = payload.get("pick", {})

        intrinsics = CameraIntrinsics(
            width=int(camera["width"]),
            height=int(camera["height"]),
            fx=float(camera["fx"]),
            fy=float(camera["fy"]),
            cx=float(camera["cx"]),
            cy=float(camera["cy"]),
        )
        frame = CameraFrame.look_at(
            origin=tuple(camera["origin"]),
            target=tuple(camera["target"]),
            up_hint=tuple(camera["up"]),
        )
        ray_origin, ray_direction = screen_click_to_ray(
            screen_x=screen_x,
            screen_y=screen_y,
            intrinsics=intrinsics,
            frame=frame,
        )
        normalized_direction = normalize(ray_direction)

        max_distance_to_ray = float(pick_settings.get("max_distance_to_ray", 0.12))
        max_screen_distance_px = pick_settings.get("max_screen_distance_px")
        hinted_point_id = pick_settings.get("hinted_point_id")
        locked_point_id = pick_settings.get("locked_point_id")
        if locked_point_id is not None:
            pick_result = self._pick_from_hint(
                int(locked_point_id),
                ray_origin,
                normalized_direction,
                intrinsics=intrinsics,
                frame=frame,
                screen_x=screen_x,
                screen_y=screen_y,
                max_screen_distance_px=float(max_screen_distance_px) if max_screen_distance_px is not None else None,
            )
            if pick_result is not None:
                return pick_result
        if max_screen_distance_px is not None:
            pick_result = self._pick_with_screen_window(
                ray_origin=ray_origin,
                ray_direction=normalized_direction,
                intrinsics=intrinsics,
                frame=frame,
                screen_x=screen_x,
                screen_y=screen_y,
                max_distance_to_ray=max_distance_to_ray,
                max_screen_distance_px=float(max_screen_distance_px),
            )
        else:
            pick_result = self._engine.pick_point(
                ray_origin=ray_origin,
                ray_direction=normalized_direction,
                max_distance_to_ray=max_distance_to_ray,
                prefer_frontmost=True,
            )
        if pick_result is None and hinted_point_id is not None:
            pick_result = self._pick_from_hint(int(hinted_point_id), ray_origin, normalized_direction)
        if pick_result is None:
            raise LookupError("No point matched the current click and threshold")
        return pick_result

    def _resolve_pick_and_roi(self, payload: Dict[str, object]):
        pick_result = self._resolve_pick(payload)
        roi_result = self._build_roi(payload, center_point_id=pick_result.point_id)
        return pick_result, roi_result

    def _build_roi(
        self,
        payload: Dict[str, object],
        *,
        center_point_id: int,
        required_point_ids: Optional[Iterable[int]] = None,
    ):
        roi = payload["roi"]
        radius_step = float(roi.get("radius_step", 0.30))
        requested_radius = float(roi["radius"])
        max_radius = float(roi["max_radius"]) if roi.get("max_radius") is not None else None
        required_radius = estimate_required_roi_radius(
            cloud=self._cloud,
            center_point_id=center_point_id,
            point_ids=required_point_ids,
            radius_padding=max(radius_step, 0.15),
        )
        if required_radius is not None:
            requested_radius = max(requested_radius, required_radius)
            if max_radius is None or max_radius < requested_radius:
                max_radius = requested_radius

        roi_config = ROIConfig(
            radius=requested_radius,
            max_points=int(roi["max_points"]),
            min_points=int(roi.get("min_points", 3000)),
            max_radius=max_radius,
            radius_step=radius_step,
        )
        center_point = self._cloud.get_point(center_point_id)
        nearest_wall = find_nearest_wall_plane(self._scene_structure.wall_planes, center_point.xyz)
        if nearest_wall is not None:
            roi_result = self._engine.build_wall_guided_roi(
                center_point_id=center_point_id,
                config=roi_config,
                wall_axis=nearest_wall["axis"],
                wall_coordinate=nearest_wall["coordinate"],
                wall_depth_margin=float(roi.get("wall_depth_margin", 0.22)),
                color_tolerance=float(roi.get("color_tolerance", 40.0)),
            )
        else:
            roi_result = self._engine.build_roi(
                center_point_id=center_point_id,
                config=roi_config,
            )

        return roi_result

    def try_pick_roi(self, payload: Dict[str, object]) -> Dict[str, object]:
        try:
            return self.pick_roi(payload)
        except LookupError as exc:
            return {"matched": False, "message": str(exc)}

    def load_scene_from_upload(
        self,
        filename: str,
        content: str,
        max_points: Optional[int] = None,
    ) -> Dict[str, object]:
        self._load_scene_source(filename, content.encode("utf-8"), max_points=max_points)
        return self.get_scene_payload()

    def load_scene_from_upload_bytes(
        self,
        filename: str,
        content: bytes,
        max_points: Optional[int] = None,
    ) -> Dict[str, object]:
        self._load_scene_source(filename, content, max_points=max_points)
        return self.get_scene_payload()

    def reload_scene(self, max_points: Optional[int] = None) -> Dict[str, object]:
        if not self._scene_source_bytes:
            raise ValueError("Current scene cannot be reloaded because the original source is unavailable")

        resolved_max_points = DEFAULT_MAX_POINTS if max_points is None else int(max_points)
        self._scene_max_points = resolved_max_points
        self._display_cloud = sample_point_cloud(self._cloud, resolved_max_points)
        self._points_payload = [serialize_point(point) for point in self._display_cloud.points]
        return self.get_scene_metadata()

    def _set_scene(
        self,
        scene_name: str,
        cloud,
        *,
        source_bytes: Optional[bytes] = None,
        max_points: Optional[int] = None,
        analysis_max_points: Optional[int] = None,
    ) -> None:
        self._scene_name = scene_name
        self._cloud = cloud
        if source_bytes is not None:
            self._scene_source_name = scene_name
            self._scene_source_bytes = source_bytes
        if max_points is not None:
            self._scene_max_points = max_points
        if analysis_max_points is not None:
            self._scene_analysis_max_points = analysis_max_points
        self._engine = SelectionEngine(self._cloud)
        self._scene_structure = analyze_scene_structure(self._cloud.points)
        self._points_payload = [serialize_point(point) for point in self._display_cloud.points]
        self._bounds = compute_bounds(self._cloud.points)

    def _load_scene_source(self, scene_name: str, source_bytes: bytes, *, max_points: Optional[int] = None) -> None:
        resolved_max_points = DEFAULT_MAX_POINTS if max_points is None else int(max_points)
        resolved_analysis_max_points = max(resolved_max_points, DEFAULT_ANALYSIS_MAX_POINTS)
        analysis_cloud = load_point_cloud_from_upload(
            scene_name,
            content_bytes=source_bytes,
            max_points=resolved_analysis_max_points,
        )
        display_cloud = sample_point_cloud(analysis_cloud, resolved_max_points)
        self._display_cloud = display_cloud
        self._set_scene(
            scene_name,
            analysis_cloud,
            source_bytes=source_bytes,
            max_points=resolved_max_points,
            analysis_max_points=resolved_analysis_max_points,
        )

    def _pick_from_hint(
        self,
        point_id: int,
        ray_origin,
        ray_direction,
        *,
        intrinsics: Optional[CameraIntrinsics] = None,
        frame: Optional[CameraFrame] = None,
        screen_x: Optional[float] = None,
        screen_y: Optional[float] = None,
        max_screen_distance_px: Optional[float] = None,
    ) -> Optional[PointPickResult]:
        try:
            point = self._cloud.get_point(point_id)
        except KeyError:
            return None

        relative = subtract(point.xyz, ray_origin)
        projection = dot(relative, ray_direction)
        if projection < 0.0:
            return None

        if (
            max_screen_distance_px is not None
            and intrinsics is not None
            and frame is not None
            and screen_x is not None
            and screen_y is not None
        ):
            screen_projection = project_point_to_screen(point.xyz, intrinsics=intrinsics, frame=frame)
            if screen_projection is None:
                return None
            point_screen_x, point_screen_y, _ = screen_projection
            pixel_distance = ((point_screen_x - screen_x) ** 2 + (point_screen_y - screen_y) ** 2) ** 0.5
            if pixel_distance > max_screen_distance_px:
                return None

        perpendicular_sq = max(0.0, dot(relative, relative) - projection * projection)
        return PointPickResult(
            point_id=point.point_id,
            xyz=point.xyz,
            distance_to_ray=perpendicular_sq ** 0.5,
            projection_length=projection,
        )

    def _pick_with_screen_window(
        self,
        *,
        ray_origin,
        ray_direction,
        intrinsics: CameraIntrinsics,
        frame: CameraFrame,
        screen_x: float,
        screen_y: float,
        max_distance_to_ray: float,
        max_screen_distance_px: float,
    ) -> Optional[PointPickResult]:
        best_pick = None
        best_key = None

        for point in self._cloud.points:
            relative = subtract(point.xyz, ray_origin)
            projection = dot(relative, ray_direction)
            if projection < 0.0:
                continue

            perpendicular_sq = max(0.0, dot(relative, relative) - projection * projection)
            distance_to_ray = perpendicular_sq ** 0.5
            if distance_to_ray > max_distance_to_ray:
                continue

            screen_projection = project_point_to_screen(point.xyz, intrinsics=intrinsics, frame=frame)
            if screen_projection is None:
                continue
            point_screen_x, point_screen_y, _ = screen_projection
            pixel_distance = ((point_screen_x - screen_x) ** 2 + (point_screen_y - screen_y) ** 2) ** 0.5
            if pixel_distance > max_screen_distance_px:
                continue

            candidate = PointPickResult(
                point_id=point.point_id,
                xyz=point.xyz,
                distance_to_ray=distance_to_ray,
                projection_length=projection,
            )
            candidate_key = (
                pixel_distance,
                projection,
                distance_to_ray,
                point.point_id,
            )
            if best_key is None or candidate_key < best_key:
                best_pick = candidate
                best_key = candidate_key

        return best_pick


def make_handler(service: DemoService):
    class DemoRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/scene":
                self._send_json(HTTPStatus.OK, service.get_scene_payload())
                return
            if parsed.path == "/api/scene/meta":
                self._send_json(HTTPStatus.OK, service.get_scene_metadata())
                return
            if parsed.path == "/api/scene/points":
                self._send_json(HTTPStatus.OK, service.get_scene_points())
                return
            self._serve_static(parsed.path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/scene/load":
                self._handle_scene_load()
                return
            if parsed.path == "/api/scene/reload":
                self._handle_scene_reload()
                return
            if parsed.path not in ("/api/pick-roi", "/api/segment-roi", "/api/pick-preview"):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length) or b"{}")
                if parsed.path == "/api/segment-roi":
                    result = service.try_segment_roi(payload)
                elif parsed.path == "/api/pick-preview":
                    result = service.try_preview_pick(payload)
                else:
                    result = service.try_pick_roi(payload)
            except json.JSONDecodeError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body"})
                return
            except KeyError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Missing field: {exc.args[0]}"})
                return
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.OK, result)

        def _handle_scene_load(self) -> None:
            content_type = self.headers.get("Content-Type", "")
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length) if content_length > 0 else b""
                if "application/json" in content_type:
                    payload = json.loads(body or b"{}")
                    result = service.load_scene_from_upload(
                        filename=str(payload["filename"]),
                        content=str(payload["content"]),
                        max_points=parse_scene_max_points(payload.get("max_points")),
                    )
                else:
                    filename = unquote(self.headers.get("X-Scene-Filename", "")).strip()
                    if not filename:
                        raise KeyError("filename")
                    result = service.load_scene_from_upload_bytes(
                        filename=filename,
                        content=body,
                        max_points=parse_scene_max_points(self.headers.get("X-Scene-Max-Points")),
                    )
            except json.JSONDecodeError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body"})
                return
            except KeyError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Missing field: {exc.args[0]}"})
                return
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.OK, result)

        def _handle_scene_reload(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length) or b"{}")
                result = service.reload_scene(max_points=parse_scene_max_points(payload.get("max_points")))
            except json.JSONDecodeError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body"})
                return
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args) -> None:
            return

        def _serve_static(self, raw_path: str) -> None:
            request_path = "/" if raw_path in ("", "/") else raw_path
            file_path = resolve_static_path(request_path)
            if file_path is None or not file_path.exists() or not file_path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Static file not found"})
                return

            content_type, _ = mimetypes.guess_type(str(file_path))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.end_headers()
            self.wfile.write(file_path.read_bytes())

        def _send_json(self, status: HTTPStatus, payload: Dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DemoRequestHandler


def resolve_static_path(request_path: str) -> Optional[Path]:
    normalized = request_path.lstrip("/") or "index.html"
    candidate = (PROJECT_ROOT / normalized).resolve()
    if candidate.is_file() and PROJECT_ROOT in candidate.parents:
        return candidate

    fallback = (WEB_ROOT / "index.html").resolve()
    if request_path == "/" and fallback.exists():
        return fallback
    return None


def compute_bounds(points: Iterable[PointRecord]) -> Dict[str, object]:
    iterator = iter(points)
    first = next(iterator)
    min_x, min_y, min_z = first.xyz
    max_x, max_y, max_z = first.xyz

    for point in iterator:
        x, y, z = point.xyz
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)

    return {"min": [min_x, min_y, min_z], "max": [max_x, max_y, max_z]}


def extract_negative_point_ids(payload: Dict[str, object]) -> list[int]:
    return extract_refine_point_ids(payload, "negative_point_ids")


def extract_positive_point_ids(payload: Dict[str, object]) -> list[int]:
    return extract_refine_point_ids(payload, "positive_point_ids")


def extract_refine_seed_point_id(payload: Dict[str, object]) -> Optional[int]:
    refine = payload.get("refine")
    if not isinstance(refine, dict):
        return None

    raw_value = refine.get("seed_point_id")
    if raw_value is None:
        return None
    return int(raw_value)


def extract_refine_point_ids(payload: Dict[str, object], field_name: str) -> list[int]:
    refine = payload.get("refine")
    if not isinstance(refine, dict):
        return []

    raw_ids = refine.get(field_name, [])
    if raw_ids is None:
        return []
    if not isinstance(raw_ids, list):
        raise ValueError(f"refine.{field_name} must be a list")

    point_ids: list[int] = []
    seen = set()
    for value in raw_ids:
        point_id = int(value)
        if point_id in seen:
            continue
        seen.add(point_id)
        point_ids.append(point_id)
    return point_ids


def estimate_required_roi_radius(
    *,
    cloud,
    center_point_id: int,
    point_ids: Optional[Iterable[int]],
    radius_padding: float,
) -> Optional[float]:
    if not point_ids:
        return None

    center_point = cloud.get_point(center_point_id)
    required_radius = 0.0
    for point_id in point_ids:
        try:
            point = cloud.get_point(point_id)
        except KeyError:
            continue
        point_radius = squared_distance(center_point.xyz, point.xyz) ** 0.5 + radius_padding
        required_radius = max(required_radius, point_radius)
    if required_radius <= 0.0:
        return None
    return required_radius


def find_nearest_wall_plane(wall_planes, xyz):
    best_plane = None
    best_distance = None
    axis_index = {"x": 0, "y": 1, "z": 2}

    for plane in wall_planes:
        distance = abs(xyz[axis_index[plane.axis]] - plane.coordinate)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_plane = {
                "axis": plane.axis,
                "side": plane.side,
                "coordinate": plane.coordinate,
                "support_ratio": plane.support_ratio,
            }
            continue
        if best_distance is not None and abs(distance - best_distance) <= 1e-9:
            if plane.support_ratio > best_plane["support_ratio"]:
                best_plane = {
                    "axis": plane.axis,
                    "side": plane.side,
                    "coordinate": plane.coordinate,
                    "support_ratio": plane.support_ratio,
                }

    return best_plane


def parse_scene_max_points(raw_value) -> Optional[int]:
    if raw_value in (None, "", "null"):
        return None
    parsed = int(raw_value)
    if parsed < 1:
        raise ValueError("scene max_points must be at least 1")
    return parsed


def serialize_point(point: PointRecord) -> Dict[str, object]:
    return {
        "point_id": point.point_id,
        "xyz": list(point.xyz),
        "rgb": list(point.rgb) if point.rgb is not None else [210, 220, 235],
    }


def serialize_point_ids(cloud, point_ids: Iterable[int]) -> list[Dict[str, object]]:
    serialized_points: list[Dict[str, object]] = []
    for point_id in point_ids:
        try:
            point = cloud.get_point(point_id)
        except KeyError:
            continue
        serialized_points.append(serialize_point(point))
    return serialized_points


def serialize_pick(pick_result: PointPickResult) -> Dict[str, object]:
    return {
        "point_id": pick_result.point_id,
        "xyz": list(pick_result.xyz),
        "distance_to_ray": pick_result.distance_to_ray,
        "projection_length": pick_result.projection_length,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Three.js point selection demo")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="Path to the demo point cloud scene")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    service = DemoService(scene_path=args.scene.resolve())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"Demo available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
