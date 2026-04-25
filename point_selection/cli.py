from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from point_selection.core import ROIConfig, SelectionEngine
from point_selection.io import load_point_cloud
from point_selection.view_adapter import CameraFrame, CameraIntrinsics, screen_click_to_ray


def parse_vector(raw_value: str) -> Tuple[float, float, float]:
    parts = [segment.strip() for segment in raw_value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Vector values must use x,y,z format")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Vector values must be numeric") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug point picking and ROI generation")
    parser.add_argument("--input", required=True, type=Path, help="Input scene JSON path")
    parser.add_argument("--ray-origin", type=parse_vector, help="Ray origin as x,y,z")
    parser.add_argument("--ray-direction", type=parse_vector, help="Ray direction as x,y,z")
    parser.add_argument("--screen-x", type=float, help="Screen click x coordinate in pixels")
    parser.add_argument("--screen-y", type=float, help="Screen click y coordinate in pixels")
    parser.add_argument("--camera-origin", type=parse_vector, help="Camera origin as x,y,z")
    parser.add_argument("--camera-target", type=parse_vector, help="Camera target as x,y,z")
    parser.add_argument("--camera-up", type=parse_vector, help="Camera up hint as x,y,z")
    parser.add_argument("--image-width", type=int, help="Viewport width in pixels")
    parser.add_argument("--image-height", type=int, help="Viewport height in pixels")
    parser.add_argument("--fx", type=float, help="Camera focal length fx")
    parser.add_argument("--fy", type=float, help="Camera focal length fy")
    parser.add_argument("--cx", type=float, help="Principal point cx")
    parser.add_argument("--cy", type=float, help="Principal point cy")
    parser.add_argument(
        "--max-distance-to-ray",
        type=float,
        default=None,
        help="Reject points farther than this perpendicular distance from the ray",
    )
    parser.add_argument("--roi-radius", required=True, type=float, help="Initial ROI radius in meters")
    parser.add_argument("--roi-max-points", required=True, type=int, help="Maximum points in ROI")
    parser.add_argument(
        "--roi-min-points",
        type=int,
        default=3000,
        help="Minimum points to reach before stopping radius expansion",
    )
    parser.add_argument(
        "--roi-max-radius",
        type=float,
        default=None,
        help="Maximum ROI radius in meters",
    )
    parser.add_argument(
        "--roi-radius-step",
        type=float,
        default=0.30,
        help="Radius increment in meters while expanding ROI",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path; stdout is used when omitted",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cloud = load_point_cloud(args.input)
    engine = SelectionEngine(cloud)
    ray_origin, ray_direction = resolve_ray(parser, args)
    pick = engine.pick_point(
        ray_origin=ray_origin,
        ray_direction=ray_direction,
        max_distance_to_ray=args.max_distance_to_ray,
    )
    if pick is None:
        parser.error("No point matched the ray and distance thresholds")

    roi = engine.build_roi(
        center_point_id=pick.point_id,
        config=ROIConfig(
            radius=args.roi_radius,
            max_points=args.roi_max_points,
            min_points=args.roi_min_points,
            max_radius=args.roi_max_radius,
            radius_step=args.roi_radius_step,
        ),
    )

    payload = {
        "pick": {
            "point_id": pick.point_id,
            "xyz": list(pick.xyz),
            "distance_to_ray": pick.distance_to_ray,
            "projection_length": pick.projection_length,
        },
        "roi": roi.to_debug_dict(),
    }
    serialized = json.dumps(payload, indent=2)

    if args.output is None:
        print(serialized)
    else:
        args.output.write_text(serialized)

    return 0


def resolve_ray(parser: argparse.ArgumentParser, args) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    has_explicit_ray = args.ray_origin is not None or args.ray_direction is not None
    has_screen_click = args.screen_x is not None or args.screen_y is not None

    if has_explicit_ray and has_screen_click:
        parser.error("Choose either explicit ray arguments or screen click arguments, not both")

    if has_explicit_ray:
        if args.ray_origin is None or args.ray_direction is None:
            parser.error("Both --ray-origin and --ray-direction are required for explicit ray mode")
        return args.ray_origin, args.ray_direction

    if has_screen_click:
        required = {
            "--screen-x": args.screen_x,
            "--screen-y": args.screen_y,
            "--camera-origin": args.camera_origin,
            "--camera-target": args.camera_target,
            "--camera-up": args.camera_up,
            "--image-width": args.image_width,
            "--image-height": args.image_height,
            "--fx": args.fx,
            "--fy": args.fy,
            "--cx": args.cx,
            "--cy": args.cy,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"Missing screen click camera arguments: {', '.join(missing)}")

        intrinsics = CameraIntrinsics(
            width=args.image_width,
            height=args.image_height,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
        )
        frame = CameraFrame.look_at(
            origin=args.camera_origin,
            target=args.camera_target,
            up_hint=args.camera_up,
        )
        return screen_click_to_ray(
            screen_x=args.screen_x,
            screen_y=args.screen_y,
            intrinsics=intrinsics,
            frame=frame,
        )

    parser.error("Provide either explicit ray arguments or screen click camera arguments")


if __name__ == "__main__":
    sys.exit(main())
