import unittest
from unittest import mock

from point_selection.classifier import BoundaryPlane, SceneStructure
from point_selection.core import PointRecord, ROIResult
from point_selection.segmenter import HeuristicRegionSegmenter, SegmentConfig, build_segmenter


def make_sparse_window_roi() -> ROIResult:
    point_ids = []
    points = []
    next_point_id = 100

    for x_index in range(8):
        for y_index in range(8):
            is_border = x_index in {0, 7} or y_index in {0, 7}
            if not is_border:
                continue
            x = round(x_index * 0.6, 3)
            y = round(y_index * 0.6, 3)
            point_ids.append(next_point_id)
            points.append(PointRecord(next_point_id, (x, y, 0.46), (90, 150, 118)))
            next_point_id += 1

    glass_points = [
        PointRecord(next_point_id, (1.2, 1.2, 0.46), (90, 150, 118)),
        PointRecord(next_point_id + 1, (2.4, 2.4, 0.46), (90, 150, 118)),
        PointRecord(next_point_id + 2, (3.0, 3.0, 0.46), (90, 150, 118)),
    ]
    leak_points = [
        PointRecord(next_point_id + 3, (1.2, 1.2, 0.05), (90, 150, 118)),
        PointRecord(next_point_id + 4, (2.4, 2.4, 0.05), (90, 150, 118)),
        PointRecord(next_point_id + 5, (3.0, 3.0, 0.05), (90, 150, 118)),
    ]
    points.extend(glass_points)
    points.extend(leak_points)
    point_ids.extend(point.point_id for point in glass_points)
    point_ids.extend(point.point_id for point in leak_points)

    return ROIResult(
        center_point_id=100,
        center_xyz=(0.0, 0.0, 0.46),
        radius=4.5,
        point_ids=point_ids,
        points=points,
        expansions=0,
        truncated=False,
    )


class HeuristicRegionSegmenterTests(unittest.TestCase):
    def test_segment_returns_seed_connected_cluster(self) -> None:
        roi = ROIResult(
            center_point_id=1,
            center_xyz=(0.0, 0.0, 0.0),
            radius=1.0,
            point_ids=[1, 2, 3, 4],
            points=[
                PointRecord(1, (0.0, 0.0, 0.0), (200, 120, 90)),
                PointRecord(2, (0.2, 0.0, 0.0), (198, 118, 92)),
                PointRecord(3, (0.4, 0.0, 0.0), (201, 121, 88)),
                PointRecord(4, (0.9, 0.9, 0.0), (120, 160, 210)),
            ],
            expansions=0,
            truncated=False,
        )

        segmenter = HeuristicRegionSegmenter()

        result = segmenter.segment(roi, seed_point_id=1)

        self.assertEqual(result.point_ids, [1, 2, 3])
        self.assertEqual(result.point_count, 3)
        self.assertEqual(result.seed_point_id, 1)
        self.assertEqual(result.method, "heuristic_region_growing")

    def test_segment_excludes_same_color_points_outside_seed_plane(self) -> None:
        roi = ROIResult(
            center_point_id=10,
            center_xyz=(0.0, 0.0, 0.0),
            radius=1.0,
            point_ids=[10, 11, 12, 13, 14],
            points=[
                PointRecord(10, (0.0, 0.0, 0.0), (180, 120, 80)),
                PointRecord(11, (0.25, 0.0, 0.0), (182, 118, 82)),
                PointRecord(12, (0.0, 0.28, 0.0), (179, 122, 79)),
                PointRecord(13, (0.25, 0.28, 0.0), (181, 121, 81)),
                PointRecord(14, (0.22, 0.05, 0.24), (181, 119, 80)),
            ],
            expansions=0,
            truncated=False,
        )

        segmenter = HeuristicRegionSegmenter(
            SegmentConfig(
                neighbor_radius=0.5,
                color_tolerance=20.0,
                plane_distance_tolerance=0.06,
            )
        )

        result = segmenter.segment(roi, seed_point_id=10)

        self.assertEqual(result.point_ids, [10, 11, 12, 13])
        self.assertNotIn(14, result.point_ids)

    def test_segment_falls_back_when_plane_support_is_insufficient(self) -> None:
        roi = ROIResult(
            center_point_id=20,
            center_xyz=(0.0, 0.0, 0.0),
            radius=1.0,
            point_ids=[20, 21, 22],
            points=[
                PointRecord(20, (0.0, 0.0, 0.0), (210, 150, 100)),
                PointRecord(21, (0.18, 0.0, 0.0), (212, 148, 101)),
                PointRecord(22, (0.36, 0.0, 0.0), (209, 151, 99)),
            ],
            expansions=0,
            truncated=False,
        )

        segmenter = HeuristicRegionSegmenter(
            SegmentConfig(
                neighbor_radius=0.3,
                color_tolerance=20.0,
                plane_distance_tolerance=0.05,
            )
        )

        result = segmenter.segment(roi, seed_point_id=20)

        self.assertEqual(result.point_ids, [20, 21, 22])

    def test_segment_uses_wall_guidance_to_extend_along_wall_surface(self) -> None:
        roi = ROIResult(
            center_point_id=30,
            center_xyz=(2.0, 1.1, 0.75),
            radius=1.0,
            point_ids=[30, 31, 32, 33, 34, 35, 36],
            points=[
                PointRecord(30, (2.0, 1.1, 0.75), (170, 120, 78)),
                PointRecord(31, (1.4, 1.1, 0.75), (170, 120, 78)),
                PointRecord(32, (2.6, 1.1, 0.75), (170, 120, 78)),
                PointRecord(33, (2.0, 1.7, 0.75), (170, 120, 78)),
                PointRecord(34, (1.5, 1.2, 0.35), (170, 120, 78)),
                PointRecord(35, (2.5, 1.2, 0.35), (170, 120, 78)),
                PointRecord(36, (2.0, 1.1, 0.75), (120, 120, 120)),
            ],
            expansions=0,
            truncated=False,
        )
        scene_structure = SceneStructure(
            min_xyz=(0.0, 0.0, 0.0),
            max_xyz=(4.0, 3.0, 2.8),
            floor_y=0.0,
            ceiling_y=3.0,
            wall_planes=[
                BoundaryPlane(axis="z", side="min", coordinate=0.0, support_ratio=0.25),
                BoundaryPlane(axis="z", side="max", coordinate=2.8, support_ratio=0.25),
            ],
        )
        segmenter = HeuristicRegionSegmenter(
            SegmentConfig(
                neighbor_radius=0.65,
                color_tolerance=20.0,
                plane_distance_tolerance=0.06,
                wall_depth_margin=0.18,
            )
        )

        result = segmenter.segment(roi, seed_point_id=30, scene_structure=scene_structure)

        self.assertEqual(result.point_ids, [30, 31, 32, 33, 34, 35])
        self.assertNotIn(36, result.point_ids)

    def test_segment_uses_wall_rectangle_prior_to_bridge_sparse_columns(self) -> None:
        roi = ROIResult(
            center_point_id=40,
            center_xyz=(0.0, 0.0, 0.75),
            radius=1.5,
            point_ids=[40, 41, 42, 43, 44, 45],
            points=[
                PointRecord(40, (0.0, 0.0, 0.75), (90, 150, 118)),
                PointRecord(41, (0.0, 0.6, 0.75), (90, 150, 118)),
                PointRecord(42, (0.0, 1.2, 0.75), (90, 150, 118)),
                PointRecord(43, (1.2, 0.0, 0.75), (90, 150, 118)),
                PointRecord(44, (1.2, 0.6, 0.75), (90, 150, 118)),
                PointRecord(45, (1.2, 1.2, 0.75), (90, 150, 118)),
            ],
            expansions=0,
            truncated=False,
        )
        scene_structure = SceneStructure(
            min_xyz=(0.0, 0.0, 0.0),
            max_xyz=(4.0, 3.0, 2.8),
            floor_y=0.0,
            ceiling_y=3.0,
            wall_planes=[
                BoundaryPlane(axis="z", side="min", coordinate=0.0, support_ratio=0.25),
            ],
        )
        segmenter = HeuristicRegionSegmenter(
            SegmentConfig(
                neighbor_radius=0.65,
                color_tolerance=20.0,
                plane_distance_tolerance=0.06,
                wall_depth_margin=0.18,
            )
        )

        result = segmenter.segment(roi, seed_point_id=40, scene_structure=scene_structure)

        self.assertEqual(result.point_ids, [40, 41, 42, 43, 44, 45])

    def test_segment_negative_point_removes_local_overreach_cluster(self) -> None:
        roi = ROIResult(
            center_point_id=40,
            center_xyz=(0.0, 0.0, 0.75),
            radius=1.5,
            point_ids=[40, 41, 42, 43, 44, 45],
            points=[
                PointRecord(40, (0.0, 0.0, 0.75), (90, 150, 118)),
                PointRecord(41, (0.0, 0.6, 0.75), (90, 150, 118)),
                PointRecord(42, (0.0, 1.2, 0.75), (90, 150, 118)),
                PointRecord(43, (1.2, 0.0, 0.75), (90, 150, 118)),
                PointRecord(44, (1.2, 0.6, 0.75), (90, 150, 118)),
                PointRecord(45, (1.2, 1.2, 0.75), (90, 150, 118)),
            ],
            expansions=0,
            truncated=False,
        )
        scene_structure = SceneStructure(
            min_xyz=(0.0, 0.0, 0.0),
            max_xyz=(4.0, 3.0, 2.8),
            floor_y=0.0,
            ceiling_y=3.0,
            wall_planes=[
                BoundaryPlane(axis="z", side="min", coordinate=0.0, support_ratio=0.25),
            ],
        )
        segmenter = HeuristicRegionSegmenter(
            SegmentConfig(
                neighbor_radius=0.65,
                color_tolerance=20.0,
                plane_distance_tolerance=0.06,
                wall_depth_margin=0.18,
            )
        )

        result = segmenter.segment(
            roi,
            seed_point_id=40,
            scene_structure=scene_structure,
            negative_point_ids=[43],
        )

        self.assertEqual(result.point_ids, [40, 41, 42])
        self.assertEqual(result.negative_point_ids, [43])
        self.assertEqual(result.removed_point_ids, [43, 44, 45])

    def test_segment_positive_point_unions_disconnected_same_surface_cluster(self) -> None:
        roi = ROIResult(
            center_point_id=60,
            center_xyz=(0.0, 0.0, 0.0),
            radius=1.5,
            point_ids=[60, 61, 62, 63],
            points=[
                PointRecord(60, (0.0, 0.0, 0.0), (150, 180, 200)),
                PointRecord(61, (0.2, 0.0, 0.0), (150, 180, 200)),
                PointRecord(62, (1.0, 0.0, 0.0), (150, 180, 200)),
                PointRecord(63, (1.2, 0.0, 0.0), (150, 180, 200)),
            ],
            expansions=0,
            truncated=False,
        )
        segmenter = HeuristicRegionSegmenter(
            SegmentConfig(
                neighbor_radius=0.35,
                color_tolerance=20.0,
                plane_distance_tolerance=0.05,
            )
        )

        base_result = segmenter.segment(roi, seed_point_id=60)
        refined_result = segmenter.segment(
            roi,
            seed_point_id=60,
            positive_point_ids=[62],
        )

        self.assertEqual(base_result.point_ids, [60, 61])
        self.assertEqual(refined_result.point_ids, [60, 61, 62, 63])
        self.assertEqual(refined_result.positive_point_ids, [62])

    def test_segment_uses_window_frame_border_support_to_fill_sparse_window_projection(self) -> None:
        roi = make_sparse_window_roi()
        scene_structure = SceneStructure(
            min_xyz=(0.0, 0.0, 0.0),
            max_xyz=(4.8, 4.8, 2.8),
            floor_y=0.0,
            ceiling_y=3.0,
            wall_planes=[
                BoundaryPlane(axis="z", side="min", coordinate=0.0, support_ratio=0.25),
            ],
        )
        segmenter = HeuristicRegionSegmenter(
            SegmentConfig(
                neighbor_radius=0.6,
                color_tolerance=20.0,
                plane_distance_tolerance=0.06,
                wall_depth_margin=0.18,
            )
        )

        result = segmenter.segment(roi, seed_point_id=100, scene_structure=scene_structure)

        self.assertIn(128, result.point_ids)
        self.assertIn(129, result.point_ids)
        self.assertIn(130, result.point_ids)

    def test_segment_wall_projection_avoids_bridging_remote_same_color_patch_across_large_gap(self) -> None:
        roi = ROIResult(
            center_point_id=70,
            center_xyz=(0.0, 0.0, 0.75),
            radius=3.5,
            point_ids=[70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81],
            points=[
                PointRecord(70, (0.0, 0.0, 0.75), (90, 150, 118)),
                PointRecord(71, (0.0, 0.6, 0.75), (90, 150, 118)),
                PointRecord(72, (0.0, 1.2, 0.75), (90, 150, 118)),
                PointRecord(73, (0.6, 0.0, 0.75), (90, 150, 118)),
                PointRecord(74, (0.6, 0.6, 0.75), (90, 150, 118)),
                PointRecord(75, (0.6, 1.2, 0.75), (90, 150, 118)),
                PointRecord(76, (2.4, 0.0, 0.75), (90, 150, 118)),
                PointRecord(77, (2.4, 0.6, 0.75), (90, 150, 118)),
                PointRecord(78, (2.4, 1.2, 0.75), (90, 150, 118)),
                PointRecord(79, (3.0, 0.0, 0.75), (90, 150, 118)),
                PointRecord(80, (3.0, 0.6, 0.75), (90, 150, 118)),
                PointRecord(81, (3.0, 1.2, 0.75), (90, 150, 118)),
            ],
            expansions=0,
            truncated=False,
        )
        scene_structure = SceneStructure(
            min_xyz=(0.0, 0.0, 0.0),
            max_xyz=(4.0, 3.0, 2.8),
            floor_y=0.0,
            ceiling_y=3.0,
            wall_planes=[
                BoundaryPlane(axis="z", side="min", coordinate=0.0, support_ratio=0.25),
            ],
        )
        segmenter = HeuristicRegionSegmenter(
            SegmentConfig(
                neighbor_radius=0.65,
                color_tolerance=20.0,
                plane_distance_tolerance=0.06,
                wall_depth_margin=0.18,
            )
        )

        result = segmenter.segment(roi, seed_point_id=70, scene_structure=scene_structure)

        self.assertEqual(result.point_ids, [70, 71, 72, 73, 74, 75])
        self.assertNotIn(76, result.point_ids)
        self.assertNotIn(77, result.point_ids)
        self.assertNotIn(78, result.point_ids)
        self.assertNotIn(79, result.point_ids)
        self.assertNotIn(80, result.point_ids)
        self.assertNotIn(81, result.point_ids)

    def test_segment_window_projection_prefers_seed_surface_over_background_wall_points(self) -> None:
        roi = make_sparse_window_roi()
        scene_structure = SceneStructure(
            min_xyz=(0.0, 0.0, 0.0),
            max_xyz=(4.8, 4.8, 2.8),
            floor_y=0.0,
            ceiling_y=3.0,
            wall_planes=[
                BoundaryPlane(axis="z", side="min", coordinate=0.0, support_ratio=0.25),
            ],
        )
        segmenter = HeuristicRegionSegmenter(
            SegmentConfig(
                neighbor_radius=0.6,
                color_tolerance=20.0,
                plane_distance_tolerance=0.06,
                wall_depth_margin=0.18,
            )
        )

        result = segmenter.segment(roi, seed_point_id=100, scene_structure=scene_structure)

        self.assertNotIn(131, result.point_ids)
        self.assertNotIn(132, result.point_ids)
        self.assertNotIn(133, result.point_ids)

    def test_build_segmenter_falls_back_when_point_sam_is_unavailable(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "POINT_SEGMENTER_BACKEND": "point_sam",
                "POINT_SAM_CHECKPOINT": "/tmp/missing-model.safetensors",
            },
            clear=False,
        ):
            segmenter, status = build_segmenter()

        self.assertIsInstance(segmenter, HeuristicRegionSegmenter)
        self.assertEqual(status.requested_backend, "point_sam")
        self.assertEqual(status.active_backend, "heuristic_region_growing")
        self.assertIn("Point-SAM", status.fallback_reason)


if __name__ == "__main__":
    unittest.main()
