import unittest

from point_selection.classifier import (
    BoundaryPlane,
    CandidateBox,
    CandidateClassification,
    SceneStructure,
    analyze_scene_structure,
    build_candidate_box,
    classify_mask_points,
)
from point_selection.core import PointRecord


def make_scene_structure() -> SceneStructure:
    return SceneStructure(
        min_xyz=(0.0, 0.0, 0.0),
        max_xyz=(4.0, 3.0, 2.8),
        floor_y=0.0,
        ceiling_y=3.0,
        wall_planes=[
            BoundaryPlane(axis="x", side="min", coordinate=0.0, support_ratio=0.2),
            BoundaryPlane(axis="x", side="max", coordinate=4.0, support_ratio=0.2),
            BoundaryPlane(axis="z", side="min", coordinate=0.0, support_ratio=0.3),
            BoundaryPlane(axis="z", side="max", coordinate=2.8, support_ratio=0.3),
        ],
    )


class MaskClassificationTests(unittest.TestCase):
    def test_analyze_scene_structure_prefers_supported_planes_over_outliers(self) -> None:
        room_points = [
            PointRecord(100 + index, (x, 0.0, z), (190, 140, 80))
            for index, (x, z) in enumerate(
                [
                    (0.0, 0.5),
                    (1.0, 0.5),
                    (2.0, 0.5),
                    (3.0, 0.5),
                    (0.0, 2.5),
                    (1.0, 2.5),
                    (2.0, 2.5),
                    (3.0, 2.5),
                ]
            )
        ]
        room_points.extend(
            [
                PointRecord(200, (0.0, 0.0, 0.0), (150, 160, 170)),
                PointRecord(201, (0.0, 1.0, 0.0), (150, 160, 170)),
                PointRecord(202, (0.0, 2.0, 0.0), (150, 160, 170)),
                PointRecord(203, (0.0, 3.0, 0.0), (150, 160, 170)),
                PointRecord(204, (3.0, 0.0, 3.0), (220, 220, 220)),
                PointRecord(205, (3.0, 1.0, 3.0), (220, 220, 220)),
                PointRecord(206, (3.0, 2.0, 3.0), (220, 220, 220)),
                PointRecord(207, (3.0, 3.0, 3.0), (220, 220, 220)),
                PointRecord(208, (-5.0, -2.0, 10.0), (255, 0, 0)),
            ]
        )

        structure = analyze_scene_structure(room_points)

        self.assertAlmostEqual(structure.floor_y, 0.0, places=3)
        wall_lookup = {(plane.axis, plane.side): plane.coordinate for plane in structure.wall_planes}
        self.assertAlmostEqual(wall_lookup[("x", "min")], 0.0, places=3)
        self.assertAlmostEqual(wall_lookup[("x", "max")], 3.0, places=3)
        self.assertAlmostEqual(wall_lookup[("z", "min")], 0.0, places=3)
        self.assertAlmostEqual(wall_lookup[("z", "max")], 3.0, places=3)

    def test_analyze_scene_structure_detects_real_sample_planes(self) -> None:
        points = [
            PointRecord(1, (0.0, 0.0, 0.7), (200, 120, 90)),
            PointRecord(2, (2.0, 0.0, 0.7), (200, 120, 90)),
            PointRecord(3, (4.0, 0.0, 0.7), (200, 120, 90)),
            PointRecord(4, (0.0, 0.0, 2.1), (200, 120, 90)),
            PointRecord(5, (2.0, 0.0, 2.1), (200, 120, 90)),
            PointRecord(6, (4.0, 0.0, 2.1), (200, 120, 90)),
            PointRecord(7, (0.0, 0.0, 0.0), (150, 160, 170)),
            PointRecord(8, (0.0, 1.5, 0.0), (150, 160, 170)),
            PointRecord(9, (0.0, 3.0, 0.0), (150, 160, 170)),
            PointRecord(10, (4.0, 0.0, 2.8), (220, 220, 220)),
            PointRecord(11, (4.0, 1.5, 2.8), (220, 220, 220)),
            PointRecord(12, (4.0, 3.0, 2.8), (220, 220, 220)),
        ]

        structure = analyze_scene_structure(points)

        self.assertAlmostEqual(structure.floor_y, 0.0, places=3)
        wall_lookup = {(plane.axis, plane.side): plane.coordinate for plane in structure.wall_planes}
        self.assertAlmostEqual(wall_lookup[("z", "min")], 0.0, places=3)
        self.assertAlmostEqual(wall_lookup[("z", "max")], 2.8, places=3)

    def test_classify_mask_identifies_door_like_region(self) -> None:
        result = classify_mask_points(
            points=[
                PointRecord(1, (1.0, 0.0, 0.0), (160, 110, 70)),
                PointRecord(2, (1.9, 0.0, 0.0), (160, 110, 70)),
                PointRecord(3, (1.0, 2.1, 0.0), (160, 110, 70)),
                PointRecord(4, (1.9, 2.1, 0.0), (160, 110, 70)),
                PointRecord(5, (1.0, 1.0, 0.08), (160, 110, 70)),
                PointRecord(6, (1.9, 1.0, 0.08), (160, 110, 70)),
            ],
            scene_structure=make_scene_structure(),
        )

        self.assertIsInstance(result, CandidateClassification)
        self.assertEqual(result.label, "door")
        self.assertTrue(result.attached_to_wall)
        self.assertGreaterEqual(result.confidence, 0.75)

        candidate_box = build_candidate_box(
            points=[
                PointRecord(1, (1.0, 0.0, 0.0), (160, 110, 70)),
                PointRecord(2, (1.9, 0.0, 0.0), (160, 110, 70)),
                PointRecord(3, (1.0, 2.1, 0.0), (160, 110, 70)),
                PointRecord(4, (1.9, 2.1, 0.0), (160, 110, 70)),
                PointRecord(5, (1.0, 1.0, 0.08), (160, 110, 70)),
                PointRecord(6, (1.9, 1.0, 0.08), (160, 110, 70)),
            ],
            classification=result,
            scene_structure=make_scene_structure(),
        )

        self.assertIsInstance(candidate_box, CandidateBox)
        self.assertEqual(candidate_box.label, "door")
        self.assertEqual(candidate_box.shape, "wall_rect")
        self.assertEqual(candidate_box.anchor_mode, "wall_plane")
        self.assertEqual(len(candidate_box.corners), 8)
        self.assertEqual(len(candidate_box.front_face), 5)
        self.assertAlmostEqual(candidate_box.min_xyz[1], 0.0, places=3)
        self.assertTrue(all(abs(corner[2] - 0.0) <= 1e-6 for corner in candidate_box.front_face[:-1]))
        self.assertGreater(candidate_box.size[1], 2.0)

    def test_classify_mask_identifies_window_like_region(self) -> None:
        result = classify_mask_points(
            points=[
                PointRecord(10, (0.0, 1.0, 1.0), (90, 150, 115)),
                PointRecord(11, (0.0, 1.0, 1.8), (90, 150, 115)),
                PointRecord(12, (0.0, 1.8, 1.0), (90, 150, 115)),
                PointRecord(13, (0.0, 1.8, 1.8), (90, 150, 115)),
                PointRecord(14, (0.12, 1.4, 1.2), (90, 150, 115)),
                PointRecord(15, (0.12, 1.4, 1.6), (90, 150, 115)),
            ],
            scene_structure=make_scene_structure(),
        )

        self.assertEqual(result.label, "window")
        self.assertTrue(result.attached_to_wall)
        self.assertGreaterEqual(result.confidence, 0.75)

        candidate_box = build_candidate_box(
            points=[
                PointRecord(10, (0.0, 1.0, 1.0), (90, 150, 115)),
                PointRecord(11, (0.0, 1.0, 1.8), (90, 150, 115)),
                PointRecord(12, (0.0, 1.8, 1.0), (90, 150, 115)),
                PointRecord(13, (0.0, 1.8, 1.8), (90, 150, 115)),
                PointRecord(14, (0.12, 1.4, 1.2), (90, 150, 115)),
                PointRecord(15, (0.12, 1.4, 1.6), (90, 150, 115)),
            ],
            classification=result,
            scene_structure=make_scene_structure(),
        )

        self.assertEqual(candidate_box.label, "window")
        self.assertEqual(candidate_box.shape, "wall_rect")
        self.assertEqual(candidate_box.anchor_mode, "wall_plane")
        self.assertGreater(candidate_box.min_xyz[1], 0.9)
        self.assertLess(candidate_box.max_xyz[1], 1.95)

    def test_build_candidate_box_keeps_offset_window_face_when_far_from_wall(self) -> None:
        points = [
            PointRecord(30, (0.5, 1.8, 0.45), (90, 150, 115)),
            PointRecord(31, (1.1, 1.8, 0.45), (90, 150, 115)),
            PointRecord(32, (0.5, 2.4, 0.45), (90, 150, 115)),
            PointRecord(33, (1.1, 2.4, 0.45), (90, 150, 115)),
            PointRecord(34, (0.5, 2.4, 0.9), (90, 150, 115)),
            PointRecord(35, (1.1, 2.4, 0.9), (90, 150, 115)),
            PointRecord(36, (0.5, 2.1, 0.7), (90, 150, 115)),
            PointRecord(37, (1.1, 2.1, 0.7), (90, 150, 115)),
        ]

        classification = classify_mask_points(points=points, scene_structure=make_scene_structure())
        candidate_box = build_candidate_box(
            points=points,
            classification=classification,
            scene_structure=make_scene_structure(),
        )

        self.assertEqual(classification.label, "window")
        self.assertEqual(candidate_box.shape, "wall_rect")
        self.assertEqual(candidate_box.anchor_mode, "mask_face")
        self.assertGreater(candidate_box.min_xyz[2], 0.3)
        self.assertLess(candidate_box.size[2], 0.6)

    def test_classify_mask_returns_unknown_for_floating_region(self) -> None:
        result = classify_mask_points(
            points=[
                PointRecord(20, (1.5, 1.1, 1.1), (120, 120, 120)),
                PointRecord(21, (2.0, 1.1, 1.1), (120, 120, 120)),
                PointRecord(22, (1.5, 1.6, 1.6), (120, 120, 120)),
                PointRecord(23, (2.0, 1.6, 1.6), (120, 120, 120)),
            ],
            scene_structure=make_scene_structure(),
        )

        self.assertEqual(result.label, "unknown")
        self.assertFalse(result.attached_to_wall)
        self.assertLessEqual(result.confidence, 0.65)


if __name__ == "__main__":
    unittest.main()
