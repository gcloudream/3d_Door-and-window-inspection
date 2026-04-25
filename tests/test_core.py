import math
import unittest

from point_selection.core import PointCloud, PointRecord, ROIConfig, SelectionEngine


class SelectionEngineTests(unittest.TestCase):
    def test_box_query_returns_points_inside_axis_aligned_bounds(self) -> None:
        cloud = PointCloud(
            [
                PointRecord(point_id=1, xyz=(0.0, 0.0, 0.0)),
                PointRecord(point_id=2, xyz=(0.2, 0.1, 0.2)),
                PointRecord(point_id=3, xyz=(0.8, 0.0, 0.0)),
                PointRecord(point_id=4, xyz=(0.1, 0.6, 0.1)),
            ]
        )

        results = cloud.box_query(
            min_xyz=(-0.1, -0.1, -0.1),
            max_xyz=(0.3, 0.3, 0.3),
        )

        self.assertEqual([point.point_id for point in results], [1, 2])

    def test_pick_point_returns_nearest_point_to_ray(self) -> None:
        cloud = PointCloud(
            [
                PointRecord(point_id=1, xyz=(1.0, 0.02, 0.0)),
                PointRecord(point_id=2, xyz=(2.0, 0.01, 0.0)),
                PointRecord(point_id=3, xyz=(1.0, 0.4, 0.0)),
            ]
        )
        engine = SelectionEngine(cloud)

        result = engine.pick_point(
            ray_origin=(0.0, 0.0, 0.0),
            ray_direction=(1.0, 0.0, 0.0),
            max_distance_to_ray=0.05,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.point_id, 2)
        self.assertTrue(math.isclose(result.distance_to_ray, 0.01, rel_tol=1e-6))

    def test_pick_point_returns_none_when_no_candidate_matches_threshold(self) -> None:
        cloud = PointCloud(
            [
                PointRecord(point_id=1, xyz=(1.0, 0.4, 0.0)),
                PointRecord(point_id=2, xyz=(2.0, 0.5, 0.0)),
            ]
        )
        engine = SelectionEngine(cloud)

        result = engine.pick_point(
            ray_origin=(0.0, 0.0, 0.0),
            ray_direction=(1.0, 0.0, 0.0),
            max_distance_to_ray=0.05,
        )

        self.assertIsNone(result)

    def test_pick_point_can_prefer_frontmost_candidate_within_ray_tolerance(self) -> None:
        cloud = PointCloud(
            [
                PointRecord(point_id=1, xyz=(1.0, 0.04, 0.0)),
                PointRecord(point_id=2, xyz=(2.0, 0.01, 0.0)),
                PointRecord(point_id=3, xyz=(3.0, 0.3, 0.0)),
            ]
        )
        engine = SelectionEngine(cloud)

        result = engine.pick_point(
            ray_origin=(0.0, 0.0, 0.0),
            ray_direction=(1.0, 0.0, 0.0),
            max_distance_to_ray=0.05,
            prefer_frontmost=True,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.point_id, 1)
        self.assertTrue(math.isclose(result.projection_length, 1.0, rel_tol=1e-6))

    def test_build_roi_expands_radius_until_minimum_point_count_is_met(self) -> None:
        cloud = PointCloud(
            [
                PointRecord(point_id=1, xyz=(0.0, 0.0, 0.0)),
                PointRecord(point_id=2, xyz=(0.40, 0.0, 0.0)),
                PointRecord(point_id=3, xyz=(0.80, 0.0, 0.0)),
                PointRecord(point_id=4, xyz=(1.10, 0.0, 0.0)),
            ]
        )
        engine = SelectionEngine(cloud)

        result = engine.build_roi(
            center_point_id=1,
            config=ROIConfig(
                radius=0.50,
                max_points=10,
                min_points=4,
                max_radius=1.20,
                radius_step=0.30,
            ),
        )

        self.assertEqual(result.center_point_id, 1)
        self.assertEqual(result.point_ids, [1, 2, 3, 4])
        self.assertTrue(math.isclose(result.radius, 1.10, rel_tol=1e-6))
        self.assertEqual(result.expansions, 2)
        self.assertFalse(result.truncated)

    def test_build_roi_truncates_to_nearest_points_when_limit_is_exceeded(self) -> None:
        cloud = PointCloud(
            [
                PointRecord(point_id=1, xyz=(0.0, 0.0, 0.0)),
                PointRecord(point_id=2, xyz=(0.1, 0.0, 0.0)),
                PointRecord(point_id=3, xyz=(0.2, 0.0, 0.0)),
                PointRecord(point_id=4, xyz=(0.3, 0.0, 0.0)),
                PointRecord(point_id=5, xyz=(0.4, 0.0, 0.0)),
            ]
        )
        engine = SelectionEngine(cloud)

        result = engine.build_roi(
            center_point_id=1,
            config=ROIConfig(
                radius=0.50,
                max_points=3,
                min_points=1,
                max_radius=0.50,
                radius_step=0.10,
            ),
        )

        self.assertEqual(result.point_ids, [1, 2, 3])
        self.assertTrue(result.truncated)

    def test_build_wall_guided_roi_expands_along_wall_plane(self) -> None:
        cloud = PointCloud(
            [
                PointRecord(point_id=1, xyz=(2.0, 1.1, 0.75), rgb=(170, 120, 78)),
                PointRecord(point_id=2, xyz=(1.4, 1.1, 0.75), rgb=(170, 120, 78)),
                PointRecord(point_id=3, xyz=(2.6, 1.1, 0.75), rgb=(170, 120, 78)),
                PointRecord(point_id=4, xyz=(2.0, 1.7, 0.75), rgb=(170, 120, 78)),
                PointRecord(point_id=5, xyz=(1.5, 1.2, 0.35), rgb=(170, 120, 78)),
                PointRecord(point_id=6, xyz=(2.5, 1.2, 0.35), rgb=(170, 120, 78)),
                PointRecord(point_id=7, xyz=(2.0, 1.1, 1.6), rgb=(170, 120, 78)),
            ]
        )
        engine = SelectionEngine(cloud)

        result = engine.build_wall_guided_roi(
            center_point_id=1,
            config=ROIConfig(
                radius=0.45,
                max_points=20,
                min_points=3,
                radius_step=0.25,
            ),
            wall_axis="z",
            wall_coordinate=0.0,
            wall_depth_margin=0.8,
            color_tolerance=20.0,
        )

        self.assertGreater(result.radius, 0.45)
        self.assertEqual(sorted(result.point_ids), [1, 2, 3, 4, 5, 6])
        self.assertNotIn(7, result.point_ids)


if __name__ == "__main__":
    unittest.main()
