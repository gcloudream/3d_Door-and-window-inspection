import unittest
from pathlib import Path
import struct

from point_selection.core import PointCloud, PointRecord
import point_selection.server as server_module
from point_selection.server import DemoService


class DemoServiceTests(unittest.TestCase):
    def test_get_scene_payload_returns_points_and_bounds(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))

        payload = service.get_scene_payload()

        self.assertEqual(payload["scene_name"], "simple_room.ply")
        self.assertEqual(payload["point_count"], 9)
        self.assertEqual(len(payload["points"]), 9)
        self.assertEqual(payload["bounds"]["min"], [0.0, 0.0, 0.0])
        self.assertEqual(payload["bounds"]["max"], [1.4, 0.92, 0.0])
        self.assertIn("segmentation", payload)
        self.assertEqual(payload["segmentation"]["active_backend"], "heuristic_region_growing")
        self.assertIn("sampling", payload)
        self.assertEqual(payload["sampling"]["max_points"], 300000)
        self.assertIn("display_max_points", payload["sampling"])
        self.assertIn("analysis_max_points", payload["sampling"])
        self.assertIn("display_point_count", payload["sampling"])
        self.assertIn("analysis_point_count", payload["sampling"])

    def test_pick_roi_payload_returns_pick_and_roi(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))

        payload = service.pick_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [0, 0, 0],
                    "target": [0, 0, 1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "roi": {"radius": 0.5, "min_points": 3, "max_points": 10},
            }
        )

        self.assertEqual(payload["pick"]["point_id"], 1)
        self.assertEqual(payload["roi"]["point_ids"], [1, 2, 3])
        self.assertEqual(payload["roi"]["radius"], 0.5)
        self.assertEqual(payload["roi"]["expansions"], 0)
        self.assertFalse(payload["roi"]["truncated"])

    def test_pick_roi_raises_when_click_hits_no_point(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))

        with self.assertRaises(LookupError):
            service.pick_roi(
                {
                    "screen_x": 0,
                    "screen_y": 0,
                    "camera": {
                        "origin": [0, 0, -2],
                        "target": [0, 0, -1],
                        "up": [0, 1, 0],
                        "width": 640,
                        "height": 480,
                        "fx": 400,
                        "fy": 400,
                        "cx": 320,
                        "cy": 240,
                    },
                    "pick": {"max_distance_to_ray": 0.000001},
                    "roi": {"radius": 0.5, "min_points": 3, "max_points": 10},
                }
            )

    def test_pick_roi_uses_hinted_point_id_for_sparse_clicks(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))

        payload = service.pick_roi(
            {
                "screen_x": 0,
                "screen_y": 0,
                "camera": {
                    "origin": [0, 0, -2],
                    "target": [0, 0, -1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.000001,
                    "hinted_point_id": 2,
                },
                "roi": {"radius": 0.25, "min_points": 1, "max_points": 10},
            }
        )

        self.assertEqual(payload["pick"]["point_id"], 2)

    def test_pick_roi_prefers_frontmost_ray_hit_before_using_hint(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))
        service._set_scene(
            "front_priority.json",
            PointCloud(
                [
                    PointRecord(1, (0.0, 0.04, 1.0), (200, 200, 200)),
                    PointRecord(2, (0.0, 0.01, 2.0), (180, 180, 180)),
                    PointRecord(3, (0.0, 0.3, 3.0), (120, 120, 120)),
                ]
            ),
        )

        payload = service.pick_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [0, 0, 0],
                    "target": [0, 0, 1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.05,
                    "hinted_point_id": 2,
                },
                "roi": {"radius": 0.2, "min_points": 1, "max_points": 10},
            }
        )

        self.assertEqual(payload["pick"]["point_id"], 1)

    def test_pick_roi_respects_screen_distance_when_front_point_is_far_from_cursor(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))
        service._set_scene(
            "screen_distance_priority.json",
            PointCloud(
                [
                    PointRecord(1, (0.28, 0.0, 1.0), (200, 200, 200)),
                    PointRecord(2, (0.01, 0.0, 2.0), (180, 180, 180)),
                    PointRecord(3, (0.0, 0.2, 2.8), (120, 120, 120)),
                ]
            ),
        )

        payload = service.pick_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [0, 0, 0],
                    "target": [0, 0, 1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.3,
                    "max_screen_distance_px": 28,
                },
                "roi": {"radius": 0.2, "min_points": 1, "max_points": 10},
            }
        )

        self.assertEqual(payload["pick"]["point_id"], 2)

    def test_pick_roi_prefers_pixel_nearest_point_within_screen_window(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))
        service._set_scene(
            "pixel_priority.json",
            PointCloud(
                [
                    PointRecord(1, (0.03, 0.0, 1.0), (200, 200, 200)),
                    PointRecord(2, (0.01, 0.0, 2.0), (180, 180, 180)),
                    PointRecord(3, (0.0, 0.2, 2.8), (120, 120, 120)),
                ]
            ),
        )

        payload = service.pick_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [0, 0, 0],
                    "target": [0, 0, 1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.05,
                    "max_screen_distance_px": 28,
                },
                "roi": {"radius": 0.2, "min_points": 1, "max_points": 10},
            }
        )

        self.assertEqual(payload["pick"]["point_id"], 2)

    def test_pick_roi_uses_locked_point_id_to_match_client_side_pick(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))
        service._set_scene(
            "locked_pick_priority.json",
            PointCloud(
                [
                    PointRecord(1, (0.0, 0.04, 1.0), (200, 200, 200)),
                    PointRecord(2, (0.0, 0.01, 2.0), (180, 180, 180)),
                ]
            ),
        )

        payload = service.pick_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [0, 0, 0],
                    "target": [0, 0, 1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.05,
                    "max_screen_distance_px": 28,
                    "locked_point_id": 1,
                },
                "roi": {"radius": 0.2, "min_points": 1, "max_points": 10},
            }
        )

        self.assertEqual(payload["pick"]["point_id"], 1)

    def test_pick_roi_ignores_locked_point_when_it_is_outside_screen_window(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))
        service._set_scene(
            "locked_pick_window_guard.json",
            PointCloud(
                [
                    PointRecord(1, (0.40, 0.0, 1.0), (200, 200, 200)),
                    PointRecord(2, (0.0, 0.0, 2.0), (180, 180, 180)),
                ]
            ),
        )

        payload = service.pick_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [0, 0, 0],
                    "target": [0, 0, 1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.5,
                    "max_screen_distance_px": 28,
                    "locked_point_id": 1,
                },
                "roi": {"radius": 0.2, "min_points": 1, "max_points": 10},
            }
        )

        self.assertEqual(payload["pick"]["point_id"], 2)

    def test_preview_pick_returns_same_result_as_click_flow(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))
        service._set_scene(
            "preview_front_priority.json",
            PointCloud(
                [
                    PointRecord(1, (0.0, 0.04, 1.0), (200, 200, 200)),
                    PointRecord(2, (0.0, 0.01, 2.0), (180, 180, 180)),
                    PointRecord(3, (0.0, 0.3, 3.0), (120, 120, 120)),
                ]
            ),
        )

        payload = service.preview_pick(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [0, 0, 0],
                    "target": [0, 0, 1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.05,
                    "max_screen_distance_px": 28,
                    "hinted_point_id": 2,
                },
            }
        )
        click_payload = service.pick_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [0, 0, 0],
                    "target": [0, 0, 1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.05,
                    "max_screen_distance_px": 28,
                    "hinted_point_id": 2,
                },
                "roi": {"radius": 0.2, "min_points": 1, "max_points": 10},
            }
        )

        self.assertTrue(payload["matched"])
        self.assertEqual(payload["pick"]["point_id"], click_payload["pick"]["point_id"])
        self.assertEqual(payload["pick"]["point_id"], 2)

    def test_preview_pick_prefers_pixel_nearest_point_within_screen_window(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))
        service._set_scene(
            "preview_pixel_priority.json",
            PointCloud(
                [
                    PointRecord(1, (0.03, 0.0, 1.0), (200, 200, 200)),
                    PointRecord(2, (0.01, 0.0, 2.0), (180, 180, 180)),
                    PointRecord(3, (0.0, 0.2, 2.8), (120, 120, 120)),
                ]
            ),
        )

        payload = service.preview_pick(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [0, 0, 0],
                    "target": [0, 0, 1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.05,
                    "max_screen_distance_px": 28,
                },
            }
        )

        self.assertTrue(payload["matched"])
        self.assertEqual(payload["pick"]["point_id"], 2)

    def test_try_preview_pick_returns_miss_payload(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))

        payload = service.try_preview_pick(
            {
                "screen_x": 0,
                "screen_y": 0,
                "camera": {
                    "origin": [0, 0, -2],
                    "target": [0, 0, -1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.000001,
                    "max_screen_distance_px": 28,
                },
            }
        )

        self.assertFalse(payload["matched"])
        self.assertIn("No point matched", payload["message"])

    def test_try_pick_roi_returns_miss_payload_instead_of_error(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))

        payload = service.try_pick_roi(
            {
                "screen_x": 0,
                "screen_y": 0,
                "camera": {
                    "origin": [0, 0, -2],
                    "target": [0, 0, -1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {"max_distance_to_ray": 0.000001},
                "roi": {"radius": 0.5, "min_points": 3, "max_points": 10},
            }
        )

        self.assertFalse(payload["matched"])
        self.assertIn("No point matched", payload["message"])

    def test_segment_roi_returns_pick_roi_and_mask(self) -> None:
        service = DemoService(scene_path=Path("sample_data/indoor_room_test.ply"))

        payload = service.segment_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [2.0, 1.5, -3.0],
                    "target": [2.0, 1.5, 0.0],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.000001,
                    "hinted_point_id": 110,
                },
                "roi": {"radius": 0.8, "min_points": 3, "max_points": 2000},
            }
        )

        self.assertTrue(payload["matched"])
        self.assertEqual(payload["pick"]["point_id"], 110)
        self.assertIn("mask", payload)
        self.assertGreaterEqual(payload["mask"]["point_count"], 1)
        self.assertIn(110, payload["mask"]["point_ids"])
        self.assertIn("classification", payload)
        self.assertIn(payload["classification"]["label"], {"door", "window", "unknown"})
        self.assertIn("candidate_box", payload)
        self.assertIn("min", payload["candidate_box"])
        self.assertIn("max", payload["candidate_box"])
        self.assertIn("size", payload["candidate_box"])
        self.assertIn("shape", payload["candidate_box"])
        self.assertIn("anchor_mode", payload["candidate_box"])
        self.assertIn("corners", payload["candidate_box"])
        self.assertIn("points", payload["roi"])
        self.assertEqual(len(payload["roi"]["points"]), len(payload["roi"]["point_ids"]))
        self.assertIn("points", payload["mask"])
        self.assertEqual(len(payload["mask"]["points"]), len(payload["mask"]["point_ids"]))

    def test_segment_roi_classifies_window_like_sample_region(self) -> None:
        service = DemoService(scene_path=Path("sample_data/indoor_room_test.ply"))

        payload = service.segment_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [2.0, 1.5, -3.0],
                    "target": [2.0, 1.5, 0.0],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.000001,
                    "hinted_point_id": 130,
                },
                "roi": {"radius": 0.8, "min_points": 3, "max_points": 2000},
            }
        )

        self.assertEqual(payload["pick"]["point_id"], 130)
        self.assertEqual(payload["classification"]["label"], "window")
        self.assertGreaterEqual(payload["classification"]["confidence"], 0.7)
        self.assertEqual(payload["candidate_box"]["label"], "window")
        self.assertGreater(payload["candidate_box"]["size"][1], 0.5)
        self.assertIn("opening_candidate", payload)
        self.assertIsNone(payload["opening_candidate"])

    def test_segment_roi_supports_negative_click_refinement(self) -> None:
        service = DemoService(scene_path=Path("sample_data/indoor_room_test.ply"))

        payload = service.segment_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [2.0, 1.5, -3.0],
                    "target": [2.0, 1.5, 0.0],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.000001,
                    "hinted_point_id": 110,
                },
                "roi": {"radius": 0.45, "min_points": 3, "max_points": 2000},
                "refine": {
                    "negative_point_ids": [118],
                },
            }
        )

        self.assertTrue(payload["matched"])
        self.assertIn("refinement", payload)
        self.assertTrue(payload["refinement"]["applied"])
        self.assertEqual(payload["refinement"]["negative_point_ids"], [118])
        self.assertGreater(payload["refinement"]["removed_point_count"], 0)
        self.assertIn(118, payload["refinement"]["removed_point_ids"])
        self.assertNotIn(118, payload["mask"]["point_ids"])
        self.assertIn("opening_candidate", payload)

    def test_segment_roi_supports_positive_click_refinement_with_seed_anchor(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))
        service._set_scene(
            "positive_refine.json",
            PointCloud(
                [
                    PointRecord(10, (0.0, 0.0, 1.0), (150, 180, 200)),
                    PointRecord(11, (0.2, 0.0, 1.0), (150, 180, 200)),
                    PointRecord(12, (1.0, 0.0, 1.0), (150, 180, 200)),
                    PointRecord(13, (1.2, 0.0, 1.0), (150, 180, 200)),
                ]
            ),
        )

        payload = service.segment_roi(
            {
                "screen_x": 0,
                "screen_y": 0,
                "camera": {
                    "origin": [0, 0, -2],
                    "target": [0, 0, -1],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.000001,
                    "hinted_point_id": 12,
                },
                "roi": {"radius": 1.5, "min_points": 1, "max_points": 20},
                "refine": {
                    "seed_point_id": 10,
                    "positive_point_ids": [12],
                },
            }
        )

        self.assertTrue(payload["matched"])
        self.assertEqual(payload["pick"]["point_id"], 12)
        self.assertEqual(payload["mask"]["seed_point_id"], 10)
        self.assertEqual(payload["mask"]["positive_point_ids"], [12])
        self.assertEqual(payload["mask"]["point_ids"], [10, 11, 12, 13])
        self.assertIn("refinement", payload)
        self.assertEqual(payload["refinement"]["positive_point_ids"], [12])
        self.assertIn("opening_candidate", payload)

    def test_pick_roi_uses_wall_guided_expansion_for_wall_attached_targets(self) -> None:
        service = DemoService(scene_path=Path("sample_data/indoor_room_test.ply"))

        payload = service.pick_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [2.0, 1.5, -3.0],
                    "target": [2.0, 1.5, 0.0],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.000001,
                    "hinted_point_id": 110,
                },
                "roi": {"radius": 0.45, "min_points": 3, "max_points": 2000},
            }
        )

        self.assertGreater(payload["roi"]["radius"], 0.45)
        self.assertGreater(len(payload["roi"]["point_ids"]), 3)

    def test_load_scene_from_upload_replaces_active_scene(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))

        payload = service.load_scene_from_upload(
            filename="uploaded.json",
            content=json_text(
                {
                    "points": [
                        {"point_id": 10, "xyz": [0.0, 0.0, 0.0], "rgb": [255, 0, 0]},
                        {"point_id": 11, "xyz": [0.0, 0.5, 0.0], "rgb": [0, 255, 0]},
                    ]
                }
            ),
        )

        self.assertEqual(payload["scene_name"], "uploaded.json")
        self.assertEqual(payload["point_count"], 2)
        self.assertEqual(payload["bounds"]["max"], [0.0, 0.5, 0.0])
        self.assertEqual(payload["points"][1]["point_id"], 11)

    def test_load_scene_from_upload_bytes_accepts_binary_pcd(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))

        payload = service.load_scene_from_upload_bytes(
            filename="uploaded.pcd",
            content=build_binary_pcd(
                rows=[
                    struct.pack("<Ifff", 0x00FF0000, 1.0, 2.0, 3.0),
                    struct.pack("<Ifff", 0x0000FF00, 4.0, 5.0, 6.0),
                ]
            ),
        )

        self.assertEqual(payload["scene_name"], "uploaded.pcd")
        self.assertEqual(payload["point_count"], 2)
        self.assertEqual(payload["bounds"]["min"], [1.0, 2.0, 3.0])
        self.assertEqual(payload["points"][0]["rgb"], [255, 0, 0])

    def test_reload_scene_resamples_current_source_with_new_density(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))

        upload_payload = service.load_scene_from_upload(
            filename="dense.json",
            content=json_text(
                {
                    "points": [
                        {"point_id": 1, "xyz": [0.0, 0.0, 0.0], "rgb": [255, 0, 0]},
                        {"point_id": 2, "xyz": [1.0, 0.0, 0.0], "rgb": [0, 255, 0]},
                        {"point_id": 3, "xyz": [2.0, 0.0, 0.0], "rgb": [0, 0, 255]},
                        {"point_id": 4, "xyz": [3.0, 0.0, 0.0], "rgb": [255, 255, 0]},
                        {"point_id": 5, "xyz": [4.0, 0.0, 0.0], "rgb": [255, 0, 255]},
                    ]
                }
            ),
            max_points=2,
        )

        self.assertEqual(upload_payload["point_count"], 2)
        self.assertEqual(upload_payload["sampling"]["max_points"], 2)

        reload_payload = service.reload_scene(max_points=4)

        self.assertEqual(reload_payload["scene_name"], "dense.json")
        self.assertEqual(reload_payload["point_count"], 3)
        self.assertEqual(reload_payload["sampling"]["max_points"], 4)
        self.assertEqual(reload_payload["sampling"]["display_point_count"], 3)
        self.assertEqual([point["point_id"] for point in service.get_scene_payload()["points"]], [1, 3, 5])

    def test_display_cloud_is_resampled_from_analysis_cloud_ids(self) -> None:
        original_analysis_max_points = server_module.DEFAULT_ANALYSIS_MAX_POINTS
        server_module.DEFAULT_ANALYSIS_MAX_POINTS = 3
        try:
            service = DemoService(scene_path=Path("sample_data/simple_room.ply"))

            payload = service.load_scene_from_upload_bytes(
                filename="dense.pcd",
                content=build_binary_pcd(
                    rows=[
                        struct.pack("<Ifff", 0x00FF0000, float(index), 0.0, 0.0)
                        for index in range(12)
                    ]
                ),
                max_points=2,
            )
        finally:
            server_module.DEFAULT_ANALYSIS_MAX_POINTS = original_analysis_max_points

        display_ids = [point["point_id"] for point in payload["points"]]
        analysis_ids = {point.point_id for point in service._cloud.points}

        self.assertEqual(payload["sampling"]["display_point_count"], 2)
        self.assertEqual(payload["sampling"]["analysis_point_count"], 3)
        self.assertEqual(display_ids, [1, 9])
        self.assertTrue(set(display_ids).issubset(analysis_ids))

    def test_segment_roi_returns_explicit_mask_points_even_when_point_is_not_in_display_payload(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))
        service.load_scene_from_upload(
            filename="sparse_display.json",
            content=json_text(
                {
                    "points": [
                        {"point_id": 1, "xyz": [0.0, 0.0, 0.0], "rgb": [255, 0, 0]},
                        {"point_id": 2, "xyz": [0.2, 0.0, 0.0], "rgb": [255, 0, 0]},
                        {"point_id": 3, "xyz": [0.4, 0.0, 0.0], "rgb": [255, 0, 0]},
                        {"point_id": 4, "xyz": [0.6, 0.0, 0.0], "rgb": [255, 0, 0]},
                    ]
                }
            ),
            max_points=2,
        )

        payload = service.segment_roi(
            {
                "screen_x": 320,
                "screen_y": 240,
                "camera": {
                    "origin": [0.2, 0, -1],
                    "target": [0.2, 0, 0],
                    "up": [0, 1, 0],
                    "width": 640,
                    "height": 480,
                    "fx": 400,
                    "fy": 400,
                    "cx": 320,
                    "cy": 240,
                },
                "pick": {
                    "max_distance_to_ray": 0.000001,
                    "hinted_point_id": 2,
                    "locked_point_id": 2,
                },
                "roi": {"radius": 1.0, "min_points": 1, "max_points": 10},
            }
        )

        display_ids = {point["point_id"] for point in service.get_scene_payload()["points"]}
        mask_ids = set(payload["mask"]["point_ids"])
        serialized_mask_ids = {point["point_id"] for point in payload["mask"]["points"]}

        self.assertTrue(mask_ids)
        self.assertEqual(mask_ids, serialized_mask_ids)
        self.assertTrue(mask_ids - display_ids)


def json_text(payload) -> str:
    import json

    return json.dumps(payload)


def build_binary_pcd(rows) -> bytes:
    header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS rgb x y z",
        "SIZE 4 4 4 4",
        "TYPE F F F F",
        "COUNT 1 1 1 1",
        f"WIDTH {len(rows)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(rows)}",
        "DATA binary",
    ]
    return ("\n".join(header_lines) + "\n").encode("ascii") + b"".join(rows)


if __name__ == "__main__":
    unittest.main()
