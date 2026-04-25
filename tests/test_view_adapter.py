import math
import unittest

from point_selection.view_adapter import CameraFrame, CameraIntrinsics, project_point_to_screen, screen_click_to_ray


class ViewAdapterTests(unittest.TestCase):
    def test_screen_click_to_ray_returns_forward_ray_for_principal_point(self) -> None:
        intrinsics = CameraIntrinsics(width=640, height=480, fx=400.0, fy=400.0, cx=320.0, cy=240.0)
        frame = CameraFrame.look_at(origin=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), up_hint=(0.0, 1.0, 0.0))

        ray_origin, ray_direction = screen_click_to_ray(
            screen_x=320.0,
            screen_y=240.0,
            intrinsics=intrinsics,
            frame=frame,
        )

        self.assertEqual(ray_origin, (0.0, 0.0, 0.0))
        self.assertTrue(math.isclose(ray_direction[0], 0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(ray_direction[1], 0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(ray_direction[2], 1.0, abs_tol=1e-6))

    def test_screen_click_to_ray_moves_right_and_up_in_world_space(self) -> None:
        intrinsics = CameraIntrinsics(width=640, height=480, fx=400.0, fy=400.0, cx=320.0, cy=240.0)
        frame = CameraFrame.look_at(origin=(1.0, 2.0, 3.0), target=(1.0, 2.0, 4.0), up_hint=(0.0, 1.0, 0.0))

        _, ray_direction = screen_click_to_ray(
            screen_x=420.0,
            screen_y=140.0,
            intrinsics=intrinsics,
            frame=frame,
        )

        expected = normalize((0.25, 0.25, 1.0))
        self.assertTrue(math.isclose(ray_direction[0], expected[0], rel_tol=1e-6))
        self.assertTrue(math.isclose(ray_direction[1], expected[1], rel_tol=1e-6))
        self.assertTrue(math.isclose(ray_direction[2], expected[2], rel_tol=1e-6))

    def test_project_point_to_screen_returns_expected_pixel_coordinates(self) -> None:
        intrinsics = CameraIntrinsics(width=640, height=480, fx=400.0, fy=400.0, cx=320.0, cy=240.0)
        frame = CameraFrame.look_at(origin=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), up_hint=(0.0, 1.0, 0.0))

        projection = project_point_to_screen(
            point_xyz=(0.25, 0.5, 2.0),
            intrinsics=intrinsics,
            frame=frame,
        )

        self.assertIsNotNone(projection)
        screen_x, screen_y, depth = projection
        self.assertTrue(math.isclose(screen_x, 370.0, rel_tol=1e-6))
        self.assertTrue(math.isclose(screen_y, 140.0, rel_tol=1e-6))
        self.assertTrue(math.isclose(depth, 2.0, rel_tol=1e-6))


def normalize(vector):
    length = math.sqrt(sum(component * component for component in vector))
    return tuple(component / length for component in vector)


if __name__ == "__main__":
    unittest.main()
