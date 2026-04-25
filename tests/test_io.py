import tempfile
import unittest
from pathlib import Path
import struct

from point_selection.io import load_point_cloud, load_point_cloud_from_bytes


class PointCloudIOTests(unittest.TestCase):
    def test_load_point_cloud_reads_ascii_ply_with_rgb(self) -> None:
        ply_content = "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 0 255 0 0",
                "1 0 0 0 255 0",
                "2 0 0 0 0 255",
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            ply_path = Path(temp_dir) / "scene.ply"
            ply_path.write_text(ply_content)

            cloud = load_point_cloud(ply_path)

        self.assertEqual(len(cloud.points), 3)
        self.assertEqual(cloud.points[0].point_id, 1)
        self.assertEqual(cloud.points[1].xyz, (1.0, 0.0, 0.0))
        self.assertEqual(cloud.points[2].rgb, (0, 0, 255))

    def test_load_point_cloud_rejects_binary_ply(self) -> None:
        ply_content = "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            ply_path = Path(temp_dir) / "scene.ply"
            ply_path.write_text(ply_content)

            with self.assertRaises(ValueError):
                load_point_cloud(ply_path)

    def test_load_point_cloud_reads_binary_pcd_with_packed_rgb(self) -> None:
        pcd_content = build_binary_pcd(
            fields=["rgb", "x", "y", "z"],
            size=[4, 4, 4, 4],
            type_names=["F", "F", "F", "F"],
            count=[1, 1, 1, 1],
            rows=[
                struct.pack("<Ifff", 0x00FF0000, 1.0, 2.0, 3.0),
                struct.pack("<Ifff", 0x0000FF00, 4.0, 5.0, 6.0),
            ],
        )

        cloud = load_point_cloud_from_bytes("scene.pcd", pcd_content)

        self.assertEqual(len(cloud.points), 2)
        self.assertEqual(cloud.points[0].xyz, (1.0, 2.0, 3.0))
        self.assertEqual(cloud.points[0].rgb, (255, 0, 0))
        self.assertEqual(cloud.points[1].rgb, (0, 255, 0))

    def test_load_point_cloud_from_bytes_samples_large_binary_pcd(self) -> None:
        pcd_content = build_binary_pcd(
            fields=["rgb", "x", "y", "z"],
            size=[4, 4, 4, 4],
            type_names=["F", "F", "F", "F"],
            count=[1, 1, 1, 1],
            rows=[
                struct.pack("<Ifff", 0x00AA0000, 0.0, 0.0, 0.0),
                struct.pack("<Ifff", 0x0000AA00, 1.0, 0.0, 0.0),
                struct.pack("<Ifff", 0x000000AA, 2.0, 0.0, 0.0),
                struct.pack("<Ifff", 0x00FFFFFF, 3.0, 0.0, 0.0),
                struct.pack("<Ifff", 0x00000000, 4.0, 0.0, 0.0),
            ],
        )

        cloud = load_point_cloud_from_bytes("scene.pcd", pcd_content, max_points=2)

        self.assertEqual(len(cloud.points), 2)
        self.assertEqual([point.point_id for point in cloud.points], [1, 4])
        self.assertEqual([point.xyz[0] for point in cloud.points], [0.0, 3.0])


if __name__ == "__main__":
    unittest.main()


def build_binary_pcd(fields, size, type_names, count, rows) -> bytes:
    header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        f"FIELDS {' '.join(fields)}",
        f"SIZE {' '.join(str(value) for value in size)}",
        f"TYPE {' '.join(type_names)}",
        f"COUNT {' '.join(str(value) for value in count)}",
        f"WIDTH {len(rows)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(rows)}",
        "DATA binary",
    ]
    return ("\n".join(header_lines) + "\n").encode("ascii") + b"".join(rows)
