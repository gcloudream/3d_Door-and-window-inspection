import json
import tempfile
import unittest
from pathlib import Path

from point_selection.cli import main


class CLITests(unittest.TestCase):
    def test_cli_writes_pick_and_roi_summary_to_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "scene.json"
            output_path = temp_path / "roi.json"
            input_path.write_text(
                json.dumps(
                    {
                        "points": [
                            {"point_id": 1, "xyz": [0.0, 0.0, 0.0]},
                            {"point_id": 2, "xyz": [0.2, 0.0, 0.0]},
                            {"point_id": 3, "xyz": [0.4, 0.0, 0.0]},
                        ]
                    }
                )
            )

            exit_code = main(
                [
                    "--input",
                    str(input_path),
                    "--ray-origin",
                    "0,0,0",
                    "--ray-direction",
                    "1,0,0",
                    "--max-distance-to-ray",
                    "0.05",
                    "--roi-radius",
                    "0.25",
                    "--roi-min-points",
                    "2",
                    "--roi-max-points",
                    "10",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text())
            self.assertEqual(payload["pick"]["point_id"], 1)
            self.assertEqual(payload["roi"]["point_ids"], [1, 2])

    def test_cli_supports_screen_click_mode_with_ascii_ply_input(self) -> None:
        ply_content = "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 1",
                "0.25 0 1",
                "0 0.25 1",
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "scene.ply"
            output_path = temp_path / "roi.json"
            input_path.write_text(ply_content)

            exit_code = main(
                [
                    "--input",
                    str(input_path),
                    "--screen-x",
                    "320",
                    "--screen-y",
                    "240",
                    "--camera-origin",
                    "0,0,0",
                    "--camera-target",
                    "0,0,1",
                    "--camera-up",
                    "0,1,0",
                    "--image-width",
                    "640",
                    "--image-height",
                    "480",
                    "--fx",
                    "400",
                    "--fy",
                    "400",
                    "--cx",
                    "320",
                    "--cy",
                    "240",
                    "--roi-radius",
                    "0.1",
                    "--roi-min-points",
                    "1",
                    "--roi-max-points",
                    "10",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text())
            self.assertEqual(payload["pick"]["point_id"], 1)
            self.assertEqual(payload["roi"]["point_ids"], [1])


if __name__ == "__main__":
    unittest.main()
