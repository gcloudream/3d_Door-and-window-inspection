import json
import subprocess
import textwrap
import unittest
from pathlib import Path


PICK_BUFFER_MODULE = Path("/Users/gengchen/Desktop/3d/web/pick-buffer.js")


class PickBufferLogicTests(unittest.TestCase):
    def run_node(self, source: str) -> dict:
        completed = subprocess.run(
            ["node", "--input-type=module", "-e", source],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_pick_color_encoding_round_trips_display_index(self) -> None:
        script = textwrap.dedent(
            f"""
            import {{ encodePickIndexToRgb, decodePickIndexFromRgb }} from {json.dumps(PICK_BUFFER_MODULE.as_posix())};

            const rgb = encodePickIndexToRgb(123456);
            const decoded = decodePickIndexFromRgb(rgb[0], rgb[1], rgb[2]);
            console.log(JSON.stringify({{ rgb, decoded }}));
            """
        )

        payload = self.run_node(script)
        self.assertEqual(payload["decoded"], 123456)
        self.assertEqual(len(payload["rgb"]), 3)

    def test_pick_search_window_prefers_nearest_non_empty_pixel(self) -> None:
        script = textwrap.dedent(
            f"""
            import {{ findNearestPickHit }} from {json.dumps(PICK_BUFFER_MODULE.as_posix())};

            const width = 5;
            const height = 5;
            const pixels = new Uint8Array(width * height * 4);

            function setPixel(x, y, index) {{
              const offset = (y * width + x) * 4;
              pixels[offset + 0] = index & 255;
              pixels[offset + 1] = (index >> 8) & 255;
              pixels[offset + 2] = (index >> 16) & 255;
              pixels[offset + 3] = 255;
            }}

            setPixel(4, 4, 7);
            setPixel(2, 1, 3);

            const hit = findNearestPickHit({{
              pixels,
              width,
              height,
              originX: 10,
              originY: 20,
              targetX: 12,
              targetY: 21,
            }});

            console.log(JSON.stringify(hit));
            """
        )

        payload = self.run_node(script)
        self.assertEqual(payload["pickIndex"], 3)
        self.assertEqual(payload["screenX"], 12)
        self.assertEqual(payload["screenY"], 21)


if __name__ == "__main__":
    unittest.main()
