import json
import subprocess
import textwrap
import unittest
from pathlib import Path


HIT_TESTING_MODULE = Path("/Users/gengchen/Desktop/3d/web/hit-testing.js")


class SceneHitLogicTests(unittest.TestCase):
    def run_node(self, source: str) -> dict:
        completed = subprocess.run(
            ["node", "--input-type=module", "-e", source],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_square_hit_distance_treats_point_sprite_as_square(self) -> None:
        script = textwrap.dedent(
            f"""
            import {{ computeSquareHitMetrics }} from {json.dumps(HIT_TESTING_MODULE.as_posix())};

            const result = computeSquareHitMetrics({{
              screenX: 108,
              screenY: 108,
              pointScreenX: 100,
              pointScreenY: 100,
              halfSizePx: 10,
            }});

            console.log(JSON.stringify(result));
            """
        )

        payload = self.run_node(script)
        self.assertEqual(payload["effectiveSquareDistance"], 0)
        self.assertGreater(payload["pixelDistance"], 10)

    def test_hint_key_prefers_frontmost_point_when_cursor_is_inside_overlapping_squares(self) -> None:
        script = textwrap.dedent(
            f"""
            import {{ buildHintCandidateKey }} from {json.dumps(HIT_TESTING_MODULE.as_posix())};

            const front = buildHintCandidateKey({{
              effectiveSquareDistance: 0,
              pixelDistance: 5,
              depth: 0.10,
              projectionLength: 1.0,
              distanceToRay: 0.02,
              pointId: 1,
              insideSquare: true,
            }});
            const back = buildHintCandidateKey({{
              effectiveSquareDistance: 0,
              pixelDistance: 1,
              depth: 0.45,
              projectionLength: 3.0,
              distanceToRay: 0.01,
              pointId: 2,
              insideSquare: true,
            }});

            console.log(JSON.stringify({{ front, back }}));
            """
        )

        payload = self.run_node(script)
        self.assertLess(payload["front"], payload["back"])


if __name__ == "__main__":
    unittest.main()
