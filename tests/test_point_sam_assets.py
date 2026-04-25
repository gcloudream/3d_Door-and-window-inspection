import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path("/Users/gengchen/Desktop/3d")


class PointSAMAssetsTests(unittest.TestCase):
    def test_point_sam_doctor_reports_fallback_when_backend_is_unavailable(self) -> None:
        env = os.environ.copy()
        env["POINT_SEGMENTER_BACKEND"] = "point_sam"
        env["POINT_SAM_CHECKPOINT"] = "/tmp/missing-model.safetensors"

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "point_sam_doctor.py")],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["segmentation"]["requested_backend"], "point_sam")
        self.assertEqual(payload["segmentation"]["active_backend"], "heuristic_region_growing")
        self.assertIn("Point-SAM", payload["segmentation"]["fallback_reason"])

    def test_shell_scripts_have_valid_bash_syntax(self) -> None:
        setup_script = PROJECT_ROOT / "scripts" / "setup_point_sam_linux.sh"
        run_script = PROJECT_ROOT / "scripts" / "run_demo_point_sam.sh"

        subprocess.run(["bash", "-n", str(setup_script)], cwd=PROJECT_ROOT, check=True)
        subprocess.run(["bash", "-n", str(run_script)], cwd=PROJECT_ROOT, check=True)

    def test_deploy_doc_mentions_required_env_vars(self) -> None:
        doc = (PROJECT_ROOT / "docs" / "point-sam-linux-deploy.md").read_text()

        self.assertIn("POINT_SAM_REPO_DIR", doc)
        self.assertIn("POINT_SAM_CHECKPOINT", doc)
        self.assertIn("POINT_SEGMENTER_BACKEND", doc)


if __name__ == "__main__":
    unittest.main()
