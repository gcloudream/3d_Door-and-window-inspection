# Three.js Demo UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Three.js demo that renders a sample point cloud, sends browser click and camera data to Python, and highlights the picked point plus ROI.

**Architecture:** A Python stdlib web server will serve both static frontend assets and JSON API endpoints. The browser app uses Three.js for rendering and stateful overlays while delegating point picking and ROI computation to the existing Python core modules.

**Tech Stack:** Python 3.9 stdlib HTTP server, unittest, Three.js, native ES modules, HTML, CSS, JavaScript

---

### Task 1: Python API Service

**Files:**
- Create: `/Users/gengchen/Desktop/3d/tests/test_server.py`
- Create: `/Users/gengchen/Desktop/3d/point_selection/server.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest
from pathlib import Path

from point_selection.server import DemoService


class DemoServiceTests(unittest.TestCase):
    def test_get_scene_payload_returns_points_and_bounds(self) -> None:
        service = DemoService(scene_path=Path("sample_data/simple_room.ply"))
        payload = service.get_scene_payload()
        self.assertEqual(payload["point_count"], 9)
        self.assertEqual(len(payload["points"]), 9)
        self.assertEqual(payload["bounds"]["min"], [0.0, 0.0, 0.0])

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'point_selection.server'`

- [ ] **Step 3: Write minimal implementation**

```python
class DemoService:
    def __init__(self, scene_path: Path) -> None:
        self._scene_path = scene_path

    def get_scene_payload(self) -> dict:
        ...

    def pick_roi(self, payload: dict) -> dict:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_server.py -v`
Expected: PASS

### Task 2: Frontend Shell

**Files:**
- Create: `/Users/gengchen/Desktop/3d/web/index.html`
- Create: `/Users/gengchen/Desktop/3d/web/styles.css`
- Create: `/Users/gengchen/Desktop/3d/web/app.js`
- Create: `/Users/gengchen/Desktop/3d/web/scene-view.js`

- [ ] **Step 1: Write the failing browser integration expectation in a smoke test**

```python
import unittest
from pathlib import Path


class FrontendSmokeTests(unittest.TestCase):
    def test_index_references_app_entry(self) -> None:
        html = Path("web/index.html").read_text()
        self.assertIn("app.js", html)
        self.assertIn("canvas-shell", html)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_web_smoke.py -v`
Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```html
<main class="app-shell">
  <section class="canvas-shell"></section>
  <aside class="info-panel"></aside>
</main>
<script type="module" src="/web/app.js"></script>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_web_smoke.py -v`
Expected: PASS

### Task 3: Wire UI to API

**Files:**
- Modify: `/Users/gengchen/Desktop/3d/web/app.js`
- Modify: `/Users/gengchen/Desktop/3d/web/scene-view.js`
- Modify: `/Users/gengchen/Desktop/3d/point_selection/server.py`

- [ ] **Step 1: Write the failing API serialization expectation**

```python
def test_pick_roi_payload_contains_debug_fields(self) -> None:
    service = DemoService(scene_path=Path("sample_data/simple_room.ply"))
    payload = service.pick_roi(...)
    self.assertIn("radius", payload["roi"])
    self.assertIn("expansions", payload["roi"])
    self.assertIn("truncated", payload["roi"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_server.py -v`
Expected: FAIL because the debug fields are missing

- [ ] **Step 3: Write minimal implementation**

```javascript
const result = await postPickRequest(payload)
sceneView.updateSelection(result)
renderInfoPanel(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS

### Task 4: Manual Verification

**Files:**
- Modify: `/Users/gengchen/Desktop/3d/README.md`

- [ ] **Step 1: Document run commands**

```bash
python3 -m point_selection.server --scene sample_data/simple_room.ply --port 8000
```

- [ ] **Step 2: Verify server starts**

Run: `python3 -m point_selection.server --scene sample_data/simple_room.ply --port 8000`
Expected: prints local URL without crashing

- [ ] **Step 3: Verify browser interaction**

Run: open `http://localhost:8000`
Expected: point cloud visible, click updates highlighted ROI and right panel
