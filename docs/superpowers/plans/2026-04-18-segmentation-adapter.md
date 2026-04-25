# Segmentation Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a replaceable ROI segmentation adapter, a `/api/segment-roi` API, and a frontend mask overlay so the demo can extract a final point-set result from a click.

**Architecture:** Keep the existing click-to-pick-to-ROI workflow unchanged, add a new backend segmentation layer that consumes `ROIResult`, and expose a new API that returns `pick + roi + mask`. The browser continues to render the base point cloud but adds a second overlay for the final mask result.

**Tech Stack:** Python 3.9 stdlib, unittest, existing point selection core, HTML, CSS, native ES modules, Three.js

---

### Task 1: Segmentation Adapter

**Files:**
- Create: `/Users/gengchen/Desktop/3d/tests/test_segmenter.py`
- Create: `/Users/gengchen/Desktop/3d/point_selection/segmenter.py`
- Modify: `/Users/gengchen/Desktop/3d/point_selection/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from point_selection.core import PointCloud, PointRecord, ROIResult
from point_selection.segmenter import HeuristicRegionSegmenter


class HeuristicRegionSegmenterTests(unittest.TestCase):
    def test_segment_returns_seed_connected_cluster(self) -> None:
        roi = ROIResult(
            center_point_id=1,
            center_xyz=(0.0, 0.0, 0.0),
            radius=1.0,
            point_ids=[1, 2, 3, 4],
            points=[
                PointRecord(1, (0.0, 0.0, 0.0), (200, 120, 90)),
                PointRecord(2, (0.2, 0.0, 0.0), (198, 118, 92)),
                PointRecord(3, (0.4, 0.0, 0.0), (201, 121, 88)),
                PointRecord(4, (0.9, 0.9, 0.0), (120, 160, 210)),
            ],
            expansions=0,
            truncated=False,
        )

        segmenter = HeuristicRegionSegmenter()
        result = segmenter.segment(roi, seed_point_id=1)

        self.assertEqual(result.point_ids, [1, 2, 3])
        self.assertEqual(result.point_count, 3)
        self.assertEqual(result.seed_point_id, 1)
        self.assertEqual(result.method, "heuristic_region_growing")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_segmenter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'point_selection.segmenter'`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentResult:
    seed_point_id: int
    point_ids: list[int]
    point_count: int
    confidence: float
    method: str


class HeuristicRegionSegmenter:
    def segment(self, roi, seed_point_id: int) -> SegmentResult:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_segmenter.py -v`
Expected: PASS

### Task 2: Backend Segment API

**Files:**
- Modify: `/Users/gengchen/Desktop/3d/tests/test_server.py`
- Modify: `/Users/gengchen/Desktop/3d/point_selection/server.py`

- [ ] **Step 1: Write the failing test**

```python
def test_segment_roi_returns_pick_roi_and_mask(self) -> None:
    service = DemoService(scene_path=Path("sample_data/indoor_room_test.ply"))

    payload = service.segment_roi(
        {
            "screen_x": 640,
            "screen_y": 360,
            "camera": {
                "origin": [2.98, 3.86, 8.29],
                "target": [2.0, 1.5, 1.4],
                "up": [0, 1, 0],
                "width": 1280,
                "height": 720,
                "fx": 772.412,
                "fy": 434.482,
                "cx": 640,
                "cy": 360,
            },
            "pick": {"max_distance_to_ray": 0.25},
            "roi": {"radius": 0.8, "min_points": 3, "max_points": 2000},
        }
    )

    self.assertTrue(payload["matched"])
    self.assertIn("mask", payload)
    self.assertGreaterEqual(payload["mask"]["point_count"], 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_server.py -v`
Expected: FAIL because `DemoService` has no `segment_roi`

- [ ] **Step 3: Write minimal implementation**

```python
class DemoService:
    def segment_roi(self, payload: Dict[str, object]) -> Dict[str, object]:
        pick_result, roi_result = self._resolve_pick_and_roi(payload)
        mask_result = self._segmenter.segment(roi_result, seed_point_id=pick_result.point_id)
        return {
            "matched": True,
            "pick": ...,
            "roi": ...,
            "mask": mask_result.to_dict(),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_server.py -v`
Expected: PASS

### Task 3: Frontend Mask Overlay

**Files:**
- Modify: `/Users/gengchen/Desktop/3d/tests/test_web_smoke.py`
- Modify: `/Users/gengchen/Desktop/3d/web/index.html`
- Modify: `/Users/gengchen/Desktop/3d/web/app.js`
- Modify: `/Users/gengchen/Desktop/3d/web/scene-view.js`

- [ ] **Step 1: Write the failing smoke test**

```python
def test_index_exposes_mask_result_fields(self) -> None:
    html = Path("/Users/gengchen/Desktop/3d/web/index.html").read_text()
    self.assertIn('id="mask-point-count"', html)
    self.assertIn('id="mask-confidence"', html)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_web_smoke.py -v`
Expected: FAIL because the new fields do not exist

- [ ] **Step 3: Write minimal implementation**

```javascript
const response = await fetch("/api/segment-roi", { ... });
sceneView.setSelection(result);
maskPointCount.textContent = String(result.mask.point_count);
maskConfidence.textContent = String(result.mask.confidence);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_web_smoke.py -v`
Expected: PASS

### Task 4: Verification and Docs

**Files:**
- Modify: `/Users/gengchen/Desktop/3d/README.md`

- [ ] **Step 1: Document the second-stage behavior**

```markdown
- `/api/segment-roi` returns pick, ROI, and final mask
- the viewer renders ROI and mask as separate overlays
```

- [ ] **Step 2: Run full automated verification**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS with all tests green

- [ ] **Step 3: Run manual browser verification**

Run:

```bash
python3 -m point_selection.server --scene sample_data/indoor_room_test.ply --port 8000
```

Expected:

- browser opens the demo
- click updates selected point
- ROI remains visible
- mask overlay appears separately
- right panel shows `Mask 点数` and `Mask 置信度`
