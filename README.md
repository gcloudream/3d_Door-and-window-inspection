# Point Selection PoC

This repository contains the first part of the click-to-extract workflow:

- ray-based point picking
- screen-click to ray conversion
- local ROI generation around the selected point
- heuristic ROI segmentation adapter for final point-set extraction
- JSON and ASCII PLY input support
- JSON-based CLI output for integration and debugging
- local Web demo with Three.js rendering and Python API

## Run tests

```bash
python3 -m unittest discover -s tests -v
```

## Run the Web demo

```bash
python3 -m point_selection.server --scene sample_data/simple_room.ply --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The demo includes:

- Three.js point cloud viewer
- orbit controls
- browser click to Python segmentation API
- in-browser point cloud file switching
- ROI highlight overlay
- final mask highlight overlay
- right-side debug and parameter panel

The second-stage API endpoint is:

- `POST /api/segment-roi`

It returns:

- `pick`: the selected point
- `roi`: the clicked local candidate region
- `mask`: the extracted final point-set result

## Optional Point-SAM backend

The project now supports an optional real Point-SAM backend with automatic fallback:

- default: heuristic segmentation backend
- optional: Point-SAM backend when the runtime is correctly configured
- fallback: if Point-SAM cannot be initialized, the server automatically falls back to the heuristic backend

To request the Point-SAM backend, set:

```bash
export POINT_SEGMENTER_BACKEND=point_sam
export POINT_SAM_CHECKPOINT=/absolute/path/to/model.safetensors
export POINT_SAM_REPO_DIR=/absolute/path/to/Point-SAM
export POINT_SAM_CONFIG_NAME=large
export POINT_SAM_DEVICE=cuda
```

Then start the demo normally:

```bash
python3 -m point_selection.server --scene sample_data/indoor_room_test.ply --port 8000
```

Notes:

- if Point-SAM is not available, the server does not crash; it falls back automatically
- the current scene payload includes a `segmentation` field describing the requested backend, active backend, and fallback reason
- Point-SAM integration is intended for a properly prepared machine with the official dependencies and checkpoint files

## Run the demo CLI

```bash
python3 -m point_selection.cli \
  --input sample_data/simple_room.json \
  --ray-origin 0,0,0 \
  --ray-direction 1,0,0 \
  --max-distance-to-ray 0.05 \
  --roi-radius 0.5 \
  --roi-min-points 3 \
  --roi-max-points 10
```

The command prints a JSON payload that includes:

- `pick`: the selected point and its distance to the ray
- `roi`: the generated local point subset, effective radius, and truncation state

## Run the screen-click demo with ASCII PLY input

```bash
python3 -m point_selection.cli \
  --input sample_data/simple_room.ply \
  --screen-x 320 \
  --screen-y 240 \
  --camera-origin 0,0,0 \
  --camera-target 0,0,1 \
  --camera-up 0,1,0 \
  --image-width 640 \
  --image-height 480 \
  --fx 400 \
  --fy 400 \
  --cx 320 \
  --cy 240 \
  --roi-radius 0.5 \
  --roi-min-points 3 \
  --roi-max-points 10
```

This mode matches the boundary we want for UI integration:

- the viewer provides `screen_x` and `screen_y`
- the viewer provides camera origin, target, up, and intrinsics
- the CLI converts them to a ray before running point pick and ROI generation
