#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POINT_SAM_VENV_DIR="${POINT_SAM_VENV_DIR:-$PROJECT_ROOT/.venv-point-sam}"
POINT_SAM_REPO_DIR="${POINT_SAM_REPO_DIR:-$PROJECT_ROOT/.vendor/Point-SAM}"
SCENE_PATH="${SCENE_PATH:-$PROJECT_ROOT/sample_data/indoor_room_test.ply}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if [ ! -d "$POINT_SAM_VENV_DIR" ]; then
  echo "ERROR: virtualenv not found at $POINT_SAM_VENV_DIR" >&2
  exit 1
fi

if [ ! -d "$POINT_SAM_REPO_DIR" ]; then
  echo "ERROR: Point-SAM repo not found at $POINT_SAM_REPO_DIR" >&2
  exit 1
fi

if [ -z "${POINT_SAM_CHECKPOINT:-}" ]; then
  echo "ERROR: POINT_SAM_CHECKPOINT is not set" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$POINT_SAM_VENV_DIR/bin/activate"

export PYTHONPATH="$POINT_SAM_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export POINT_SEGMENTER_BACKEND="${POINT_SEGMENTER_BACKEND:-point_sam}"
export POINT_SAM_REPO_DIR
export POINT_SAM_DEVICE="${POINT_SAM_DEVICE:-cuda}"

exec python -m point_selection.server --scene "$SCENE_PATH" --host "$HOST" --port "$PORT"
