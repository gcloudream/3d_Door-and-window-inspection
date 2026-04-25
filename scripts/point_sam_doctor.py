#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from point_selection.segmenter import build_segmenter


def main() -> int:
    _, status = build_segmenter()
    payload = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "segmentation": status.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
