#!/usr/bin/env python
# step3_blender_roi_compose.py
# Thin wrapper to run Step 3 (ROI re-render + composite) inside Blender.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import blender_steps as _bs  # noqa: E402

if __name__ == "__main__":
    _bs.render_roi_and_compose()

