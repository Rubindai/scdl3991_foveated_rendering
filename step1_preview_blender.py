#!/usr/bin/env python
# step1_preview_blender.py
# Thin wrapper to run Step 1 (preview) inside Blender.

import sys
from pathlib import Path

# Ensure repo root (this file's dir) is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import blender_steps as _bs  # noqa: E402

if __name__ == "__main__":
    _bs.render_preview()

