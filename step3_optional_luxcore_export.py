#!/usr/bin/env python
# step3_optional_luxcore_export.py
# Purpose: Run inside Windows Blender to export LuxCore SDL (render.cfg + scene.scn).
# - Requires BlendLuxCore add-on. Uses FileSaver to write into export/ next to the .blend.

import os
import bpy
import addon_utils
from pathlib import Path

from logging_utils import get_logger

REPO_ROOT = Path(__file__).resolve().parent
BLEND_DIR = Path(bpy.path.abspath("//")).resolve()
EXPORT_DIR = (REPO_ROOT / "export").resolve()
LOG_PATH = (REPO_ROOT / "out" / "scdl_pipeline.log").resolve()
LOGGER = get_logger("scdl_step3_export", LOG_PATH)


def log_info(msg: str):
    LOGGER.info(msg)


def log_warn(msg: str):
    LOGGER.warning(msg)


def log_error(msg: str):
    LOGGER.error(msg)

# Export-time resolution header (can be overridden later by pyluxcore)
RES_X = int(os.environ.get("SCDL_EXPORT_RES_X", "1024"))
RES_Y = int(os.environ.get("SCDL_EXPORT_RES_Y", "1024"))

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure BlendLuxCore is enabled
try:
    addon_utils.enable("BlendLuxCore", default_set=True, persistent=True)
except Exception as e:
    log_warn(f"Could not enable BlendLuxCore automatically: {e}")

# Switch to LuxCore engine and export SDL
bpy.context.scene.render.engine = "LUXCORE"
bpy.context.scene.render.resolution_x = RES_X
bpy.context.scene.render.resolution_y = RES_Y
bpy.context.scene.render.resolution_percentage = 100

# Enable FileSaver via official BlendLuxCore properties
ok = False
try:
    lc = bpy.context.scene.luxcore
    cfg = getattr(lc, "config", None)
    if cfg is not None:
        cfg.use_filesaver = True
        cfg.filesaver_format = "TXT"  # writes human-readable render.cfg + scene.scn
        cfg.filesaver_path = str(EXPORT_DIR)
        ok = True
except Exception as e:
    log_warn(f"[Filesaver] Config failed: {e}")
if not ok:
    log_warn("WARNING: Could not configure FileSaver automatically. Enable it in UI: Render Properties → LuxCore → Tools → Only write LuxCore scene.")
    log_info(f"Suggested FileSaver path: {EXPORT_DIR}")

# Trigger a still render; when FileSaver is enabled, it writes render.cfg/scene.scn instead of rendering pixels
bpy.ops.render.render(write_still=True)

log_info(f"[OK] Exported LuxCore SDL to: {EXPORT_DIR}")
log_info("Check for 'render.cfg' and 'scene.scn'.")
