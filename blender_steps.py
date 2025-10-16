#!/usr/bin/env python
# blender_steps.py
# Purpose: Run inside Windows Blender to perform Step 1 (preview) render.

import os
import bpy
from pathlib import Path

from logging_utils import get_logger

try:
    import addon_utils
except Exception:  # Blender guards; addon_utils is Blender-specific
    addon_utils = None  # type: ignore


# Common dirs: keep outputs under the repo so Windows/WSL see the same files.
REPO_ROOT = Path(__file__).resolve().parent
BLEND_DIR = Path(bpy.path.abspath("//")).resolve()
OUT_DIR = (REPO_ROOT / "out").resolve()
EXPORT_DIR = (REPO_ROOT / "export").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _enum_engines():
    try:
        return [it.identifier for it in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items]
    except Exception:
        return []

LOGGER = get_logger("scdl_blender", OUT_DIR / "scdl_pipeline.log")


def _ensure_cycles_addon():
    try:
        if getattr(bpy.app.build_options, "cycles", True) is False:
            log_warn("[Engine] This Blender build has Cycles disabled at compile time.")
            return
        prefs = getattr(bpy.context, "preferences", None)
        if prefs and 'cycles' in prefs.addons:
            return
        if addon_utils is None:
            log_warn("[Engine] addon_utils unavailable; cannot auto-enable Cycles add-on.")
            return
        try:
            addon_utils.enable("cycles", default_set=True)  # type: ignore[attr-defined]
            log_info("[Engine] Enabled Cycles add-on for this session.")
        except Exception as exc:
            log_warn(f"[Engine] Could not enable Cycles add-on automatically: {exc}")
    except Exception as exc:
        log_warn(f"[Engine] Failed to check Cycles availability: {exc}")


def _configure_cycles_device(scene: bpy.types.Scene) -> None:
    desired = os.environ.get("SCDL_CYCLES_DEVICE", "").strip()
    if not desired:
        return
    desired_upper = desired.upper()
    try:
        prefs_all = getattr(bpy.context, "preferences", None)
        if prefs_all is None:
            log_warn("[Cycles] Preferences unavailable; cannot configure device.")
            return
        cycles_addon = prefs_all.addons.get('cycles')
        if cycles_addon is None:
            log_warn("[Cycles] Cycles add-on preferences not found; device override skipped.")
            return
        prefs = cycles_addon.preferences
    except Exception as exc:
        log_warn(f"[Cycles] Could not access preferences: {exc}")
        return

    if desired_upper == 'CPU':
        try:
            scene.cycles.device = 'CPU'
        except Exception as exc:
            log_warn(f"[Cycles] Failed to set CPU device: {exc}")
        devices = prefs.get_devices()
        for dev_type, dev_list in devices:
            for dev in dev_list:
                try:
                    dev.use = (dev_type == 'CPU')
                except Exception:
                    pass
        log_info("[Cycles] Using CPU device for rendering.")
        return

    available_types = [dev_type for dev_type, _ in prefs.get_devices()]
    target_type: str | None = None
    if desired_upper in available_types:
        target_type = desired_upper
    elif desired_upper == 'GPU':
        target_type = next((dt for dt in available_types if dt != 'CPU'), None)

    if not target_type:
        log_warn(f"[Cycles] Requested device '{desired}' not available; options: {available_types}")
        return

    try:
        prefs.compute_device_type = target_type
    except Exception as exc:
        log_warn(f"[Cycles] Failed to set compute device '{target_type}': {exc}")

    devices = prefs.get_devices()
    any_enabled = False
    for dev_type, dev_list in devices:
        for dev in dev_list:
            use = (dev_type == target_type)
            try:
                dev.use = use
            except Exception:
                pass
            any_enabled = any_enabled or use

    try:
        scene.cycles.device = 'GPU'
    except Exception as exc:
        log_warn(f"[Cycles] Failed to set GPU device on scene: {exc}")
        return

    if any_enabled:
        log_info(f"[Cycles] Using {target_type} devices for rendering.")
    else:
        log_warn(f"[Cycles] No devices enabled for type '{target_type}'.")


def log_info(msg: str):
    LOGGER.info(msg)


def log_warn(msg: str):
    LOGGER.warning(msg)


def log_error(msg: str):
    LOGGER.error(msg)


# =====================
# Step 1: Preview render
# =====================
def render_preview():
    scene = bpy.context.scene

    # Preview settings
    PREVIEW_SHORT = int(os.environ.get("SCDL_PREVIEW_SHORT", "448"))
    PREVIEW_SPP = int(os.environ.get("SCDL_PREVIEW_SPP", "16"))
    PREVIEW_DENOISE = int(os.environ.get("SCDL_PREVIEW_DENOISE", "1")) != 0
    PREVIEW_PATH = (OUT_DIR / "preview.png")

    # Keep preview aspect ratio equal to final render to avoid ROI mapping issues
    rw, rh = int(scene.render.resolution_x), int(scene.render.resolution_y)
    if rw <= 0 or rh <= 0:
        rw, rh = 1920, 1080
    if rw >= rh:
        ph = PREVIEW_SHORT
        pw = int(round(ph * (rw / rh)))
    else:
        pw = PREVIEW_SHORT
        ph = int(round(pw * (rh / rw)))
    scene.render.resolution_x = pw
    scene.render.resolution_y = ph
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    try:
        _ensure_cycles_addon()
        scene.render.engine = 'CYCLES'
        _configure_cycles_device(scene)
    except Exception as exc:
        engines = set(_enum_engines())
        log_error(f"[Engine] Could not set Cycles; available engines: {sorted(engines)}")
        msg = f"Cycles render engine not available; cannot render preview ({exc})."
        log_error(msg)
        raise RuntimeError(msg) from exc

    try:
        scene.cycles.samples = PREVIEW_SPP
        scene.cycles.use_adaptive_sampling = True
        if PREVIEW_DENOISE:
            for vl in scene.view_layers:
                try:
                    vl.cycles.use_denoising = True
                except Exception:
                    pass
    except Exception as exc:
        log_error(f"Failed to configure Cycles preview settings: {exc}")
        raise

    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(PREVIEW_PATH)
    bpy.ops.render.render(write_still=True)
    log_info(f"[OK] Preview saved → {PREVIEW_PATH}")




def main():
    step = os.environ.get('SCDL_BLEND_STEP', 'preview').strip().lower()
    if step in ('preview', 'step1', '1'):
        render_preview()
    else:
        log_warn(f"[WARN] Unknown SCDL_BLEND_STEP={step}; this helper only supports 'preview'.")
        render_preview()


if __name__ == "__main__":
    main()
