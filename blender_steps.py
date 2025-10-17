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
            raise RuntimeError("[Engine] This Blender build has Cycles disabled at compile time.")
        prefs = getattr(bpy.context, "preferences", None)
        if prefs and 'cycles' in prefs.addons:
            return
        if addon_utils is None:
            raise RuntimeError("[Engine] addon_utils unavailable; cannot auto-enable Cycles add-on.")
        try:
            addon_utils.enable("cycles", default_set=True)  # type: ignore[attr-defined]
            log_info("[Engine] Enabled Cycles add-on for this session.")
        except Exception as exc:
            raise RuntimeError(f"[Engine] Could not enable Cycles add-on automatically: {exc}")
    except Exception as exc:
        raise RuntimeError(f"[Engine] Failed to check Cycles availability: {exc}")


def _configure_cycles_device(scene: bpy.types.Scene) -> None:
    desired = os.environ.get("SCDL_CYCLES_DEVICE", "").strip()
    if not desired:
        return
    desired_upper = desired.upper()
    try:
        prefs_all = getattr(bpy.context, "preferences", None)
        if prefs_all is None:
            raise RuntimeError("[Cycles] Preferences unavailable; cannot configure device.")
        cycles_addon = prefs_all.addons.get('cycles')
        if cycles_addon is None:
            raise RuntimeError("[Cycles] Cycles add-on preferences not found; device override skipped.")
        prefs = cycles_addon.preferences
    except Exception as exc:
        raise RuntimeError(f"[Cycles] Could not access preferences: {exc}")

    if desired_upper == 'CPU':
        try:
            scene.cycles.device = 'CPU'
        except Exception as exc:
            raise RuntimeError(f"[Cycles] Failed to set CPU device: {exc}")
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
        raise RuntimeError(f"[Cycles] Requested device '{desired}' not available; options: {available_types}")

    try:
        prefs.compute_device_type = target_type
    except Exception as exc:
        raise RuntimeError(f"[Cycles] Failed to set compute device '{target_type}': {exc}")

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
        raise RuntimeError(f"[Cycles] Failed to set GPU device on scene: {exc}")

    if any_enabled:
        log_info(f"[Cycles] Using {target_type} devices for rendering.")
    else:
        raise RuntimeError(f"[Cycles] No devices enabled for type '{target_type}'.")


def log_info(msg: str):
    LOGGER.info(msg)


def log_warn(msg: str):
    LOGGER.warning(msg)


def log_error(msg: str):
    LOGGER.error(msg)


# Choose the fastest denoiser available (OPTIX on NVIDIA, OIDN otherwise)
def _configure_best_denoiser(scene: bpy.types.Scene) -> None:
    try:
        cycles = scene.cycles
    except Exception:
        raise RuntimeError("[Cycles] Scene has no Cycles settings; cannot configure denoiser.")
    try:
        cycles.denoiser = "OPTIX"
    except Exception as exc:
        raise RuntimeError(f"[Cycles] Failed to set OptiX denoiser: {exc}")
    if getattr(cycles, "denoiser", None) != "OPTIX":
        raise RuntimeError("[Cycles] OptiX denoiser not active after assignment.")
    log_info("[Cycles] Preview denoiser → OPTIX")

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
    PREVIEW_ADAPTIVE_THRESHOLD = os.environ.get("SCDL_PREVIEW_ADAPTIVE_THRESHOLD", "").strip()
    PREVIEW_ADAPTIVE_MIN = os.environ.get("SCDL_PREVIEW_MIN_SPP", "").strip()

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
        if PREVIEW_ADAPTIVE_THRESHOLD:
            try:
                scene.cycles.adaptive_threshold = max(0.0, float(PREVIEW_ADAPTIVE_THRESHOLD))
                log_info(f"[Cycles] Preview adaptive threshold → {scene.cycles.adaptive_threshold:.4f}")
            except Exception as exc:
                raise RuntimeError(f"[Cycles] Failed to set preview adaptive threshold: {exc}")
        if PREVIEW_ADAPTIVE_MIN:
            try:
                scene.cycles.adaptive_min_samples = max(1, min(PREVIEW_SPP, int(float(PREVIEW_ADAPTIVE_MIN))))
                log_info(f"[Cycles] Preview adaptive min samples → {scene.cycles.adaptive_min_samples}")
            except Exception as exc:
                raise RuntimeError(f"[Cycles] Failed to set preview adaptive minimum samples: {exc}")
        # Preview noise controls tuned for saliency stability
        # Optional: disable caustics (reduces fireflies without harming edges)
        if int(os.environ.get("SCDL_PREVIEW_CAUSTICS", "0")) == 0:
            try:
                scene.cycles.caustics_reflective = False
                scene.cycles.caustics_refractive = False
            except Exception as exc:
                raise RuntimeError(f"[Cycles] Failed to disable preview caustics: {exc}")
        # Optional: mild filter glossy to calm specular noise
        bg = os.environ.get("SCDL_PREVIEW_BLUR_GLOSSY", "").strip()
        if bg:
            try:
                scene.cycles.blur_glossy = float(bg)
            except Exception as exc:
                raise RuntimeError(f"[Cycles] Failed to set preview blur glossy value: {exc}")
        # Optional: light clamping (set only if provided)
        cd = os.environ.get("SCDL_PREVIEW_CLAMP_DIRECT", "").strip()
        ci = os.environ.get("SCDL_PREVIEW_CLAMP_INDIRECT", "").strip()
        if cd:
            try:
                scene.cycles.sample_clamp_direct = max(0.0, float(cd))
            except Exception as exc:
                raise RuntimeError(f"[Cycles] Failed to set preview direct clamp: {exc}")
        if ci:
            try:
                scene.cycles.sample_clamp_indirect = max(0.0, float(ci))
            except Exception as exc:
                raise RuntimeError(f"[Cycles] Failed to set preview indirect clamp: {exc}")
        if PREVIEW_DENOISE:
            try:
                scene.cycles.use_denoising = True
            except Exception as exc:
                raise RuntimeError(f"[Cycles] Failed to enable preview denoising: {exc}")
            for vl in scene.view_layers:
                try:
                    vl.cycles.use_denoising = True
                except Exception as exc:
                    raise RuntimeError(f"[Cycles] Failed to enable preview denoising on view layer '{vl.name}': {exc}")
            _configure_best_denoiser(scene)
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
        raise RuntimeError(f"Unsupported SCDL_BLEND_STEP={step}; this helper only supports 'preview'.")


if __name__ == "__main__":
    main()
