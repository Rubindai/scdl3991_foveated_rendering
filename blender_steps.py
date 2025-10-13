#!/usr/bin/env python
# blender_steps.py
# Purpose: Run inside Windows Blender to perform step1 preview OR step3 ROI+composite,
# selected via the env var SCDL_BLEND_STEP={preview|final}.

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
        scene.cycles.use_adaptive_sampling = False
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


# ======================================
# Step 3: ROI re-render and composite
# ======================================
def render_roi_and_compose():
    scene = bpy.context.scene

    preview_path = (OUT_DIR / 'preview.png')
    mask_png_path = (OUT_DIR / 'user_importance_preview.png')
    bbox_path = (OUT_DIR / 'roi_bbox.txt')
    final_path = (OUT_DIR / 'final.png')
    base_path = (BLEND_DIR / '__base.png')

    if not preview_path.exists():
        raise FileNotFoundError(f"Preview not found: {preview_path}")
    if not bbox_path.exists():
        raise FileNotFoundError(f"ROI bbox not found: {bbox_path}. Run DINO step first.")

    # Load normalized bbox (nx0 nx1 ny0 ny1)
    txt = bbox_path.read_text().strip().split()
    if len(txt) != 4:
        raise RuntimeError(f"Malformed roi_bbox.txt: {bbox_path}")
    nx0, nx1, ny0, ny1 = [float(x) for x in txt]
    # Clamp to [0,1] for safety
    nx0 = max(0.0, min(1.0, nx0)); nx1 = max(0.0, min(1.0, nx1))
    ny0 = max(0.0, min(1.0, ny0)); ny1 = max(0.0, min(1.0, ny1))

    # Force Cycles for ROI re-render to match the composite pipeline expectations
    try:
        _ensure_cycles_addon()
        scene.render.engine = 'CYCLES'
        _configure_cycles_device(scene)
    except Exception as exc:
        log_error(f"[Engine] Could not configure Cycles for ROI render: {exc}")
        raise

    # Compute aspect-preserving placement of preview inside render frame
    rw, rh = int(scene.render.resolution_x), int(scene.render.resolution_y)
    img_preview = bpy.data.images.load(str(preview_path))
    try:
        pw, ph = int(img_preview.size[0]), int(img_preview.size[1])
    except Exception:
        pw, ph = rw, rh
    pw = max(1, pw); ph = max(1, ph)
    scale = min(rw / pw, rh / ph)
    sw, sh = int(round(pw * scale)), int(round(ph * scale))
    offx, offy = (rw - sw) * 0.5, (rh - sh) * 0.5

    # Adjust ROI bbox from preview space → render space (account for padding)
    fx0 = offx + nx0 * pw * scale
    fx1 = offx + nx1 * pw * scale
    fy0 = offy + ny0 * ph * scale
    fy1 = offy + ny1 * ph * scale
    fnx0 = float(max(0.0, min(1.0, fx0 / rw)))
    fnx1 = float(max(0.0, min(1.0, fx1 / rw)))
    fny0 = float(max(0.0, min(1.0, fy0 / rh)))
    fny1 = float(max(0.0, min(1.0, fy1 / rh)))

    # Film and border
    scene.render.use_border = True
    scene.render.use_crop_to_border = False  # keep full-frame for easy compositing
    scene.render.border_min_x = fnx0
    scene.render.border_max_x = fnx1
    # Blender uses bottom-left origin for border Y
    scene.render.border_min_y = 1.0 - fny1
    scene.render.border_max_y = 1.0 - fny0
    scene.render.film_transparent = True

    # Configure ROI sampling budget for Cycles
    try:
        scene.cycles.samples = int(os.environ.get("SCDL_ROI_CYCLES_SPP", "512"))
        scene.cycles.use_adaptive_sampling = bool(int(os.environ.get("SCDL_ROI_CYCLES_ADAPTIVE", "1")))
        scene.cycles.adaptive_threshold = float(os.environ.get("SCDL_ROI_CYCLES_THRESHOLD", "0.05"))
        # Enable denoising for ROI
        for vl in scene.view_layers:
            try:
                vl.cycles.use_denoising = True
            except Exception:
                pass
    except Exception:
        pass

    # 1) Render a quick full-frame base as background to avoid "unrendered" look
    scene.render.use_border = False
    scene.render.film_transparent = False
    try:
        # Modest base samples; denoise on
        base_spp = int(os.environ.get("SCDL_BASE_CYCLES_SPP", "128"))
        scene.cycles.samples = base_spp
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = float(os.environ.get("SCDL_BASE_CYCLES_THRESHOLD", "0.08"))
        for vl in scene.view_layers:
            try:
                vl.cycles.use_denoising = True
            except Exception:
                pass
    except Exception:
        pass
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(base_path)
    bpy.ops.render.render(write_still=True)

    # 2) ROI heavy pass with transparency
    scene.render.use_border = True
    scene.render.film_transparent = True

    roi_path = (BLEND_DIR / '__roi.png')
    _ensure_parent(roi_path)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(roi_path)
    bpy.ops.render.render(write_still=True)

    # Composite: base + ROI with soft mask (avoid hard rectangle)
    _ensure_parent(final_path)
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear(); nt.links.clear()

    # Load images
    n_base = nt.nodes.new('CompositorNodeImage'); n_base.image = bpy.data.images.load(str(base_path))
    n_roi  = nt.nodes.new('CompositorNodeImage'); n_roi.image  = bpy.data.images.load(str(roi_path))
    # Mask: use user_importance_preview.png (contrast-stretched), scale to render size
    n_mask = nt.nodes.new('CompositorNodeImage')
    if mask_png_path.exists():
        n_mask.image = bpy.data.images.load(str(mask_png_path))
    else:
        # Fallback to preview as neutral mask (will act like no-op)
        n_mask.image = img_preview

    # Scale mask to render size
    n_scale_mask = nt.nodes.new('CompositorNodeScale')
    try:
        n_scale_mask.space = 'ABSOLUTE'
    except Exception:
        pass
    n_scale_mask.inputs['X'].default_value = float(rw)
    n_scale_mask.inputs['Y'].default_value = float(rh)

    # Mix ROI over base using soft mask (per-pixel)
    n_mix = nt.nodes.new('CompositorNodeMixRGB')
    n_mix.blend_type = 'MIX'
    n_mix.use_alpha = True

    n_comp = nt.nodes.new('CompositorNodeComposite')

    # Some Blender versions name sockets differently ("Image" vs "Color1/Color2").
    # Use positional indices for the MixRGB compositor node: [0]=Fac, [1]=A, [2]=B.
    mix_in_a = n_mix.inputs[1]
    mix_in_b = n_mix.inputs[2]
    fac_in   = n_mix.inputs[0]

    nt.links.new(n_base.outputs['Image'], mix_in_a)
    nt.links.new(n_mask.outputs['Image'], n_scale_mask.inputs['Image'])
    nt.links.new(n_roi.outputs['Image'],  mix_in_b)
    nt.links.new(n_scale_mask.outputs['Image'], fac_in)
    nt.links.new(n_mix.outputs['Image'], n_comp.inputs['Image'])

    scene.render.filepath = str(final_path)
    bpy.ops.render.render(write_still=True)

    # Cleanup
    try:
        bpy.data.images.remove(n_base.image)
        bpy.data.images.remove(n_roi.image)
        if n_mask and n_mask.image:
            bpy.data.images.remove(n_mask.image)
    except Exception:
        pass
    for p in (roi_path, base_path):
        try:
            p.unlink()
        except Exception:
            pass

    log_info(f"[OK] Saved final → {final_path}")


def main():
    step = os.environ.get('SCDL_BLEND_STEP', 'preview').strip().lower()
    if step in ('preview', 'step1', '1'):
        render_preview()
    elif step in ('final', 'roi', 'compose', 'step3', '3'):
        render_roi_and_compose()
    else:
        log_warn(f"[WARN] Unknown SCDL_BLEND_STEP={step}; defaulting to 'preview'.")
        render_preview()


if __name__ == "__main__":
    main()
