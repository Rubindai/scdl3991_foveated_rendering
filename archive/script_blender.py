#!/usr/bin/env python
"""
Foveated pipeline run from Windows Blender, with optional BlendLuxCore.

Overview:
  1) Render a low-cost preview at downscaled resolution (fast; Eevee)
  2) Run the DINO script to compute a saliency mask
  3) Convert the mask to a normalized ROI bounding box
  4) Render a high-quality ROI in one pass (film transparent + border). If
     BlendLuxCore is available, prefer its adaptive sampling (dynamic SPP).
  5) Composite the ROI over the base (preview or low-spp base) and save.

Notes:
  - Run Blender on Windows for GPU rendering (Cycles/OptiX). The DINO step
    is invoked via script_pipeline_dino.py, which self-bounces to WSL Python
    when needed, so no WSL-specific logic is required here.
  - If LuxCore is not installed, the script falls back to Cycles with border
    rendering for the ROI. This keeps the pipeline working everywhere.
  - NumPy is optional: the mask is read from .npy if present, else PNG.
"""

# ======================= CONFIG (edit these) =======================
import subprocess
import os
import shutil
import bpy
import time
import sys
import platform
from typing import Optional, Tuple, List

def proj_path(rel: str) -> str:
    """Resolve a path relative to the .blend file directory (Linux/headless safe)."""
    rel = rel.lstrip("/\\")
    return bpy.path.abspath(f"//{rel}")

OUT_FINAL = proj_path("out/final_foveated.png")

# Engine and quality configuration
# Prefer LuxCore for ROI (dynamic SPP). Enforce LuxCore availability.
PREFER_LUXCORE = True
REQUIRE_LUXCORE = True  # fail if LuxCore isn't available (no Cycles fallback)

# Use Eevee for preview to keep it snappy. Base can be preview or a low-spp render.
USE_EEVEE_FOR_PREVIEW = True
USE_PREVIEW_AS_BASE = True   # If True, composite ROI over the preview directly

# Cycles quality knobs (fallback mode)
LOW_SPP = 16
HIGH_SPP = 384

# Cycles adaptive sampling
USE_ADAPTIVE_FOR_BASE = True
BASE_ADAPTIVE_THRESHOLD = 0.05  # higher = faster, noisier
ROI_ADAPTIVE_THRESHOLD = 0.01   # lower = cleaner in ROI

# Tile sizes (best-effort; Cycles may ignore depending on device)
CPU_TILE_SIZE = 32
GPU_TILE_SIZE = 256

# Preview (DINO input) rendered by Blender
PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540
PREVIEW_SPP = 16
# must match IMG_PATH in script_pipeline_dino.py
PREVIEW_PATH = proj_path("out/preview.png")

# LuxCore halting (to avoid infinite sessions). Applied when engine_hint='LUXCORE'
# and before each LuxCore render in this script.
ROI_MAX_SPP = 1500           # cap samples for ROI pass
ROI_MAX_TIME = 60            # seconds cap for ROI pass
ROI_NOISE_THRESHOLD = 0.02   # optional noise-based halt (lower = cleaner)
BASE_MAX_SPP = 128           # if a LuxCore base render is used
BASE_MAX_TIME = 20
BASE_NOISE_THRESHOLD = 0.05

# Preview halting if preview must run on LuxCore (when Eevee unavailable)
PREVIEW_MAX_SPP = 64
PREVIEW_MAX_TIME = 5
PREVIEW_NOISE_THRESHOLD = 0.10

def resolve_ext_python() -> str:
    """Pick a Python to run the DINO script.
    - On Windows (recommended), use Blender's own Python so the script can
      re-exec into WSL automatically if needed.
    - On non-Windows, use system Python as a fallback.
    """
    if platform.system() == "Windows":
        return sys.executable or (shutil.which("python") or "python")
    found = shutil.which("python") or shutil.which("python3")
    if found:
        return found
    return bpy.app.binary_path_python

# Paths for DINO
DINOV3_PYTHON = resolve_ext_python()
DINOV3_SCRIPT = proj_path("script_pipeline_dino.py")

# DINO outputs we read inside Blender
MASK_PNG_PATH = proj_path("out/mask.png")
MASK_NPY_PATH = proj_path("out/mask.npy")

# ROI shaping
THRESHOLD = 0.5     # fallback cutoff in [0,1] if auto-threshold unavailable
PAD_FRAC = 0.02    # normalized padding around bbox (2%)
ROI_AREA_BUDGET = 0.20  # target fraction of pixels for ROI when auto-thresholding
# ================================================================


def ensure_dir(p):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)


def enum_engines() -> List[str]:
    """Return available render engines identifiers (e.g., 'BLENDER_EEVEE', 'CYCLES', 'LUXCORE')."""
    try:
        items = bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items
        return [it.identifier for it in items]
    except Exception:
        return []


def set_engine(engine_id: str) -> bool:
    """Set scene render engine if available. Returns True on success."""
    if engine_id not in enum_engines():
        return False
    try:
        bpy.context.scene.render.engine = engine_id
        return True
    except Exception:
        return False


def configure_cycles_devices(prefer_gpu=True):
    """Enable GPU+CPU when available; fallback to CPU-only if needed.
    Returns a dict with selected mode/backend/devices for logging.
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences

    # Try GPU backends in order of preference
    backends = ["OPTIX", "CUDA", "HIP",
                "METAL", "ONEAPI"] if prefer_gpu else []
    selected_backend = None
    for be in backends:
        try:
            prefs.compute_device_type = be
            try:
                prefs.get_devices()
            except Exception:
                pass
            gpus = [d for d in prefs.devices if d.type != 'CPU']
            if gpus:
                selected_backend = be
                # Enable all GPUs and also CPU for combined rendering
                for d in prefs.devices:
                    d.use = True
                scene.cycles.device = 'GPU'
                break
        except Exception:
            continue

    if not selected_backend:
        # Fallback: CPU only
        scene.cycles.device = 'CPU'
        try:
            prefs.compute_device_type = 'NONE'
            try:
                prefs.get_devices()
            except Exception:
                pass
            for d in prefs.devices:
                d.use = (d.type == 'CPU')
        except Exception:
            pass

    # Tile size hint
    try:
        if scene.cycles.device == 'GPU':
            scene.cycles.tile_size = GPU_TILE_SIZE
        else:
            scene.cycles.tile_size = CPU_TILE_SIZE
    except Exception:
        pass

    # Persistent data to reuse BVH/shaders across renders
    try:
        scene.render.use_persistent_data = True
    except Exception:
        pass

    # Build info for logging
    try:
        devices = []
        for d in prefs.devices:
            if d.use:
                devices.append(
                    f"{getattr(d, 'name', 'unknown')}({getattr(d, 'type', '?')})")
        info = {
            'mode': scene.cycles.device,
            'backend': selected_backend or 'NONE',
            'devices': devices,
        }
    except Exception:
        info = {'mode': scene.cycles.device,
                'backend': 'UNKNOWN', 'devices': []}
    return info


def log_luxcore_diagnostics(prefix: str = "[LuxCore]") -> None:
    try:
        engines = enum_engines()
    except Exception:
        engines = []
    print(f"{prefix} Engines available: {engines}")
    # Try pyluxcore import to surface binary issues
    try:
        import importlib  # type: ignore
        pylx = importlib.import_module('pyluxcore')
        ver = getattr(pylx, 'Version', lambda: 'unknown')()
        print(f"{prefix} pyluxcore import: OK (Version={ver})")
    except Exception as e:
        print(f"{prefix} pyluxcore import: FAILED ({e})")


def configure_luxcore(scene) -> bool:
    """Use LuxCore if already available. No add-on enable attempts required on 4.5.
    Returns True when engine is set to 'LUXCORE'.
    """
    # Respect preselected engine (e.g., started with -E LUXCORE)
    if getattr(scene.render, 'engine', None) == 'LUXCORE':
        return True

    # Try to set immediately if registered
    if 'LUXCORE' in enum_engines():
        try:
            scene.render.engine = 'LUXCORE'
            return True
        except Exception:
            pass

    # Poll briefly for registration (extensions can finalize after load)
    for _ in range(30):
        if 'LUXCORE' in enum_engines():
            try:
                scene.render.engine = 'LUXCORE'
                return True
            except Exception:
                break
        time.sleep(0.1)

    # Diagnostics
    log_luxcore_diagnostics()
    return False


def set_luxcore_halting(scene,
                        spp: int | None = None,
                        time_s: int | None = None,
                        noise_threshold: float | int | None = None) -> None:
    """Set BlendLuxCore halt conditions using official property paths.
    Applies to scene and all view layers to ensure exporter picks it up.
    - spp: max samples (int)
    - time_s: max time in seconds (int)
    - noise_threshold: either 0..1 float or 0..255 int (converted to 0..255)
    """
    try:
        def _apply(halt_pg):
            if halt_pg is None:
                return
            halt_pg.enable = True
            # samples
            if spp is not None:
                halt_pg.use_samples = True
                halt_pg.samples = int(max(2, spp))
            else:
                halt_pg.use_samples = False
            # time
            if time_s is not None:
                halt_pg.use_time = True
                halt_pg.time = int(max(1, time_s))
            else:
                halt_pg.use_time = False
            # noise threshold (0..255 integer scale in BlendLuxCore)
            if noise_threshold is not None:
                v = noise_threshold
                if isinstance(v, float) and v <= 1.0:
                    v = int(max(0, min(255, round(v * 256))))
                else:
                    v = int(max(0, min(255, v)))
                halt_pg.use_noise_thresh = True
                halt_pg.noise_thresh = v
                # Reasonable defaults for checks
                try:
                    halt_pg.noise_thresh_warmup = max(16, min(256, halt_pg.noise_thresh_warmup))
                    halt_pg.noise_thresh_step = max(16, min(256, halt_pg.noise_thresh_step))
                except Exception:
                    pass
            else:
                halt_pg.use_noise_thresh = False

        lc = getattr(scene, 'luxcore', None)
        if lc is not None:
            _apply(getattr(lc, 'halt', None))
        # Apply to all view layers too (exporter may take per-layer settings)
        for vl in scene.view_layers:
            vl_lc = getattr(vl, 'luxcore', None)
            if vl_lc is not None:
                _apply(getattr(vl_lc, 'halt', None))
    except Exception:
        pass


def pick_eevee_id() -> Optional[str]:
    """Return the Eevee engine identifier present in this Blender build."""
    engines = set(enum_engines())
    if 'BLENDER_EEVEE_NEXT' in engines:
        return 'BLENDER_EEVEE_NEXT'
    if 'BLENDER_EEVEE' in engines:
        return 'BLENDER_EEVEE'
    return None


def render_preview(scene, width, height, spp, out_path):
    """Render a low-res preview at explicit size; then restore scene settings.
    Uses Eevee by default for speed, but falls back if unavailable.
    """
    ensure_dir(out_path)
    orig = dict(rx=scene.render.resolution_x,
                ry=scene.render.resolution_y,
                rp=scene.render.resolution_percentage,
                engine=scene.render.engine,
                spp=scene.cycles.samples,
                use_border=scene.render.use_border,
                crop=scene.render.use_crop_to_border,
                film=scene.render.film_transparent)
    # Preview settings
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.film_transparent = False
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    # Prefer Eevee for preview regardless of LuxCore requirement; switch back later.
    eevee_id = pick_eevee_id()
    if USE_EEVEE_FOR_PREVIEW and eevee_id and set_engine(eevee_id):
        pass
    else:
        # Fallback to Cycles if available, otherwise keep current engine (could be LuxCore)
        if 'CYCLES' in enum_engines() and set_engine('CYCLES'):
            scene.cycles.samples = int(spp)
        else:
            # If we are on LuxCore for preview, enforce short halting to keep it quick
            if getattr(scene.render, 'engine', '') == 'LUXCORE':
                set_luxcore_halting(scene,
                                    spp=PREVIEW_MAX_SPP,
                                    time_s=PREVIEW_MAX_TIME,
                                    noise_threshold=PREVIEW_NOISE_THRESHOLD)
    scene.render.filepath = out_path
    t0 = time.time()
    print(f"[Preview] Engine={scene.render.engine} size={width}x{height}")
    bpy.ops.render.render(write_still=True)
    print(f"[Preview] Wrote {out_path} in {time.time()-t0:.2f}s")
    # Restore
    try:
        scene.render.engine = orig["engine"]
    except Exception:
        pass
    scene.render.resolution_x = orig["rx"]
    scene.render.resolution_y = orig["ry"]
    scene.render.resolution_percentage = orig["rp"]
    scene.cycles.samples = orig["spp"]
    scene.render.use_border = orig["use_border"]
    scene.render.use_crop_to_border = orig["crop"]
    scene.render.film_transparent = orig["film"]


def run_dinov3():
    """Run the DINO script via external Python.
    On Windows, the script will re-exec inside WSL automatically to use
    WSL Python + Torch. Paths are passed via environment.
    """
    env = os.environ.copy()
    env.setdefault('SCDL_PREVIEW', PREVIEW_PATH)
    env.setdefault('SCDL_OUT_DIR', os.path.dirname(PREVIEW_PATH))
    env.setdefault('SCDL_DINO_REPO', proj_path('dinov3'))
    env.setdefault('SCDL_DINO_WEIGHTS', proj_path('dinov3_vits16_pretrain_lvd1689m-08c60483.pth'))
    # Ensure the WSL Python path is known to the DINO script for bouncing
    env.setdefault('SCDL_WSL_PYTHON', '/home/rubin/anaconda3/bin/python')

    cmd = [DINOV3_PYTHON, DINOV3_SCRIPT]
    print("[DINO] Running:", " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise RuntimeError("dinov3_mask.py failed.")
    print(r.stdout)


def bbox_norm_from_mask_png(mask_png_path, thr=0.5, pad_frac=0.02) -> Optional[Tuple[float, float, float, float]]:
    """
    Read a grayscale (or RGB) mask PNG via Blender's image loader (no numpy).
    Returns normalized bbox (nx0,nx1,ny0,ny1) in [0,1]. Uses the R channel.
    """
    if not os.path.isfile(mask_png_path):
        raise FileNotFoundError(f"Mask PNG not found: {mask_png_path}")
    img = bpy.data.images.load(mask_png_path)
    W, H = img.size[0], img.size[1]   # width, height
    # Read pixel buffer in one shot for speed
    total = 4 * W * H
    buf = [0.0] * total
    try:
        img.pixels.foreach_get(buf)
    except Exception:
        # Fallback to accessing .pixels directly
        buf = list(img.pixels)

    found = False
    min_x, max_x = W, -1
    min_y, max_y = H, -1

    # Scan red channel only; linear scan is faster in Python than nested loops
    # Layout: RGBA floats, row-major, bottom-to-top
    # pixel_index = i // 4, x = pixel_index % W, y = pixel_index // W
    for i in range(0, total, 4):
        r = buf[i]
        if r >= thr:
            found = True
            p = i // 4
            x = p % W
            y = p // W
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

    # Free the image datablock
    bpy.data.images.remove(img)

    if not found:
        return None

    # Convert to normalized [0,1] + padding
    nx0 = min_x / W
    nx1 = (max_x + 1) / W
    ny0 = min_y / H
    ny1 = (max_y + 1) / H

    nx0 = max(0.0, nx0 - pad_frac)
    nx1 = min(1.0, nx1 + pad_frac)
    ny0 = max(0.0, ny0 - pad_frac)
    ny1 = min(1.0, ny1 + pad_frac)
    return (nx0, nx1, ny0, ny1)


def bbox_norm_from_mask(mask_npy_path, mask_png_path, thr=0.5, pad_frac=0.02) -> Optional[Tuple[float, float, float, float]]:
    """Prefer fast NumPy .npy mask when available inside Blender's Python.
    Falls back to PNG via Blender's image loader when NumPy is unavailable.
    The mask is expected to be HxW in [0,1].
    """
    # Try NumPy path
    try:
        import numpy as np  # type: ignore
        if os.path.isfile(mask_npy_path):
            m = np.load(mask_npy_path)
            if m.ndim == 3:
                # Reduce to single channel if needed
                m = m[..., 0]
            H, W = int(m.shape[0]), int(m.shape[1])
            ys, xs = np.where(m >= float(thr))
            if xs.size == 0:
                return None
            min_x = int(xs.min())
            max_x = int(xs.max())
            min_y = int(ys.min())
            max_y = int(ys.max())
            nx0 = max(0.0, min_x / W - pad_frac)
            nx1 = min(1.0, (max_x + 1) / W + pad_frac)
            ny0 = max(0.0, min_y / H - pad_frac)
            ny1 = min(1.0, (max_y + 1) / H + pad_frac)
            return (nx0, nx1, ny0, ny1)
    except Exception:
        pass
    # PNG fallback (no NumPy required)
    return bbox_norm_from_mask_png(mask_png_path, thr=thr, pad_frac=pad_frac)


def render_to_png(scene, spp, out_path, use_border=False, border_norm=None, film_transparent=False,
                  use_adaptive=False, adaptive_threshold=None, engine_hint: Optional[str] = None):
    """Generic render helper for any engine.
    - For Cycles, SPP/adaptive settings are applied here.
    - For LuxCore, per-spp/adaptive are expected to be set in the add-on; this
      function still sets border and film options at the Blender level.
    """
    # Switch engine if requested
    if engine_hint:
        set_engine(engine_hint)

    # Apply per-engine knobs
    prev_tf = None
    ip = None
    if bpy.context.scene.render.engine == 'CYCLES':
        bpy.context.scene.cycles.samples = int(spp)
        try:
            bpy.context.scene.cycles.use_adaptive_sampling = bool(use_adaptive)
            if adaptive_threshold is not None:
                bpy.context.scene.cycles.adaptive_threshold = float(adaptive_threshold)
        except Exception:
            pass
    elif bpy.context.scene.render.engine == 'LUXCORE':
        # Set LuxCore halting to ensure finite sessions
        # Choose limits based on whether this is a base or ROI render (heuristic: border implies ROI)
        is_roi = bool(use_border and border_norm)
        if is_roi:
            set_luxcore_halting(scene,
                                spp=ROI_MAX_SPP,
                                time_s=ROI_MAX_TIME,
                                noise_threshold=ROI_NOISE_THRESHOLD)
        else:
            set_luxcore_halting(scene,
                                spp=BASE_MAX_SPP,
                                time_s=BASE_MAX_TIME,
                                noise_threshold=BASE_NOISE_THRESHOLD)
        # Ensure RGBA output when we expect transparency; BlendLuxCore needs camera imagepipeline flag
        cam = getattr(scene, 'camera', None)
        cam_data = getattr(cam, 'data', None) if cam else None
        ip = getattr(getattr(cam_data, 'luxcore', None), 'imagepipeline', None)
        prev_tf = None
        try:
            if film_transparent and ip is not None and hasattr(ip, 'transparent_film'):
                prev_tf = ip.transparent_film
                ip.transparent_film = True
        except Exception:
            pass

    # Border and film options (engine-agnostic)
    scene.render.use_border = bool(use_border)
    scene.render.use_crop_to_border = False
    if use_border and border_norm:
        nx0, nx1, ny0, ny1 = border_norm
        # Blender uses bottom-left origin for border:
        scene.render.border_min_x = nx0
        scene.render.border_max_x = nx1
        scene.render.border_min_y = 1.0 - ny1
        scene.render.border_max_y = 1.0 - ny0
    scene.render.film_transparent = bool(film_transparent)
    scene.render.filepath = out_path
    t0 = time.time()
    # Best-effort ETA: print initial settings; exact ETA not accessible in background.
    if scene.render.engine == 'LUXCORE':
        try:
            hp = scene.luxcore.halt
            print(f"[LuxCore] Halting: enabled={hp.enable} use_samples={hp.use_samples} samples={hp.samples} "
                  f"use_time={hp.use_time} time={hp.time}s use_noise={hp.use_noise_thresh} noise={hp.noise_thresh}")
        except Exception:
            print("[LuxCore] Halting: (could not read properties)")
    print(f"[Render {scene.render.engine}] Starting → {out_path}")
    bpy.ops.render.render(write_still=True)
    # Restore camera pipeline transparent flag if we toggled it
    try:
        if bpy.context.scene.render.engine == 'LUXCORE' and prev_tf is not None and ip is not None:
            ip.transparent_film = prev_tf
    except Exception:
        pass
    dt = time.time() - t0
    print(f"[Render {scene.render.engine}] Done {out_path} in {dt:.2f}s")


def composite_base_roi_to_file(scene, base_path, roi_path, out_path):
    """
    Build a tiny Compositor graph:
      [Image base] + [Image roi] -> [Alpha Over] -> [Composite]
    Then render once to write out_path (no 3D render needed).
    """
    ensure_dir(out_path)
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    nt.links.clear()

    n_base = nt.nodes.new("CompositorNodeImage")
    n_base.image = bpy.data.images.load(base_path)
    n_roi = nt.nodes.new("CompositorNodeImage")
    n_roi.image = bpy.data.images.load(roi_path)
    n_over = nt.nodes.new("CompositorNodeAlphaOver")
    n_over.premul = 1
    n_comp = nt.nodes.new("CompositorNodeComposite")

    nt.links.new(n_base.outputs['Image'], n_over.inputs[1])   # background
    nt.links.new(n_roi.outputs['Image'],  n_over.inputs[2])   # foreground
    nt.links.new(n_over.outputs['Image'], n_comp.inputs['Image'])

    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = out_path
    # This "render" only executes the compositor using the two Image nodes
    bpy.ops.render.render(write_still=True)

    # Clean loaded images (optional)
    bpy.data.images.remove(n_base.image)
    bpy.data.images.remove(n_roi.image)


def main():
    ensure_dir(OUT_FINAL)
    scene = bpy.context.scene
    scene.render.resolution_percentage = 100
    lux_ok = False
    if PREFER_LUXCORE:
        lux_ok = configure_luxcore(scene)
        if lux_ok:
            print("[Engine] Using LuxCore for ROI (dynamic adaptive sampling).")
        else:
            if REQUIRE_LUXCORE:
                raise RuntimeError("LuxCore/BlendLuxCore not available in this Blender session. See diagnostics above.")
            print("[Engine] LuxCore not available; falling back to Cycles.")

    if not lux_ok and not REQUIRE_LUXCORE:
        # Prepare Cycles (device auto-config and logging)
        set_engine('CYCLES')
        scene.cycles.use_adaptive_sampling = False  # default; set per render
        dev_info = configure_cycles_devices(prefer_gpu=True)
        print(f"[Cycles] Mode={dev_info['mode']} Backend={dev_info['backend']} Devices={dev_info['devices']}")
        try:
            dn = getattr(scene.cycles, 'denoiser', 'N/A')
            print(f"[Cycles] Denoiser={dn}")
        except Exception:
            pass
        print(f"[Settings] Base: spp={LOW_SPP} adaptive={USE_ADAPTIVE_FOR_BASE} thr={BASE_ADAPTIVE_THRESHOLD}; "
              f"ROI: spp={HIGH_SPP} adaptive=True thr={ROI_ADAPTIVE_THRESHOLD}")

    # Avoid requiring OpenImageDenoiser: disable denoising or switch to NLM
    try:
        # Prefer no denoising to keep compatibility
        if hasattr(scene.cycles, 'denoiser'):
            # Use built-in NLM denoiser to avoid OIDN dependency
            try:
                scene.cycles.denoiser = 'NLM'
            except Exception:
                pass
        # Disable view layer denoising if enabled
        for vl in scene.view_layers:
            try:
                vl.cycles.use_denoising = False
            except Exception:
                pass
        # Older properties (best-effort)
        for attr in ('use_denoising', 'use_preview_denoising'):
            if hasattr(scene.cycles, attr):
                try:
                    setattr(scene.cycles, attr, False)
                except Exception:
                    pass
    except Exception:
        pass

    # Final resolution comes from the .blend
    # (No dependency on numpy here.)

    # 1) Low-res preview for DINO (Eevee if available)
    render_preview(scene, PREVIEW_WIDTH, PREVIEW_HEIGHT,
                   PREVIEW_SPP, PREVIEW_PATH)

    # 2) Run DINO (dinov3_mask.py reads PREVIEW_PATH via its constants and writes mask.png)
    run_dinov3()
    print("Dino completed")
    # 3) Load mask and compute threshold dynamically to hit ROI_AREA_BUDGET when possible
    thr_used = THRESHOLD
    try:
        import numpy as _np  # type: ignore
        if os.path.isfile(MASK_NPY_PATH):
            m = _np.load(MASK_NPY_PATH)
            area = float(max(0.01, min(0.95, ROI_AREA_BUDGET)))
            thr_used = float(_np.quantile(m.astype(_np.float32), 1.0 - area))
            # Clamp to sensible range
            thr_used = float(max(0.01, min(0.99, thr_used)))
            print(f"[Mask] Auto-threshold from ROI_AREA_BUDGET={ROI_AREA_BUDGET:.0%} → thr={thr_used:.3f}")
    except Exception:
        print(f"[Mask] NumPy unavailable in Blender; using fallback THRESHOLD={THRESHOLD}")

    # Compute normalized ROI bbox at chosen threshold
    bbox_norm = bbox_norm_from_mask(
        MASK_NPY_PATH, MASK_PNG_PATH, thr=thr_used, pad_frac=PAD_FRAC)
    if bbox_norm is None:
        raise RuntimeError(
            f"Mask produced empty ROI at THR={thr_used:.3f}. Increase AREA_BUDGET in script_pipeline_dino.py or adjust ROI_AREA_BUDGET/THRESHOLD.")
    nx0, nx1, ny0, ny1 = bbox_norm
    print(
        f"[Mask] BBox norm: x=({nx0:.3f},{nx1:.3f}) y=({ny0:.3f},{ny1:.3f}) area≈{(nx1-nx0)*(ny1-ny0):.1%}")

    # Temp files (next to the .blend)
    base_path = bpy.path.abspath("//__fovea_base.png")
    roi_path = bpy.path.abspath("//__fovea_roi.png")

    # 4) Render
    if USE_PREVIEW_AS_BASE:
        # One heavy pass: ROI only (film transparent) over the preview image
        if lux_ok:
            render_to_png(scene, HIGH_SPP, roi_path, use_border=True,
                          border_norm=bbox_norm, film_transparent=True,
                          use_adaptive=True, adaptive_threshold=ROI_ADAPTIVE_THRESHOLD,
                          engine_hint='LUXCORE')
        elif not REQUIRE_LUXCORE:
            # Fallback only if LuxCore not required; otherwise we already raised
            render_to_png(scene, HIGH_SPP, roi_path, use_border=True,
                          border_norm=bbox_norm, film_transparent=True,
                          use_adaptive=True, adaptive_threshold=ROI_ADAPTIVE_THRESHOLD,
                          engine_hint='CYCLES')

        # Composite ROI over preview
        composite_base_roi_to_file(scene, PREVIEW_PATH, roi_path, OUT_FINAL)
        try:
            os.remove(roi_path)
        except Exception:
            pass
    else:
        # Two-pass: quick base (never LuxCore) + heavy ROI (LuxCore)
        eevee_id = pick_eevee_id()
        if USE_EEVEE_FOR_PREVIEW and eevee_id and set_engine(eevee_id):
            render_to_png(scene, PREVIEW_SPP, base_path, use_border=False,
                          border_norm=None, film_transparent=False,
                          engine_hint=eevee_id)
        else:
            # Fall back to Cycles for base (fast), not LuxCore
            render_to_png(scene, LOW_SPP, base_path, use_border=False,
                          border_norm=None, film_transparent=False,
                          use_adaptive=USE_ADAPTIVE_FOR_BASE, adaptive_threshold=BASE_ADAPTIVE_THRESHOLD,
                          engine_hint='CYCLES')

        # ROI heavy pass
        if lux_ok:
            render_to_png(scene, HIGH_SPP, roi_path, use_border=True,
                          border_norm=bbox_norm, film_transparent=True,
                          use_adaptive=True, adaptive_threshold=ROI_ADAPTIVE_THRESHOLD,
                          engine_hint='LUXCORE')
        elif not REQUIRE_LUXCORE:
            render_to_png(scene, HIGH_SPP, roi_path, use_border=True,
                          border_norm=bbox_norm, film_transparent=True,
                          use_adaptive=True, adaptive_threshold=ROI_ADAPTIVE_THRESHOLD,
                          engine_hint='CYCLES')

        # Composite and cleanup
        composite_base_roi_to_file(scene, base_path, roi_path, OUT_FINAL)
        try:
            os.remove(base_path)
            os.remove(roi_path)
        except Exception:
            pass

    print(f"[OK] Saved final foveated image → {OUT_FINAL}")


if __name__ == "__main__":
    main()
