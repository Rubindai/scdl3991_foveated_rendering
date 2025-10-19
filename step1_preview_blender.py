# step1_preview_blender.py  -- Blender 4.5.2 LTS, RTX 3060, no-fallback
import os
from pathlib import Path
import bpy

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "preview.png"

# ---------- Scene / Engine ----------
scene = bpy.context.scene
scene.render.engine = "CYCLES"

# ---------- Force GPU + OPTIX (no fallback) ----------
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
except KeyError:
    raise RuntimeError("[Step1] Cycles addon not found; cannot render.")

prefs.compute_device_type = "OPTIX"                      # hard requirement
# Populate device list and enable only OPTIX devices
prefs.get_devices()
optix_devices = [d for d in prefs.devices if getattr(d, "type", "") == "OPTIX"]
if not optix_devices:
    raise RuntimeError("[Step1] No OPTIX-capable device detected. "
                       "This configuration forbids falling back to CUDA/CPU.")

# Disable all devices first, then enable OPTIX devices explicitly
for d in prefs.devices:
    d.use = False
for d in optix_devices:
    d.use = True

# Ensure Cycles uses GPU
for s in bpy.data.scenes:
    s.cycles.device = "GPU"

# ---------- Resolution: preserve aspect; snap to multiples of 16 ----------
# (helps downstream DINOv3 ViT-L/16 patching; doesn’t change FOV)
target_short = 448  # preview speed/quality pivot; fixed to avoid silent variability
rx, ry = max(1, scene.render.resolution_x), max(1, scene.render.resolution_y)
if rx >= ry:
    new_y = target_short
    new_x = int(round(target_short * (rx / ry)))
else:
    new_x = target_short
    new_y = int(round(target_short * (ry / rx)))

# snap to multiples of 16 (never zero)
def _snap16(v): return max(16, (v // 16) * 16)
scene.render.resolution_x = _snap16(new_x)
scene.render.resolution_y = _snap16(new_y)
scene.render.resolution_percentage = 100
scene.render.film_transparent = False  # Step2 expects RGB, not alpha

# ---------- Sampling & Denoising (preview-optimized) ----------
c = scene.cycles
c.samples = 16                               # render max samples (preview)
c.use_adaptive_sampling = True               # finish clean regions early
c.adaptive_threshold = 0.04                  # typical ranges documented; fast preview
c.adaptive_min_samples = 4

# Render denoiser: OPTIX (explicit, no auto)
c.use_denoising = True
c.denoiser = "OPTIX"                         # require OPTIX denoiser
# Also enable per-view-layer flag (API stores denoise settings here)
for vl in scene.view_layers:
    vl.cycles.use_denoising = True
    # store passes can help denoiser; cheap to keep on
    vl.cycles.denoising_store_passes = True

# ---------- Variance controls (keep look stable, speed up convergence) ----------
c.caustics_reflective = False
c.caustics_refractive = False
c.blur_glossy = 0.2                           # "Filter Glossy"

# Optional: clamp very hot paths; OFF by default to avoid tonal shifts
if os.getenv("SCDL_PREVIEW_CLAMP", "0") == "1":
    c.sample_clamp_direct = 10.0
    c.sample_clamp_indirect = 10.0

# Deterministic previews help reproducibility in Step2
c.seed = int(os.getenv("SCDL_PREVIEW_SEED", "42"))

# ---------- Color management ----------
# Keep whatever the file uses (AgX is default in Blender 4.x).
# Do NOT force different view transforms; Step2 should see what the user sees.

# ---------- Output ----------
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "8"  # sRGB 8-bit is fine for Step2 saliency
scene.render.filepath = str(OUT_PNG)

# ---------- Render ----------
bpy.ops.render.render(write_still=True)
