#!/usr/bin/env python3
# Step 3 — ROI render + composite (no banding + focus-by-mask)
#
# What this does
# 1) Renders a BASE frame (full) and an ROI frame using Render Border (+ Film Transparent).
# 2) Composites ROI over BASE **without banding** using Alpha Over and the ROI alpha.
#    Factor = (scaled importance mask) × (ROI alpha), so outside the border = 0.
# 3) (Default) Blurs the background by mixing a Gaussian blur with
#    (1 - mask) so the ROI stays sharp and the rest falls out of focus.
# 4) Saves the Composite result to out/final.png.
#
# Env vars (sane defaults included):
#   SCDL_PROJECT_DIR        (.)
#   SCDL_BASE_CYCLES_SPP    (128)
#   SCDL_ROI_CYCLES_SPP     (512)
#   SCDL_BASE_DENOISE       (1)
#   SCDL_ROI_DENOISE        (1)
#   SCDL_ROI_BORDER_PAD     (0.02)   # grow ROI on all sides to avoid edge misses
#   SCDL_FOCUS_MODE         (defocus) # off | dof | defocus
#   # DOF (camera) controls
#   SCDL_FOCUS_FSTOP        (1.8)
#   SCDL_FOCUS_DISTANCE     (0.0)
#   SCDL_FOCUS_OBJECT       ("")
#   # Defocus (post) controls
#   SCDL_DEFOCUS_FSTOP      (12.0)   # lower = stronger blur (128 ≈ none)
#   SCDL_DEFOCUS_ZSCALE     (1.0)
#   SCDL_DEFOCUS_MAXBLUR    (32.0)
#
# Inputs from previous steps (in ./out):
#   preview.png (human check only)
#   user_importance_preview.png (grayscale mask @ preview size)
#   roi_bbox.txt  → x0 x1 y0 y1 in [0,1]
# Output:
#   final.png (composited + optional blur)

from __future__ import annotations
import os
from pathlib import Path
import bpy

# ----------------------------- helpers -----------------------------


def _getenv_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return str(v) if v is not None else str(default)


def _getenv_int(name: str, default: int) -> int:
    try:
        return int(float(_getenv_str(name, default)))
    except Exception:
        return int(default)


def _getenv_float(name: str, default: float) -> float:
    try:
        return float(_getenv_str(name, default))
    except Exception:
        return float(default)


def _set_denoise(enabled: bool):
    scn = bpy.context.scene
    if hasattr(scn, 'cycles') and hasattr(scn.cycles, 'use_denoising'):
        scn.cycles.use_denoising = bool(enabled)
    if hasattr(bpy.context.view_layer, 'cycles') and hasattr(bpy.context.view_layer.cycles, 'use_denoising'):
        bpy.context.view_layer.cycles.use_denoising = bool(enabled)


def _read_bbox_norm(path: Path):
    vals = [float(x) for x in Path(path).read_text().strip().split()]
    if len(vals) != 4:
        raise RuntimeError(
            f"Bad bbox in {path}: expected 4 floats (x0 x1 y0 y1)")
    x0, x1, y0, y1 = vals
    # clamp & order
    x0, x1 = max(0.0, min(1.0, min(x0, x1))), max(0.0, min(1.0, max(x0, x1)))
    y0, y1 = max(0.0, min(1.0, min(y0, y1))), max(0.0, min(1.0, max(y0, y1)))
    return x0, x1, y0, y1


def _render_to(filepath: Path, spp: int, denoise: bool, *, use_border=False, border=None, transparent=False):
    scn = bpy.context.scene
    scn.render.use_compositing = False
    scn.render.image_settings.file_format = 'PNG'
    scn.render.filepath = str(filepath)
    scn.render.use_border = bool(use_border)
    scn.render.use_crop_to_border = False
    if border is not None:
        bx0, bx1, by0, by1 = border
        scn.render.border_min_x = float(bx0)
        scn.render.border_max_x = float(bx1)
        scn.render.border_min_y = float(by0)
        scn.render.border_max_y = float(by1)
    scn.render.film_transparent = bool(transparent)

    if hasattr(scn, 'cycles'):
        scn.cycles.samples = int(spp)
    _set_denoise(denoise)

    bpy.ops.render.render(write_still=True, use_viewport=False)


def _enable_camera_dof(fstop: float, focus_dist: float, focus_obj_name: str):
    cam = bpy.context.scene.camera
    if not cam:
        return
    dof = cam.data.dof
    dof.use_dof = True
    dof.aperture_fstop = float(fstop)
    if focus_obj_name:
        obj = bpy.data.objects.get(focus_obj_name)
        if obj is not None:
            dof.focus_object = obj
    if focus_dist > 0:
        dof.focus_distance = float(focus_dist)


def _build_compositor(base_path: Path, roi_path: Path, mask_path: Path, *,
                      add_defocus: bool, defocus_fstop: float, defocus_zscale: float, defocus_maxblur: float):
    scn = bpy.context.scene
    scn.use_nodes = True
    scn.render.use_compositing = True
    nt = scn.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links

    # Images
    n_base = N.new('CompositorNodeImage')
    n_base.image = bpy.data.images.load(str(base_path))
    n_roi = N.new('CompositorNodeImage')
    n_roi.image = bpy.data.images.load(str(roi_path))
    n_mask = N.new('CompositorNodeImage')
    n_mask.image = bpy.data.images.load(str(mask_path))
    if n_mask.image.colorspace_settings:
        n_mask.image.colorspace_settings.name = 'Non-Color'

    # Scale mask to the current render size (keeps alignment with base/roi)
    n_scale = N.new('CompositorNodeScale')
    n_scale.space = 'RENDER_SIZE'
    L.new(n_mask.outputs['Image'], n_scale.inputs['Image'])

    # Split ROI alpha (1 inside border, 0 outside)
    n_sep = N.new('CompositorNodeSepRGBA')
    L.new(n_roi.outputs['Image'], n_sep.inputs['Image'])

    # Factor = scaled_mask * roi_alpha  (prevents band/seams)
    n_mul = N.new('CompositorNodeMath')
    n_mul.operation = 'MULTIPLY'
    L.new(n_scale.outputs['Image'], n_mul.inputs[0])
    L.new(n_sep.outputs['A'],       n_mul.inputs[1])

    # Apply that factor as ROI alpha
    n_seta = N.new('CompositorNodeSetAlpha')
    L.new(n_roi.outputs['Image'], n_seta.inputs['Image'])
    L.new(n_mul.outputs['Value'], n_seta.inputs['Alpha'])

    # Composite without banding: Alpha Over (base bg, masked ROI fg)
    n_over = N.new('CompositorNodeAlphaOver')
    L.new(n_base.outputs['Image'], n_over.inputs[1])
    L.new(n_seta.outputs['Image'], n_over.inputs[2])
    out_socket = n_over.outputs['Image']

    # Optional background blur using blurred composite mixed by (1 - mask)
    if add_defocus:
        n_one = N.new('CompositorNodeValue')
        n_one.outputs[0].default_value = 1.0
        n_sub = N.new('CompositorNodeMath')
        n_sub.operation = 'SUBTRACT'
        L.new(n_one.outputs[0],         n_sub.inputs[0])
        L.new(n_scale.outputs['Image'], n_sub.inputs[1])

        sharp_socket = out_socket
        n_blur = N.new('CompositorNodeBlur')
        n_blur.filter_type = 'FAST_GAUSS'
        n_blur.use_relative = False
        blur_px = max(0, int(round(defocus_maxblur)))
        n_blur.size_x = blur_px
        n_blur.size_y = blur_px
        L.new(sharp_socket, n_blur.inputs['Image'])

        n_mix = N.new('CompositorNodeMixRGB')
        n_mix.blend_type = 'MIX'
        n_mix.use_alpha = False
        L.new(n_sub.outputs['Value'], n_mix.inputs['Fac'])
        L.new(sharp_socket,          n_mix.inputs[1])
        L.new(n_blur.outputs['Image'], n_mix.inputs[2])
        out_socket = n_mix.outputs['Image']

    n_comp = N.new('CompositorNodeComposite')
    L.new(out_socket, n_comp.inputs['Image'])


# ------------------------------ main ------------------------------

def main():
    project_dir = Path(_getenv_str('SCDL_PROJECT_DIR', '.')).resolve()
    out_dir = project_dir / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_png = out_dir / 'user_importance_preview.png'
    bbox_txt = out_dir / 'roi_bbox.txt'
    final_png = out_dir / 'final.png'
    base_tmp = out_dir / '__base.png'
    roi_tmp = out_dir / '__roi.png'

    base_spp = _getenv_int('SCDL_BASE_CYCLES_SPP', 128)
    roi_spp = _getenv_int('SCDL_ROI_CYCLES_SPP', 512)
    base_denoise = bool(_getenv_int('SCDL_BASE_DENOISE', 1))
    roi_denoise = bool(_getenv_int('SCDL_ROI_DENOISE', 1))

    pad = _getenv_float('SCDL_ROI_BORDER_PAD', 0.02)

    # default defocus per request
    focus_mode = _getenv_str('SCDL_FOCUS_MODE', 'defocus').lower()
    f_fstop = _getenv_float('SCDL_FOCUS_FSTOP', 1.8)
    f_dist = _getenv_float('SCDL_FOCUS_DISTANCE', 0.0)
    f_obj = _getenv_str('SCDL_FOCUS_OBJECT', '')

    d_fstop = _getenv_float('SCDL_DEFOCUS_FSTOP', 12.0)
    d_zscale = _getenv_float('SCDL_DEFOCUS_ZSCALE', 1.0)
    d_maxblur = _getenv_float('SCDL_DEFOCUS_MAXBLUR', 32.0)

    # DOF first (applies to both base and roi renders)
    if focus_mode == 'dof':
        _enable_camera_dof(f_fstop, f_dist, f_obj)

    # ROI border
    x0, x1, y0, y1 = _read_bbox_norm(bbox_txt)
    if pad > 0:
        x0 = max(0.0, x0 - pad)
        x1 = min(1.0, x1 + pad)
        y0 = max(0.0, y0 - pad)
        y1 = min(1.0, y1 + pad)

    # Renders
    _render_to(base_tmp, base_spp, base_denoise,
               use_border=False, border=None, transparent=False)
    _render_to(roi_tmp,  roi_spp,  roi_denoise,  use_border=True,
               border=(x0, x1, y0, y1), transparent=True)

    # Compositor (no banding + optional defocus)
    _build_compositor(base_tmp, roi_tmp, mask_png,
                      add_defocus=(focus_mode == 'defocus'),
                      defocus_fstop=d_fstop, defocus_zscale=d_zscale, defocus_maxblur=d_maxblur)

    # Save composite to final.png
    scn = bpy.context.scene
    scn.render.use_compositing = True
    scn.render.image_settings.file_format = 'PNG'
    scn.render.filepath = str(final_png)
    bpy.ops.render.render(write_still=True, use_viewport=False)

    # cleanup
    for p in (base_tmp, roi_tmp):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


if __name__ == '__main__':
    main()
