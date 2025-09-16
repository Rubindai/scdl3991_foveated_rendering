Pipeline (matches the numbered callouts in the diagram):

1. Base render → preview.
- Run in Blender (Windows). Renders a fast, low-cost preview image for the scene, keeping render aspect. Output: `out/preview.png`.
- Script: `step1_preview_blender.py`.
- Default project: `blender_files/cookie.blend` (override via `BLEND_FILE` or CLI).
- Logging: control Blender-side logging with `SCDL_LOG_MODE` (`stdout`, `file`, `both`) and optional `SCDL_LOG_FILE`.

2. Semantic saliency → foveation mask.
- Run in WSL. DINOv3 computes a content-aware saliency/importance map from the preview and saves both the mask and a normalized ROI bbox.
- Outputs: `out/user_importance.npy`, `out/user_importance_preview.png`, `out/roi_bbox.txt`.
- Script: `step2_dino_mask.py`.

3. Guided re-render and composition → final.
- Run in Blender (Windows). Uses the ROI bbox to selectively re-render masked regions at higher quality and composites those patches over a modest base render.
- Output: `out/final.png`.
- Script: `step3_blender_roi_compose.py`.

Optional: LuxCore single-pass final (WSL).
- Instead of step 3 in Blender, export LuxCore SDL and do a single LuxCore render that consumes the per-pixel importance map (when supported by the build).
- The launcher only runs this branch when `SCDL_USE_WSL_FINAL=1`; set `SCDL_FORCE_LUXCORE_EXPORT=1` if you still want the SDL files without the WSL render.
- Scripts: `step3_optional_luxcore_export.py` (Blender) and `step4_optional_luxcore_render.py` (WSL).
