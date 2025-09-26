Pipeline (matches the numbered callouts in the diagram):

1. Preview render → `out/preview.png`
- Runs in Windows Blender and produces a quick, low-sample render that preserves the final aspect ratio.
- Inputs: scene `.blend` file (defaults to `blender_files/cookie.blend`, override via CLI or `BLEND_FILE`), runtime overrides such as `SCDL_CYCLES_DEVICE`.
- Outputs: `out/preview.png` plus log entries controlled by `SCDL_LOG_MODE`/`SCDL_LOG_FILE`.
- Script: `step1_preview_blender.py` → `blender_steps.render_preview()`.

2. Saliency analysis → ROI mask & bbox
- Runs in WSL. DINOv3 analyses the preview to generate a per-pixel importance mask and a normalized ROI bounding box (0–1 coordinates).
- Inputs: `out/preview.png`, DINO config/weights defined in `.scdl.env` or environment variables.
- Outputs:
  - `out/user_importance.npy` (numeric mask)
  - `out/user_importance_preview.png` (mask overlay for visualization)
  - `out/roi_bbox.txt` (normalized ROI bounds)
  - `out/mask_overlay.png` (preview image tinted by the mask for sanity checks)
- Script: `step2_dino_mask.py`.

3. ROI re-render & composite → `out/final.png`
- Runs in Windows Blender. Renders a modest full-frame “base” and a high-quality ROI pass, then composites the ROI over the base using the mask from Step 2.
- Inputs: `out/preview.png`, `out/roi_bbox.txt`, `out/user_importance_preview.png` (if present), plus scene `.blend` file and quality knobs (`SCDL_ROI_*`, `SCDL_BASE_CYCLES_*`).
- Output: `out/final.png`.
- Script: `step3_blender_roi_compose.py`.

Optional branch: LuxCore export & WSL render
- Replaces the Blender composite with a LuxCore FileSaver export and an optional WSL final render when `SCDL_USE_WSL_FINAL=1` (set `SCDL_FORCE_LUXCORE_EXPORT=1` to export without rendering).
- Step 3 (export) inputs: scene `.blend`, BlendLuxCore add-on, mask artifacts if referenced by the export script. Outputs: `export/render.cfg`, `export/scene.scn`.
- Step 4 (WSL render) inputs: exported SDL files, `out/user_importance.npy`, LuxCore configuration/overrides. Output: `out/final.png` (only when the WSL render runs).
- Scripts: `step3_optional_luxcore_export.py` (Windows Blender) and `step4_optional_luxcore_render.py` (WSL).
