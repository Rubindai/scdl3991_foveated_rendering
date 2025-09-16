# Scene-Driven Composite LuxCore Pipeline

## Overview
This repository hosts a hybrid Windows/WSL rendering pipeline that produces a saliency-guided final image. Blender (running on Windows) generates a fast preview, WSL computes a DINO-based importance map, and a selective high-quality re-render produces the final composite. An optional LuxCore branch exports SDL from Blender and performs the final render directly under WSL. The full stage-by-stage breakdown now lives in `pipeline.md`.

## Prerequisites
- Windows Blender installation accessible from WSL (configure `BLENDER_EXE` in `.scdl.env`).
- WSL environment with Python 3.9+, PyTorch (CUDA optional), torchvision, timm, imageio, numpy, and pyluxcore (for the optional final render).
- Blend files stored under `blender_files/` (default `cookie.blend`).
- DINOv3 repo + weights (paths configurable via `.scdl.env`; upstream: https://github.com/facebookresearch/dinov3, mirror weights: https://drive.google.com/drive/folders/1Gg6it9iF08VrFRwQVmbzWuUv2o6JTtlH?usp=sharing).

## Setup
1. Review and edit `.scdl.env` to match your environment paths, Blender executable, and stage defaults (preview ROI area, DINO weights, `CONDA_ENV`, LuxCore toggles).
2. Place your working `.blend` files in `blender_files/` or override `BLEND_FILE` via CLI/environment.
3. Ensure the `dinov3` submodule/clone and weights are present as referenced by `.scdl.env`.
4. (Optional) Install BlendLuxCore inside Blender if you plan to use the LuxCore export/render path.

## Running the Pipeline
```bash
# Default: preview → DINO mask → Blender ROI composite
./run_windows_blender_wsl.sh

# Force LuxCore final render and keep the exported SDL
SCDL_USE_WSL_FINAL=1 ./run_windows_blender_wsl.sh

# Only export LuxCore SDL (no final render)
SCDL_FORCE_LUXCORE_EXPORT=1 ./run_windows_blender_wsl.sh
```
- Pass a different blend file with `./run_windows_blender_wsl.sh blender_files/your_scene.blend`.
- Each stage logs to stdout and/or `out/scdl_pipeline.log` depending on `SCDL_LOG_MODE` / `SCDL_LOG_FILE`.
- Persistent defaults live in `.scdl.env`; toggle LuxCore export with `SCDL_USE_WSL_FINAL` / `SCDL_FORCE_LUXCORE_EXPORT` and set the WSL Python environment via `CONDA_ENV`.

## Logging
All Python stages (Blender preview/final, DINO mask, LuxCore export/render) honor:
- `SCDL_LOG_MODE`: `stdout`, `file`, or `both` (default).
- `SCDL_LOG_FILE`: override log path; defaults to `out/scdl_pipeline.log` in the project root.

## Directory Layout
- `blender_files/` – source .blend projects.
- `out/` – preview/final renders, masks, logs.
- `export/` – LuxCore SDL output (render.cfg/scene.scn) when enabled.
- `pipeline.md` – detailed description of each pipeline stage.

## Troubleshooting
- Missing preview? Re-run the preview stage or check Blender logs in `out/scdl_pipeline.log`.
- DINO import errors? Verify `dinov3` path/weights in `.scdl.env` and Python dependencies in WSL.
- LuxCore failures? Confirm BlendLuxCore add-on is installed for export and pyluxcore is available in WSL.

For stage-by-stage explanations and tunable parameters, refer to `pipeline.md` and inline comments within the scripts.
