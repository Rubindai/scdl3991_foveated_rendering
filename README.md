# Scene-Driven Composite Pipeline

## Overview
This repository hosts a hybrid Windows/WSL rendering pipeline that produces a saliency-guided final image. Blender (running on Windows) generates a fast preview, WSL computes a DINO-based importance map, and a selective high-quality Cycles re-render produces the final composite. The full stage-by-stage breakdown lives in `pipeline.md`.

## Prerequisites
- Windows Blender installation accessible from WSL (configure `BLENDER_EXE` in `.scdl.env`).
- WSL environment with Python 3.9+, PyTorch (CUDA optional), torchvision, timm, imageio, and numpy.
- Blend files stored under `blender_files/` (default `cookie.blend`).
- DINOv3 repo + weights (paths configurable via `.scdl.env`; upstream: https://github.com/facebookresearch/dinov3, mirror weights: https://drive.google.com/drive/folders/1Gg6it9iF08VrFRwQVmbzWuUv2o6JTtlH?usp=sharing).

## Setup
1. Review and edit `.scdl.env` to match your environment paths, Blender executable, and stage defaults (preview ROI area, DINO weights).
2. Place your working `.blend` files in `blender_files/` or override `BLEND_FILE` via CLI/environment.
3. Ensure the `dinov3` submodule/clone and weights are present as referenced by `.scdl.env`.

## Running the Pipeline
```bash
# Preview → DINO mask → Blender ROI composite
./run_windows_blender_wsl.sh
```
- Pass a different blend file with `./run_windows_blender_wsl.sh blender_files/your_scene.blend`.
- Each stage logs to stdout and/or `out/scdl_pipeline.log` depending on `SCDL_LOG_MODE` / `SCDL_LOG_FILE`.
- Persistent defaults live in `.scdl.env`; adjust logging, device selection, and render quality there.

## Logging
All Python stages (Blender preview/final and DINO mask) honor:
- `SCDL_LOG_MODE`: `stdout`, `file`, or `both` (default).
- `SCDL_LOG_FILE`: override log path; defaults to `out/scdl_pipeline.log` in the project root.

## Directory Layout
- `blender_files/` – source .blend projects.
- `out/` – preview/final renders, masks, logs.
- `pipeline.md` – detailed description of each pipeline stage.

## Troubleshooting
- Missing preview? Re-run the preview stage or check Blender logs in `out/scdl_pipeline.log`.
- DINO import errors? Verify `dinov3` path/weights in `.scdl.env` and Python dependencies in WSL.

For stage-by-stage explanations and tunable parameters, refer to `pipeline.md` and inline comments within the scripts.
