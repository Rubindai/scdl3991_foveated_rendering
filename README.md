# Scene-Driven Composite Pipeline

## Overview
This repository hosts a hybrid Windows/WSL rendering pipeline that produces a saliency-guided foveated render in a single pass. Blender (running on Windows) generates a fast preview, and WSL computes a DINO-based importance map from it. A final, efficient render pass in Blender then uses this mask to vary rendering quality across the image, focusing detail on important regions without needing to composite multiple images.


## Installation

This project uses [Conda](https://docs.conda.io/en/latest/) to manage the Python environment for the analysis scripts (Step 2).

1.  **Create the Conda Environment:**
    From the project root directory, run the following command to create the environment from the provided file:
    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the Environment:**
    Before running the main pipeline script, you must activate the Conda environment:
    ```bash
    conda activate scdl-foveated
    ```

This will install all the necessary Python dependencies (PyTorch, OpenCV, OpenEXR, etc.).

## Prerequisites
- Windows Blender installation accessible from WSL.
- A local clone of the DINOv3 repository and its pre-trained weights.
- Conda installed on your system.

## Configuration
1. Review and edit `.scdl.env` to match your environment paths, especially `BLENDER_EXE`.
2. Place your working `.blend` files in `blender_files/` or override the `BLEND_FILE` environment variable.
3. Ensure the `dinov3` repository and weights paths in `.scdl.env` are correct.

## Running the Pipeline
```bash
# Preview → DINO mask → Single-pass foveated render
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
