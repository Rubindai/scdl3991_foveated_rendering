# Scene-Driven Foveated Rendering

## Overview
- Preview, saliency analysis, and final render are split into three audited stages (`step1_preview_blender.py`, `step2_dino_mask.py`, `step3_singlepass_foveated.py`).
- Blender 4.5.4 LTS and an NVIDIA RTX 3060 (OPTIX) are hard requirements; the code path aborts immediately if either prerequisite is missing.
- Shared utilities live in the `scdl/` package: colourised logging, GPU-aware timing, system validation, and path helpers.

## Hardware & Software Requirements
- **Blender:** 4.5.4 LTS (Linux build recommended; Windows/WSL fallback supported).
- **GPU:** NVIDIA RTX 3060 with OPTIX support (no CUDA/CPU fallback).
- **Python environment:** Python 3.10 with CUDA 12.1 (provided via `environment.yml`).
- **Optional WSL setup:** Required only when using the Windows fallback launcher.
- Local copy of `dinov3` weights (HF-format directory) placed wherever `SCDL_DINO_LOCAL_DIR` points.
- Python extras bundled in the Conda env: `OpenEXR`/`Imath` for EXR export and `opencv-python` for the optional saliency preview.
- If your GPU string differs (e.g., "RTX 4070"), set `SCDL_EXPECTED_GPU` in `.scdl.env` to the identifying substring.

## Environment Setup
```bash
conda env create --file environment.yml
conda activate scdl-foveated
```

Keep the environment up to date with:

```bash
conda env update --file environment.yml --prune
```

The `.scdl.env` file stores runtime configuration (paths, sampling budgets, logging options) and is sourced automatically by all launcher scripts.

## Running the Pipeline
Preferred (native Linux):
```bash
./run_linux_blender.sh                      # uses BLEND_FILE from .scdl.env
./run_linux_blender.sh path/to.scene.blend
```

Fallback (Windows Blender invoked from WSL):
```bash
./run_windows_blender_wsl.sh                # uses BLEND_FILE from .scdl.env
./run_windows_blender_wsl.sh path/to.scene.blend
```

Set `SCDL_LINUX_BLENDER_EXE=/path/to/blender` (or `BLENDER_EXE` for Windows) inside `.scdl.env` when the binary is not already on your `PATH`.

The launcher clears `out/` on every run unless `SCDL_CLEAR_OUT_DIR=0` is set in `.scdl.env`, ensures `CONDA_DEFAULT_ENV=scdl-foveated`, and streams all Blender output through the structured logger.

Each stage:
1. Validates platform requirements (Blender version, OPTIX devices, CUDA capability, Flash Attention).
2. Emits colour-coded logs to stderr and `out/scdl_pipeline.log` (configurable via `SCDL_LOG_MODE` / `SCDL_LOG_FILE`).
3. Records per-stage timings using GPU-synchronised `scdl.StageTimer` so inference and rendering durations are accurate.

Outputs land in `out/` (`preview.png`, `user_importance.{npy,exr}`, optional `user_importance_preview.png`, `final.png`, `scdl_pipeline.log`). Override the target directory via `SCDL_OUT_DIR`, or point Step 2 at an alternate preview with `SCDL_PREVIEW_PATH`.

## Benchmarking the Outputs
To quantify how closely the foveated render matches the full baseline, run the benchmarking utility described in `benchmark/README.md`. The standalone script compares `out/final.png` against `out_full_render/final.png` and reports MSE, PSNR, SSIM, and histogram distances, writing a markdown summary for record keeping.

## Baseline Full Render (Non-foveated)

The repository also ships a baseline render path that reuses the same Blender environment but skips all foveation steps. Use it to generate a ground-truth frame for quality comparisons and timing studies.

```bash
./run_linux_full_render.sh                  # native Linux baseline render
./run_linux_full_render.sh path/to.scene.blend
```

If you need to target Windows Blender, `run_full_render_wsl.sh` mirrors the same behaviour (loading `.scdl.env`, enforcing OPTIX, capturing timings).

Outputs are written to `out_full_render/`:
- `final.png` (16-bit PNG, full-frame render with uniform sampling).
- `full_render.log` (structured log with device validation and timings).

Sampling configuration inherits the same Step 3 knobs from `.scdl.env`:
- `SCDL_FOVEATED_BASE_SPP`, `SCDL_FOVEATED_MIN_SPP`, `SCDL_FOVEATED_MAX_SPP`, `SCDL_ADAPTIVE_THRESHOLD`, `SCDL_ADAPTIVE_MIN_SAMPLES`, `SCDL_FOVEATED_FILTER_GLOSSY`, `SCDL_FOVEATED_CAUSTICS`.
- Baseline-only controls: set `SCDL_BASELINE_SPP=<int>` to force an explicit sample count, or `SCDL_BASELINE_SPP_SCALE=<float>` to scale the base samples (useful when you want a slower ground truth while keeping the foveated render fast).

Because the baseline render shares the same validation guards (Blender 4.5.4, RTX 3060 OPTIX) and logging, it makes a reliable reference for the benchmarking script.

## Key Configuration Knobs
- **Preview (Step 1)** — `SCDL_PREVIEW_SHORT`, `SCDL_PREVIEW_SPP`, `SCDL_PREVIEW_MIN_SPP`, `SCDL_PREVIEW_ADAPTIVE_THRESHOLD`, `SCDL_PREVIEW_DENOISE`, `SCDL_PREVIEW_BLUR_GLOSSY`, optional `SCDL_PREVIEW_CLAMP_DIRECT/INDIRECT`, `SCDL_PREVIEW_SEED`.
- **DINO Saliency (Step 2)** — `SCDL_DINO_LOCAL_DIR`, `SCDL_DINO_SIZE`, `SCDL_PERC_LO/HI`, `SCDL_MASK_GAMMA`, `SCDL_MORPH_K`, optional `SCDL_MASK_DEVICE` (defaults to `cuda:0` and must remain CUDA).
- **Final Render (Step 3)** — `SCDL_FOVEATED_BASE_SPP/MIN_SPP/MAX_SPP`, `SCDL_ADAPTIVE_THRESHOLD`, `SCDL_ADAPTIVE_MIN_SAMPLES`, `SCDL_FOVEATED_FILTER_GLOSSY`.
- **System** — `SCDL_EXPECTED_GPU` (defaults to `RTX 3060`) controls which OPTIX/CUDA device the pipeline accepts.

All scripts reject non-OPTIX paths (`SCDL_CYCLES_DEVICE` must be `OPTIX`) and any CPU/CUDA fallbacks.

## Logging & Timing
- Colour palette highlights `[env]`, `[device]`, `[timer]`, and `[step]` tags for quick scanning.
- Timing records include stage-by-stage runtime plus consolidated summaries at the end of Steps 2 and 3.
- Step 3 prints the saliency-derived thresholds (`lo`, `hi`, `gamma`) and mask coverage before reporting `effective_samples`, making it easy to confirm the adaptive sample scaling matches expectations.
- Setting `SCDL_LOG_MODE=file` disables console output and writes exclusively to the configured log file.

## Reference Run (`blender_files/cookie.blend`)
- Blender 4.5.4 LTS + RTX 3060 Laptop GPU + Conda `scdl-foveated`.
- `out/preview.png`: 784×448 deterministic preview (16 spp, adaptive min 4).
- `out/user_importance.{npy,exr}`: saliency mask at preview resolution with optional grayscale overlay `out/user_importance_preview.png`.
- `out/final.png`: 16-bit single-pass foveated render with three materials receiving the `SCDL_FoveationMix` group.
- Representative timings from the verification pass: preview ≈1 s, saliency ≈13 s, final render ≈4 s (varies with GPU load).

## Verification Workflow
- Minimal quick check: `python -m compileall scdl step1_preview_blender.py step2_dino_mask.py step3_singlepass_foveated.py`.
- Full validation (requires Blender 4.5.4 + RTX 3060):
  1. `./run_linux_blender.sh` to execute the end-to-end pipeline (or `./run_windows_blender_wsl.sh` when Windows Blender is required).
  2. Inspect `out/scdl_pipeline.log` for device validation, timing summaries, the saliency threshold line (`[env] thresholds … coverage=…`), and the reported `effective_samples`/material injection count.
  3. Confirm `out/final.png` quality while reviewing `user_importance_preview.png` (if enabled).

See `pipeline.md` for a deeper stage-by-stage breakdown and parameter reference.
