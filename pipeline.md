# Pipeline Breakdown

All stages enforce the same invariants:
- Blender must be 4.5.2 LTS.
- The active render device must be an NVIDIA RTX 3060 using OPTIX; CUDA/CPU/CPU+OIDN fallbacks are rejected.
- Logs are routed through `scdl.logging.get_logger`, yielding `[step]`, `[env]`, `[device]`, and `[timer]` tagged output in both the terminal and `out/scdl_pipeline.log`.
- GPU-aware timings are captured with `scdl.StageTimer` so reported durations include CUDA synchronisation points.

## Stage Overview

| Stage | Script | Inputs | Key Outputs | Notes |
|-------|--------|--------|-------------|-------|
| Preview | `step1_preview_blender.py` | `.blend`, `.scdl.env` | `out/preview.png` | Validates Blender/OPTIX, resizes frame to multiple-of-16 grid, enforces deterministic sampling. |
| Saliency | `step2_dino_mask.py` | `out/preview.png`, DINOv3 weights | `out/user_importance.npy`, `out/user_importance_mask.exr`, optional `out/user_importance_preview.png` | Requires CUDA Flash Attention, BF16, RTX 3060. Runs fully offline (no HF downloads). |
| Final Render | `step3_singlepass_foveated.py` | `.blend`, saliency mask artefacts | `out/final.png` | Injects foveation node group into every eligible material and renders in a single Cycles pass. |

The `run_windows_blender_wsl.sh` launcher orchestrates these stages, clearing `out/` by default (set `SCDL_CLEAR_OUT_DIR=0` to retain previous artefacts) while funnelling Blender output through the structured logger.

## Stage 1 – Preview Render (`out/preview.png`)
- **Validation**
  - Confirms `SCDL_CYCLES_DEVICE` is `OPTIX`.
  - Enables only OPTIX devices and verifies the RTX 3060 is active.
  - Logs Blender build hash and whether the session runs headless.
- **Configuration**
  - Resizes the render to keep the short side at `SCDL_PREVIEW_SHORT` (default 448) snapped to 16 px tiles.
  - Applies deterministic sampling: `SCDL_PREVIEW_SPP`, `SCDL_PREVIEW_MIN_SPP`, adaptive sampling thresholds, OPTIX denoiser, glossy filtering, optional clamping, reproducible `SCDL_PREVIEW_SEED`.
- **Output**
  - Writes `out/preview.png` (PNG, opaque, 8-bit) used as input to Stage 2.

## Stage 2 – DINOv3 Saliency (`out/user_importance*`)
- **Validation**
  - Optional `SCDL_MASK_DEVICE` must remain on CUDA (defaults to `cuda:0`).
  - Calls `torch_device_summary("RTX 3060")` and `ensure_flash_attention()` to guarantee Flash Attention availability.
  - Logs PyTorch, Transformers, and Kornia versions alongside the active device name.
- **Processing**
  1. Loads the preview PNG into GPU memory (`TorchVision`), with timing tracked via `StageTimer`.
  2. Preprocesses with a locally cached DINOv3 processor (`SCDL_DINO_LOCAL_DIR`, resize controlled by `SCDL_DINO_SIZE`).
  3. Runs a single BF16 forward pass with Flash Attention.
  4. Converts CLS-patch cosine similarities into a dense saliency map.
  5. Performs percentile stretch (`SCDL_PERC_LO`, `SCDL_PERC_HI`), gamma shaping (`SCDL_MASK_GAMMA`), and then morphological closing + Gaussian blur with kernel size `SCDL_MORPH_K`, all inside `torch.inference_mode()` to avoid autograd overhead.
- **Output**
  - Binary artefacts: `user_importance.npy` (float32 H×W) and `user_importance_mask.exr` (single-channel float).
  - Optional preview (`SCDL_USER_IMPORTANCE_PREVIEW`) written via OpenCV in grayscale or viridis overlay.
  - Accepts `SCDL_PREVIEW_PATH` to override the preview input and `SCDL_OUT_DIR` to redirect artefacts.
- **Timing**
  - `env.validate`, `model.load_processor`, `preview.load`, `dino.preprocess`, `dino.forward`, `dino.post`, `io.save`, and `preview.write` stages captured in the log.
- **Dependencies**
  - Requires `OpenEXR`/`Imath` for EXR export and `opencv-python` for overlay generation (all packaged in the Conda environment).

## Stage 3 – Single-Pass Foveated Render (`out/final.png`)
- **Validation**
  - Enforces OPTIX-only rendering and re-validates the RTX 3060.
  - Checks the existence of both `user_importance_mask.exr` and `user_importance.npy`.
- **Configuration**
  - Adaptive sampling budgets controlled by `SCDL_FOVEATED_BASE_SPP`, `SCDL_FOVEATED_MIN_SPP`, `SCDL_FOVEATED_MAX_SPP`, and `SCDL_FOVEATED_AUTOSPP`.
  - Material injection:
    - Builds/loads `SCDL_FoveationMix` node group.
    - Adds or refreshes a simplified Principled BSDF (LQ shader), preserving base colour/normal links.
    - Screenspace mask sampling via `TexCoord (Window)` → EXR mask (non-color data).
  - Derives low/high thresholds (20/80 quantiles) and fixed gamma 2.2 from `user_importance.npy`, logging `[env] thresholds … coverage=…` before rendering.
  - Coverage from the NPY mask biases the effective adaptive sample count while remaining bounded by min/max.
- **Output**
  - Final render written to `out/final.png` (PNG, 16-bit).
  - Console + log emit material injection count, `[env] effective_samples=…`, and duration for configuration, injection, and render execution.

## Failure Modes
- Any violation of hardware requirements (missing OPTIX, wrong GPU, disabled BF16) raises a `RuntimeError` with a `[ERROR]` log entry.
- Missing artefacts between stages (e.g., absent `preview.png`, EXR mask) abort the run before rendering commences.
- Preview and final render scripts refuse to continue if `SCDL_CYCLES_DEVICE` is set to anything other than `OPTIX`, guaranteeing that no silent CPU/CUDA fallback occurs.

## Verification Checklist
1. Run `python -m compileall scdl step1_preview_blender.py step2_dino_mask.py step3_singlepass_foveated.py` to confirm syntax.
2. Execute the pipeline via `./run_windows_blender_wsl.sh`.
3. Inspect `out/scdl_pipeline.log` for:
   - `[device] OPTIX REQUIRED ...` and `[device] CUDA ...` entries.
   - `[timer]` sections per stage (indicates GPU synchronisation succeeded).
   - Saliency stats (`[env] thresholds … coverage=…`) and `[env] effective_samples=…` before the final render.
   - `[OK]` lines confirming all artefacts were written.

## Reference Run (`blender_files/cookie.blend`)
- Preview: `784×448`, `[env] samples=16`, `[env] adaptive_min_samples=4`.
- Saliency: mask resolution matches the preview, `user_importance_preview.png` saved in grayscale, DINO timings ~1.0 s (preprocess + forward + post) with Flash Attention active.
- Final render: three materials receive `SCDL_FoveationMix`, `effective_samples=104`, render completes in ≈3.7 s with OPTIX denoising.
- Outputs: `out/preview.png`, `out/user_importance.{npy,exr}`, optional overlay, and `out/final.png` plus consolidated timings in `out/scdl_pipeline.log`.
