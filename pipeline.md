# Pipeline Breakdown

All stages enforce the same invariants:
- Blender must be 4.5.4 LTS.
- The active render device must be an NVIDIA RTX 3060 using OPTIX; CUDA/CPU/CPU+OIDN fallbacks are rejected.
- The expected GPU substring comes from `SCDL_EXPECTED_GPU` (defaults to `RTX 3060`) and is checked in every stage.
- Logs are routed through `scdl.logging.get_logger`, yielding `[step]`, `[env]`, `[device]`, and `[timer]` tagged output in both the terminal and `out/scdl_pipeline.log`.
- GPU-aware timings are captured with `scdl.StageTimer` so reported durations include CUDA synchronisation points.

## Stage Overview

| Stage | Script | Inputs | Key Outputs | Notes |
|-------|--------|--------|-------------|-------|
| Preview | `step1_preview_blender.py` | `.blend`, `.scdl.env` | `out/preview.png` | Validates Blender/OPTIX, resizes frame to multiple-of-16 grid, enforces deterministic sampling. |
| Saliency | `step2_dino_mask.py` | `out/preview.png`, DINOv3 weights | `out/user_importance.npy`, `out/user_importance_mask.exr`, optional `out/user_importance_preview.png` | Requires CUDA Flash Attention, BF16, RTX 3060. Runs fully offline (no HF downloads). |
| Final Render | `step3_singlepass_foveated.py` | `.blend`, saliency mask artefacts | `out/final.png` | Injects foveation node group into every eligible material (excluding Volume, Holdout, and Toon) and renders in a single Cycles pass. |

The `run_linux_blender.sh` launcher orchestrates these stages on native Ubuntu (clearing `out/` by default; set `SCDL_CLEAR_OUT_DIR=0` to retain artefacts) while funnelling Blender output through the structured logger. When Windows Blender is required, `run_windows_blender_wsl.sh` provides equivalent behaviour through WSL.

`run_linux_blender.sh` runs the original three-stage flow (two Blender launches + external DINO). Step 2 uses the external Python environment (`SCDL_PYTHON_EXE`/`python`).

## Stage 1 – Preview Render (`out/preview.png`)
- **Validation**
  - Confirms `SCDL_CYCLES_DEVICE` is `OPTIX`.
  - Enables only OPTIX devices and verifies the GPU string matches `SCDL_EXPECTED_GPU`.
  - Logs Blender build hash and whether the session runs headless.
- **Configuration**
  - Resizes the render to keep the short side at `SCDL_PREVIEW_SHORT` (default 448) snapped to 16 px tiles.
  - Applies deterministic sampling: `SCDL_PREVIEW_SPP`, `SCDL_PREVIEW_MIN_SPP`, adaptive sampling thresholds, OPTIX denoiser, glossy filtering, optional clamping, reproducible `SCDL_PREVIEW_SEED`.
- **Output**
  - Writes `out/preview.png` (PNG, opaque, 8-bit) used as input to Stage 2.

## Stage 2 – DINOv3 Saliency (`out/user_importance*`)
- **Validation**
  - Optional `SCDL_MASK_DEVICE` must remain on CUDA (defaults to `cuda:0`).
  - Calls `torch_device_summary` with `SCDL_EXPECTED_GPU` and `ensure_flash_attention()` to guarantee Flash Attention availability.
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
  - Enforces OPTIX-only rendering and re-validates the GPU against `SCDL_EXPECTED_GPU`.
  - Checks the existence of both `user_importance_mask.exr` and `user_importance.npy`.
- **Configuration**
  - Adaptive sampling budgets controlled by `SCDL_FOVEATED_BASE_SPP`, `SCDL_FOVEATED_MIN_SPP`, `SCDL_FOVEATED_MAX_SPP`, and `SCDL_FOVEATED_AUTOSPP`. The coverage-aware curve uses `SCDL_FOVEATED_AUTOSPP_FLOOR` (default 0.35), `SCDL_FOVEATED_AUTOSPP_TAPER` (default 0.65, tapers the global sample scale toward the minimum when HQ coverage is tiny), and `SCDL_FOVEATED_AUTOSPP_GAMMA` (default 1.35). The curve now keys off the HQ footprint plus a weighted mid band (`SCDL_FOVEATED_IMPORTANCE_MID_WEIGHT`, default 0.35) instead of the raw mask mean, and `SCDL_FOVEATED_MIN_FLOOR_FRACTION` (default 0.5) still caps how far the floor can fall.
  - Additional runtime trims:
    - `SCDL_FOVEATED_ADAPTIVE_BOOST` (default 1.35) raises the adaptive threshold as coverage shrinks so LQ-dominant frames converge faster.
    - `SCDL_FOVEATED_ADAPTIVE_MIN_FRACTION` lets the adaptive-min budget shrink in proportion to coverage (floor keeps at least one sample).
    - Bounce taper: `SCDL_FOVEATED_MAX_BOUNCES` / `SCDL_FOVEATED_TRANSPARENT_BOUNCES` (defaults 8) are scaled by coverage via `SCDL_FOVEATED_BOUNCE_TAPER` (default 0.55) and bounded by `SCDL_FOVEATED_BOUNCE_FLOOR` (default 4) so LQ-heavy scenes take shorter light paths.
    - LQ throttle: `SCDL_FOVEATED_LQ_SAMPLE_BIAS` (default 0.85) and `SCDL_FOVEATED_LQ_MIN_SCALE` (default 0.35) multiply the autospp curve, adaptive-min floor, bounce taper, and light-tree threshold when the LQ footprint dominates, yielding much lower SPP in defocused scenes while keeping HQ regions intact.
    - Light-tree controls: `SCDL_FOVEATED_LIGHT_TREE` (default on), `SCDL_FOVEATED_LIGHT_SAMPLING_THRESHOLD` (default 0.01), and `SCDL_FOVEATED_LIGHT_THRESHOLD_BOOST` (default 2.0) which scales the light sampling threshold up as HQ coverage shrinks (and now scales harder when the LQ throttle is active) to cull negligible lights sooner.
  - Material injection:
    - Builds/loads `SCDL_FoveationMix` node group.
    - If the group is already wired to the material output (e.g., when re-running inside the same Blender session), thresholds/mask/LQ branch are refreshed instead of stacking another copy.
    - Adds or refreshes a simplified LQ shader. By default `SCDL_FOVEATED_LQ_DIFFUSE=1` swaps the LQ branch to a minimal diffuse BSDF; set it to `0` to use the lightweight Principled fallback. To avoid double-evaluating heavy texture graphs, the LQ branch uses cheap defaults; set `SCDL_FOVEATED_LQ_REUSE_TEXTURES=1` if you prefer the previous behaviour.
    - Screenspace mask sampling via `TexCoord (Window)` → EXR mask (non-color data).
  - Derives low/high thresholds from coverage-aware percentiles (`SCDL_FOVEATED_AUTO_THRESH=1`): `AUTO_LO_PUSH` raises the low percentile as coverage shrinks (expanding LQ), `AUTO_HI_PULL` adjusts the high percentile with coverage, and `AUTO_MASK_BIAS_GAIN` adds extra bias when coverage is tiny. Logging shows raw/derived percentiles, applied bias, and realized coverage for HQ/LQ.
  - Coverage from the NPY mask (mean + HQ/mid/LQ fractions) biases the effective adaptive sample count while remaining bounded by min/max; the log reports the taper factor, the resulting scale, the LQ throttle, bounces, SPP floor, and the effective sample count.
- **Output**
  - Final render written to `out/final.png` (PNG, 16-bit).
  - Console + log emit material injection count, `[env] effective_samples=…`, and duration for configuration, injection, and render execution.

## Failure Modes
- Any violation of hardware requirements (missing OPTIX, wrong GPU, disabled BF16) raises a `RuntimeError` with a `[ERROR]` log entry.
- Missing artefacts between stages (e.g., absent `preview.png`, EXR mask) abort the run before rendering commences.
- Preview and final render scripts refuse to continue if `SCDL_CYCLES_DEVICE` is set to anything other than `OPTIX`, guaranteeing that no silent CPU/CUDA fallback occurs.

## Verification Checklist
1. Run `python -m compileall scdl step1_preview_blender.py step2_dino_mask.py step3_singlepass_foveated.py` to confirm syntax.
2. Execute the pipeline via `./run_linux_blender.sh` (or `./run_windows_blender_wsl.sh` when running the Windows Blender build).
3. Inspect `out/scdl_pipeline.log` for:
   - `[device] OPTIX REQUIRED ...` and `[device] CUDA ...` entries.
   - `[timer]` sections per stage (indicates GPU synchronisation succeeded).
   - Saliency stats (`[env] thresholds … coverage=… hq=… mid=… lq=…`) and the `[env] effective_samples=…` line that now reports the LQ throttle factor.
   - `[OK]` lines confirming all artefacts were written.

## Reference Run (`blender_files/cookie.blend`)
- Preview: `784×448`, `[env] samples=16`, `[env] adaptive_min_samples=4`.
- Saliency: mask resolution matches the preview, `user_importance_preview.png` saved in grayscale, DINO timings ~1.0 s (preprocess + forward + post) with Flash Attention active.
- Final render: three materials receive `SCDL_FoveationMix`, `effective_samples≈104` on the reference mask (lower when the LQ throttle kicks in), render completes in ≈3.7 s with OPTIX denoising.
- Outputs: `out/preview.png`, `out/user_importance.{npy,exr}`, optional overlay, and `out/final.png` plus consolidated timings in `out/scdl_pipeline.log`.
