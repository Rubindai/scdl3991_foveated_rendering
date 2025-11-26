# Benchmark Sweep

Generates LPIPS, SSIM, and PSNR curves for foveated vs. baseline renders, using the final full render (`out_full_render/final.png`) as the reference point.

## Quick Start
```bash
conda activate scdl-foveated
python benchmark/benchmark_sweep.py \
  --spp-start 0 \
  --spp-step 32
```

This:
- Uses your settings to derive the end of each curve (baseline cap = clamped `SCDL_BASELINE_SPP`, foveated cap = clamped `SCDL_FOVEATED_BASE_SPP`; both start at 0, and any SPP above the cap is skipped).
- Renders both modes at the requested SPP levels (skipping levels above each mode’s effective cap).
- Forces foveated sweeps to use exact SPP counts (`SCDL_FOVEATED_AUTOSPP=0`) for clean x-axis control.
- Captures the actual rays fired per level using the Cycles Debug Sample Count pass, writing `samples_baseline.json` / `samples_foveated.json` and plotting `samples.png`.
- Computes LPIPS/SSIM/PSNR against `out_full_render/final.png`.
- Writes results to `benchmark_sweep_out/metrics.json` and plots `lpips.png`, `ssim.png`, `psnr.png` in the same directory.

Set `--no-render` to reuse existing renders in `benchmark_sweep_out/renders/`.

## Requirements
- Baseline reference already rendered (run `./run_linux_full_render.sh` first).
- Mask artefacts present (`out/user_importance.npy` / `user_importance_mask.exr`) for foveated sweeps.
- Blender path via `SCDL_LINUX_BLENDER_EXE` or `BLENDER_EXE`.

### Environment knobs
- Baseline cap: clamped `SCDL_BASELINE_SPP` (default 512) with optional `SCDL_BASELINE_SPP_SCALE`, bounded by `SCDL_BASELINE_MIN_SPP`/`SCDL_BASELINE_MAX_SPP`.
- Foveated cap: clamped `SCDL_FOVEATED_BASE_SPP` (default 192) bounded by `SCDL_FOVEATED_MIN_SPP`/`SCDL_FOVEATED_MAX_SPP`.
- `BLEND_FILE` is read from `.scdl.env` if `--blend` is omitted.
