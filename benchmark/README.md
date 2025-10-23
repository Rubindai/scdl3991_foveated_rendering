# Benchmarking Foveated vs. Full Renders

This directory contains utilities for quantifying how closely the foveated render matches the full baseline. The entry-point script, `benchmark_images.py`, compares two rendered frames and reports common image-similarity metrics.

## Prerequisites
- Python 3.10+ (the same Conda env used for the pipeline already ships Pillow and NumPy).
- Completed renders saved to:
  - `out_full_render/final.png` (baseline, produced by `run_full_render_wsl.sh`)
  - `out/final.png` (foveated pipeline output)

> **Tip:** If you keep multiple variants, use the `--full-name` / `--foveated-name` flags to target specific files.

## Usage

```bash
conda activate scdl-foveated
python benchmark/benchmark_images.py
```

Default behaviour:
- Looks for the baseline image in `/home/rubin/uni/scdl3991_foveated_rendering/out_full_render`.
- Looks for the foveated image in `/home/rubin/uni/scdl3991_foveated_rendering/out`.
- Writes `benchmark_results.md` next to the script (i.e., inside `benchmark/`).

### CLI Options

| Flag | Description |
|------|-------------|
| `--full-dir PATH` | Override the directory containing the full baseline image. |
| `--foveated-dir PATH` | Override the directory containing the foveated image. |
| `--output-dir PATH` | Redirect where the markdown report is written. |
| `--full-name NAME` | Force a specific filename inside `--full-dir`. |
| `--foveated-name NAME` | Force a specific filename inside `--foveated-dir`. |

## Reported Metrics
- **MSE** — Mean Squared Error (0.0 = identical, lower is better).
- **PSNR (dB)** — Peak Signal-to-Noise Ratio (higher is better).
- **SSIM** — Structural Similarity Index (1.0 = identical).
- **Histogram L1 Distance** — Per-channel colour histogram difference (0.0 = identical, lower is better).

Results appear both in the terminal (as a monospace table) and in `benchmark_results.md`, making it easy to track changes over time or share results.

## Error Handling
The script exits cleanly with descriptive messages when:
- Expected directories are missing.
- Multiple candidate images are found (unless a filename override is supplied).
- Image dimensions or channel layouts differ.

Resolve the issue and re-run the command; the markdown report is regenerated on every successful invocation.
