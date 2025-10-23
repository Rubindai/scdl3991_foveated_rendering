#!/usr/bin/env python3
"""Benchmark foveated render output against full baseline ground truth."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".exr"}


class BenchmarkError(RuntimeError):
    """Raised when the benchmark cannot proceed."""


@dataclass(frozen=True)
class BenchmarkConfig:
    """Input/output configuration for the benchmark."""

    full_dir: Path
    foveated_dir: Path
    output_dir: Path
    report_path: Path
    full_name: Optional[str]
    foveated_name: Optional[str]


@dataclass(frozen=True)
class MetricResult:
    """Computed metric with an optional note."""

    name: str
    value: float
    note: Optional[str] = None


def parse_args(argv: Optional[Sequence[str]] = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-dir",
        default="/home/rubin/uni/scdl3991_foveated_rendering/out_full_render",
        help="Directory containing the full render image",
    )
    parser.add_argument(
        "--foveated-dir",
        default="/home/rubin/uni/scdl3991_foveated_rendering/out",
        help="Directory containing the foveated render image",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for benchmark outputs (default: alongside this script)",
    )
    parser.add_argument(
        "--full-name",
        default=None,
        help="Specific filename to use within --full-dir (default: auto-detect)",
    )
    parser.add_argument(
        "--foveated-name",
        default=None,
        help="Specific filename to use within --foveated-dir (default: auto-detect)",
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    return BenchmarkConfig(
        full_dir=Path(args.full_dir).expanduser().resolve(),
        foveated_dir=Path(args.foveated_dir).expanduser().resolve(),
        output_dir=output_dir,
        report_path=output_dir / "benchmark_results.md",
        full_name=args.full_name,
        foveated_name=args.foveated_name,
    )


def find_single_image(directory: Path, preferred_name: Optional[str] = None) -> Path:
    if not directory.exists():
        raise BenchmarkError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise BenchmarkError(f"Expected a directory but found: {directory}")

    if preferred_name:
        candidate = directory / preferred_name
        if not candidate.exists():
            raise BenchmarkError(f"Requested image '{preferred_name}' not found in {directory}")
        if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise BenchmarkError(f"Unsupported image extension for {candidate}")
        if not candidate.is_file():
            raise BenchmarkError(f"Requested image is not a file: {candidate}")
        return candidate

    images = sorted(
        [
            path
            for path in directory.iterdir()
            if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file()
        ]
    )
    if not images:
        raise BenchmarkError(f"No supported images found in {directory}")
    preferred_defaults = ("final.png", "final.jpg", "final.jpeg", "final.tif", "final.exr")
    for name in preferred_defaults:
        candidate = directory / name
        if candidate in images:
            return candidate

    if len(images) > 1:
        raise BenchmarkError(f"Multiple images found in {directory}; expected exactly one")
    return images[0]


def load_image(path: Path) -> np.ndarray:
    try:
        img = Image.open(path)
    except Exception as exc:
        raise BenchmarkError(f"Failed to open image {path}: {exc}") from exc

    # Convert to RGB(A) or grayscale depending on mode
    if img.mode not in {"L", "RGB", "RGBA"}:
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")

    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[..., None]

    bit_depth = 8
    if arr.dtype == np.uint16:
        bit_depth = 16
    elif arr.dtype == np.float32 or arr.dtype == np.float64:
        bit_depth = 1  # already normalized

    if bit_depth == 16:
        arr = arr.astype(np.float32) / 65535.0
    elif bit_depth == 8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)

    return arr


def ensure_same_shape(arr1: np.ndarray, arr2: np.ndarray) -> None:
    if arr1.shape != arr2.shape:
        raise BenchmarkError(f"Image shapes do not match: {arr1.shape} vs {arr2.shape}")


def mean_squared_error(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(np.mean((img1 - img2) ** 2))


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = mean_squared_error(img1, img2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    if img1.ndim == 3 and img1.shape[2] > 1:
        channels = [
            _ssim_channel(img1[..., i], img2[..., i], C1=C1, C2=C2)
            for i in range(img1.shape[2])
        ]
        return float(np.mean(channels))
    return _ssim_channel(img1[..., 0], img2[..., 0], C1=C1, C2=C2)


def _ssim_channel(channel1: np.ndarray, channel2: np.ndarray, *, C1: float, C2: float) -> float:
    mu1 = channel1.mean()
    mu2 = channel2.mean()
    sigma1 = channel1.var()
    sigma2 = channel2.var()
    covariance = np.mean((channel1 - mu1) * (channel2 - mu2))

    numerator = (2 * mu1 * mu2 + C1) * (2 * covariance + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator / denominator)


def histogram_l1_distance(img1: np.ndarray, img2: np.ndarray, bins: int = 256) -> float:
    if img1.shape[2] == 1:
        img1 = np.repeat(img1, 3, axis=2)
    if img2.shape[2] == 1:
        img2 = np.repeat(img2, 3, axis=2)

    distances = []
    for channel in range(3):
        h1, _ = np.histogram(img1[..., channel], bins=bins, range=(0.0, 1.0), density=True)
        h2, _ = np.histogram(img2[..., channel], bins=bins, range=(0.0, 1.0), density=True)
        distances.append(0.5 * np.sum(np.abs(h1 - h2)))
    return float(np.mean(distances))


def format_table(metrics: Iterable[MetricResult]) -> str:
    headers = ("Metric", "Value", "Notes")
    rows = [
        (metric.name, f"{metric.value:.6f}" if math.isfinite(metric.value) else "∞", metric.note or "")
        for metric in metrics
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]

    def fmt_row(row: Sequence[str]) -> str:
        return " | ".join(item.ljust(widths[idx]) for idx, item in enumerate(row))

    lines = [fmt_row(headers), "-+-".join("-" * width for width in widths)]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def write_markdown_report(report_path: Path, metrics: Sequence[MetricResult], full_image: Path, foveated_image: Path) -> None:
    lines = [
        "# Render Benchmark Results",
        "",
        f"- **Full render:** `{full_image.as_posix()}`",
        f"- **Foveated render:** `{foveated_image.as_posix()}`",
        "",
        "| Metric | Value | Notes |",
        "|--------|-------|-------|",
    ]
    for metric in metrics:
        value_str = f"{metric.value:.6f}" if math.isfinite(metric.value) else "&infin;"
        note = metric.note or ""
        lines.append(f"| {metric.name} | {value_str} | {note} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        config = parse_args(argv)
        full_image_path = find_single_image(config.full_dir, config.full_name)
        foveated_image_path = find_single_image(config.foveated_dir, config.foveated_name)

        full_img = load_image(full_image_path)
        foveated_img = load_image(foveated_image_path)
        ensure_same_shape(full_img, foveated_img)

        mse_value = mean_squared_error(full_img, foveated_img)
        psnr_value = psnr(full_img, foveated_img)
        ssim_value = ssim(full_img, foveated_img)
        hist_distance = histogram_l1_distance(full_img, foveated_img)

        metrics = [
            MetricResult("MSE", mse_value, note="0.0 = identical (lower is better)"),
            MetricResult("PSNR (dB)", psnr_value, note="Higher is better"),
            MetricResult("SSIM", ssim_value, note="1.0 = identical"),
            MetricResult("Histogram L1 Distance", hist_distance, note="0.0 = identical (lower is better)"),
        ]

        print(format_table(metrics))
        write_markdown_report(config.report_path, metrics, full_image_path, foveated_image_path)
        print(f"\nMarkdown report written to {config.report_path}")
        return 0
    except BenchmarkError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - safety net
        print(f"[ERROR] Unexpected failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
