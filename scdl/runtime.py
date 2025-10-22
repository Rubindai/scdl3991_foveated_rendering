#!/usr/bin/env python3
"""Runtime validation helpers for Blender and PyTorch environments."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import List, Optional


try:
    import bpy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - Blender unavailable in some contexts
    bpy = None  # type: ignore


@dataclass(frozen=True)
class BlenderInfo:
    """Snapshot of Blender runtime information."""

    version_string: str
    build_commit: str
    python_version: str
    is_background: bool


def require_blender_version(expected: tuple[int, int, int]) -> BlenderInfo:
    """Ensure Blender is available and matches the expected version."""

    if bpy is None:
        raise RuntimeError("Blender's Python API (bpy) is not available in this interpreter.")

    if bpy.app.version != expected:
        raise RuntimeError(
            f"Blender version mismatch: expected {'.'.join(map(str, expected))}, "
            f"found {bpy.app.version_string}"
        )
    return BlenderInfo(
        version_string=bpy.app.version_string,
        build_commit=getattr(bpy.app, "build_hash", "unknown"),
        python_version=bpy.app.version_string.split()[-1] if " " in bpy.app.version_string else platform.python_version(),
        is_background=bpy.app.background,
    )


def cycles_devices() -> List[str]:
    """Return human-readable descriptions of enabled Cycles devices."""

    if bpy is None:
        raise RuntimeError("Blender API not available; cannot query Cycles devices.")

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.get_devices()
    devices = []
    for device in prefs.devices:
        status = "ENABLED" if getattr(device, "use", False) else "DISABLED"
        dtype = getattr(device, "type", "UNKNOWN")
        devices.append(f"{status} {dtype} {device.name}")
    return devices


def ensure_optix_device(expected_substring: str) -> List[str]:
    """Enable only OPTIX devices and ensure one matches the expected GPU."""

    if bpy is None:
        raise RuntimeError("Blender API not available; cannot enforce OPTIX requirements.")

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()

    matched_devices: List[str] = []
    for device in prefs.devices:
        is_optix = getattr(device, "type", "") == "OPTIX"
        device.use = bool(is_optix)
        if is_optix and expected_substring.lower() in device.name.lower():
            matched_devices.append(device.name)

    if not matched_devices:
        raise RuntimeError(
            f"No OPTIX device matching '{expected_substring}' detected. "
            "This pipeline forbids falling back to CUDA or CPU."
        )
    return matched_devices


def set_cycles_scene_device() -> None:
    """Force every scene to use GPU rendering for Cycles."""

    if bpy is None:
        raise RuntimeError("Blender API not available; cannot set Cycles device.")

    for scene in bpy.data.scenes:
        scene.cycles.device = "GPU"


def blender_python_version() -> str:
    """Return Blender's embedded Python version string."""

    return platform.python_version()


def torch_device_summary(expected_substring: str, *, require_bf16: bool = True) -> str:
    """Validate that PyTorch sees the correct GPU and capabilities."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)

    if expected_substring.lower() not in device_name.lower():
        raise RuntimeError(
            f"Expected CUDA device containing '{expected_substring}', found '{device_name}'."
        )

    major, minor = torch.cuda.get_device_capability(device_index)
    if major < 8:
        raise RuntimeError(
            f"Requires NVIDIA Ampere or newer (compute capability 8.x). Found {major}.{minor}."
        )

    if require_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 is not supported by this CUDA build or GPU.")

    return device_name


def ensure_flash_attention() -> None:
    """Check that Flash Attention SDP backend is usable."""

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    B, H, T, D = 1, 4, 256, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        result = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    if not torch.isfinite(result).all():
        raise RuntimeError("Flash Attention probe produced non-finite output.")


def platform_summary() -> str:
    """Return a concise platform summary string."""

    return f"{platform.system()} {platform.release()} ({platform.machine()})"


__all__ = [
    "BlenderInfo",
    "cycles_devices",
    "ensure_optix_device",
    "ensure_flash_attention",
    "platform_summary",
    "require_blender_version",
    "set_cycles_scene_device",
    "torch_device_summary",
]
