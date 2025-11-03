#!/usr/bin/env python3
"""Step 3 — Single-pass foveated render for Blender 4.5.4 LTS + RTX 3060 (OPTIX only)."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scdl import StageTimer, TimerRecord, get_logger, get_pipeline_paths
from scdl.logging import log_devices, log_environment
from scdl.runtime import (
    cycles_devices,
    ensure_optix_device,
    require_blender_version,
    set_cycles_scene_device,
)


@dataclass(frozen=True)
class FoveatedConfig:
    """Render configuration derived from environment variables."""

    base_samples: int
    min_samples: int
    max_samples: int
    adaptive_threshold: float
    adaptive_min_samples: int
    filter_glossy: float
    clamp_direct: Optional[float]
    clamp_indirect: Optional[float]
    auto_samples: bool
    caustics: bool

    @classmethod
    def from_env(cls) -> "FoveatedConfig":
        """Parse and validate Step 3 configuration."""

        base_samples = _env_positive_int("SCDL_FOVEATED_BASE_SPP", default=192)
        min_samples = _env_positive_int("SCDL_FOVEATED_MIN_SPP", default=32)
        max_samples = _env_positive_int("SCDL_FOVEATED_MAX_SPP", default=768)
        adaptive_threshold = float(os.getenv("SCDL_ADAPTIVE_THRESHOLD", "0.02"))
        adaptive_min_samples = _env_positive_int("SCDL_ADAPTIVE_MIN_SAMPLES", default=min_samples)
        filter_glossy = float(os.getenv("SCDL_FOVEATED_FILTER_GLOSSY", "0.8"))
        clamp_direct = _env_optional_float("SCDL_FOVEATED_CLAMP_DIRECT")
        clamp_indirect = _env_optional_float("SCDL_FOVEATED_CLAMP_INDIRECT")
        auto_samples = _env_bool("SCDL_FOVEATED_AUTOSPP", default=True)
        caustics = _env_bool("SCDL_FOVEATED_CAUSTICS", default=False)

        if min_samples > base_samples:
            raise RuntimeError("[Step3] SCDL_FOVEATED_MIN_SPP cannot exceed SCDL_FOVEATED_BASE_SPP.")
        if max_samples < base_samples:
            raise RuntimeError("[Step3] SCDL_FOVEATED_MAX_SPP cannot be lower than SCDL_FOVEATED_BASE_SPP.")
        if adaptive_min_samples < min_samples:
            raise RuntimeError("[Step3] SCDL_ADAPTIVE_MIN_SAMPLES must be ≥ SCDL_FOVEATED_MIN_SPP.")

        return cls(
            base_samples=base_samples,
            min_samples=min_samples,
            max_samples=max_samples,
            adaptive_threshold=adaptive_threshold,
            adaptive_min_samples=adaptive_min_samples,
            filter_glossy=filter_glossy,
            clamp_direct=clamp_direct,
            clamp_indirect=clamp_indirect,
            auto_samples=auto_samples,
            caustics=caustics,
        )


def _env_positive_int(name: str, *, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"[Step3] {name} must be an integer (received '{raw}').") from exc
    if value <= 0:
        raise RuntimeError(f"[Step3] {name} must be greater than zero (received {value}).")
    return value


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"[Step3] {name} must be truthy/falsy (received '{raw}').")


def _env_optional_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"[Step3] {name} must be a float (received '{raw}').") from exc


def thresholds(np_path: Path) -> Tuple[float, float, float, float]:
    """Compute mask thresholds and coverage statistics."""

    arr = np.load(np_path).astype(np.float32)
    flat = arr[np.isfinite(arr)].ravel()
    if flat.size == 0:
        return 0.25, 0.75, 2.2, 0.0

    q20, q80 = np.quantile(flat, [0.2, 0.8])
    lo = float(q20)
    hi = float(q80)
    gamma = 2.2
    coverage = float(flat.mean())
    return lo, hi, gamma, coverage


def ensure_foveation_group() -> bpy.types.NodeTree:
    """Create (or fetch) the node group that blends HQ/LQ shading based on the mask."""

    name = "SCDL_FoveationMix"
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]

    group = bpy.data.node_groups.new(name, "ShaderNodeTree")
    interface = group.interface
    interface.new_socket(name="HQ Shader", in_out="INPUT", socket_type="NodeSocketShader", description="Full-quality shader")
    interface.new_socket(name="LQ Shader", in_out="INPUT", socket_type="NodeSocketShader", description="Simplified shader")
    interface.new_socket(name="Mask", in_out="INPUT", socket_type="NodeSocketFloat", description="Foveation mask 0..1")
    interface.new_socket(name="LoThr", in_out="INPUT", socket_type="NodeSocketFloat", description="Lower threshold")
    interface.new_socket(name="HiThr", in_out="INPUT", socket_type="NodeSocketFloat", description="Upper threshold")
    interface.new_socket(name="Gamma", in_out="INPUT", socket_type="NodeSocketFloat", description="Gamma shaping")
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader", description="Output")

    n_in = group.nodes.new("NodeGroupInput")
    n_out = group.nodes.new("NodeGroupOutput")
    map_range = group.nodes.new("ShaderNodeMapRange")
    map_range.clamp = True
    map_range.data_type = "FLOAT"
    power = group.nodes.new("ShaderNodeMath")
    power.operation = "POWER"
    mix = group.nodes.new("ShaderNodeMixShader")

    group.links.new(n_in.outputs["Mask"], map_range.inputs["Value"])
    group.links.new(n_in.outputs["LoThr"], map_range.inputs["From Min"])
    group.links.new(n_in.outputs["HiThr"], map_range.inputs["From Max"])
    group.links.new(map_range.outputs["Result"], power.inputs[0])
    group.links.new(n_in.outputs["Gamma"], power.inputs[1])

    group.links.new(n_in.outputs["LQ Shader"], mix.inputs[1])
    group.links.new(n_in.outputs["HQ Shader"], mix.inputs[2])
    group.links.new(power.outputs[0], mix.inputs["Fac"])
    group.links.new(mix.outputs["Shader"], n_out.inputs["Shader"])
    return group


def _norm(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


def get_input(node: bpy.types.Node, *names: str):
    """Find a node input socket by name, tolerant to whitespace differences."""

    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    by_norm = {_norm(sock.name): sock for sock in node.inputs}
    for name in names:
        socket = by_norm.get(_norm(name))
        if socket is not None:
            return socket
    return None


def link_or_copy(
    node_tree: bpy.types.NodeTree,
    src_node: bpy.types.Node,
    src_names: Iterable[str],
    dst_node: bpy.types.Node,
    dst_name: str,
) -> None:
    """Link source socket to destination or copy the default value."""

    src = get_input(src_node, *src_names)
    dst = get_input(dst_node, dst_name)
    if not dst:
        return
    if src and src.is_linked:
        node_tree.links.new(src.links[0].from_socket, dst)
    elif src:
        dst.default_value = getattr(src, "default_value", dst.default_value)


def build_lq_principled(node_tree: bpy.types.NodeTree, ref_principled: bpy.types.Node) -> bpy.types.Node:
    """Construct a simplified Principled BSDF for low-importance regions."""

    lq = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    lq.label = "LQ_Principled"

    roughness = get_input(lq, "Roughness")
    if roughness:
        roughness.default_value = 0.6
    for name in ("Specular", "Specular IOR Level", "IOR Level"):
        socket = get_input(lq, name)
        if socket:
            socket.default_value = 0.2

    link_or_copy(node_tree, ref_principled, ["Base Color"], lq, "Base Color")
    link_or_copy(node_tree, ref_principled, ["Roughness"], lq, "Roughness")
    link_or_copy(node_tree, ref_principled, ["Normal"], lq, "Normal")

    src_met = get_input(ref_principled, "Metallic")
    dst_met = get_input(lq, "Metallic")
    if src_met and dst_met:
        if src_met.is_linked:
            node_tree.links.new(src_met.links[0].from_socket, dst_met)
        else:
            dst_met.default_value = 0.5 * src_met.default_value
    return lq


def find_principled(node_tree: bpy.types.NodeTree) -> bpy.types.Node:
    """Return the first Principled BSDF in the material, creating one if absent."""

    for node in node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return node_tree.nodes.new("ShaderNodeBsdfPrincipled")


def load_mask_image(path: Path) -> bpy.types.Image:
    """Load the EXR mask as a non-color data image."""

    image = bpy.data.images.load(str(path), check_existing=True)
    image.colorspace_settings.name = "Non-Color"
    return image


def material_is_prohibited(material: bpy.types.Material) -> bool:
    """Skip materials where foveation should not be injected."""

    name = (material.name or "").lower()
    return any(token in name for token in ("volume", "holdout", "toon"))


def inject_group_into_material(
    material: bpy.types.Material,
    mask_image: bpy.types.Image,
    group: bpy.types.NodeTree,
    lo: float,
    hi: float,
    gamma: float,
    logger,
) -> bool:
    """Insert the foveation node group into a material, returning success."""

    if not material or not material.use_nodes or material_is_prohibited(material):
        return False

    node_tree = material.node_tree
    output = next(
        (node for node in node_tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output),
        None,
    )
    if not output or not output.inputs.get("Surface") or not output.inputs["Surface"].links:
        return False

    original_link = output.inputs["Surface"].links[0]
    original_socket = original_link.from_socket

    tex_coords = node_tree.nodes.new("ShaderNodeTexCoord")
    tex_coords.label = "Coords"
    tex_image = node_tree.nodes.new("ShaderNodeTexImage")
    tex_image.image = mask_image
    tex_image.label = "FoveaMask"
    tex_image.interpolation = "Linear"
    tex_image.extension = "CLIP"
    tex_image.projection = "FLAT"
    tex_image.image.colorspace_settings.name = "Non-Color"
    node_tree.links.new(tex_coords.outputs["Window"], tex_image.inputs["Vector"])

    group_inst = node_tree.nodes.new("ShaderNodeGroup")
    group_inst.node_tree = group
    group_inst.label = "FoveationMix"
    ref_principled = find_principled(node_tree)
    lq_principled = build_lq_principled(node_tree, ref_principled)

    mask_output = tex_image.outputs.get("Alpha") or tex_image.outputs.get("Color")
    if mask_output is None:
        logger.warning("[WARN] Material '%s' mask has no usable output; skipping", material.name)
        return False

    node_tree.links.new(original_socket, group_inst.inputs["HQ Shader"])
    node_tree.links.new(lq_principled.outputs["BSDF"], group_inst.inputs["LQ Shader"])
    node_tree.links.new(mask_output, group_inst.inputs["Mask"])
    group_inst.inputs["LoThr"].default_value = float(lo)
    group_inst.inputs["HiThr"].default_value = float(hi)
    group_inst.inputs["Gamma"].default_value = float(gamma)

    node_tree.links.remove(original_link)
    node_tree.links.new(group_inst.outputs["Shader"], output.inputs["Surface"])
    return True


def configure_cycles(scene: bpy.types.Scene, cfg: FoveatedConfig, coverage: float) -> int:
    """Apply Cycles render settings and return the effective sample count."""

    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"

    cycles = scene.cycles
    cycles.device = "GPU"
    cycles.use_fast_gi = False
    cycles.max_bounces = 8
    cycles.transparent_max_bounces = 8
    cycles.caustics_reflective = cfg.caustics
    cycles.caustics_refractive = cfg.caustics

    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = cfg.adaptive_threshold
    cycles.adaptive_min_samples = cfg.adaptive_min_samples

    base_samples = cfg.base_samples
    if cfg.auto_samples:
        scaled = int(base_samples * (0.5 + 0.5 * max(0.0, min(1.0, coverage))))
        effective = max(cfg.min_samples, min(cfg.max_samples, scaled))
    else:
        effective = base_samples
    cycles.samples = effective

    cycles.use_denoising = True
    cycles.denoiser = "OPTIX"
    for view_layer in scene.view_layers:
        view_layer.cycles.use_denoising = True

    cycles.blur_glossy = cfg.filter_glossy
    if cfg.clamp_direct is not None:
        cycles.sample_clamp_direct = cfg.clamp_direct
    if cfg.clamp_indirect is not None:
        cycles.sample_clamp_indirect = cfg.clamp_indirect

    return effective


def summarize_timings(logger, timings: Iterable[TimerRecord]) -> None:
    """Log a consolidated summary of recorded durations."""

    entries = list(timings)
    if not entries:
        return
    logger.info("[step] Timing summary")
    for record in entries:
        logger.info("[timer] %s %.3fs", record.name, record.seconds)


def main() -> None:
    """Execute the single-pass foveated render inside Blender."""

    paths = get_pipeline_paths(Path(__file__).resolve().parent)
    logger = get_logger("scdl.step3.foveated", default_path=paths.log_file)
    logger.info("[step] Step3 single-pass render starting")

    info = require_blender_version((4, 5, 4))
    logger.info("[system] Blender %s (commit %s)", info.version_string, info.build_commit)

    device_override = os.getenv("SCDL_CYCLES_DEVICE", "OPTIX").strip().upper()
    if device_override != "OPTIX":
        raise RuntimeError(f"[Step3] SCDL_CYCLES_DEVICE must be OPTIX (received '{device_override}').")

    expected_gpu = os.getenv("SCDL_EXPECTED_GPU", "RTX 3060")
    matched_devices = ensure_optix_device(expected_gpu)
    set_cycles_scene_device()
    devices = cycles_devices()
    log_devices(logger, devices)
    if matched_devices:
        log_devices(logger, [f"OPTIX REQUIRED {name}" for name in matched_devices])

    cfg = FoveatedConfig.from_env()
    log_environment(
        logger,
        {
            "base_samples": cfg.base_samples,
            "min_samples": cfg.min_samples,
            "max_samples": cfg.max_samples,
            "adaptive_threshold": cfg.adaptive_threshold,
            "adaptive_min_samples": cfg.adaptive_min_samples,
            "filter_glossy": cfg.filter_glossy,
            "clamp_direct": cfg.clamp_direct,
            "clamp_indirect": cfg.clamp_indirect,
            "auto_samples": cfg.auto_samples,
            "caustics": cfg.caustics,
            "expected_gpu": expected_gpu,
        },
    )

    if not paths.mask_exr.exists():
        raise RuntimeError(f"[Step3] Mask image not found at {paths.mask_exr}.")
    if not paths.mask_npy.exists():
        raise RuntimeError(f"[Step3] Mask array not found at {paths.mask_npy}.")

    timings: List[TimerRecord] = []

    with StageTimer("mask.load", logger=logger, records=timings):
        lo, hi, gamma, coverage = thresholds(paths.mask_npy)
        mask_image = load_mask_image(paths.mask_exr)
        logger.info(
            "[env] thresholds lo=%.4f hi=%.4f gamma=%.2f coverage=%.3f",
            lo,
            hi,
            gamma,
            coverage,
        )

    scene = bpy.context.scene
    scene.render.filepath = str(paths.final)

    with StageTimer("cycles.configure", logger=logger, records=timings):
        effective_samples = configure_cycles(scene, cfg, coverage)
        logger.info("[env] effective_samples=%d", effective_samples)

    group = ensure_foveation_group()
    injected = 0

    with StageTimer("materials.inject", logger=logger, records=timings):
        for material in bpy.data.materials:
            try:
                if inject_group_into_material(material, mask_image, group, lo, hi, gamma, logger):
                    injected += 1
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("[WARN] Material '%s' injection failed: %s", material.name, exc)
    logger.info("[step] Injected foveation group into %d materials", injected)

    with StageTimer("render.execute", logger=logger, records=timings):
        bpy.ops.render.render(write_still=True)
    logger.info("[OK] Final render saved to %s", paths.final.as_posix())

    summarize_timings(logger, timings)


if __name__ == "__main__":
    main()
