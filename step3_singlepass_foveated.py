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
    auto_bias: float
    auto_gamma: float
    auto_taper: float
    caustics: bool
    reuse_textures: bool
    lq_diffuse: bool
    mask_bias: float
    auto_thresholds: bool
    auto_lo_push: float
    auto_hi_pull: float
    auto_mask_bias_gain: float
    adaptive_boost: float
    adaptive_min_fraction: float
    light_tree: bool
    light_sampling_threshold: float
    light_threshold_boost: float
    importance_mid_weight: float
    min_floor_fraction: float
    max_bounces: int
    transparent_max_bounces: int
    bounce_floor: int
    bounce_taper: float
    lq_sample_bias: float
    lq_min_scale: float

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
        auto_bias = _env_clamped_float("SCDL_FOVEATED_AUTOSPP_FLOOR", default=0.35, min_value=0.0, max_value=1.0)
        auto_gamma = _env_clamped_float("SCDL_FOVEATED_AUTOSPP_GAMMA", default=1.35, min_value=0.5, max_value=4.0)
        auto_taper = _env_clamped_float("SCDL_FOVEATED_AUTOSPP_TAPER", default=0.65, min_value=0.1, max_value=1.0)
        caustics = _env_bool("SCDL_FOVEATED_CAUSTICS", default=False)
        reuse_textures = _env_bool("SCDL_FOVEATED_LQ_REUSE_TEXTURES", default=False)
        lq_diffuse = _env_bool("SCDL_FOVEATED_LQ_DIFFUSE", default=True)
        mask_bias = _env_clamped_float("SCDL_FOVEATED_MASK_BIAS", default=0.08, min_value=-0.5, max_value=0.5)
        auto_thresholds = _env_bool("SCDL_FOVEATED_AUTO_THRESH", default=True)
        auto_lo_push = _env_clamped_float("SCDL_FOVEATED_AUTO_LO_PUSH", default=0.25, min_value=0.0, max_value=0.6)
        auto_hi_pull = _env_clamped_float("SCDL_FOVEATED_AUTO_HI_PULL", default=0.15, min_value=0.0, max_value=0.6)
        auto_mask_bias_gain = _env_clamped_float("SCDL_FOVEATED_AUTO_MASK_BIAS_GAIN", default=0.12, min_value=0.0, max_value=0.5)
        adaptive_boost = _env_clamped_float("SCDL_FOVEATED_ADAPTIVE_BOOST", default=1.35, min_value=0.0, max_value=6.0)
        adaptive_min_fraction = _env_clamped_float(
            "SCDL_FOVEATED_ADAPTIVE_MIN_FRACTION", default=0.55, min_value=0.1, max_value=1.0
        )
        light_tree = _env_bool("SCDL_FOVEATED_LIGHT_TREE", default=True)
        light_sampling_threshold = _env_clamped_float(
            "SCDL_FOVEATED_LIGHT_SAMPLING_THRESHOLD", default=0.01, min_value=0.0, max_value=1.0
        )
        light_threshold_boost = _env_clamped_float(
            "SCDL_FOVEATED_LIGHT_THRESHOLD_BOOST", default=2.0, min_value=0.0, max_value=8.0
        )
        importance_mid_weight = _env_clamped_float(
            "SCDL_FOVEATED_IMPORTANCE_MID_WEIGHT", default=0.35, min_value=0.0, max_value=1.0
        )
        min_floor_fraction = _env_clamped_float(
            "SCDL_FOVEATED_MIN_FLOOR_FRACTION", default=0.5, min_value=0.1, max_value=1.0
        )
        lq_sample_bias = _env_clamped_float(
            "SCDL_FOVEATED_LQ_SAMPLE_BIAS", default=0.85, min_value=0.0, max_value=2.0
        )
        lq_min_scale = _env_clamped_float(
            "SCDL_FOVEATED_LQ_MIN_SCALE", default=0.35, min_value=0.05, max_value=1.0
        )
        max_bounces = _env_positive_int("SCDL_FOVEATED_MAX_BOUNCES", default=8)
        transparent_max_bounces = _env_positive_int("SCDL_FOVEATED_TRANSPARENT_BOUNCES", default=max_bounces)
        bounce_floor = _env_positive_int("SCDL_FOVEATED_BOUNCE_FLOOR", default=4)
        bounce_taper = _env_clamped_float("SCDL_FOVEATED_BOUNCE_TAPER", default=0.55, min_value=0.1, max_value=1.0)

        if min_samples > base_samples:
            raise RuntimeError("[Step3] SCDL_FOVEATED_MIN_SPP cannot exceed SCDL_FOVEATED_BASE_SPP.")
        if max_samples < base_samples:
            raise RuntimeError("[Step3] SCDL_FOVEATED_MAX_SPP cannot be lower than SCDL_FOVEATED_BASE_SPP.")
        if adaptive_min_samples < min_samples:
            raise RuntimeError("[Step3] SCDL_ADAPTIVE_MIN_SAMPLES must be ≥ SCDL_FOVEATED_MIN_SPP.")
        if bounce_floor > max_bounces or bounce_floor > transparent_max_bounces:
            raise RuntimeError("[Step3] SCDL_FOVEATED_BOUNCE_FLOOR cannot exceed max/transparent bounce limits.")

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
            auto_bias=auto_bias,
            auto_gamma=auto_gamma,
            auto_taper=auto_taper,
            caustics=caustics,
            reuse_textures=reuse_textures,
            lq_diffuse=lq_diffuse,
            mask_bias=mask_bias,
            auto_thresholds=auto_thresholds,
            auto_lo_push=auto_lo_push,
            auto_hi_pull=auto_hi_pull,
            auto_mask_bias_gain=auto_mask_bias_gain,
            adaptive_boost=adaptive_boost,
            adaptive_min_fraction=adaptive_min_fraction,
            light_tree=light_tree,
            light_sampling_threshold=light_sampling_threshold,
            light_threshold_boost=light_threshold_boost,
            importance_mid_weight=importance_mid_weight,
            min_floor_fraction=min_floor_fraction,
            lq_sample_bias=lq_sample_bias,
            lq_min_scale=lq_min_scale,
            max_bounces=max_bounces,
            transparent_max_bounces=transparent_max_bounces,
            bounce_floor=bounce_floor,
            bounce_taper=bounce_taper,
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


def _env_clamped_float(name: str, *, default: float, min_value: float, max_value: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"[Step3] {name} must be a float (received '{raw}').") from exc
    if not (min_value <= value <= max_value):
        raise RuntimeError(
            f"[Step3] {name} must be between {min_value} and {max_value} (received {value})."
        )
    return value


@dataclass(frozen=True)
class MaskStats:
    """Summary statistics derived from the user saliency mask."""

    lo: float
    hi: float
    gamma: float
    coverage: float  # mean value across the mask (legacy coverage metric)
    coverage_hq: float
    coverage_lq: float
    coverage_mid: float
    p_lo: float
    p_hi: float
    mask_bias_used: float
    raw_lo: float
    raw_hi: float


def apply_threshold_bias(lo: float, hi: float, bias: float) -> Tuple[float, float]:
    """Shift thresholds while keeping them ordered and clamped."""

    lo_b = max(0.0, min(1.0, lo + bias))
    hi_b = max(lo_b + 1e-6, min(1.0, hi + bias))
    return lo_b, hi_b


def mask_statistics(np_path: Path, cfg: FoveatedConfig) -> MaskStats:
    """Compute thresholds, gamma, and coverage metrics from the saliency mask."""

    arr = np.load(np_path).astype(np.float32)
    flat = arr[np.isfinite(arr)].ravel()
    if flat.size == 0:
        return MaskStats(
            lo=0.25,
            hi=0.75,
            gamma=2.2,
            coverage=0.0,
            coverage_hq=0.0,
            coverage_lq=0.0,
            coverage_mid=0.0,
            p_lo=0.2,
            p_hi=0.8,
            mask_bias_used=cfg.mask_bias,
            raw_lo=0.25,
            raw_hi=0.75,
        )

    coverage = float(flat.mean())

    base_p_lo, base_p_hi = 0.2, 0.8
    if cfg.auto_thresholds:
        # Push the low threshold upward (bigger LQ) as coverage shrinks.
        p_lo = min(0.9, base_p_lo + (1.0 - coverage) * cfg.auto_lo_push)
        # Pull the high threshold upward modestly; keep spacing above p_lo.
        p_hi_candidate = base_p_hi + (coverage - 0.5) * cfg.auto_hi_pull
        p_hi = max(p_lo + 0.05, min(0.99, p_hi_candidate))
        bias_dynamic = cfg.mask_bias + (1.0 - coverage) * cfg.auto_mask_bias_gain
    else:
        p_lo = base_p_lo
        p_hi = base_p_hi
        bias_dynamic = cfg.mask_bias

    q_lo_base, q_hi_base = np.quantile(flat, [p_lo, p_hi])

    q_hi = float(q_hi_base)
    q_lo = float(q_lo_base)

    # Maintain ordering before bias.
    if q_hi <= q_lo:
        q_hi = q_lo + 1e-4

    lo, hi = apply_threshold_bias(q_lo, q_hi, bias_dynamic)
    if hi <= lo:
        hi = min(1.0, lo + 1e-4)

    coverage_hq = float((flat >= hi).mean())
    coverage_lq = float((flat <= lo).mean())
    coverage_mid = max(0.0, min(1.0, 1.0 - coverage_hq - coverage_lq))

    return MaskStats(
        lo=lo,
        hi=hi,
        gamma=2.2,
        coverage=coverage,
        coverage_hq=coverage_hq,
        coverage_lq=coverage_lq,
        coverage_mid=coverage_mid,
        p_lo=float(p_lo),
        p_hi=float(p_hi),
        mask_bias_used=float(bias_dynamic),
        raw_lo=float(q_lo_base),
        raw_hi=float(q_hi_base),
    )


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


def _socket_scalar_default(socket: Optional[bpy.types.NodeSocket], fallback: float) -> float:
    if socket is None:
        return fallback
    try:
        return float(socket.default_value)
    except Exception:
        return fallback


def _socket_color_default(
    socket: Optional[bpy.types.NodeSocket], fallback: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    if socket is None:
        return fallback
    try:
        raw = tuple(socket.default_value)
    except Exception:
        return fallback
    if len(raw) >= 4:
        return tuple(float(x) for x in raw[:4])
    if len(raw) == 3:
        return float(raw[0]), float(raw[1]), float(raw[2]), 1.0
    return fallback


def build_lq_diffuse(
    node_tree: bpy.types.NodeTree,
    ref_principled: bpy.types.Node,
    *,
    reuse_textures: bool,
) -> bpy.types.Node:
    """Construct a lightweight diffuse BSDF for low-importance regions."""

    diffuse = node_tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.label = "LQ_Diffuse"
    diffuse.inputs["Roughness"].default_value = 0.9

    if reuse_textures:
        link_or_copy(node_tree, ref_principled, ["Base Color"], diffuse, "Color")
        link_or_copy(node_tree, ref_principled, ["Normal"], diffuse, "Normal")
    else:
        fallback_color = _socket_color_default(
            get_input(ref_principled, "Base Color"),
            fallback=(0.5, 0.5, 0.5, 1.0),
        )
        color = get_input(diffuse, "Color")
        if color:
            color.default_value = fallback_color
    return diffuse


def build_lq_principled(
    node_tree: bpy.types.NodeTree,
    ref_principled: bpy.types.Node,
    *,
    reuse_textures: bool,
) -> bpy.types.Node:
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

    if reuse_textures:
        link_or_copy(node_tree, ref_principled, ["Base Color"], lq, "Base Color")
        link_or_copy(node_tree, ref_principled, ["Roughness"], lq, "Roughness")
        link_or_copy(node_tree, ref_principled, ["Normal"], lq, "Normal")
    else:
        fallback_color = _socket_color_default(
            get_input(ref_principled, "Base Color"),
            fallback=(0.55, 0.55, 0.55, 1.0),
        )
        base_color = get_input(lq, "Base Color")
        if base_color:
            base_color.default_value = fallback_color

        src_roughness = get_input(ref_principled, "Roughness")
        dst_roughness = get_input(lq, "Roughness")
        if dst_roughness:
            dst_roughness.default_value = max(
                dst_roughness.default_value,
                _socket_scalar_default(src_roughness, dst_roughness.default_value),
            )

    src_met = get_input(ref_principled, "Metallic")
    dst_met = get_input(lq, "Metallic")
    if src_met and dst_met:
        if src_met.is_linked and reuse_textures:
            node_tree.links.new(src_met.links[0].from_socket, dst_met)
        else:
            dst_met.default_value = 0.5 * _socket_scalar_default(src_met, dst_met.default_value)
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


def _find_existing_group_instance(node_tree: bpy.types.NodeTree, group: bpy.types.NodeTree) -> Optional[bpy.types.Node]:
    for node in node_tree.nodes:
        if node.type == "GROUP" and getattr(node, "node_tree", None) == group:
            return node
    return None


def _configure_mask_node(tex_image: bpy.types.ShaderNodeTexImage, mask_image: bpy.types.Image) -> None:
    tex_image.image = mask_image
    tex_image.label = tex_image.label or "FoveaMask"
    tex_image.interpolation = "Linear"
    tex_image.extension = "CLIP"
    tex_image.projection = "FLAT"
    tex_image.image.colorspace_settings.name = "Non-Color"


def _build_mask_nodes(node_tree: bpy.types.NodeTree, mask_image: bpy.types.Image) -> Tuple[bpy.types.Node, bpy.types.ShaderNodeTexImage]:
    tex_coords = node_tree.nodes.new("ShaderNodeTexCoord")
    tex_coords.label = "Coords"
    tex_image = node_tree.nodes.new("ShaderNodeTexImage")
    _configure_mask_node(tex_image, mask_image)
    node_tree.links.new(tex_coords.outputs["Window"], tex_image.inputs["Vector"])
    return tex_coords, tex_image


def _ensure_mask_link(
    node_tree: bpy.types.NodeTree,
    group_inst: bpy.types.Node,
    mask_image: bpy.types.Image,
) -> bool:
    """Refresh or create the mask texture feeding a foveation group."""

    mask_input = group_inst.inputs.get("Mask")
    if mask_input and mask_input.is_linked:
        link = mask_input.links[0]
        src_node = getattr(link.from_socket, "node", None)
        if src_node and src_node.type == "TEX_IMAGE":
            _configure_mask_node(src_node, mask_image)
            return True

    for node in node_tree.nodes:
        if node.type == "TEX_IMAGE" and (node.label == "FoveaMask" or _norm(node.name).startswith("foveamask")):
            _configure_mask_node(node, mask_image)
            if mask_input and not mask_input.is_linked:
                mask_output = node.outputs.get("Alpha") or node.outputs.get("Color")
                if mask_output:
                    node_tree.links.new(mask_output, mask_input)
                    return True
            elif mask_input:
                return True
    return False


def _ensure_lq_link(
    node_tree: bpy.types.NodeTree,
    group_inst: bpy.types.Node,
    ref_principled: bpy.types.Node,
    *,
    reuse_textures: bool,
    lq_diffuse: bool,
) -> None:
    """Make sure the LQ shader input matches the requested simplification."""

    lq_input = group_inst.inputs.get("LQ Shader")
    if lq_input is None:
        return

    current = None
    if lq_input.is_linked:
        current = getattr(lq_input.links[0].from_socket, "node", None)

    wants_diffuse = lq_diffuse
    if wants_diffuse and (current is None or current.type != "BSDF_DIFFUSE"):
        lq_shader = build_lq_diffuse(node_tree, ref_principled, reuse_textures=reuse_textures)
        node_tree.links.new(lq_shader.outputs["BSDF"], lq_input)
    elif not wants_diffuse and (current is None or current.type != "BSDF_PRINCIPLED"):
        lq_shader = build_lq_principled(node_tree, ref_principled, reuse_textures=reuse_textures)
        node_tree.links.new(lq_shader.outputs["BSDF"], lq_input)


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
    reuse_textures: bool,
    lq_diffuse: bool,
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

    existing = _find_existing_group_instance(node_tree, group)
    if existing and output.inputs["Surface"].links and output.inputs["Surface"].links[0].from_node == existing:
        ref_principled = find_principled(node_tree)
        _ensure_lq_link(node_tree, existing, ref_principled, reuse_textures=reuse_textures, lq_diffuse=lq_diffuse)
        updated_mask = _ensure_mask_link(node_tree, existing, mask_image)
        if not updated_mask:
            _, tex_image = _build_mask_nodes(node_tree, mask_image)
            mask_output = tex_image.outputs.get("Alpha") or tex_image.outputs.get("Color")
            if mask_output and existing.inputs.get("Mask"):
                node_tree.links.new(mask_output, existing.inputs["Mask"])

        existing.inputs["LoThr"].default_value = float(lo)
        existing.inputs["HiThr"].default_value = float(hi)
        existing.inputs["Gamma"].default_value = float(gamma)
        return True

    tex_coords, tex_image = _build_mask_nodes(node_tree, mask_image)

    group_inst = node_tree.nodes.new("ShaderNodeGroup")
    group_inst.node_tree = group
    group_inst.label = "FoveationMix"
    ref_principled = find_principled(node_tree)
    if lq_diffuse:
        lq_shader = build_lq_diffuse(node_tree, ref_principled, reuse_textures=reuse_textures)
        lq_output_name = "BSDF"
    else:
        lq_shader = build_lq_principled(node_tree, ref_principled, reuse_textures=reuse_textures)
        lq_output_name = "BSDF"

    mask_output = tex_image.outputs.get("Alpha") or tex_image.outputs.get("Color")
    if mask_output is None:
        logger.warning("[WARN] Material '%s' mask has no usable output; skipping", material.name)
        return False

    node_tree.links.new(original_socket, group_inst.inputs["HQ Shader"])
    node_tree.links.new(lq_shader.outputs[lq_output_name], group_inst.inputs["LQ Shader"])
    node_tree.links.new(mask_output, group_inst.inputs["Mask"])
    group_inst.inputs["LoThr"].default_value = float(lo)
    group_inst.inputs["HiThr"].default_value = float(hi)
    group_inst.inputs["Gamma"].default_value = float(gamma)

    node_tree.links.remove(original_link)
    node_tree.links.new(group_inst.outputs["Shader"], output.inputs["Surface"])
    return True


@dataclass(frozen=True)
class SamplingResult:
    """Effective sampling and adaptive settings applied to the scene."""

    effective_samples: int
    sample_scale: float
    lq_scale: float
    adaptive_threshold: float
    adaptive_min_samples: int
    light_threshold: float
    importance: float
    taper: float
    min_floor: int
    max_bounces: int
    transparent_bounces: int


def configure_cycles(
    scene: bpy.types.Scene, cfg: FoveatedConfig, stats: MaskStats
) -> SamplingResult:
    """Apply Cycles render settings and return the effective sampling stats."""

    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"

    cycles = scene.cycles
    cycles.device = "GPU"
    cycles.use_fast_gi = False
    cycles.caustics_reflective = cfg.caustics
    cycles.caustics_refractive = cfg.caustics

    base_samples = cfg.base_samples
    hq_clamped = max(0.0, min(1.0, stats.coverage_hq))
    lq_clamped = max(0.0, min(1.0, stats.coverage_lq))
    mid_clamped = max(0.0, min(1.0, stats.coverage_mid if hasattr(stats, "coverage_mid") else 1.0 - hq_clamped - lq_clamped))
    # Emphasise the HQ footprint and down-weight mid-importance coverage to slash samples when the fovea is tiny.
    importance = max(0.0, min(1.0, hq_clamped + cfg.importance_mid_weight * mid_clamped))
    # LQ-heavy scenes should fall harder toward the minimum budget; apply a multiplicative throttle.
    lq_scale = max(cfg.lq_min_scale, 1.0 - cfg.lq_sample_bias * lq_clamped)
    # Taper pulls the global sample factor toward the minimum budget for low-coverage scenes.
    taper = cfg.auto_taper + (1.0 - cfg.auto_taper) * importance
    sample_scale = 1.0
    min_floor = cfg.min_samples
    if cfg.auto_samples:
        coverage_term = importance ** cfg.auto_gamma
        sample_scale = taper * (cfg.auto_bias + (1.0 - cfg.auto_bias) * coverage_term) * lq_scale
        min_floor_base = int(
            round(cfg.min_samples * (cfg.min_floor_fraction + (1.0 - cfg.min_floor_fraction) * importance))
        )
        min_floor = max(1, min(cfg.min_samples, int(round(min_floor_base * lq_scale))))
        min_floor = max(1, min(cfg.min_samples, min_floor))
        scaled = int(round(base_samples * sample_scale))
        effective = max(min_floor, min(cfg.max_samples, scaled))
    else:
        effective = base_samples
        sample_scale = effective / base_samples if base_samples else 1.0
        taper = 1.0
        min_floor = cfg.min_samples
    cycles.samples = effective

    adaptive_min_scale = cfg.adaptive_min_fraction + (1.0 - cfg.adaptive_min_fraction) * importance
    adaptive_min = int(round(cfg.adaptive_min_samples * adaptive_min_scale * lq_scale))
    adaptive_min = max(1, min(cfg.adaptive_min_samples, adaptive_min, effective))
    adaptive_min = min(adaptive_min, effective)

    adaptive_scale = 1.0 + (1.0 - importance) * cfg.adaptive_boost
    adaptive_scale *= 1.0 + (1.0 - lq_scale) * 0.35
    adaptive_threshold = cfg.adaptive_threshold * adaptive_scale

    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = adaptive_threshold
    cycles.adaptive_min_samples = adaptive_min

    bounce_scale = (cfg.bounce_taper + (1.0 - cfg.bounce_taper) * importance) * (0.8 + 0.2 * lq_scale)
    max_bounces = int(round(cfg.max_bounces * bounce_scale))
    max_bounces = max(cfg.bounce_floor, min(cfg.max_bounces, max_bounces))
    transparent_bounces = int(round(cfg.transparent_max_bounces * bounce_scale))
    transparent_bounces = max(cfg.bounce_floor, min(cfg.transparent_max_bounces, transparent_bounces))
    cycles.max_bounces = max_bounces
    cycles.transparent_max_bounces = transparent_bounces

    cycles.use_denoising = True
    cycles.denoiser = "OPTIX"
    for view_layer in scene.view_layers:
        view_layer.cycles.use_denoising = True

    cycles.blur_glossy = cfg.filter_glossy
    if cfg.clamp_direct is not None:
        cycles.sample_clamp_direct = cfg.clamp_direct
    if cfg.clamp_indirect is not None:
        cycles.sample_clamp_indirect = cfg.clamp_indirect

    light_threshold = cfg.light_sampling_threshold
    if hasattr(cycles, "use_light_tree"):
        cycles.use_light_tree = cfg.light_tree
    if hasattr(cycles, "light_sampling_threshold"):
        boosted_threshold = light_threshold * (1.0 + (1.0 - importance) * cfg.light_threshold_boost)
        boosted_threshold *= 1.0 + (1.0 - lq_scale) * 0.5
        light_threshold = min(1.0, boosted_threshold)
        cycles.light_sampling_threshold = light_threshold

    return SamplingResult(
        effective_samples=effective,
        sample_scale=sample_scale,
        lq_scale=lq_scale,
        adaptive_threshold=adaptive_threshold,
        adaptive_min_samples=adaptive_min,
        light_threshold=light_threshold,
        importance=importance,
        taper=taper,
        min_floor=min_floor,
        max_bounces=max_bounces,
        transparent_bounces=transparent_bounces,
    )


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

    skip_validate = os.getenv("SCDL_SKIP_STEP3_VALIDATE", "0").strip().lower() in {"1", "true", "yes", "on"}
    matched_devices = []
    devices = []

    if skip_validate:
        logger.info("[system] Validation skipped (SCDL_SKIP_STEP3_VALIDATE=1)")
        expected_gpu = os.getenv("SCDL_EXPECTED_GPU", "RTX 3060")
    else:
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
            "auto_bias": cfg.auto_bias,
            "auto_gamma": cfg.auto_gamma,
            "auto_taper": cfg.auto_taper,
            "caustics": cfg.caustics,
            "expected_gpu": expected_gpu,
            "reuse_textures": cfg.reuse_textures,
            "lq_diffuse": cfg.lq_diffuse,
            "mask_bias": cfg.mask_bias,
            "auto_thresholds": cfg.auto_thresholds,
            "auto_lo_push": cfg.auto_lo_push,
            "auto_hi_pull": cfg.auto_hi_pull,
            "auto_mask_bias_gain": cfg.auto_mask_bias_gain,
            "adaptive_boost": cfg.adaptive_boost,
            "adaptive_min_fraction": cfg.adaptive_min_fraction,
            "light_tree": cfg.light_tree,
            "light_sampling_threshold": cfg.light_sampling_threshold,
            "light_threshold_boost": cfg.light_threshold_boost,
            "importance_mid_weight": cfg.importance_mid_weight,
            "min_floor_fraction": cfg.min_floor_fraction,
            "lq_sample_bias": cfg.lq_sample_bias,
            "lq_min_scale": cfg.lq_min_scale,
            "max_bounces": cfg.max_bounces,
            "transparent_max_bounces": cfg.transparent_max_bounces,
            "bounce_floor": cfg.bounce_floor,
            "bounce_taper": cfg.bounce_taper,
        },
    )

    if not paths.mask_exr.exists():
        raise RuntimeError(f"[Step3] Mask image not found at {paths.mask_exr}.")
    if not paths.mask_npy.exists():
        raise RuntimeError(f"[Step3] Mask array not found at {paths.mask_npy}.")

    timings: List[TimerRecord] = []

    with StageTimer("mask.load", logger=logger, records=timings):
        mask_stats = mask_statistics(paths.mask_npy, cfg)
        lo, hi, gamma = mask_stats.lo, mask_stats.hi, mask_stats.gamma
        mask_image = load_mask_image(paths.mask_exr)
        logger.info(
            "[env] thresholds lo=%.4f hi=%.4f gamma=%.2f coverage=%.3f hq=%.3f mid=%.3f lq=%.3f (raw_lo=%.4f raw_hi=%.4f bias=%.3f bias_used=%.3f p_lo=%.2f p_hi=%.2f)",
            mask_stats.lo,
            mask_stats.hi,
            mask_stats.gamma,
            mask_stats.coverage,
            mask_stats.coverage_hq,
            mask_stats.coverage_mid,
            mask_stats.coverage_lq,
            mask_stats.raw_lo,
            mask_stats.raw_hi,
            cfg.mask_bias,
            mask_stats.mask_bias_used,
            mask_stats.p_lo,
            mask_stats.p_hi,
        )

    scene = bpy.context.scene
    scene.render.filepath = str(paths.final)

    with StageTimer("cycles.configure", logger=logger, records=timings):
        sampling = configure_cycles(scene, cfg, mask_stats)
        logger.info(
            "[env] effective_samples=%d (floor=%d scale=%.3f lq_scale=%.3f coverage=%.3f mid=%.3f hq=%.3f taper=%.3f importance=%.3f bounces=%d/%d adaptive_threshold=%.4f adaptive_min=%d light_tree=%s light_thresh=%.3f)",
            sampling.effective_samples,
            sampling.min_floor,
            sampling.sample_scale,
            sampling.lq_scale,
            mask_stats.coverage,
            mask_stats.coverage_mid,
            mask_stats.coverage_hq,
            sampling.taper,
            sampling.importance,
            sampling.max_bounces,
            sampling.transparent_bounces,
            sampling.adaptive_threshold,
            sampling.adaptive_min_samples,
            cfg.light_tree,
            sampling.light_threshold,
        )

    group = ensure_foveation_group()
    injected = 0

    with StageTimer("materials.inject", logger=logger, records=timings):
        for material in bpy.data.materials:
            try:
                if inject_group_into_material(
                    material,
                    mask_image,
                    group,
                    lo,
                    hi,
                    gamma,
                    cfg.reuse_textures,
                    cfg.lq_diffuse,
                    logger,
                ):
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
