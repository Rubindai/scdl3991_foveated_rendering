#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root regardless of where this script is called from
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IS_TTY=0
if [ -t 1 ]; then
  IS_TTY=1
fi

ENV_LOADED=0

# ======== CONFIG ========
# Optional: auto-load repo-local env overrides (e.g., SCDL_USE_GPU, SCDL_HALTTIME, etc.)
if [ -f .scdl.env ]; then
  set -a
  # shellcheck disable=SC1090
  . ./.scdl.env
  set +a
  ENV_LOADED=1
fi

LOG_MODE="${SCDL_LOG_MODE:-both}"
LOG_MODE="${LOG_MODE,,}"
LOG_PATH="${SCDL_LOG_FILE:-$PWD/out/scdl_pipeline.log}"
ANSI_STRIP_EXPR=$'s/\x1B\[[0-9;]*[A-Za-z]//g'
case "$LOG_MODE" in
  both)
    mkdir -p "$(dirname -- "$LOG_PATH")"
    exec > >(tee >(sed -u -E "$ANSI_STRIP_EXPR" >> "$LOG_PATH")) 2>&1
    ;;
  file)
    mkdir -p "$(dirname -- "$LOG_PATH")"
    : > "$LOG_PATH"
    exec > >(sed -u -E "$ANSI_STRIP_EXPR" >> "$LOG_PATH") 2>&1
    ;;
  stdout)
    ;;
  *)
    LOG_MODE="both"
    mkdir -p "$(dirname -- "$LOG_PATH")"
    exec > >(tee >(sed -u -E "$ANSI_STRIP_EXPR" >> "$LOG_PATH")) 2>&1
    ;;
esac

USE_COLOR=0
if { [ "$LOG_MODE" = "stdout" ] || [ "$LOG_MODE" = "both" ]; } && [ "$IS_TTY" -eq 1 ]; then
  USE_COLOR=1
fi

if [ "$USE_COLOR" -eq 1 ]; then
  COLOR_INFO=$'\033[0;36m'
  COLOR_WARN=$'\033[0;33m'
  COLOR_ERROR=$'\033[0;31m'
  COLOR_STEP=$'\033[1;34m'
  COLOR_CMD=$'\033[0;32m'
  COLOR_ENV=$'\033[0;35m'
  COLOR_OK=$'\033[0;32m'
  COLOR_RESET=$'\033[0m'
else
  COLOR_INFO=""
  COLOR_WARN=""
  COLOR_ERROR=""
  COLOR_STEP=""
  COLOR_CMD=""
  COLOR_ENV=""
  COLOR_OK=""
  COLOR_RESET=""
fi

log_emit() {
  local color="$1"; shift
  local prefix="$1"; shift
  local message="$*"
  if [ -n "$prefix" ]; then
    if [ "$USE_COLOR" -eq 1 ] && [ -n "$color" ]; then
      printf '%b%s%b %s\n' "$color" "$prefix" "$COLOR_RESET" "$message"
    else
      printf '%s %s\n' "$prefix" "$message"
    fi
  else
    if [ "$USE_COLOR" -eq 1 ] && [ -n "$color" ]; then
      printf '%b%s%b\n' "$color" "$message" "$COLOR_RESET"
    else
      printf '%s\n' "$message"
    fi
  fi
}

log_info() {
  local msg="$*"
  if [[ "$msg" == \[OK\]* ]]; then
    msg="${msg#\[OK\]}"
    msg="${msg# }"
    log_emit "$COLOR_OK" "[OK]" "$msg"
  else
    log_emit "$COLOR_INFO" "[INFO]" "$msg"
  fi
}
log_warn() { log_emit "$COLOR_WARN" "[WARN]" "$*"; }
log_error() { log_emit "$COLOR_ERROR" "[ERROR]" "$*"; }
log_step() { local tag="[$1]"; shift; log_emit "$COLOR_STEP" "$tag" "$*"; }
log_cmd() {
  if [ "$USE_COLOR" -eq 1 ] && [ -n "$COLOR_CMD" ]; then
    printf '%b[CMD]%b %b%s%b\n' "$COLOR_CMD" "$COLOR_RESET" "$COLOR_CMD" "$*" "$COLOR_RESET"
  else
    printf '[CMD] %s\n' "$*"
  fi
}
log_env() { log_emit "$COLOR_ENV" "[env]" "$*"; }
log_done() { log_emit "$COLOR_OK" "[DONE]" "$*"; }

filter_blender_output() {
  local line stripped
  while IFS= read -r line || [ -n "$line" ]; do
    # Normalize carriage returns that Blender occasionally emits during progress updates
    stripped="${line//$'\r'/}"
    if [ -z "$stripped" ]; then
      printf '\n'
      continue
    fi
    case "$stripped" in
      Saved:*)
        log_info "$stripped"
        ;;
      BlendLuxCore*|Updating\ device\ list|Created\ history\ step*|Blender\ *|Read\ blend:*|Time:*Saving:*)
        log_info "$stripped"
        ;;
      Fra:*)
        printf '  %s\n' "$stripped"
        ;;
      WARN\ *|Warning:*)
        log_warn "$stripped"
        ;;
      Error:*|ERROR:*|Traceback*)
        log_error "$stripped"
        ;;
      *)
        printf '%s\n' "$stripped"
        ;;
    esac
  done
}

run_blender_cmd() {
  local status
  set +e
  "$@" 2>&1 | filter_blender_output
  status=${PIPESTATUS[0]}
  set -e
  return $status
}

die() {
  log_error "$*"
  exit 1
}

if [ "$ENV_LOADED" = "1" ]; then
  log_env "Loading .scdl.env overrides"
fi
# Windows Blender path (as seen from WSL). For Windows path D:\\Programs\\blender\\blender.exe
# the WSL path is /mnt/d/Programs/blender/blender.exe
BLENDER_EXE="${BLENDER_EXE:-/mnt/d/Programs/blender/blender.exe}"

# Project file (relative to repo root)
# Priority: CLI arg > BLEND_FILE from env/.scdl.env > default
BLEND_FILE="${1:-${BLEND_FILE:-blender_files/cookie.blend}}"
PREVIEW_SCRIPT="step1_preview_blender.py"              # renders out/preview.png next to the .blend
FINAL_SCRIPT="step3_blender_roi_compose.py"            # ROI + composite to out/final.png
EXPORT_SCRIPT="step3_optional_luxcore_export.py"       # exports render.cfg + scene.scn to ./export
WSL_RENDER_SCRIPT="step4_optional_luxcore_render.py"  # optional LuxCore final in WSL
WSL_DINO_SCRIPT="step2_dino_mask.py"

# Optional: conda env for the WSL render script
if [ -z "${CONDA_EXE:-}" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
  elif [ -x "$HOME/miniconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniconda3/bin/conda"
  elif [ -x "$HOME/anaconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/anaconda3/bin/conda"
  else
    CONDA_EXE=""
  fi
fi
if [ -n "$CONDA_EXE" ]; then
  log_info "Using conda executable: $CONDA_EXE"
else
  log_info "CONDA_EXE not set; will rely on system Python for WSL steps."
fi
CONDA_ENV="${CONDA_ENV:-base}"   # must have torch, timm, pyluxcore, imageio, numpy
# ========================

# Preflight checks
command -v wslpath >/dev/null 2>&1 || die "wslpath not found. Run under WSL."
[ -f "$BLEND_FILE" ] || die "Blend file not found: $BLEND_FILE"
[ -f "$EXPORT_SCRIPT" ] || die "Exporter script missing: $EXPORT_SCRIPT"
[ -f "$WSL_RENDER_SCRIPT" ] || die "WSL render script missing: $WSL_RENDER_SCRIPT"
[ -f "$WSL_DINO_SCRIPT" ] || die "WSL DINO script missing: $WSL_DINO_SCRIPT"
[ -f "$BLENDER_EXE" ] || die "Blender.exe not found at $BLENDER_EXE. Set BLENDER_EXE=/mnt/<drive>/path/to/blender.exe"

TOTAL_STEPS=3

# Skip the LuxCore export unless the WSL render (step 4) is requested or explicitly forced
SHOULD_EXPORT=0
if [ "${SCDL_USE_WSL_FINAL:-0}" = "1" ] || [ "${SCDL_FORCE_LUXCORE_EXPORT:-0}" = "1" ]; then
  SHOULD_EXPORT=1
  TOTAL_STEPS=4
fi

STEP=1
log_step "${STEP}/${TOTAL_STEPS}" "Rendering preview via Windows Blender..."

# Convert Linux paths to Windows for Blender.exe invocation
WIN_BLEND_FILE="$(wslpath -w "$PWD/$BLEND_FILE")"
WIN_PREVIEW_SCRIPT="$(wslpath -w "$PWD/$PREVIEW_SCRIPT")"
BLENDER_ARGS=(--addons cycles)
if [ -n "${SCDL_CYCLES_DEVICE:-}" ]; then
  log_info "SCDL_CYCLES_DEVICE=${SCDL_CYCLES_DEVICE}"
else
  log_info "SCDL_CYCLES_DEVICE not set; using Blender default device selection."
fi

PREVIEW_CMD=("$BLENDER_EXE" "$WIN_BLEND_FILE" -b -E CYCLES "${BLENDER_ARGS[@]}" -P "$WIN_PREVIEW_SCRIPT")
if [ -n "${SCDL_CYCLES_DEVICE:-}" ]; then
  PREVIEW_CMD+=(-- --cycles-device "${SCDL_CYCLES_DEVICE}")
fi
printf -v _cmd_str ' %q' "${PREVIEW_CMD[@]}"
log_cmd "${_cmd_str# }"
run_blender_cmd "${PREVIEW_CMD[@]}" || die "Blender preview step failed."

# Verify preview
[ -f out/preview.png ]   || die "Missing out/preview.png after Blender preview."

STEP=$((STEP + 1))
log_step "${STEP}/${TOTAL_STEPS}" "Computing DINO mask in WSL..."

# Ensure the Python scripts resolve project paths to this repo
export SCDL_PROJECT_DIR="$PWD"
python "$WSL_DINO_SCRIPT" || die "DINO mask step failed."
[ -f out/user_importance.npy ] || die "Missing out/user_importance.npy after DINO."

if [ "$SHOULD_EXPORT" = "1" ]; then
  STEP=$((STEP + 1))
  log_step "${STEP}/${TOTAL_STEPS}" "Exporting LuxCore SDL via Windows Blender..."

  WIN_EXPORT_SCRIPT="$(wslpath -w "$PWD/$EXPORT_SCRIPT")"
  EXPORT_CMD=("$BLENDER_EXE" "$WIN_BLEND_FILE" -b -E CYCLES "${BLENDER_ARGS[@]}" -P "$WIN_EXPORT_SCRIPT")
  if [ -n "${SCDL_CYCLES_DEVICE:-}" ]; then
    EXPORT_CMD+=(-- --cycles-device "${SCDL_CYCLES_DEVICE}")
  fi
  printf -v _cmd_str ' %q' "${EXPORT_CMD[@]}"
  log_cmd "${_cmd_str# }"
  run_blender_cmd "${EXPORT_CMD[@]}" || die "Blender export step failed."

  # Locate exported SDL; some BlendLuxCore versions place files in subfolders
  CFG_FOUND=""
  SCN_FOUND=""
  if [ -f export/render.cfg ]; then CFG_FOUND="export/render.cfg"; fi
  if [ -f export/scene.scn ];  then SCN_FOUND="export/scene.scn"; fi
  if [ -z "$CFG_FOUND" ] || [ -z "$SCN_FOUND" ]; then
    log_info "Searching for exported SDL inside ./export ..."
    [ -z "$CFG_FOUND" ] && CFG_FOUND="$(find export -type f -name 'render.cfg' -print0 2>/dev/null | xargs -0 -r ls -t 2>/dev/null | head -n1 || true)"
    [ -z "$SCN_FOUND" ] && SCN_FOUND="$(find export -type f -name 'scene.scn'  -print0 2>/dev/null | xargs -0 -r ls -t 2>/dev/null | head -n1 || true)"
  fi
  [ -n "$CFG_FOUND" ] || die "Missing export/render.cfg after Blender export."
  [ -n "$SCN_FOUND" ]  || die "Missing export/scene.scn after Blender export."
  log_info "Using SDL: $CFG_FOUND (scene: $SCN_FOUND)"

  # Export the exact cfg path so the WSL render uses the correct resolver base
  export SCDL_CFG_PATH="$(realpath "$CFG_FOUND")"
else
  log_info "Skipping LuxCore export (SCDL_USE_WSL_FINAL=0). Set SCDL_FORCE_LUXCORE_EXPORT=1 to force it."
fi

# Step 4: optional LuxCore render inside WSL (triggered via SCDL_USE_WSL_FINAL=1)
if [ "${SCDL_USE_WSL_FINAL:-0}" = "1" ]; then
  STEP=$((STEP + 1))
  log_step "${STEP}/${TOTAL_STEPS}" "Running LuxCore final render in WSL..."
  [ -n "${SCDL_CFG_PATH:-}" ] || die "SCDL_CFG_PATH not set; rerun with successful LuxCore export."
  # Activate conda env and run
  if [ -x "$CONDA_EXE" ]; then
    # Some conda activate.d scripts assume unset vars are allowed; disable nounset temporarily
    set +u
    # shellcheck disable=SC1090
    source "$("$CONDA_EXE" info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    set -u
  else
    log_warn "CONDA_EXE not found; using system python."
  fi
  python "$WSL_RENDER_SCRIPT"
else
  STEP=$((STEP + 1))
  log_step "${STEP}/${TOTAL_STEPS}" "Rendering ROI + composite in Windows Blender..."
  WIN_FINAL_SCRIPT="$(wslpath -w "$PWD/$FINAL_SCRIPT")"
  FINAL_CMD=("$BLENDER_EXE" "$WIN_BLEND_FILE" -b -E CYCLES "${BLENDER_ARGS[@]}" -P "$WIN_FINAL_SCRIPT")
  if [ -n "${SCDL_CYCLES_DEVICE:-}" ]; then
    FINAL_CMD+=(-- --cycles-device "${SCDL_CYCLES_DEVICE}")
  fi
  printf -v _cmd_str ' %q' "${FINAL_CMD[@]}"
  log_cmd "${_cmd_str# }"
  run_blender_cmd "${FINAL_CMD[@]}" || die "Blender final (ROI) step failed."
fi

# Verify final output
[ -f out/final.png ] || die "Final render not found at out/final.png"

log_done "Outputs: out/preview.png, out/user_importance.npy, export/render.cfg, export/scene.scn, out/final.png"
