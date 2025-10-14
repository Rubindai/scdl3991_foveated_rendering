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
log_cmd_array() {
  local formatted
  printf -v formatted ' %q' "$@"
  log_cmd "${formatted# }"
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
      Updating\ device\ list|Created\ history\ step*|Blender\ *|Read\ blend:*|Time:*Saving:*)
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
FINAL_SCRIPT="step3_singlepass_foveated.py"            # Single-pass foveated render to out/final.png
WSL_DINO_SCRIPT="step2_dino_mask.py"

# ========================

# Preflight checks
command -v wslpath >/dev/null 2>&1 || die "wslpath not found. Run under WSL."
[ -f "$BLEND_FILE" ] || die "Blend file not found: $BLEND_FILE"
[ -f "$WSL_DINO_SCRIPT" ] || die "WSL DINO script missing: $WSL_DINO_SCRIPT"
[ -f "$BLENDER_EXE" ] || die "Blender.exe not found at $BLENDER_EXE. Set BLENDER_EXE=/mnt/<drive>/path/to/blender.exe"

# Base stage count matches the documented pipeline steps (preview, DINO, final composite).
TOTAL_STEPS=3

# ----- Step 1: Preview render in Windows Blender -----
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
log_cmd_array "${PREVIEW_CMD[@]}"
run_blender_cmd "${PREVIEW_CMD[@]}" || die "Blender preview step failed."

# Verify preview
[ -f out/preview.png ]   || die "Missing out/preview.png after Blender preview."

# ----- Step 2: DINO mask in WSL -----
STEP=$((STEP + 1))
log_step "${STEP}/${TOTAL_STEPS}" "Computing DINO mask in WSL..."

# Ensure the Python scripts resolve project paths to this repo
export SCDL_PROJECT_DIR="$PWD"
DINO_CMD=(conda run -n scdl-foveated python "$WSL_DINO_SCRIPT")
log_cmd_array "${DINO_CMD[@]}"
"${DINO_CMD[@]}" || die "DINO mask step failed."
[ -f out/user_importance.npy ] || die "Missing out/user_importance.npy after DINO."


# ----- Step 3: ROI + composite in Windows Blender -----
STEP=$((STEP + 1))
log_step "${STEP}/${TOTAL_STEPS}" "Rendering final image (single-pass foveated) in Windows Blender..."
WIN_FINAL_SCRIPT="$(wslpath -w "$PWD/$FINAL_SCRIPT")"
FINAL_CMD=("$BLENDER_EXE" "$WIN_BLEND_FILE" -b -E CYCLES "${BLENDER_ARGS[@]}" -P "$WIN_FINAL_SCRIPT")
if [ -n "${SCDL_CYCLES_DEVICE:-}" ]; then
  FINAL_CMD+=(-- --cycles-device "${SCDL_CYCLES_DEVICE}")
fi
log_cmd_array "${FINAL_CMD[@]}"
run_blender_cmd "${FINAL_CMD[@]}" || die "Blender final (ROI) step failed."

# Verify final output
[ -f out/final.png ] || die "Final render not found at out/final.png"

declare -a OUTPUT_PATHS
OUTPUT_PATHS+=("out/preview.png")
OUTPUT_PATHS+=("out/user_importance.npy")
OUTPUT_PATHS+=("out/final.png")

if [ "${#OUTPUT_PATHS[@]}" -gt 0 ]; then
  printf -v _outputs_str '%s, ' "${OUTPUT_PATHS[@]}"
  _outputs_str="${_outputs_str%, }"
  log_done "Outputs: ${_outputs_str}"
else
  log_done "Pipeline finished with no outputs tracked."
fi
