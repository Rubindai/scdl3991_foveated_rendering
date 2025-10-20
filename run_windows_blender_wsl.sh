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

OUT_DIR="$PWD/out"
OUT_CLEAR_NOTE=""
if [ "${SCDL_CLEAR_OUT_DIR:-1}" != "0" ]; then
  if [ -d "$OUT_DIR" ] && [ "$OUT_DIR" != "$PWD" ] && [ "$OUT_DIR" != "/" ]; then
    rm -rf -- "$OUT_DIR"
    OUT_CLEAR_NOTE="Cleared existing output directory: $OUT_DIR"
  else
    OUT_CLEAR_NOTE="Initialized output directory: $OUT_DIR"
  fi
else
  OUT_CLEAR_NOTE="Skipped clearing output directory (SCDL_CLEAR_OUT_DIR=0)."
fi
mkdir -p "$OUT_DIR"

if [ "${CONDA_DEFAULT_ENV:-}" != "scdl-foveated" ]; then
  printf '[ERROR] Activate the scdl-foveated conda env before running (CONDA_DEFAULT_ENV=%s).\n' "${CONDA_DEFAULT_ENV:-none}" >&2
  exit 1
fi

PIPELINE_START=$(date +%s)
declare -a STEP_NAMES=()
declare -a STEP_DURATIONS=()
declare -a STEP_REPORTED_TIMES=()
BLENDER_TIME_FILE="$(mktemp)"

cleanup_temp() {
  if [ -f "$BLENDER_TIME_FILE" ]; then
    rm -f "$BLENDER_TIME_FILE"
  fi
}
trap cleanup_temp EXIT

format_duration() {
  local total=${1:-0}
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  if [ "$hours" -gt 0 ]; then
    printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
  else
    printf '%02d:%02d' "$minutes" "$seconds"
  fi
}

record_step_duration() {
  local name="$1"
  local seconds="$2"
  STEP_NAMES+=("$name")
  STEP_DURATIONS+=("$seconds")
  local reported=""
  if [ -s "$BLENDER_TIME_FILE" ]; then
    reported="$(tail -n 1 "$BLENDER_TIME_FILE")"
  fi
  STEP_REPORTED_TIMES+=("$reported")
  : > "$BLENDER_TIME_FILE"
}

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

if [ -n "$OUT_CLEAR_NOTE" ]; then
  log_info "$OUT_CLEAR_NOTE"
fi

filter_blender_output() {
  local line stripped
  local time_file="$BLENDER_TIME_FILE"
  while IFS= read -r line || [ -n "$line" ]; do
    # Normalize carriage returns that Blender occasionally emits during progress updates
    stripped="${line//$'\r'/}"
    if [ -z "$stripped" ]; then
      printf '\n'
      continue
    fi
    case "$stripped" in
      \[INFO\]*)
        local msg="${stripped#\[INFO\]}"
        msg="${msg# }"
        if [[ "$msg" == Time:* ]]; then
          if [ -n "$time_file" ]; then
            printf '%s\n' "$msg" > "$time_file"
          fi
        fi
        log_info "$msg"
        ;;
      \[WARN\]*)
        local msg="${stripped#\[WARN\]}"
        msg="${msg# }"
        log_warn "$msg"
        ;;
      \[ERROR\]*|\[ERR\]*)
        local msg="$stripped"
        if [[ "$msg" == \[ERROR\]* ]]; then
          msg="${msg#\[ERROR\]}"
        else
          msg="${msg#\[ERR\]}"
        fi
        msg="${msg# }"
        log_error "$msg"
        ;;
      Saved:*)
        log_info "$stripped"
        ;;
      Updating\ device\ list|Created\ history\ step*|Blender\ *|Read\ blend:*)
        log_info "$stripped"
        ;;
      Time:*)
        if [ -n "$time_file" ]; then
          printf '%s\n' "$stripped" > "$time_file"
        fi
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

WIN_BLEND_FILE="$(wslpath -w "$PWD/$BLEND_FILE")"

# ----- Step 1: Preview render in Windows Blender -----
STEP=1
log_step "${STEP}/${TOTAL_STEPS}" "Rendering preview via Windows Blender..."

# Convert Linux paths to Windows for Blender.exe invocation
WIN_PREVIEW_SCRIPT="$(wslpath -w "$PWD/$PREVIEW_SCRIPT")"
BLENDER_ARGS=(--factory-startup --addons cycles)
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
: > "$BLENDER_TIME_FILE"
_step_start=$(date +%s)
run_blender_cmd "${PREVIEW_CMD[@]}" || die "Blender preview step failed."
record_step_duration "Preview render" $(( $(date +%s) - _step_start ))

# Verify preview
[ -f out/preview.png ]   || die "Missing out/preview.png after Blender preview."

# ----- Step 2: DINO mask in WSL -----
STEP=$((STEP + 1))
log_step "${STEP}/${TOTAL_STEPS}" "Computing DINO mask in WSL..."

: > "$BLENDER_TIME_FILE"
# Ensure the Python scripts resolve project paths to this repo
export SCDL_PROJECT_DIR="$PWD"
export PYTHONNOUSERSITE=1
export PYTHONWARNINGS="default"

if [ -n "${SCDL_DINO_PYTHON:-}" ]; then
  DINO_CMD=("${SCDL_DINO_PYTHON}" "$WSL_DINO_SCRIPT")
  log_info "Using explicit SCDL_DINO_PYTHON=${SCDL_DINO_PYTHON}"
else
  DINO_CMD=(python "$WSL_DINO_SCRIPT")
fi

log_cmd_array "${DINO_CMD[@]}"
: > "$BLENDER_TIME_FILE"
_step_start=$(date +%s)
"${DINO_CMD[@]}" || die "DINO mask step failed."
record_step_duration "DINO mask" $(( $(date +%s) - _step_start ))
[ -f out/user_importance.npy ] || die "Missing out/user_importance.npy after DINO."
[ -f out/user_importance_mask.exr ] || die "Missing out/user_importance_mask.exr after DINO."


# ----- Step 3: Single-pass foveated render in Windows Blender -----
STEP=$((STEP + 1))
log_step "${STEP}/${TOTAL_STEPS}" "Rendering final image (single-pass foveated) in Windows Blender..."
WIN_FINAL_SCRIPT="$(wslpath -w "$PWD/$FINAL_SCRIPT")"
FINAL_CMD=("$BLENDER_EXE" "$WIN_BLEND_FILE" -b -E CYCLES "${BLENDER_ARGS[@]}" -P "$WIN_FINAL_SCRIPT")
if [ -n "${SCDL_CYCLES_DEVICE:-}" ]; then
  FINAL_CMD+=(-- --cycles-device "${SCDL_CYCLES_DEVICE}")
fi
log_cmd_array "${FINAL_CMD[@]}"
: > "$BLENDER_TIME_FILE"
_step_start=$(date +%s)
run_blender_cmd "${FINAL_CMD[@]}" || die "Blender final step failed."
record_step_duration "Final render" $(( $(date +%s) - _step_start ))

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

if [ "${#STEP_NAMES[@]}" -gt 0 ]; then
  log_info "Timing summary:"
  total_duration=$(( $(date +%s) - PIPELINE_START ))
  for idx in "${!STEP_NAMES[@]}"; do
    step_name="${STEP_NAMES[$idx]}"
    step_secs="${STEP_DURATIONS[$idx]}"
    step_report="${STEP_REPORTED_TIMES[$idx]}"
    local_line="  ${step_name}: $(format_duration "$step_secs")"
    if [ -n "$step_report" ]; then
      case "$step_report" in
        Time:*)
          extract="${step_report#Time: }"
          extract="${extract%% *}"
          local_line+=" (Blender: $extract)"
          ;;
        *)
          local_line+=" ($step_report)"
          ;;
      esac
    fi
    log_info "$local_line"
  done
  log_info "  Total: $(format_duration "$total_duration")"
fi
