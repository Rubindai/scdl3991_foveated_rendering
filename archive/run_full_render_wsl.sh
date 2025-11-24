#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root regardless of invocation directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IS_TTY=0
if [ -t 1 ]; then
  IS_TTY=1
fi

ENV_LOADED=0

append_wslenv_literals() {
  local key entry current
  current=":${WSLENV:-}:"
  for key in "$@"; do
    entry="${key}/l"
    case "$current" in
      *":${entry}:"*|*":${key}:"*)
        continue
        ;;
      *)
        if [ -n "${WSLENV:-}" ]; then
          WSLENV="${WSLENV}:${entry}"
        else
          WSLENV="${entry}"
        fi
        current=":${WSLENV:-}:"
        ;;
    esac
  done
  export WSLENV
}

# ======== CONFIG ========
if [ -f .scdl.env ]; then
  set -a
  # shellcheck disable=SC1090
  . ./.scdl.env
  set +a
  ENV_LOADED=1
fi

if [ "$ENV_LOADED" = "1" ]; then
  mapfile -t __scdl_wslenv_keys < <(env | awk -F= '/^SCDL_/ {print $1}' | sort -u)
  if [ "${#__scdl_wslenv_keys[@]}" -gt 0 ]; then
    append_wslenv_literals "${__scdl_wslenv_keys[@]}"
  fi
  unset __scdl_wslenv_keys
fi

LOG_MODE="${SCDL_LOG_MODE:-both}"
LOG_MODE="${LOG_MODE,,}"
LOG_PATH="${SCDL_LOG_FILE:-$PWD/out_full_render/full_render.log}"
ANSI_STRIP_EXPR=$'s/\x1B\[[0-9;]*[A-Za-z]//g'

OUT_DIR="$PWD/out_full_render"
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

BLENDER_EXE="${BLENDER_EXE:-/mnt/d/Programs/blender/blender.exe}"
BLEND_FILE_ARG=""
if [ $# -ge 1 ]; then
  BLEND_FILE_ARG="$1"
fi
BLEND_FILE_ENV="${BLEND_FILE:-}"
if [ -n "$BLEND_FILE_ARG" ]; then
  BLEND_FILE="$BLEND_FILE_ARG"
elif [ -n "$BLEND_FILE_ENV" ]; then
  BLEND_FILE="$BLEND_FILE_ENV"
else
  BLEND_FILE="blender_files/cookie.blend"
fi
export BLEND_FILE
BASELINE_SCRIPT="full_render_baseline.py"

command -v wslpath >/dev/null 2>&1 || die "wslpath not found. Run under WSL."
[ -f "$BLEND_FILE" ] || die "Blend file not found: $BLEND_FILE"
[ -f "$BASELINE_SCRIPT" ] || die "Baseline Blender script missing: $BASELINE_SCRIPT"
[ -f "$BLENDER_EXE" ] || die "Blender.exe not found at $BLENDER_EXE. Set BLENDER_EXE=/mnt/<drive>/path/to/blender.exe"

REQUIRED_BLENDER_VERSION="4.5.4"
BLENDER_VERSION_LINE="$("$BLENDER_EXE" --version 2>/dev/null | head -n 1 || true)"
if [[ "$BLENDER_VERSION_LINE" != *"$REQUIRED_BLENDER_VERSION"* ]]; then
  die "Blender version mismatch. Expected $REQUIRED_BLENDER_VERSION LTS, got '${BLENDER_VERSION_LINE:-unknown}'. Update BLENDER_EXE to the 4.5.4 LTS binary."
fi

TOTAL_STEPS=1
WIN_BLEND_FILE="$(wslpath -w "$PWD/$BLEND_FILE")"
WIN_BASELINE_SCRIPT="$(wslpath -w "$PWD/$BASELINE_SCRIPT")"

STEP=1
log_step "${STEP}/${TOTAL_STEPS}" "Rendering full baseline via Windows Blender..."

BLENDER_ARGS=(--factory-startup --addons cycles)
if [ -n "${SCDL_CYCLES_DEVICE:-}" ]; then
  log_info "SCDL_CYCLES_DEVICE=${SCDL_CYCLES_DEVICE}"
else
  log_info "SCDL_CYCLES_DEVICE not set; using Blender default device selection."
fi
log_info "Blend file: $BLEND_FILE"

if [ -z "${SCDL_PROJECT_DIR:-}" ]; then
  export SCDL_PROJECT_DIR="$PWD"
fi

BASELINE_CMD=("$BLENDER_EXE" "$WIN_BLEND_FILE" -b -E CYCLES "${BLENDER_ARGS[@]}" -P "$WIN_BASELINE_SCRIPT")
if [ -n "${SCDL_CYCLES_DEVICE:-}" ]; then
  BASELINE_CMD+=(-- --cycles-device "${SCDL_CYCLES_DEVICE}")
fi
log_cmd_array "${BASELINE_CMD[@]}"
: > "$BLENDER_TIME_FILE"
_step_start=$(date +%s)
run_blender_cmd "${BASELINE_CMD[@]}" || die "Blender baseline render failed."
record_step_duration "Baseline render" $(( $(date +%s) - _step_start ))

[ -f "$OUT_DIR/final.png" ] || die "Baseline render not found at $OUT_DIR/final.png"

declare -a OUTPUT_PATHS
OUTPUT_PATHS+=("$OUT_DIR/final.png")
if [ "$LOG_MODE" != "stdout" ]; then
  OUTPUT_PATHS+=("$LOG_PATH")
fi

if [ "${#OUTPUT_PATHS[@]}" -gt 0 ]; then
  printf -v _outputs_str '%s, ' "${OUTPUT_PATHS[@]}"
  _outputs_str="${_outputs_str%, }"
  log_done "Outputs: ${_outputs_str}"
else
  log_done "Baseline run finished with no outputs tracked."
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
