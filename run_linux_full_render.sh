#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IS_TTY=0
if [ -t 1 ]; then
  IS_TTY=1
fi

ENV_LOADED=0
if [ -f .scdl.env ]; then
  set -a
  # shellcheck disable=SC1090
  . ./.scdl.env
  set +a
  ENV_LOADED=1
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
    OUT_CLEAR_NOTE="Cleared existing baseline output directory: $OUT_DIR"
  else
    OUT_CLEAR_NOTE="Initialized baseline output directory: $OUT_DIR"
  fi
else
  OUT_CLEAR_NOTE="Skipped clearing baseline output directory (SCDL_CLEAR_OUT_DIR=0)."
fi
mkdir -p "$OUT_DIR"

if [ "${CONDA_DEFAULT_ENV:-}" != "scdl-foveated" ]; then
  printf '[ERROR] Activate the scdl-foveated conda env before running (CONDA_DEFAULT_ENV=%s).\n' "${CONDA_DEFAULT_ENV:-none}" >&2
  exit 1
fi

declare -a STEP_NAMES=()
declare -a STEP_DURATIONS=()
declare -a STEP_REPORTED_TIMES=()
BLENDER_TIME_FILE="$(mktemp)"
PIPELINE_START=$(date +%s)

cleanup_temp() {
  if [ -f "$BLENDER_TIME_FILE" ]; then
    rm -f "$BLENDER_TIME_FILE"
  fi
}
trap cleanup_temp EXIT

log_emit() {
  local color="$1"; shift
  local prefix="$1"; shift
  local message="$*"
  if [ "$IS_TTY" -eq 1 ] && [ -n "$color" ]; then
    printf '%b%s%b %s\n' "$color" "$prefix" $'\033[0m' "$message"
  else
    printf '%s %s\n' "$prefix" "$message"
  fi
}

COLOR_INFO=""; COLOR_WARN=""; COLOR_ERROR=""; COLOR_STEP=""; COLOR_CMD=""; COLOR_ENV=""; COLOR_OK=""
if [ "$IS_TTY" -eq 1 ] && { [ "$LOG_MODE" = "both" ] || [ "$LOG_MODE" = "stdout" ]; }; then
  COLOR_INFO=$'\033[0;36m'
  COLOR_WARN=$'\033[0;33m'
  COLOR_ERROR=$'\033[0;31m'
  COLOR_STEP=$'\033[1;34m'
  COLOR_CMD=$'\033[0;32m'
  COLOR_ENV=$'\033[0;35m'
  COLOR_OK=$'\033[0;32m'
fi

log_info() { log_emit "$COLOR_INFO" "[INFO]" "$*"; }
log_warn() { log_emit "$COLOR_WARN" "[WARN]" "$*"; }
log_error() { log_emit "$COLOR_ERROR" "[ERROR]" "$*"; }
log_step() { log_emit "$COLOR_STEP" "[$1]" "$2"; }
log_env() { log_emit "$COLOR_ENV" "[env]" "$*"; }
log_done() { log_emit "$COLOR_OK" "[DONE]" "$*"; }

log_cmd() {
  if [ "$IS_TTY" -eq 1 ] && [ -n "$COLOR_CMD" ]; then
    printf '%b[CMD]%b %b%s%b\n' "$COLOR_CMD" $'\033[0m' "$COLOR_CMD" "$*" $'\033[0m'
  else
    printf '[CMD] %s\n' "$*"
  fi
}

log_cmd_array() {
  local formatted
  printf -v formatted ' %q' "$@"
  log_cmd "${formatted# }"
}

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
        msg="${msg#\[ERROR\]}"
        msg="${msg#\[ERR\]}"
        msg="${msg# }"
        log_error "$msg"
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

summarize_pipeline() {
  local total=$(( $(date +%s) - PIPELINE_START ))
  log_done "Baseline render finished in ${total}s"
  local count=${#STEP_NAMES[@]}
  if [ "$count" -eq 0 ]; then
    return
  fi
  log_info "Stage timings:"
  local i
  for ((i = 0; i < count; i++)); do
    local extra="${STEP_REPORTED_TIMES[$i]}"
    if [ -n "$extra" ]; then
      log_info "  ${STEP_NAMES[$i]}: ${STEP_DURATIONS[$i]}s (${extra})"
    else
      log_info "  ${STEP_NAMES[$i]}: ${STEP_DURATIONS[$i]}s"
    fi
  done
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

if [ -n "$OUT_CLEAR_NOTE" ]; then
  log_info "$OUT_CLEAR_NOTE"
fi
if [ -n "${SCDL_EXPECTED_GPU:-}" ]; then
  log_env "SCDL_EXPECTED_GPU=${SCDL_EXPECTED_GPU}"
fi

die() {
  log_error "$*"
  exit 1
}

BLENDER_BIN="${SCDL_LINUX_BLENDER_EXE:-${BLENDER_EXE:-blender}}"
if ! command -v "$BLENDER_BIN" >/dev/null 2>&1; then
  die "Blender executable not found. Set SCDL_LINUX_BLENDER_EXE or BLENDER_EXE to the Linux binary."
fi

REQUIRED_BLENDER_VERSION="4.5.4"
BLENDER_VERSION_LINE="$("$BLENDER_BIN" --version 2>/dev/null | head -n 1 || true)"
if [[ "$BLENDER_VERSION_LINE" != *"$REQUIRED_BLENDER_VERSION"* ]]; then
  die "Blender version mismatch. Expected $REQUIRED_BLENDER_VERSION LTS, got '${BLENDER_VERSION_LINE:-unknown}'. Update BLENDER_EXE/SCDL_LINUX_BLENDER_EXE to the 4.5.4 LTS binary."
fi

BLEND_FILE="${1:-${BLEND_FILE:-blender_files/cookie.blend}}"
[ -f "$BLEND_FILE" ] || die "Blend file not found: $BLEND_FILE"
BASELINE_SCRIPT="$SCRIPT_DIR/full_render_baseline.py"
[ -f "$BASELINE_SCRIPT" ] || die "Baseline script missing: $BASELINE_SCRIPT"

log_step "1/1" "Rendering baseline in Linux Blender..."

BASELINE_CMD=("$BLENDER_BIN" "$BLEND_FILE" -b -E CYCLES --factory-startup --addons cycles -P "$BASELINE_SCRIPT")
if [ -n "${SCDL_CYCLES_DEVICE:-}" ]; then
  BASELINE_CMD+=(-- --cycles-device "${SCDL_CYCLES_DEVICE}")
fi
log_cmd_array "${BASELINE_CMD[@]}"
: > "$BLENDER_TIME_FILE"
_step_start=$(date +%s)
run_blender_cmd "${BASELINE_CMD[@]}" || die "Baseline render step failed."
record_step_duration "Baseline render" $(( $(date +%s) - _step_start ))

[ -f "$OUT_DIR/final.png" ] || die "Baseline render not found at $OUT_DIR/final.png"

summarize_pipeline

log_info "Outputs written:"
if [ -e "$OUT_DIR/final.png" ]; then
  log_info "  $OUT_DIR/final.png"
fi
if [ -e "$LOG_PATH" ]; then
  log_info "  $LOG_PATH"
fi

exit 0
