#!/usr/bin/env python3
"""Shared logging helpers for the SCDL pipeline."""

from __future__ import annotations

import atexit
import io
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

PathLike = Union[str, os.PathLike]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class _StreamTee(io.TextIOBase):
    """Minimal TextIO wrapper that mirrors writes to multiple streams."""

    def __init__(self, streams: List[io.TextIOBase]):
        super().__init__()
        self._streams = streams

    def write(self, s: str) -> int:  # type: ignore[override]
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def writelines(self, lines) -> None:  # type: ignore[override]
        for line in lines:
            self.write(line)

    def flush(self) -> None:  # type: ignore[override]
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    @property
    def encoding(self) -> str:  # type: ignore[override]
        for stream in self._streams:
            enc = getattr(stream, "encoding", None)
            if enc:
                return enc
        return "utf-8"

    def fileno(self) -> int:  # type: ignore[override]
        streams_with_fd = [s for s in self._streams if hasattr(s, "fileno")]
        if len(streams_with_fd) == 1:
            return streams_with_fd[0].fileno()  # type: ignore[return-value]
        raise OSError("tee stream has no single file descriptor")

    def close(self) -> None:  # type: ignore[override]
        # Avoid closing underlying streams automatically (they are managed elsewhere).
        pass


class _StdIORouter:
    """Redirects stdout/stderr so plain prints honor the logging mode."""

    def __init__(self) -> None:
        self._original_stdout = sys.__stdout__
        self._original_stderr = sys.__stderr__
        self._file: Optional[io.TextIOBase] = None
        self._log_path: Optional[Path] = None
        self._targets: set[str] = set()
        atexit.register(self.restore)

    def configure(self, log_path: Path, targets: set[str]) -> None:
        # Reconfigure only if something actually changed.
        if self._log_path == log_path and self._targets == targets and ("file" not in targets or (self._file and not self._file.closed)):
            return

        self.restore(close_file=True)
        self._log_path = log_path
        self._targets = set(targets)

        file_stream: Optional[io.TextIOBase] = None
        if "file" in targets:
            _ensure_parent(log_path)
            file_stream = open(log_path, "a", encoding="utf-8", buffering=1)
            self._file = file_stream

        stdout_target = self._build_stream(file_stream, self._original_stdout if "stdout" in targets else None, self._original_stdout)
        stderr_target = self._build_stream(file_stream, self._original_stderr if "stdout" in targets else None, self._original_stderr)

        sys.stdout = stdout_target
        sys.stderr = stderr_target

    def restore(self, close_file: bool = True) -> None:
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

        if close_file and self._file:
            try:
                self._file.flush()
            finally:
                self._file.close()
        self._file = None
        self._log_path = None
        self._targets = set()

    def _build_stream(
        self,
        file_stream: Optional[io.TextIOBase],
        console_stream: Optional[io.TextIOBase],
        fallback_stream: io.TextIOBase,
    ) -> io.TextIOBase:
        streams = [s for s in (console_stream, file_stream) if s is not None]
        if not streams:
            return fallback_stream
        if len(streams) == 1:
            return streams[0]
        return _StreamTee(streams)


_STDIO_ROUTER = _StdIORouter()


_ANSI_RESET = "\033[0m"
_LEVEL_NAMES = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRIT",
}
_LEVEL_COLORS = {
    logging.DEBUG: "\033[90m",   # dim grey
    logging.INFO: "\033[36m",    # cyan
    logging.WARNING: "\033[33m", # yellow
    logging.ERROR: "\033[31m",   # red
    logging.CRITICAL: "\033[41m",  # red background
}
_TAG_COLORS = {
    "[OK]": "\033[32m",
    "[DINO]": "\033[35m",
    "[Cycles]": "\033[36m",
    "[Engine]": "\033[36m",
    "[WARN]": "\033[33m",
    "[ERROR]": "\033[31m",
    "[env]": "\033[35m",
    "[CMD]": "\033[90m",
    "[DONE]": "\033[32m",
}


def _supports_color(stream: io.TextIOBase) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class _ConsoleFormatter(logging.Formatter):
    """Pretty console formatter with optional ANSI highlights."""

    def __init__(self, use_color: bool):
        super().__init__("%(message)s")
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        message = super().format(record)
        # Prefix with level tag and color when appropriate
        level_tag = _LEVEL_NAMES.get(record.levelno, record.levelname)
        prefix_text = f"[{level_tag}]"
        prefix = prefix_text
        if message.startswith(prefix_text):
            remainder = message[len(prefix_text):]
            if not remainder or remainder[0].isspace():
                message = remainder.lstrip()
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno)
            if color:
                prefix = f"{color}{prefix_text}{_ANSI_RESET}"

        if self._use_color:
            message = self._colorize_tags(message)

        if "\n" in message:
            indent = " " * (len(prefix_text) + 1)
            lines = message.splitlines()
            message = lines[0] + "".join("\n" + indent + line for line in lines[1:])

        return f"{prefix} {message}"

    def _colorize_tags(self, message: str) -> str:
        for tag, color in _TAG_COLORS.items():
            if tag in message:
                message = message.replace(tag, f"{color}{tag}{_ANSI_RESET}")
        return message


def get_logger(name: str, default_path: PathLike) -> logging.Logger:
    """Configure a logger that honors SCDL_LOG_MODE/SCDL_LOG_FILE."""
    mode = os.environ.get("SCDL_LOG_MODE", "both").strip().lower()
    log_file_override = os.environ.get("SCDL_LOG_FILE", "").strip()

    if mode not in {"stdout", "file", "both"}:
        mode = "both"

    targets = {mode}
    if mode == "both":
        targets = {"stdout", "file"}

    default_path = Path(default_path)
    log_path = Path(log_file_override).expanduser() if log_file_override else default_path

    _STDIO_ROUTER.configure(log_path, targets)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers to avoid duplicate logs when Blender reuses the interpreter
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    if "file" in targets:
        _ensure_parent(log_path)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)

    if "stdout" in targets or not logger.handlers:
        stream = sys.__stderr__
        sh = logging.StreamHandler(stream=stream)
        sh.setFormatter(_ConsoleFormatter(use_color=_supports_color(stream)))
        logger.addHandler(sh)

    return logger


__all__ = ["get_logger"]
