#!/usr/bin/env python3
"""Timing utilities with optional GPU synchronisation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, MutableSequence, Optional

import logging


@dataclass
class TimerRecord:
    """Recorded duration for a named stage."""

    name: str
    seconds: float
    metadata: dict[str, object] = field(default_factory=dict)


def format_duration(seconds: float) -> str:
    """Return `MM:SS.mmm` (or `HH:MM:SS.mmm`) formatted duration."""

    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


class StageTimer:
    """Context manager that times pipeline stages and logs/report durations."""

    def __init__(
        self,
        name: str,
        *,
        logger: Optional[logging.Logger] = None,
        records: Optional[MutableSequence[TimerRecord]] = None,
        sync_callbacks: Optional[Iterable[Callable[[], None]]] = None,
        auto_log: bool = True,
    ) -> None:
        self._name = name
        self._logger = logger
        self._records = records
        self._sync_callbacks: List[Callable[[], None]] = list(sync_callbacks or [])
        self._auto_log = auto_log
        self._start: Optional[float] = None
        self.elapsed: Optional[float] = None

    def add_sync_callback(self, callback: Callable[[], None]) -> None:
        """Append a synchronisation callback executed before the timer stops."""

        self._sync_callbacks.append(callback)

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end_time = time.perf_counter()
        self.elapsed = (end_time - self._start) if self._start is not None else 0.0

        for callback in self._sync_callbacks:
            try:
                callback()
            except Exception as sync_error:
                if self._logger:
                    self._logger.warning("[WARN] timer sync failed for %s: %s", self._name, sync_error)

        if self._records is not None:
            self._records.append(TimerRecord(self._name, self.elapsed))

        if self._logger and self._auto_log and exc_type is None:
            self._logger.info("[timer] %s %s", self._name, format_duration(self.elapsed or 0.0))


__all__ = ["StageTimer", "TimerRecord", "format_duration"]
