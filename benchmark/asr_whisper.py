"""Stub for Whisper v3 — not yet implemented in this framework."""
from __future__ import annotations

from .config import AsrResult, Sample


def transcribe_whisper_v3(sample: Sample, pace: bool = True) -> AsrResult:
    return AsrResult("whisper_v3", "", None, None, "not_implemented")
