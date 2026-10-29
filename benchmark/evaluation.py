"""Persistent evaluation evidence and reference-aware reports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .annotation_metrics import SCORING_VERSION, describe_segments, language_scores, speaker_agreement
from .config import AsrResult, Sample
from .metrics import normalize_text, normalize_characters


def evaluation_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_evaluation.jsonl")


def variant_id(sample: Sample, service: str) -> str:
    if sample.evaluation_options is None:
        return ""
    identity = {"options": sample.evaluation_options.to_dict(), "service": service,
                "requested_language": sample.locale if sample.evaluation_options.language_mode == "monolingual" else None}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def evaluation_record(sample: Sample, result: AsrResult) -> dict:
    options = sample.evaluation_options
    if options is None:
        raise ValueError("Evaluation record requires explicit run options")
    segments = result.segments or (
        [{"text": result.hypothesis, "speaker": None, "language": None, "source": "transcript_only"}]
        if result.hypothesis else [])
    raw_segments = result.segments
    try:
        if not isinstance(segments, list):
            raise ValueError("segments must be a list")
        for segment in segments:
            if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
                raise ValueError("each segment must contain string text")
            if segment.get("speaker") is not None and not isinstance(segment["speaker"], (str, int)):
                raise ValueError("speaker must be a string/integer or unavailable")
            if segment.get("language") is not None and not isinstance(segment["language"], str):
                raise ValueError("detected language must be a string or unavailable")
            start, end = segment.get("start_s"), segment.get("end_s")
            if start is not None or end is not None:
                if (not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) for v in (start, end))
                        or not 0 <= start <= end <= sample.duration_s + .1):
                    raise ValueError("invalid/out-of-bounds segment timestamps")
        normalize = normalize_characters if sample.locale.startswith(("zh-", "yue-")) else normalize_text
        if not result.error and normalize(" ".join(s["text"] for s in segments)) != normalize(result.hypothesis):
            raise ValueError("final segment text disagrees with combined transcript")
    except ValueError as exc:
        result.error = result.error or f"invalid_annotation: {exc}"
        segments = []
    references = sample.reference_segments or []
    return {
        "schema_version": 1, "scoring_version": SCORING_VERSION,
        "dataset": sample.dataset, "sample_id": sample.sample_id, "service": result.service,
        "variant_id": variant_id(sample, result.service), "options": options.to_dict(),
        "requested_language": sample.locale if options.language_mode == "monolingual" else None,
        "dataset_locale": sample.locale, "scenario": sample.scenario,
        "language_configuration": ("language_hint" if result.service in ("realtime_mai_2_api", "realtime_mai_2", "azure_whisper")
                                   and options.language_mode == "monolingual" else
                                   "specified_locale" if options.language_mode == "monolingual" else options.lid_strategy),
        "reference_available": sample.reference_available,
        "reference": sample.reference if sample.reference_available else None,
        "reference_language": sample.reference_language, "reference_segments": references,
        "manifest_hash": sample.manifest_hash,
        "pcm_sha256": hashlib.sha256(sample.pcm16_mono_16k).hexdigest(),
        "hypothesis": result.hypothesis, "segments": segments,
        "raw_annotation_evidence": json.dumps(raw_segments, ensure_ascii=False) if raw_segments is not None else None,
        "error": result.error, "request_id": result.request_id, "session_id": result.session_id,
        "description": describe_segments(segments),
        "wder": speaker_agreement(references, segments) if options.score_wder and not result.error else None,
        "lid": language_scores(sample.reference_language if sample.scenario == "single-language" else None,
                               references, segments, options.lid_strategy)
        if not result.error else {"status": "service_error"},
    }


def build_evaluation_report(csv_path: Path, output: Path, *, details: bool = False,
                            tag: str = "") -> None:
    from .evaluation_report import build_reports
    build_reports([csv_path], output, details=details, tag=tag)


def build_combined_evaluation_report(csv_paths: list[Path], output: Path, *,
                                     retry_csv_paths: tuple[Path, ...] = ()) -> None:
    from .evaluation_report import build_reports
    build_reports(csv_paths, output, details=True, retry_csv_paths=retry_csv_paths)
