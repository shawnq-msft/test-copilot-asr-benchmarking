"""Local preflight and exact artifact validation; never calls Azure."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import datasets_loader as loader
from benchmark import config
from benchmark.config import DEFAULT_SERVICES, LOCALES, SERVICES, STREAMING_SERVICES, Sample, assert_credentials
from benchmark.runner import CSV_FIELDS


def metadata_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_run.json")


def collect_samples(languages: list[str], limit: int, seed: int) -> list[Sample]:
    samples: list[Sample] = []
    for locale in languages:
        datasets = loader.list_datasets(locale)
        if not datasets:
            raise ValueError(f"No datasets for {locale} under {loader.DATA_ROOT}")
        for dataset in datasets:
            samples.extend(loader.iter_samples(dataset, limit=limit, seed=seed, strict=True))
    keys = [(sample.dataset, sample.sample_id) for sample in samples]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Selection has no samples or duplicate sample identities")
    return samples


def check_selection(services: list[str], languages: list[str], limit: int,
                    workers: int, phrases: list[str]) -> list[str]:
    if not services or any(service not in SERVICES for service in services):
        raise ValueError("Select at least one known service")
    if len(set(services)) != len(services):
        raise ValueError("Duplicate services are not allowed")
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("Select at least one distinct locale")
    if limit < 0 or workers < 1:
        raise ValueError("--max-per-dataset must be >=0 (0 means all); --workers must be >=1")
    if "realtime_mai_2" in services:
        import azure.cognitiveservices.speech as sdk
        if not hasattr(sdk.SpeechConfig, "model"):
            raise ValueError("realtime_mai_2 requires Speech SDK >=1.52.0; install requirements.txt")
        if phrases:
            raise ValueError("Phrase lists are not verified for realtime_mai_2; remove --phrase-list")
    warnings = []
    if "whisper_v3" in services:
        warnings.append("whisper_v3 is a stub and will fail; choose an explicit subset to exclude it")
    if "fast_mai" in services:
        warnings.append("fast_mai selects retired MAI-Transcribe-1 and is expected to fail; "
                        "it is retained only for explicit legacy diagnostics")
    enhanced = set(services) & {"fast_llm", "fast_mai", "fast_mai_1_5", "fast_mai_2"}
    if enhanced and config.LLM_AZURE_REGION not in config.ENHANCED_REST_REGIONS:
        warnings.append(
            f"REST enhanced mode is not documented in configured region {config.LLM_AZURE_REGION!r}; "
            "it can return HTTP 400 'Enhanced mode ... not supported'. "
            "Use a resource with matching key/endpoint in a supported REST region. "
            "Southeast Asia supports both REST enhanced mode and MAI streaming "
            "(regions documentation checked 2026-10-10); no automatic fallback is applied.")
    if "realtime_mai_2" in services and config.AZURE_SPEECH_REGION not in config.MAI_STREAMING_REGIONS:
        warnings.append(
            f"MAI streaming is not documented in configured region {config.AZURE_SPEECH_REGION!r}. "
            "Use a matching resource/key/endpoint in a documented streaming region; "
            "Southeast Asia also supports REST enhanced mode (checked 2026-10-10).")
    return warnings


def make_metadata(samples: list[Sample], services: list[str], *, pace: bool,
                  seed: int, limit: int, workers: int,
                  phrases: list[str], phrase_weight: float) -> dict:
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "speech_sdk": version("azure-cognitiveservices-speech"),
        "data_root": str(loader.DATA_ROOT.resolve()),
        "services": services,
        "languages": sorted({sample.locale for sample in samples}),
        "pace": pace,
        "seed": seed,
        "max_per_dataset": limit,
        "workers": workers,
        "phrases": phrases,
        "phrase_weight": phrase_weight,
        "expected_keys": [[sample.dataset, sample.sample_id, service]
                          for sample in samples for service in services],
    }


def check_resume(csv_path: Path, metadata: dict) -> None:
    if not csv_path.is_file():
        raise ValueError(f"Resume CSV does not exist: {csv_path}")
    if csv_path.with_suffix(csv_path.suffix + ".strict").exists():
        raise ValueError("Cannot resume normalized results; use a fresh run and preserve the strict backup")
    saved_path = metadata_path(csv_path)
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        for field in ("data_root", "services", "languages", "pace", "seed",
                      "max_per_dataset", "phrases", "phrase_weight", "expected_keys"):
            if saved.get(field) != metadata[field]:
                raise ValueError(f"Resume selection differs in {field}; use a fresh run")
    else:
        raise ValueError("Legacy resume has no run metadata; use a fresh run to ensure selection/pacing consistency")
    expected = {tuple(key) for key in metadata["expected_keys"]}
    with csv_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError("Resume CSV schema does not match this runner")
        keys = [(row["dataset"], row["sample_id"], row["service"]) for row in reader]
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise ValueError("Resume CSV contains duplicate or unexpected sample/service rows")
    words_path = csv_path.with_name(csv_path.stem + "_words.jsonl")
    if words_path.exists():
        entries = [json.loads(line) for line in words_path.read_text(encoding="utf-8").splitlines() if line]
        word_keys = [(entry["dataset"], entry["sample_id"], entry.get("service", "realtime"))
                     for entry in entries]
        if len(set(word_keys)) != len(word_keys) or not set(word_keys) <= set(keys):
            raise ValueError("Resume sidecar is inconsistent with CSV; preserve evidence and start a fresh run")


def validate_artifacts(csv_path: Path, *, require_reports: bool = True,
                       allow_legacy: bool = False) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    service_errors: list[dict] = []
    saved_path = metadata_path(csv_path)
    metadata = json.loads(saved_path.read_text(encoding="utf-8")) if saved_path.exists() else None
    if metadata is not None:
        expected_keys = metadata["expected_keys"]
        if (not isinstance(metadata["pace"], bool) or not isinstance(expected_keys, list)
                or not expected_keys
                or any(not isinstance(key, list) or len(key) != 3
                       or any(not isinstance(part, str) or not part for part in key)
                       for key in expected_keys)):
            raise ValueError("Invalid run metadata: expected nonempty sample/service keys and boolean pacing")
        if len({tuple(key) for key in expected_keys}) != len(expected_keys):
            raise ValueError("Duplicate expected keys in run metadata")
    if metadata is None:
        (warnings if allow_legacy else errors).append(
            "Missing run metadata: exact expected coverage and pacing cannot be verified")
    elif not metadata["pace"]:
        warnings.append("Unpaced run: latency values must not be used for comparisons")
    with csv_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"dataset", "sample_id", "service", "error", "wer", "sentence_error"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("CSV lacks required identity/scoring columns")
        rows = list(reader)
    keys = [(row["dataset"], row["sample_id"], row["service"]) for row in rows]
    counts = Counter(keys)
    if not rows:
        errors.append("CSV contains no results")
    if any(count != 1 for count in counts.values()):
        errors.append("Duplicate CSV sample/service keys")
    expected = {tuple(key) for key in metadata["expected_keys"]} if metadata else set(keys)
    missing = sorted(expected - set(keys))
    unexpected = sorted(set(keys) - expected)
    if missing:
        errors.append(f"Missing {len(missing)} expected sample/service rows")
    if unexpected:
        errors.append(f"Unexpected {len(unexpected)} sample/service rows")
    for row, key in zip(rows, keys):
        if not row.get("request_id") or (row["service"] in STREAMING_SERVICES and not row.get("session_id")):
            warnings.append(f"{key}: correlation ID unavailable; do not invent one")
        if row["error"]:
            service_errors.append({"key": key, "error": row["error"],
                                   "request_id": row.get("request_id", ""),
                                   "session_id": row.get("session_id", "")})
            continue
        for field in ("wer", "sentence_error", "ins", "del", "sub", "ref_len"):
            value = row.get(field, "")
            try:
                number = float(value)
                valid = math.isfinite(number) and number >= 0
                if field != "wer":
                    valid = valid and number.is_integer()
                if field == "sentence_error":
                    valid = valid and number in (0, 1)
                if not valid:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"{key}: missing/invalid {field}")
        for field in ("first_latency_ms", "lbl_ms", "upl_ms", "upl_self_ms"):
            value = row.get(field)
            if value not in ("", None):
                try:
                    if not math.isfinite(float(value)):
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(f"{key}: invalid {field}")
        if row["service"] in STREAMING_SERVICES and not row.get("upl_ms"):
            warnings.append(f"{key}: speech-anchored UPL unavailable")
    words_path = csv_path.with_name(csv_path.stem + "_words.jsonl")
    streaming = {key for key in keys if key[2] in STREAMING_SERVICES}
    rows_by_key = dict(zip(keys, rows))
    if streaming and not words_path.exists():
        (warnings if allow_legacy else errors).append("Missing streaming evidence sidecar")
    if words_path.exists():
        entries = [json.loads(line) for line in words_path.read_text(encoding="utf-8").splitlines() if line]
        word_keys = [(entry["dataset"], entry["sample_id"], entry.get("service", "realtime"))
                     for entry in entries]
        if len(set(word_keys)) != len(word_keys):
            errors.append("Duplicate streaming evidence keys")
        if set(word_keys) - streaming:
            errors.append("Streaming evidence has unexpected sample/service keys")
        if streaming - set(word_keys):
            (warnings if allow_legacy else errors).append("Streaming evidence coverage is incomplete")
        for entry in entries:
            if not isinstance(entry.get("words"), list):
                errors.append("Streaming evidence words must be a list")
            evidence = entry.get("streaming_evidence")
            entry_key = (entry["dataset"], entry["sample_id"], entry.get("service", "realtime"))
            row = rows_by_key.get(entry_key)
            if row is not None and (row.get("error") or "") != (entry.get("error") or ""):
                errors.append(f"{entry_key}: CSV and sidecar disagree on service error")
            if not entry.get("error") and entry.get("service") == "realtime_mai_2":
                if not evidence or not evidence.get("completed") or not evidence.get("final_count"):
                    errors.append("Successful MAI row lacks completed final evidence")
                elif evidence.get("audio_bytes_sent") != evidence.get("audio_bytes_expected"):
                    errors.append("Successful MAI row did not send all expected audio bytes")
    if require_reports:
        for suffix in ("_report.md", "_error_analysis.md"):
            report = csv_path.with_name(csv_path.stem + suffix)
            if not report.exists() or not report.stat().st_size:
                errors.append(f"Missing/empty report: {report.name}")
    strict_path = csv_path.with_suffix(csv_path.suffix + ".strict")
    if strict_path.exists():
        with strict_path.open(encoding="utf-8", newline="") as source:
            strict_rows = list(csv.DictReader(source))
        strict_keys = [(row["dataset"], row["sample_id"], row["service"]) for row in strict_rows]
        if Counter(strict_keys) != counts:
            errors.append("Strict backup row identities differ from normalized CSV")
        immutable = ("reference", "hypothesis", "error", "request_id", "session_id",
                     "first_latency_ms", "lbl_ms", "upl_ms", "upl_self_ms", "upl_anchor")
        strict_by_key = dict(zip(strict_keys, strict_rows))
        for key, row in zip(keys, rows):
            if key in strict_by_key and any(row.get(field) != strict_by_key[key].get(field) for field in immutable):
                errors.append(f"{key}: normalization changed non-scoring evidence")
    return {
        "ok": not errors and not service_errors,
        "coverage_verified": metadata is not None and not missing and not unexpected
                             and len(counts) == len(rows),
        "csv": str(csv_path.resolve()), "rows": len(rows), "expected_rows": len(expected),
        "missing_keys": missing, "unexpected_keys": unexpected,
        "errors": errors, "service_errors": service_errors, "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--languages", nargs="+", default=list(LOCALES))
    preflight.add_argument("--services", nargs="+", default=list(DEFAULT_SERVICES), choices=SERVICES)
    preflight.add_argument("--max-per-dataset", type=int, default=30)
    preflight.add_argument("--workers", type=int, default=4)
    preflight.add_argument("--seed", type=int, default=42)
    preflight.add_argument("--no-pace", action="store_true")
    preflight.add_argument("--testdata-dir", type=Path)
    preflight.add_argument("--output-dir", type=Path, default=Path("results"))
    preflight.add_argument("--phrase-list", nargs="*", default=[])
    preflight.add_argument("--phrase-weight", type=float, default=1.0)
    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("csv_path", type=Path)
    artifacts.add_argument("--rows-only", action="store_true")
    artifacts.add_argument("--allow-legacy", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            if not 0.0 <= args.phrase_weight <= 2.0:
                raise ValueError("--phrase-weight must be between 0.0 and 2.0")
            if args.testdata_dir:
                loader.DATA_ROOT = args.testdata_dir
            warnings = check_selection(args.services, args.languages, args.max_per_dataset,
                                       args.workers, args.phrase_list)
            assert_credentials(tuple(args.services))
            samples = collect_samples(args.languages, args.max_per_dataset, args.seed)
            output_parent = args.output_dir.resolve()
            while not output_parent.exists():
                output_parent = output_parent.parent
            if not output_parent.is_dir() or not os.access(output_parent, os.W_OK):
                raise ValueError("Output parent is not writable")
            result = {"ok": True, "samples": len(samples), "expected_rows": len(samples) * len(args.services),
                      "warnings": warnings, "authentication_verified": False,
                      "speech_sdk": version("azure-cognitiveservices-speech"),
                      "pace": not args.no_pace}
        else:
            result = validate_artifacts(args.csv_path, require_reports=not args.rows_only,
                                        allow_legacy=args.allow_legacy)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"ok": False, "errors": [str(exc)]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
