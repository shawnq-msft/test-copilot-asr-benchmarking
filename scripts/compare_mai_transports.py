"""Paired MAI SDK/Realtime API experiment using the existing benchmark runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import config
from benchmark.datasets_loader import iter_samples
from benchmark.metrics import aggregate
from benchmark.runner import run_benchmark
from benchmark.eval_options import EvaluationOptions
from scripts.validate_benchmark import (check_selection, make_metadata, metadata_path,
                                        validate_artifacts)

TRANSPORTS = ("realtime_mai_2", "realtime_mai_2_api")


def run_trial(samples, path: Path, services: tuple[str, ...], chunk_ms: int) -> tuple[dict, list[dict]]:
    options = (EvaluationOptions() if services == ("realtime",) else
               EvaluationOptions(language_mode="multilingual", lid_strategy="native"))
    metadata = make_metadata(samples, list(services), pace=True, seed=42, limit=0,
                             workers=1, phrases=[], phrase_weight=1.0, chunk_ms=chunk_ms,
                             evaluation_options=options)
    metadata_path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    run_benchmark(samples, path, services=services, workers=1, pace=True, retries=0,
                  sequential=True, chunk_ms=chunk_ms, evaluation_options=options)
    validation = validate_artifacts(path, require_reports=False)
    path.with_name(path.stem + "_validation.json").write_text(json.dumps(validation, indent=2),
                                                            encoding="utf-8")
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    evidence = {}
    for line in path.with_name(path.stem + "_words.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        evidence[(entry["dataset"], entry["sample_id"], entry["service"])] = entry
    for row in rows:
        sidecar = evidence[(row["dataset"], row["sample_id"], row["service"])]
        timing = sidecar.get("streaming_evidence") or {}
        row["timing"] = timing
        row["csv_path"] = str(path)
        first_text = timing.get("first_text_s")
        setup = timing.get("setup_ms")
        start = row.get("speech_start_s")
        row["first_text_from_audio_ms"] = first_text * 1000 if first_text is not None else None
        row["first_text_from_speech_ms"] = (first_text * 1000 - float(start) * 1000
                                           if first_text is not None and start not in ("", None) else None)
        row["call_to_first_text_ms"] = (setup + first_text * 1000
                                        if setup is not None and first_text is not None else None)
        row["setup_ms"] = setup
        for field in ("lbl_ms", "upl_ms", "first_latency_ms", "wer"):
            row[field] = float(row[field]) if row.get(field) not in ("", None) else None
        if not row["error"]:
            if (not timing.get("completed") or not timing.get("final_count")
                    or timing.get("audio_bytes_sent") != timing.get("audio_bytes_expected")
                    or timing.get("pace") is not True or timing.get("chunk_ms") != chunk_ms):
                raise ValueError(f"Incomplete or mismatched timing evidence in {path}")
    return validation, rows


def summarize(records: list[dict]) -> dict:
    metrics = ("setup_ms", "first_text_from_audio_ms", "first_text_from_speech_ms",
               "call_to_first_text_ms", "lbl_ms", "upl_ms", "wer")
    summary = {"attempts": len(records), "failures": [row for row in records if row["error"]],
               "groups": [], "paired": []}
    for chunk in sorted({row["chunk_ms"] for row in records}):
        for corpus in ("all", "xiaomi", "commands"):
            selected = [row for row in records if row["chunk_ms"] == chunk
                        and (corpus == "all" or row["corpus"] == corpus)]
            if not selected:
                continue
            pairs = {}
            for row in selected:
                pairs.setdefault((row["repetition"], row["dataset"], row["sample_id"]), {})[row["service"]] = row
            valid_pairs = [pair for pair in pairs.values()
                           if all(service in pair and not pair[service]["error"] for service in TRANSPORTS)]
            for service in TRANSPORTS:
                rows = [pair[service] for pair in valid_pairs]
                summary["groups"].append({
                    "chunk_ms": chunk, "corpus": corpus, "service": service, "n": len(rows),
                    "metrics": {metric: aggregate([row[metric] for row in rows if row[metric] is not None])
                                for metric in metrics},
                })
            paired_metrics = {}
            for metric in metrics:
                eligible = [pair for pair in valid_pairs if all(pair[s][metric] is not None for s in TRANSPORTS)]
                differences = [pair[TRANSPORTS[1]][metric] - pair[TRANSPORTS[0]][metric] for pair in eligible]
                by_sample = {}
                for pair, difference in zip(eligible, differences):
                    row = pair[TRANSPORTS[0]]
                    by_sample.setdefault((row["dataset"], row["sample_id"]), []).append(difference)
                paired_metrics[metric] = {
                    "api_minus_sdk": aggregate(differences),
                    "api_lower_count": sum(value < 0 for value in differences),
                    "sample_median_differences": [
                        {"dataset": key[0], "sample_id": key[1], "n": len(values),
                         "api_minus_sdk": statistics.median(values)}
                        for key, values in sorted(by_sample.items())],
                }
            summary["paired"].append({"chunk_ms": chunk, "corpus": corpus,
                                      "pairs": len(pairs), "successful_pairs": len(valid_pairs),
                                      "metrics": paired_metrics})
    return summary


def write_summary(output: Path, records: list[dict]) -> None:
    summary = summarize(records)
    (output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# MAI Speech SDK versus Realtime API",
        "",
        "Both transports use the same resource/key, unmodified PCM16 16 kHz mono audio, "
        "matched paced chunk sizes, automatic language detection, and one fresh connection per request.",
        "Execution order alternates within matched pairs. Warm-ups and baseline anchor calls are excluded. "
        "There are no automatic retries, normalization, VAD trimming, or selected best runs.",
        "",
        "Realtime API finalization uses an explicit commit at recording end; SDK finalization closes the "
        "push stream at the same audio boundary. This compares those documented workflows, not isolated "
        "transport overhead or identical server endpointing policies.",
        "",
        "First text means first nonempty SDK recognizing/recognized text, or API intermediate/delta/completed text. "
        "API deltas are appended verbatim; intermediate suffixes replace earlier suffixes; completed is the final transcript.",
        "First speech-relative text and UPL use the same frozen baseline-realtime word boundaries per recording. "
        "These are estimated boundaries, not manually labeled ground truth.",
        "",
        "Setup is call start to audio-send start. API setup includes session.created/session.updated; "
        "SDK start may return before its network handshake completes. Thus call-to-first-text is also reported "
        "to avoid hiding connection/setup differences. LBL includes explicit-commit or EOF finalization.",
        "",
        f"Measured attempts: {summary['attempts']}; failures: {len(summary['failures'])}. "
        "Tables use only successful matched pairs and show median / p90 in milliseconds. "
        "Negative paired API-minus-SDK differences favor API. Repetitions are not independent utterances.",
        f"Missing request correlation IDs: {sum(not row.get('request_id') for row in records)}; "
        f"missing session IDs: {sum(not row.get('session_id') for row in records)}. "
        "See per-pair validation warnings; missing IDs are not synthesized.",
        "",
        "| Chunk ms | Corpus | Transport | N | Setup | First text from audio | First text from speech | Call to first text | LBL | UPL | Mean strict WER |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        def fmt(name: str) -> str:
            metric = group["metrics"][name]
            return f"{metric['median']:.1f} / {metric['p90']:.1f}" if metric["n"] else "-"
        wer = group["metrics"]["wer"]["mean"]
        wer_text = f"{wer:.4f}" if wer is not None else "-"
        lines.append(
            f"| {group['chunk_ms']} | {group['corpus']} | {group['service']} | {group['n']} | "
            + " | ".join(fmt(name) for name in ("setup_ms", "first_text_from_audio_ms",
                                                "first_text_from_speech_ms", "call_to_first_text_ms",
                                                "lbl_ms", "upl_ms"))
            + f" | {wer_text} |")
    lines.extend(["", "## Paired differences",
                  "| Chunk ms | Corpus | Metric | Pairs | Median API minus SDK ms | API lower count |",
                  "|---:|---|---|---:|---:|---:|"])
    for group in summary["paired"]:
        for name in ("first_text_from_audio_ms", "call_to_first_text_ms", "lbl_ms", "upl_ms"):
            metric = group["metrics"][name]
            delta = metric["api_minus_sdk"]
            if delta["n"]:
                lines.append(f"| {group['chunk_ms']} | {group['corpus']} | {name} | {delta['n']} | "
                             f"{delta['median']:.1f} | {metric['api_lower_count']} |")
    if summary["failures"]:
        lines.extend(["", "## Failed calls (excluded as entire pairs)", ""])
        for row in summary["failures"]:
            filename = Path(row["csv_path"]).name
            lines.extend([
                f"- {row['service']}, {row['chunk_ms']} ms, {row['sample_id']}: "
                f"[CSV]({filename})",
                f"  - Error: `{row['error']}`",
                f"  - Request ID: `{row.get('request_id') or 'unavailable'}`",
                f"  - Session ID: `{row.get('session_id') or 'unavailable'}`",
            ])
    lines.extend(["", "## Limitations",
                  "- One region/resource/client network and a small fixed recording set; no universal speed claim.",
                  "- No concurrent sessions, connection reuse, manual speech labels, or server-side processing telemetry.",
                  "- 100 ms and 20 ms are separate controlled configurations, not an unfair cross-chunk transport comparison.",
                  "- Configured MAI deployment/model is recorded; identical backend versions/routes are not externally verified.",
                  "- Inspect per-sample paired differences in comparison.json; pooled medians mix recording lengths.",
                  "",
                  "Sources: [Realtime API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming-realtime) "
                  "and [Speech SDK](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming-speech-sdk)."])
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--commands", type=int, default=5)
    parser.add_argument("--chunk-ms", nargs="+", type=int, default=[100, 20])
    args = parser.parse_args()
    if args.repetitions <= 0 or args.commands < 0 or any(chunk <= 0 for chunk in args.chunk_ms):
        parser.error("Repetitions/chunks must be positive; command count must be nonnegative")
    if len(set(args.chunk_ms)) != len(args.chunk_ms):
        parser.error("Chunk sizes must be distinct")
    check_selection(list(TRANSPORTS), ["en-US", "en-GB"], 0, 1, [])
    config.assert_credentials(TRANSPORTS)
    xiaomi = iter_samples("en-US_xiaomi-ionia", limit=0, seed=42, strict=True)
    commands = iter_samples("en-GB_smart-commands", limit=args.commands, seed=42, strict=True) if args.commands else []
    samples = xiaomi + commands
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "region": config.AZURE_SPEECH_REGION,
        "sdk_endpoint": config.AZURE_SPEECH_ENDPOINT, "api_endpoint": config.MAI_REALTIME_ENDPOINT,
        "sdk_model": config.MAI_STREAMING_MODEL, "api_deployment": config.MAI_REALTIME_DEPLOYMENT,
        "sdk_version": version("azure-cognitiveservices-speech"), "websockets_version": version("websockets"),
        "repetitions": args.repetitions, "chunk_ms": args.chunk_ms, "pace": True,
        "retries": 0, "seed": 42, "warmups_excluded": True,
        "expected_pairs": len(samples) * args.repetitions * len(args.chunk_ms),
        "samples": [{"dataset": s.dataset, "sample_id": s.sample_id, "duration_s": s.duration_s,
                     "reference": s.reference, "pcm_sha256": hashlib.sha256(s.pcm16_mono_16k).hexdigest()}
                    for s in samples],
    }
    (args.output_dir / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validation, _ = run_trial(samples, args.output_dir / "anchors.csv", ("realtime",), 100)
    if not validation["ok"] or any(s.first_word_start_s is None or s.last_word_end_s is None for s in samples):
        raise SystemExit("Baseline anchor validation failed; evidence preserved; paired trials not started")
    for entry, sample in zip(manifest["samples"], samples):
        entry.update(speech_start_s=sample.first_word_start_s, speech_end_s=sample.last_word_end_s,
                     boundary_fix=sample.boundary_fix)
    (args.output_dir / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for chunk in args.chunk_ms:
        validation, _ = run_trial([xiaomi[0]], args.output_dir / f"warmup_{chunk}.csv", TRANSPORTS, chunk)
        if not validation["ok"]:
            raise SystemExit("Transport warm-up failed; evidence preserved; measured trials not started")
    records = []
    pair_index = 0
    rng = random.Random(42)
    for repetition in range(args.repetitions):
        ordered = list(samples)
        rng.shuffle(ordered)
        for sample_index, sample in enumerate(ordered):
            chunks = args.chunk_ms if (repetition + sample_index) % 2 == 0 else args.chunk_ms[::-1]
            for chunk in chunks:
                # Balance order independently within each chunk-size condition.
                services = TRANSPORTS if (repetition + sample_index) % 2 == 0 else TRANSPORTS[::-1]
                path = args.output_dir / f"pair_{pair_index:03d}_{chunk}ms.csv"
                print(f"Pair {pair_index + 1}/{manifest['expected_pairs']}: {sample.dataset}/{sample.sample_id}, "
                      f"{chunk}ms, {' -> '.join(services)}", flush=True)
                validation, rows = run_trial([sample], path, services, chunk)
                if validation["errors"]:
                    raise SystemExit(f"Structural validation failed in {path}; experiment stopped")
                for row in rows:
                    row.update(repetition=repetition, chunk_ms=chunk, order=list(services),
                               pair_index=pair_index, corpus="xiaomi" if sample in xiaomi else "commands")
                    records.append(row)
                with (args.output_dir / "trials.jsonl").open("a", encoding="utf-8") as destination:
                    for row in rows:
                        destination.write(json.dumps(row) + "\n")
                pair_index += 1
    expected = manifest["expected_pairs"] * 2
    if len(records) != expected:
        raise SystemExit(f"Coverage mismatch: {len(records)} of {expected} rows")
    write_summary(args.output_dir, records)
    validation = {"ok": not any(row["error"] for row in records), "rows": len(records),
                  "expected_rows": expected, "pairs": pair_index,
                  "failed_rows": sum(bool(row["error"]) for row in records)}
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation))
    print(f"Comparison: {args.output_dir / 'comparison.md'}")
    if not validation["ok"]:
        raise SystemExit("Experiment includes failed calls; only successful matched pairs are summarized")


if __name__ == "__main__":
    main()
