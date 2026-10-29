from __future__ import annotations

import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from . import boundary_fix
from .asr_fast import transcribe_fast_default
from .asr_llm_speech import (transcribe_fast_llm,
                             transcribe_fast_mai_1_5, transcribe_fast_mai_2)
from .asr_realtime import (transcribe_realtime, transcribe_realtime_refine,
                           transcribe_realtime_mai_2, transcribe_realtime_diarization)
from .asr_whisper import transcribe_whisper_v3, transcribe_azure_whisper
from .asr_mai_realtime_api import transcribe_realtime_mai_2_api
from .config import DEFAULT_SERVICES, STREAMING_SERVICES, AsrResult, Sample
from .metrics import wer, wer_breakdown, sentence_error, cer_breakdown, normalize_characters
from .eval_options import EvaluationOptions
from .evaluation import evaluation_path, evaluation_record, variant_id
from .datasets_loader import prepare_evaluation

ADAPTERS = {
    "fast_default": transcribe_fast_default,
    "fast_llm": transcribe_fast_llm,
    "fast_mai_1_5": transcribe_fast_mai_1_5,
    "fast_mai_2": transcribe_fast_mai_2,
    "realtime": transcribe_realtime,
    "realtime_refine": transcribe_realtime_refine,
    "realtime_mai_2": transcribe_realtime_mai_2,
    "realtime_mai_2_api": transcribe_realtime_mai_2_api,
    "realtime_diarization": transcribe_realtime_diarization,
    "whisper_v3": transcribe_whisper_v3,
    "azure_whisper": transcribe_azure_whisper,
}

CSV_FIELDS = [
    "dataset", "sample_id", "service", "locale", "duration_s",
    "reference", "hypothesis",
    "wer", "sentence_error",
    "ins", "del", "sub", "ref_len",
    "first_latency_ms", "lbl_ms",
    "upl_ms", "upl_self_ms", "upl_anchor",
    "speech_start_s", "speech_end_s", "boundary_fix_action",
    "first_word_start_s", "last_word_end_s",
    "vad_truncated_s",
    "request_id", "session_id",
    "error",
    "variant_id", "language_mode", "lid_strategy", "reference_available",
    "cer", "char_ins", "char_del", "char_sub", "ref_chars",
]


def _call_with_retries(adapter, sample: Sample, pace: bool, retries: int,
                       phrases: tuple[str, ...] = (),
                       phrase_weight: float = 1.0,
                       chunk_ms: int | None = None) -> AsrResult:
    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            kwargs = {"pace": pace}
            if chunk_ms is not None:
                kwargs["chunk_ms"] = chunk_ms
            if phrases and adapter is not transcribe_whisper_v3:
                kwargs.update(phrases=phrases, phrase_weight=phrase_weight)
            res: AsrResult = adapter(sample, **kwargs)
            if res.error and attempt < retries:
                last_err = res.error
                time.sleep(1.5 * (attempt + 1))
                continue
            return res
        except Exception as e:
            last_err = f"exception: {e!r}"
            time.sleep(1.5 * (attempt + 1))
    return AsrResult(adapter.__name__.replace("transcribe_", ""), "", None, None,
                     last_err or "unknown")


def _to_row(sample: Sample, res: AsrResult) -> dict:
    char_scores = (cer_breakdown(sample.reference, res.hypothesis)
                   if not res.error and sample.reference_available else
                   dict.fromkeys(("cer", "char_ins", "char_del", "char_sub", "ref_chars"), ""))
    if res.error or not sample.reference_available:
        breakdown = {"ins": "", "dele": "", "sub": "", "ref_len": ""}
        wer_val = ""
        ser_val = ""
    else:
        breakdown = wer_breakdown(sample.reference, res.hypothesis)
        wer_val = round(wer(sample.reference, res.hypothesis), 4)
        ser_val = sentence_error(sample.reference, res.hypothesis)
        if sample.locale.startswith(("zh-", "yue-")):
            ser_val = int(normalize_characters(sample.reference) != normalize_characters(res.hypothesis))
    speech_start = sample.first_word_start_s
    speech_end = sample.last_word_end_s
    fix_action = (getattr(sample, "boundary_fix", None) or {}).get("action", "")
    return {
        **char_scores,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "service": res.service,
        "locale": sample.locale,
        "duration_s": round(sample.duration_s, 3),
        "reference": sample.reference if sample.reference_available else "",
        "hypothesis": res.hypothesis,
        "wer": wer_val,
        "sentence_error": ser_val,
        "ins": breakdown["ins"] if not res.error else "",
        "del": breakdown["dele"] if not res.error else "",
        "sub": breakdown["sub"] if not res.error else "",
        "ref_len": breakdown["ref_len"] if not res.error else "",
        "first_latency_ms": round(res.first_latency_ms, 1) if res.first_latency_ms is not None else "",
        "lbl_ms": round(res.lbl_ms, 1) if res.lbl_ms is not None else "",
        "upl_ms": round(res.upl_ms, 1) if res.upl_ms is not None else "",
        "upl_self_ms": round(res.upl_self_ms, 1) if res.upl_self_ms is not None else "",
        "upl_anchor": res.upl_anchor or "",
        "speech_start_s": round(speech_start, 3) if speech_start is not None else "",
        "speech_end_s": round(speech_end, 3) if speech_end is not None else "",
        "boundary_fix_action": fix_action,
        "first_word_start_s": round(res.first_word_start_s, 3) if res.first_word_start_s is not None else "",
        "last_word_end_s": round(res.last_word_end_s, 3) if res.last_word_end_s is not None else "",
        "vad_truncated_s": round(res.vad_truncated_s, 3) if res.vad_truncated_s is not None else "",
        "request_id": res.request_id or "",
        "session_id": res.session_id or "",
        "error": res.error or "",
        "variant_id": variant_id(sample, res.service),
        "language_mode": sample.evaluation_options.language_mode if sample.evaluation_options else "legacy",
        "lid_strategy": sample.evaluation_options.lid_strategy or "" if sample.evaluation_options else "",
        "reference_available": str(sample.reference_available).lower(),
    }


def _run_sample(sample: Sample, services: tuple[str, ...], pace: bool,
                retries: int, words_writer=None,
                phrases: tuple[str, ...] = (),
                phrase_weight: float = 1.0, chunk_ms: int = 100, evaluation_writer=None) -> list[dict]:
    rows: list[dict] = []

    def record_streaming(res: AsrResult) -> None:
        if evaluation_writer is not None:
            evaluation_writer(evaluation_record(sample, res))
        if words_writer is None:
            return
        words_writer({
            **_to_row(sample, res),
            "words": res.words or [],
            "raw_first_word_start_s": res.first_word_start_s,
            "raw_last_word_end_s": res.last_word_end_s,
            "anchored_first_word_start_s": sample.first_word_start_s,
            "anchored_last_word_end_s": sample.last_word_end_s,
            "boundary_fix": sample.boundary_fix,
            "streaming_evidence": res.streaming_evidence,
        })

    if "realtime" in services:
        res = _call_with_retries(transcribe_realtime, sample, pace, retries,
                     phrases, phrase_weight, chunk_ms)
        if not res.error and sample.reference_available and not (
                sample.evaluation_options and (sample.evaluation_options.diarization
                                               or sample.evaluation_options.language_mode == "multilingual")):
            anchored_first, anchored_last, fix_meta = boundary_fix.decide(
                sample.reference, res.hypothesis, res.words)
            sample.first_word_start_s = anchored_first
            sample.last_word_end_s = anchored_last
            sample.boundary_fix = fix_meta
        else:
            sample.first_word_start_s = sample.last_word_end_s = None
            sample.boundary_fix = {"action": "skip", "reason": f"realtime error: {res.error}" if res.error
                                   else "Reference-based anchoring disabled for this evaluation"}
        record_streaming(res)
        rows.append(_to_row(sample, res))

    for service in services:
        if service != "realtime" and service in STREAMING_SERVICES:
            res = _call_with_retries(ADAPTERS[service], sample, pace, retries,
                                     phrases, phrase_weight, chunk_ms)
            record_streaming(res)
            rows.append(_to_row(sample, res))

    fast_services = [s for s in services if s not in STREAMING_SERVICES]
    if fast_services:
        with ThreadPoolExecutor(max_workers=len(fast_services)) as pool:
            futs = {pool.submit(_call_with_retries, ADAPTERS[s], sample, pace, retries,
                                phrases, phrase_weight): s
                    for s in fast_services}
            for fut in as_completed(futs):
                res = fut.result()
                if evaluation_writer is not None:
                    evaluation_writer(evaluation_record(sample, res))
                rows.append(_to_row(sample, res))

    return rows


def _load_done(csv_path: Path) -> set[tuple[str, str, str]]:
    if not csv_path.exists():
        return set()
    done: set[tuple[str, str, str]] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            done.add((r["dataset"], r["sample_id"], r["service"]))
    return done


def run_benchmark(samples: list[Sample], csv_path: Path,
                  services: tuple[str, ...] = DEFAULT_SERVICES,
                  workers: int = 8, pace: bool = True,
                  retries: int = 2, sequential: bool = False,
                  phrases: tuple[str, ...] = (),
                  phrase_weight: float = 1.0, chunk_ms: int = 100,
                  evaluation_options: EvaluationOptions | None = None) -> Path:
    if evaluation_options is not None:
        prepare_evaluation(samples, services, evaluation_options, phrases=phrases)
    if csv_path.exists():
        expected_variants = {(s.dataset, s.sample_id, service): variant_id(s, service)
                             for s in samples for service in services}
        with csv_path.open(encoding="utf-8", newline="") as existing:
            for row in csv.DictReader(existing):
                if row.get("variant_id", "") != expected_variants.get((row["dataset"], row["sample_id"], row["service"])):
                    raise ValueError("Resume configuration variant differs; use a fresh run")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not services or any(service not in ADAPTERS for service in services):
        raise ValueError("Select at least one known service")
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    if csv_path.with_suffix(".csv.strict").exists():
        raise ValueError("Cannot resume a normalized CSV; use a fresh run")
    done = _load_done(csv_path)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    lock = threading.Lock()
    words_lock = threading.Lock()

    pending_samples = [
        s for s in samples
        if not all((s.dataset, s.sample_id, svc) in done for svc in services)
    ]
    if done:
        by_key = {(sample.dataset, sample.sample_id): sample for sample in pending_samples}
        with csv_path.open(encoding="utf-8", newline="") as existing:
            for row in csv.DictReader(existing):
                sample = by_key.get((row["dataset"], row["sample_id"]))
                if sample is not None and row["service"] == "realtime":
                    sample.first_word_start_s = (float(row["speech_start_s"])
                                                 if row.get("speech_start_s") else None)
                    sample.last_word_end_s = (float(row["speech_end_s"])
                                              if row.get("speech_end_s") else None)
                    sample.boundary_fix = {"action": row.get("boundary_fix_action", "")}

    f = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        f.flush()

    words_path = csv_path.with_name(csv_path.stem + "_words.jsonl")
    words_f = (words_path.open("a", encoding="utf-8")
               if any(service in STREAMING_SERVICES for service in services) else None)
    words_done: set[tuple[str, str, str]] = set()
    eval_f = (evaluation_path(csv_path).open("a", encoding="utf-8")
              if evaluation_options is not None else None)

    def evaluation_writer(obj: dict) -> None:
        if eval_f is not None:
            with words_lock:
                eval_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                eval_f.flush()
    if words_path.exists():
        with words_path.open(encoding="utf-8") as existing_words:
            for line in existing_words:
                entry = json.loads(line)
                words_done.add((entry["dataset"], entry["sample_id"],
                                entry.get("service", "realtime")))

    def write_rows(rows: list[dict]):
        with lock:
            for r in rows:
                if (r["dataset"], r["sample_id"], r["service"]) in done:
                    continue
                writer.writerow(r)
            f.flush()

    def words_writer(obj: dict) -> None:
        if words_f is None:
            return
        with words_lock:
            key = (obj["dataset"], obj["sample_id"], obj["service"])
            if key in words_done:
                return
            words_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            words_f.flush()
            words_done.add(key)

    def run_pending(sample: Sample) -> list[dict]:
        pending = tuple(service for service in services
                        if (sample.dataset, sample.sample_id, service) not in done)
        return _run_sample(sample, pending, pace, retries, words_writer, phrases, phrase_weight, chunk_ms,
                           evaluation_writer if eval_f is not None else None)

    try:
        if sequential or workers <= 1:
            for s in tqdm(pending_samples, desc="benchmark"):
                write_rows(run_pending(s))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(run_pending, s) for s in pending_samples]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="benchmark"):
                    write_rows(fut.result())
    finally:
        f.close()
        if words_f is not None:
            words_f.close()
        if eval_f is not None:
            eval_f.close()
    return csv_path
