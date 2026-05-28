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
from .asr_llm_speech import transcribe_fast_llm, transcribe_fast_mai
from .asr_realtime import transcribe_realtime, transcribe_realtime_refine
from .asr_whisper import transcribe_whisper_v3
from .config import SERVICES, AsrResult, Sample
from .metrics import wer, wer_breakdown, sentence_error

ADAPTERS = {
    "fast_default": transcribe_fast_default,
    "fast_llm": transcribe_fast_llm,
    "fast_mai": transcribe_fast_mai,
    "realtime": transcribe_realtime,
    "realtime_refine": transcribe_realtime_refine,
    "whisper_v3": transcribe_whisper_v3,
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
    "error",
]


def _call_with_retries(adapter, sample: Sample, pace: bool, retries: int) -> AsrResult:
    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            res: AsrResult = adapter(sample, pace=pace)
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
    if res.error:
        breakdown = {"ins": "", "dele": "", "sub": "", "ref_len": ""}
        wer_val = ""
        ser_val = ""
    else:
        breakdown = wer_breakdown(sample.reference, res.hypothesis)
        wer_val = round(wer(sample.reference, res.hypothesis), 4)
        ser_val = sentence_error(sample.reference, res.hypothesis)
    speech_start = sample.first_word_start_s
    speech_end = sample.last_word_end_s
    fix_action = (getattr(sample, "boundary_fix", None) or {}).get("action", "")
    return {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "service": res.service,
        "locale": sample.locale,
        "duration_s": round(sample.duration_s, 3),
        "reference": sample.reference,
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
        "error": res.error or "",
    }


def _run_sample(sample: Sample, services: tuple[str, ...], pace: bool,
                retries: int, words_writer=None) -> list[dict]:
    rows: list[dict] = []

    if "realtime" in services:
        res = _call_with_retries(transcribe_realtime, sample, pace, retries)
        if not res.error:
            anchored_first, anchored_last, fix_meta = boundary_fix.decide(
                sample.reference, res.hypothesis, res.words)
            sample.first_word_start_s = anchored_first
            sample.last_word_end_s = anchored_last
            sample.boundary_fix = fix_meta
            if words_writer is not None:
                bd = wer_breakdown(sample.reference, res.hypothesis)
                ref_len = max(bd["ref_len"], 1)
                wer_val = (bd["ins"] + bd["dele"] + bd["sub"]) / ref_len
                words_writer({
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "reference": sample.reference,
                    "hypothesis": res.hypothesis,
                    "wer": round(wer_val, 4),
                    "ins": bd["ins"], "del": bd["dele"], "sub": bd["sub"],
                    "ref_len": bd["ref_len"],
                    "words": res.words or [],
                    "raw_first_word_start_s": res.first_word_start_s,
                    "raw_last_word_end_s": res.last_word_end_s,
                    "anchored_first_word_start_s": anchored_first,
                    "anchored_last_word_end_s": anchored_last,
                    "boundary_fix": fix_meta,
                })
        else:
            sample.boundary_fix = {"action": "skip", "reason": f"realtime error: {res.error}"}
        rows.append(_to_row(sample, res))

    if "realtime_refine" in services:
        res = _call_with_retries(transcribe_realtime_refine, sample, pace, retries)
        rows.append(_to_row(sample, res))

    fast_services = [s for s in services if s not in ("realtime", "realtime_refine")]
    if fast_services:
        with ThreadPoolExecutor(max_workers=len(fast_services)) as pool:
            futs = {pool.submit(_call_with_retries, ADAPTERS[s], sample, pace, retries): s
                    for s in fast_services}
            for fut in as_completed(futs):
                rows.append(_to_row(sample, fut.result()))

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
                  services: tuple[str, ...] = SERVICES,
                  workers: int = 8, pace: bool = True,
                  retries: int = 2, sequential: bool = False) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(csv_path)
    new_file = not csv_path.exists()
    lock = threading.Lock()
    words_lock = threading.Lock()

    pending_samples = [
        s for s in samples
        if not all((s.dataset, s.sample_id, svc) in done for svc in services)
    ]

    f = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        f.flush()

    words_path = csv_path.with_name(csv_path.stem + "_words.jsonl")
    words_f = words_path.open("a", encoding="utf-8") if "realtime" in services else None

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
            words_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            words_f.flush()

    try:
        if sequential or workers <= 1:
            for s in tqdm(pending_samples, desc="benchmark"):
                write_rows(_run_sample(s, services, pace, retries, words_writer))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_run_sample, s, services, pace, retries, words_writer)
                        for s in pending_samples]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="benchmark"):
                    write_rows(fut.result())
    finally:
        f.close()
        if words_f is not None:
            words_f.close()
    return csv_path
