from __future__ import annotations

import json
import time

import requests

from .audio_utils import chunk_pcm, truncate_pcm, wav_header_placeholder
from .config import (AZURE_SPEECH_KEY, FAST_API_VERSION,
                     REALTIME_CHUNK_MS, SEGMENTATION_SILENCE_MS,
                     AsrResult, Sample, rest_base_url)

BOUNDARY = "----AzureBenchBoundary9876543210"


def _endpoint() -> str:
    return f"{rest_base_url()}?api-version={FAST_API_VERSION}"


def _multipart_preamble(definition: str) -> bytes:
    return (
        f"--{BOUNDARY}\r\n".encode()
        + b'Content-Disposition: form-data; name="definition"\r\n'
        + b"Content-Type: application/json\r\n\r\n"
        + definition.encode()
        + b"\r\n"
        + f"--{BOUNDARY}\r\n".encode()
        + b'Content-Disposition: form-data; name="audio"; filename="audio.wav"\r\n'
        + b"Content-Type: audio/wav\r\n\r\n"
        + wav_header_placeholder()
    )


def _multipart_epilogue() -> bytes:
    return f"\r\n--{BOUNDARY}--\r\n".encode()


def _streaming_upload(pcm: bytes, definition_obj: dict, url: str,
                      pace: bool, service: str,
                      sample: Sample,
                      key: str = None,
                      vad_end_s: float | None = None) -> AsrResult:
    if key is None:
        key = AZURE_SPEECH_KEY
    if vad_end_s is not None:
        pcm = truncate_pcm(pcm, vad_end_s)
    chunks = chunk_pcm(pcm, REALTIME_CHUNK_MS)
    definition = json.dumps(definition_obj)

    streaming_start_ts: list[float] = []
    audio_start_ts: list[float] = []
    end_of_audio_ts: list[float] = []

    def body():
        streaming_start_ts.append(time.perf_counter())
        yield _multipart_preamble(definition)
        chunk_dt = REALTIME_CHUNK_MS / 1000.0
        t0 = time.perf_counter()
        for i, c in enumerate(chunks):
            if i == 0:
                audio_start_ts.append(time.perf_counter())
            yield c
            if pace:
                target = t0 + (i + 1) * chunk_dt
                slack = target - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
        end_of_audio_ts.append(time.perf_counter())
        yield _multipart_epilogue()

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
        "Transfer-Encoding": "chunked",
    }
    try:
        resp = requests.post(url, headers=headers, data=body(),
                             timeout=300, stream=True)
    except Exception as e:
        return AsrResult(service, "", None, None, f"http_error: {e!r}")

    first_response_ts = time.perf_counter()
    try:
        body_bytes = resp.content
    except Exception as e:
        return AsrResult(service, "", None, None, f"read_body: {e!r}")
    final_response_ts = time.perf_counter()

    s_start = streaming_start_ts[0] if streaming_start_ts else first_response_ts
    audio_start = audio_start_ts[0] if audio_start_ts else s_start
    eoa = end_of_audio_ts[0] if end_of_audio_ts else final_response_ts
    first_latency_ms = (first_response_ts - s_start) * 1000.0
    lbl_ms = (final_response_ts - eoa) * 1000.0

    if resp.status_code != 200:
        return AsrResult(service, "", first_latency_ms, lbl_ms,
                         f"status={resp.status_code} body={body_bytes[:300]!r}")
    try:
        data = json.loads(body_bytes.decode("utf-8"))
    except Exception as e:
        return AsrResult(service, "", first_latency_ms, lbl_ms, f"json_parse: {e!r}")

    text = ""
    if data.get("combinedPhrases"):
        text = data["combinedPhrases"][0].get("text", "") or ""

    upl_self_ms = None
    phrases = data.get("phrases") or []
    if phrases:
        last = phrases[-1]
        off_ms = last.get("offsetMilliseconds")
        dur_ms = last.get("durationMilliseconds")
        if off_ms is not None and dur_ms is not None:
            end_of_speech_wall = audio_start + (off_ms + dur_ms) / 1000.0
            upl_self_ms = (final_response_ts - end_of_speech_wall) * 1000.0

    if sample.last_word_end_s is not None:
        end_of_speech_wall = audio_start + sample.last_word_end_s
        upl_ms = (final_response_ts - end_of_speech_wall) * 1000.0
        upl_anchor = "realtime"
    else:
        upl_ms = upl_self_ms
        upl_anchor = "self" if upl_self_ms is not None else None

    return AsrResult(service, text, first_latency_ms, lbl_ms, None,
                     upl_ms=upl_ms, upl_self_ms=upl_self_ms,
                     upl_anchor=upl_anchor,
                     vad_truncated_s=vad_end_s)


def transcribe_fast_default(sample: Sample, pace: bool = True) -> AsrResult:
    definition = {"locales": [sample.locale]}
    vad_end_s = None
    if sample.last_word_end_s is not None:
        vad_end_s = sample.last_word_end_s + SEGMENTATION_SILENCE_MS / 1000.0
    return _streaming_upload(sample.pcm16_mono_16k, definition,
                             _endpoint(), pace, "fast_default", sample,
                             vad_end_s=vad_end_s)
