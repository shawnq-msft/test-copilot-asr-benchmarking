"""Azure OpenAI Whisper; separate from the historical Whisper v3 stub."""
from __future__ import annotations

import time
import threading
from collections.abc import Iterator
from urllib.parse import quote, urlsplit

import requests

from . import config
from .asr_fast import _response_request_id
from .audio_utils import chunk_pcm, pcm_to_wav_bytes
from .config import AsrResult, Sample

_request_lock = threading.Lock()
_next_request = 0.0


def _wait_for_request_slot() -> None:
    global _next_request
    if config.WHISPER_AZURE_REQUESTS_PER_MINUTE < 1:
        raise ValueError("WHISPER_AZURE_REQUESTS_PER_MINUTE must be positive")
    with _request_lock:
        now = time.perf_counter()
        time.sleep(max(0, _next_request - now))
        _next_request = time.perf_counter() + 60 / config.WHISPER_AZURE_REQUESTS_PER_MINUTE + .1


def transcribe_whisper_v3(sample: Sample, pace: bool = True) -> AsrResult:
    return AsrResult("whisper_v3", "", None, None, "not_implemented")


def azure_whisper_url() -> str:
    endpoint = config.WHISPER_AZURE_ENDPOINT
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise ValueError("WHISPER_AZURE_ENDPOINT must be an HTTPS resource root URL")
    if not config.WHISPER_AZURE_DEPLOYMENT.strip():
        raise ValueError("WHISPER_AZURE_DEPLOYMENT must identify an existing Whisper deployment")
    return (f"{endpoint.rstrip('/')}/openai/deployments/"
            f"{quote(config.WHISPER_AZURE_DEPLOYMENT, safe='')}/audio/transcriptions"
            f"?api-version={config.WHISPER_API_VERSION}")


class _WhisperUpload:
    def __init__(self, sample: Sample, pace: bool) -> None:
        self.pcm = sample.pcm16_mono_16k
        self.pace = pace
        self.audio_start: float | None = None
        self.end_of_audio: float | None = None
        self.boundary = "----AzureWhisperBenchmark"
        language = sample.locale.split("-")[0]
        self.prefix = (
            f"--{self.boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n"
            f"{language}\r\n--{self.boundary}\r\n"
            'Content-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'
            f"--{self.boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + pcm_to_wav_bytes(self.pcm)[:44]
        self.suffix = f"\r\n--{self.boundary}--\r\n".encode()

    def __len__(self) -> int:
        return len(self.prefix) + len(self.pcm) + len(self.suffix)

    def __iter__(self) -> Iterator[bytes]:
        yield self.prefix
        self.audio_start = time.perf_counter()
        sent = 0
        for chunk in chunk_pcm(self.pcm, config.REALTIME_CHUNK_MS):
            yield chunk
            sent += len(chunk)
            if self.pace:
                target = self.audio_start + sent / (config.TARGET_SR * config.TARGET_SAMPWIDTH)
                time.sleep(max(0, target - time.perf_counter()))
        self.end_of_audio = time.perf_counter()
        yield self.suffix


def transcribe_azure_whisper(sample: Sample, pace: bool = True) -> AsrResult:
    service = "azure_whisper"
    url = azure_whisper_url()
    if len(sample.pcm16_mono_16k) + 44 > 25_000_000:
        return AsrResult(service, "", None, None, "Azure Whisper file exceeds 25 MB; no automatic splitting")
    body = _WhisperUpload(sample, pace)
    _wait_for_request_slot()
    started = time.perf_counter()
    try:
        response = requests.post(
            url, headers={"api-key": config.WHISPER_AZURE_KEY,
                          "Content-Type": f"multipart/form-data; boundary={body.boundary}"},
            data=body, timeout=300,
        )
    except requests.RequestException as exc:
        return AsrResult(service, "", None, None, f"http_error: {exc!r}")
    finished = time.perf_counter()
    first_ms = (finished - started) * 1000
    lbl_ms = (finished - body.end_of_audio) * 1000 if body.end_of_audio is not None else None
    request_id = _response_request_id(response) or None
    if response.status_code != 200:
        return AsrResult(service, "", first_ms, lbl_ms,
                         f"status={response.status_code} body={response.content[:500]!r}",
                         request_id=request_id)
    try:
        payload = response.json()
    except ValueError as exc:
        return AsrResult(service, "", first_ms, lbl_ms, f"json_parse: {exc}", request_id=request_id)
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        return AsrResult(service, "", first_ms, lbl_ms, "invalid_response: missing string text",
                         request_id=request_id)
    text = payload["text"]
    if not text.strip():
        return AsrResult(service, "", first_ms, lbl_ms, "no_final: empty Whisper transcript",
                         request_id=request_id)
    if body.audio_start is None or body.end_of_audio is None:
        return AsrResult(service, text, first_ms, lbl_ms, "incomplete_upload: audio was not fully consumed",
                         request_id=request_id)
    upl_ms = ((finished - body.audio_start - sample.last_word_end_s) * 1000
              if sample.last_word_end_s is not None else None)
    return AsrResult(
        service, text, first_ms, lbl_ms, request_id=request_id,
        upl_ms=upl_ms, upl_anchor="realtime" if upl_ms is not None else None,
        segments=[{"text": text, "speaker": None, "language": None,
                   "source": "azure_whisper_final", "raw": payload}],
    )
