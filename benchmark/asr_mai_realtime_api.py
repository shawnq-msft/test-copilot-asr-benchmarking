"""MAI Realtime WebSocket API, with explicit end-of-recording commit."""
from __future__ import annotations

import asyncio
import base64
import json
import time
from urllib.parse import urlsplit

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from .audio_utils import chunk_pcm
from .config import (AZURE_SPEECH_KEY, MAI_REALTIME_DEPLOYMENT, MAI_REALTIME_ENDPOINT,
                     REALTIME_CHUNK_MS, TARGET_SR, AsrResult, Sample)

SERVICE = "realtime_mai_2_api"
PREFIX = "conversation.item.input_audio_transcription."


def realtime_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ("https", "wss") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("MAI_REALTIME_ENDPOINT must be a documented HTTPS/WSS resource root URL")
    return parsed._replace(scheme="wss", path="/mai/v1/realtime",
                           query="intent=transcription").geturl()


def session_update(deployment: str) -> dict:
    if not deployment.strip():
        raise ValueError("MAI_REALTIME_DEPLOYMENT must identify a deployed MAI streaming model")
    return {"type": "session.update", "session": {
        "type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": TARGET_SR},
            "transcription": {"model": deployment},
            "turn_detection": None, "noise_reduction": None,
        }}}}


class TranscriptEvents:
    def __init__(self) -> None:
        self.delta = ""
        self.intermediate = ""
        self.transcript: str | None = None
        self.first_text: float | None = None
        self.first_partial: float | None = None
        self.last_final: float | None = None
        self.partial_count = 0
        self.delta_count = 0
        self.item_id: str | None = None

    def accept(self, event: dict, now: float) -> None:
        kind = event.get("type")
        if kind == "error" or kind == PREFIX + "failed":
            raise ValueError(f"Realtime API error: {json.dumps(event.get('error', event))}")
        if kind not in (PREFIX + "delta", PREFIX + "intermediate", PREFIX + "completed"):
            return
        item_id = event.get("item_id")
        if item_id:
            if self.item_id is not None and item_id != self.item_id:
                raise ValueError("Unexpected multiple items for a single-commit recording")
            self.item_id = item_id
        field = {"delta": "delta", "intermediate": "intermediate", "completed": "transcript"}[
            kind.removeprefix(PREFIX)]
        text = event.get(field)
        if not isinstance(text, str):
            raise ValueError(f"Realtime API {kind} lacks string {field}")
        if text and self.first_text is None:
            self.first_text = now
        if kind == PREFIX + "delta":
            self.delta += text
            self.intermediate = ""
            self.delta_count += 1
        elif kind == PREFIX + "intermediate":
            self.intermediate = text
            self.partial_count += 1
            if text and self.first_partial is None:
                self.first_partial = now
        else:
            self.transcript = text
            self.last_final = now


async def _transcribe(sample: Sample, pace: bool, chunk_ms: int) -> AsrResult:
    call_start = time.perf_counter()
    url = realtime_url(MAI_REALTIME_ENDPOINT)
    update = session_update(MAI_REALTIME_DEPLOYMENT)
    if sample.evaluation_options and sample.evaluation_options.language_mode == "monolingual":
        update["session"]["audio"]["input"]["transcription"]["language"] = sample.locale.split("-")[0]
    state = TranscriptEvents()
    evidence: dict = {
        "model": MAI_REALTIME_DEPLOYMENT, "pace": pace, "chunk_ms": chunk_ms,
        "finalization": "explicit input_audio_buffer.commit at end of recording",
        "audio_bytes_sent": 0, "audio_bytes_expected": len(sample.pcm16_mono_16k),
        "completed": False, "final_count": 0, "events": [],
    }
    session_id = None
    request_id = None
    audio_start = None
    end_of_audio = None
    error = None

    try:
        async with connect(url, additional_headers={"api-key": AZURE_SPEECH_KEY},
                           open_timeout=20, close_timeout=5, max_size=2**22) as ws:
            if ws.response is not None:
                request_id = ws.response.headers.get("apim-request-id")

            async def receive() -> dict:
                event = json.loads(await ws.recv())
                if not isinstance(event, dict):
                    raise ValueError("Realtime API event must be a JSON object")
                now = time.perf_counter()
                evidence["events"].append({"call_elapsed_s": now - call_start, **event})
                state.accept(event, now)
                return event

            async def wait_for(kind: str) -> dict:
                async with asyncio.timeout(20):
                    while True:
                        event = await receive()
                        if event.get("type") == kind:
                            return event

            created = await wait_for("session.created")
            session_id = created.get("session", {}).get("id")
            await ws.send(json.dumps(update))
            updated = await wait_for("session.updated")
            negotiated = updated["session"]["audio"]["input"]
            evidence["negotiated_input"] = negotiated
            if (negotiated["format"] != update["session"]["audio"]["input"]["format"]
                    or negotiated["transcription"]["model"] != MAI_REALTIME_DEPLOYMENT
                    or negotiated["transcription"].get("language") != update["session"]["audio"]["input"]["transcription"].get("language")
                    or negotiated.get("turn_detection") is not None):
                raise ValueError("Realtime API did not confirm requested format/model/manual commit")

            async def drain() -> None:
                async with asyncio.timeout(sample.duration_s + 30):
                    while state.transcript is None:
                        await receive()

            audio_start = time.perf_counter()
            evidence["setup_ms"] = (audio_start - call_start) * 1000
            receiver = asyncio.create_task(drain())
            try:
                for index, chunk in enumerate(chunk_pcm(sample.pcm16_mono_16k, chunk_ms)):
                    if receiver.done():
                        await receiver
                        raise ValueError("Realtime API completed before all audio was sent")
                    await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                              "audio": base64.b64encode(chunk).decode("ascii")}))
                    evidence["audio_bytes_sent"] += len(chunk)
                    if pace:
                        await asyncio.sleep(max(0, audio_start + (index + 1) * chunk_ms / 1000
                                                - time.perf_counter()))
                end_of_audio = time.perf_counter()
                evidence["commit_s"] = end_of_audio - audio_start
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await receiver
            finally:
                if not receiver.done():
                    receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
            if not state.transcript:
                raise ValueError("Realtime API completed without a final transcript")
            if state.delta != state.transcript:
                raise ValueError("Completed transcript differs from accumulated finalized deltas")
            evidence["completed"] = True
            evidence["final_count"] = 1
    except (WebSocketException, OSError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        error = f"realtime_api_failed: {type(exc).__name__}: {exc}"
        response = getattr(exc, "response", None)
        if response is not None:
            request_id = response.headers.get("apim-request-id")
            evidence["http_status"] = response.status_code
            evidence["http_body"] = response.body.decode("utf-8", errors="replace")

    evidence["partial_count"] = state.partial_count
    evidence["delta_count"] = state.delta_count
    evidence["item_id"] = state.item_id
    for field, moment in (("first_text_s", state.first_text), ("first_partial_s", state.first_partial),
                          ("last_final_s", state.last_final), ("end_of_audio_s", end_of_audio)):
        evidence[field] = moment - audio_start if moment is not None and audio_start is not None else None
    first = lbl = upl = None
    if not error and audio_start is not None and state.last_final is not None:
        if state.first_partial is not None and sample.first_word_start_s is not None:
            first = (state.first_partial - audio_start - sample.first_word_start_s) * 1000
        if end_of_audio is not None:
            lbl = (state.last_final - end_of_audio) * 1000
        if sample.last_word_end_s is not None:
            upl = (state.last_final - audio_start - sample.last_word_end_s) * 1000
    return AsrResult(SERVICE, state.transcript or "", first, lbl, error,
                     request_id=f"apim-request-id={request_id}" if request_id else None,
                     session_id=session_id, upl_ms=upl,
                     upl_anchor="realtime" if upl is not None else None,
                     streaming_evidence=evidence,
                     segments=[{"text": state.transcript, "speaker": None, "language": None,
                                "source": "mai_api_completed", "start_s": None, "end_s": None}]
                     if state.transcript else [])


def transcribe_realtime_mai_2_api(sample: Sample, pace: bool = True,
                                 phrases: tuple[str, ...] = (), phrase_weight: float = 1.0,
                                 chunk_ms: int = REALTIME_CHUNK_MS) -> AsrResult:
    if phrases or phrase_weight != 1.0:
        raise ValueError("Phrase biasing is not enabled for the MAI Realtime API")
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    return asyncio.run(_transcribe(sample, pace, chunk_ms))
