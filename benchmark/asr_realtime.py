from __future__ import annotations

import json
import threading
import time

import azure.cognitiveservices.speech as speechsdk

from .audio_utils import chunk_pcm
from .config import (AZURE_SPEECH_ENDPOINT, AZURE_SPEECH_KEY,
                     AZURE_SPEECH_REGION, REALTIME_CHUNK_MS,
                     REFINE_AZURE_ENDPOINT, REFINE_AZURE_KEY,
                     REFINE_AZURE_REGION, SEGMENTATION_SILENCE_MS,
                     TARGET_CHANNELS, TARGET_SAMPWIDTH,
                     TARGET_SR, AsrResult, Sample)


def _extract_words(result_json_str: str) -> list[dict]:
    try:
        data = json.loads(result_json_str)
    except Exception:
        return []
    nbest = data.get("NBest") or []
    if not nbest:
        return []
    raw_words = nbest[0].get("Words") or []
    out: list[dict] = []
    for w in raw_words:
        off = w.get("Offset")
        dur = w.get("Duration")
        if off is None or dur is None:
            continue
        out.append({
            "text": w.get("Word") or "",
            "start_s": off / 1e7,
            "end_s": (off + dur) / 1e7,
            "confidence": w.get("Confidence"),
        })
    return out


def _result_request_id(result, *, allow_result_id_fallback: bool = False) -> str | None:
    prop_id = getattr(speechsdk.PropertyId, "SpeechServiceResponse_RequestId", None)
    if prop_id is not None:
        try:
            request_id = result.properties.get_property(prop_id)
            if request_id:
                return f"RequestId={request_id}"
        except Exception:
            pass
    result_id = getattr(result, "result_id", None)
    if allow_result_id_fallback and result_id:
        return f"ResultId={result_id}"
    return None


def _run_realtime(sample: Sample, *, service_name: str, key: str,
                  endpoint: str, region: str, pace: bool,
                  post_refinement: bool,
                  segmentation_silence_ms: int | None = None) -> AsrResult:
    if endpoint:
        speech_config = speechsdk.SpeechConfig(subscription=key, endpoint=endpoint)
    else:
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = sample.locale
    speech_config.output_format = speechsdk.OutputFormat.Detailed
    speech_config.request_word_level_timestamps()
    if post_refinement:
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption,
            "PostRefinement",
        )
    if segmentation_silence_ms is not None:
        speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            str(segmentation_silence_ms),
        )

    stream_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=TARGET_SR,
        bits_per_sample=TARGET_SAMPWIDTH * 8,
        channels=TARGET_CHANNELS,
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,
                                            audio_config=audio_config)

    state = {
        "audio_start": None,
        "first_partial": None,
        "last_final": None,
        "first_word_start_s": None,
        "last_word_end_s": None,
        "words": [],
        "finals": [],
        "error": None,
        "request_ids": [],
        "session_id": None,
    }
    session_done = threading.Event()

    def remember_request_id(result, *, allow_result_id_fallback: bool = False) -> None:
        request_id = _result_request_id(
            result,
            allow_result_id_fallback=allow_result_id_fallback,
        )
        if request_id and request_id not in state["request_ids"]:
            state["request_ids"].append(request_id)

    def remember_session_id(evt) -> None:
        session_id = getattr(evt, "session_id", None)
        if session_id and state["session_id"] is None:
            state["session_id"] = session_id

    def on_recognizing(evt):
        remember_session_id(evt)
        remember_request_id(evt.result)
        if state["first_partial"] is None and evt.result.text:
            state["first_partial"] = time.perf_counter()

    def on_recognized(evt):
        remember_session_id(evt)
        remember_request_id(evt.result, allow_result_id_fallback=True)
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
            state["last_final"] = time.perf_counter()
            state["finals"].append(evt.result.text)
            words = _extract_words(evt.result.json)
            if words:
                state["words"].extend(words)
                if state["first_word_start_s"] is None:
                    state["first_word_start_s"] = words[0]["start_s"]
                state["last_word_end_s"] = words[-1]["end_s"]

    def on_canceled(evt):
        remember_session_id(evt)
        if getattr(evt, "result", None) is not None:
            remember_request_id(evt.result, allow_result_id_fallback=True)
        if evt.reason != speechsdk.CancellationReason.EndOfStream:
            state["error"] = f"canceled: {evt.reason} {evt.error_details}"
        session_done.set()

    def on_session_started(evt):
        remember_session_id(evt)

    def on_session_stopped(evt):
        remember_session_id(evt)
        session_done.set()

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_started.connect(on_session_started)
    recognizer.session_stopped.connect(on_session_stopped)

    try:
        recognizer.start_continuous_recognition_async().get()
    except Exception as e:
        return AsrResult(service_name, "", None, None, f"start_failed: {e!r}")

    chunks = chunk_pcm(sample.pcm16_mono_16k, REALTIME_CHUNK_MS)
    chunk_dt = REALTIME_CHUNK_MS / 1000.0

    state["audio_start"] = time.perf_counter()
    try:
        t0 = time.perf_counter()
        for i, c in enumerate(chunks):
            push_stream.write(c)
            if pace:
                target = t0 + (i + 1) * chunk_dt
                slack = target - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
    finally:
        push_stream.close()

    end_of_audio = time.perf_counter()
    session_done.wait(timeout=15.0 if post_refinement else 10.0)
    try:
        recognizer.stop_continuous_recognition_async().get()
    except Exception:
        pass

    if state["error"]:
        return AsrResult(service_name, "".join(state["finals"]), None, None,
                         state["error"],
                         request_id="; ".join(state["request_ids"]) or None,
                         session_id=state["session_id"])

    audio_start = state["audio_start"]

    first_lat_ms = None
    if state["first_partial"] is not None and state["first_word_start_s"] is not None:
        speech_start_wall = audio_start + state["first_word_start_s"]
        first_lat_ms = (state["first_partial"] - speech_start_wall) * 1000.0

    lbl_ms = ((state["last_final"] - end_of_audio) * 1000.0
              if state["last_final"] else None)

    upl_ms = None
    if state["last_final"] is not None and state["last_word_end_s"] is not None:
        end_of_speech_wall = audio_start + state["last_word_end_s"]
        upl_ms = (state["last_final"] - end_of_speech_wall) * 1000.0

    hypothesis = "".join(state["finals"])
    return AsrResult(service_name, hypothesis, first_lat_ms, lbl_ms, None,
                     request_id="; ".join(state["request_ids"]) or None,
                     session_id=state["session_id"],
                     upl_ms=upl_ms,
                     upl_self_ms=upl_ms,
                     upl_anchor="realtime",
                     first_word_start_s=state["first_word_start_s"],
                     last_word_end_s=state["last_word_end_s"],
                     words=state["words"] or None)


def transcribe_realtime(sample: Sample, pace: bool = True) -> AsrResult:
    return _run_realtime(sample, service_name="realtime",
                         key=AZURE_SPEECH_KEY,
                         endpoint=AZURE_SPEECH_ENDPOINT,
                         region=AZURE_SPEECH_REGION,
                         pace=pace, post_refinement=False,
                         segmentation_silence_ms=SEGMENTATION_SILENCE_MS)


def transcribe_realtime_refine(sample: Sample, pace: bool = True) -> AsrResult:
    return _run_realtime(sample, service_name="realtime_refine",
                         key=REFINE_AZURE_KEY,
                         endpoint=REFINE_AZURE_ENDPOINT,
                         region=REFINE_AZURE_REGION,
                         pace=pace, post_refinement=True,
                         segmentation_silence_ms=SEGMENTATION_SILENCE_MS)
