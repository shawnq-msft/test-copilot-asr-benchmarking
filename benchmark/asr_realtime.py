from __future__ import annotations

import json
import threading
import time

import azure.cognitiveservices.speech as speechsdk

from .audio_utils import chunk_pcm
from .config import (AZURE_SPEECH_ENDPOINT, AZURE_SPEECH_KEY,
                     AZURE_SPEECH_REGION, REALTIME_CHUNK_MS,
                     MAI_STREAMING_MODEL,
                     REFINE_AZURE_ENDPOINT, REFINE_AZURE_KEY,
                     REFINE_AZURE_REGION, SEGMENTATION_SILENCE_MS,
                     TARGET_CHANNELS, TARGET_SAMPWIDTH,
                     TARGET_SR, AsrResult, Sample)


def _extract_words(result_json_str: str) -> list[dict]:
    try:
        data = json.loads(result_json_str)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Invalid streaming result JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Streaming result JSON must be an object")
    nbest = data.get("NBest") or []
    if not nbest:
        return []
    if not isinstance(nbest, list) or not isinstance(nbest[0], dict):
        raise ValueError("Invalid streaming NBest payload")
    raw_words = nbest[0].get("Words") or []
    if not isinstance(raw_words, list) or any(not isinstance(word, dict) for word in raw_words):
        raise ValueError("Invalid streaming word list")
    out: list[dict] = []
    for w in raw_words:
        off = w.get("Offset")
        dur = w.get("Duration")
        if off is None or dur is None:
            continue
        if not isinstance(off, (int, float)) or not isinstance(dur, (int, float)) or off < 0 or dur < 0:
            raise ValueError("Invalid streaming word offsets")
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
                  segmentation_silence_ms: int | None = None,
                  phrases: tuple[str, ...] = (),
                  phrase_weight: float = 1.0,
                  model: str | None = None,
                  chunk_ms: int = REALTIME_CHUNK_MS) -> AsrResult:
    call_start = time.perf_counter()
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    if model and not hasattr(speechsdk.SpeechConfig, "model"):
        return AsrResult(service_name, "", None, None,
                         "MAI streaming requires Speech SDK >=1.52.0 with SpeechConfig.model")
    if model and phrases:
        return AsrResult(service_name, "", None, None,
                         "Phrase-list support is not verified for MAI streaming; remove --phrase-list")
    if endpoint:
        speech_config = speechsdk.SpeechConfig(subscription=key, endpoint=endpoint)
    else:
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    if model:
        speech_config.model = model
    else:
        if sample.evaluation_options is None or sample.evaluation_options.language_mode == "monolingual":
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
    recognizer_args = {"speech_config": speech_config, "audio_config": audio_config}
    if sample.evaluation_options and sample.evaluation_options.language_mode == "multilingual" and not model:
        recognizer_args["auto_detect_source_language_config"] = (
            speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=list(sample.evaluation_options.candidates)))
        if sample.evaluation_options.lid_strategy == "continuous":
            speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous")
    diarization = service_name == "realtime_diarization"
    if diarization:
        recognizer = speechsdk.transcription.ConversationTranscriber(**recognizer_args)
        recognizing_signal, recognized_signal = recognizer.transcribing, recognizer.transcribed
        start_recognition = recognizer.start_transcribing_async
        stop_recognition = recognizer.stop_transcribing_async
    else:
        recognizer = speechsdk.SpeechRecognizer(**recognizer_args)
        recognizing_signal, recognized_signal = recognizer.recognizing, recognizer.recognized
        start_recognition = recognizer.start_continuous_recognition_async
        stop_recognition = recognizer.stop_continuous_recognition_async
    if phrases:
        phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
        for phrase in phrases:
            phrase_list.addPhrase(phrase)
        phrase_list.setWeight(phrase_weight)

    state = {
        "audio_start": None,
        "first_partial": None,
        "first_text": None,
        "last_final": None,
        "first_word_start_s": None,
        "last_word_end_s": None,
        "words": [],
        "finals": [],
        "error": None,
        "request_ids": [],
        "session_id": None,
        "partial_count": 0,
        "completed": False,
        "result_statuses": [],
        "cancellations": [],
    }
    session_done = threading.Event()
    segments = []

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
        if evt.result.text:
            state["partial_count"] += 1
            if state["first_text"] is None:
                state["first_text"] = time.perf_counter()

    def on_recognized(evt):
        remember_session_id(evt)
        remember_request_id(evt.result, allow_result_id_fallback=True)
        state["result_statuses"].append(str(evt.result.reason))
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
            state["last_final"] = time.perf_counter()
            if state["first_text"] is None:
                state["first_text"] = state["last_final"]
            state["finals"].append(evt.result.text)
            try:
                words = _extract_words(evt.result.json)
                detected = None
                if sample.evaluation_options and sample.evaluation_options.language_mode == "multilingual" and not model:
                    detected = speechsdk.AutoDetectSourceLanguageResult(evt.result).language or None
                offset, duration = getattr(evt.result, "offset", None), getattr(evt.result, "duration", None)
                segments.append({
                    "text": evt.result.text, "speaker": getattr(evt.result, "speaker_id", None),
                    "language": detected, "source": "sdk_final_result",
                    "start_s": offset / 1e7 if offset is not None else None,
                    "end_s": (offset + duration) / 1e7 if offset is not None and duration is not None else None,
                    "raw_json": evt.result.json,
                })
            except (ValueError, TypeError, RuntimeError) as exc:
                state["error"] = str(exc)
                session_done.set()
                return
            if words:
                state["words"].extend(words)
                if state["first_word_start_s"] is None:
                    state["first_word_start_s"] = words[0]["start_s"]
                state["last_word_end_s"] = words[-1]["end_s"]

    def on_canceled(evt):
        remember_session_id(evt)
        state["cancellations"].append({"reason": str(evt.reason), "details": evt.error_details})
        if getattr(evt, "result", None) is not None:
            remember_request_id(evt.result, allow_result_id_fallback=True)
        if evt.reason != speechsdk.CancellationReason.EndOfStream:
            state["error"] = f"canceled: {evt.reason} {evt.error_details}"
        else:
            state["completed"] = True
        session_done.set()

    def on_session_started(evt):
        remember_session_id(evt)

    def on_session_stopped(evt):
        remember_session_id(evt)
        state["completed"] = True
        session_done.set()

    recognizing_signal.connect(on_recognizing)
    recognized_signal.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_started.connect(on_session_started)
    recognizer.session_stopped.connect(on_session_stopped)

    try:
        start_recognition().get()
    except Exception as e:
        push_stream.close()
        return AsrResult(service_name, "", None, None, f"start_failed: {e!r}")

    chunks = chunk_pcm(sample.pcm16_mono_16k, chunk_ms)
    chunk_dt = chunk_ms / 1000.0

    state["audio_start"] = time.perf_counter()
    bytes_sent = 0
    try:
        t0 = time.perf_counter()
        for i, c in enumerate(chunks):
            if session_done.is_set():
                state["error"] = state["error"] or "premature_end: session ended before all audio was sent"
                break
            push_stream.write(c)
            bytes_sent += len(c)
            if pace:
                target = t0 + (i + 1) * chunk_dt
                slack = target - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
    except Exception as exc:
        state["error"] = f"stream_failed: {exc!r}"
    finally:
        end_of_audio = time.perf_counter()
        try:
            push_stream.close()
        except Exception as exc:
            state["error"] = state["error"] or f"close_failed: {exc!r}"

    if not session_done.wait(timeout=15.0 if post_refinement else 10.0):
        state["error"] = state["error"] or "timeout: no terminal streaming event after audio closed"
    try:
        stop_recognition().get()
    except Exception as exc:
        state["error"] = state["error"] or f"stop_failed: {exc!r}"

    if not state["finals"] and not state["error"]:
        state["error"] = "no_final: stream ended without a recognized final transcript"

    audio_start = state["audio_start"]

    first_lat_ms = None
    first_word = (sample.first_word_start_s if model and sample.first_word_start_s is not None
                  else state["first_word_start_s"])
    if state["first_partial"] is not None and first_word is not None:
        speech_start_wall = audio_start + first_word
        first_lat_ms = (state["first_partial"] - speech_start_wall) * 1000.0

    lbl_ms = ((state["last_final"] - end_of_audio) * 1000.0
              if state["last_final"] else None)

    upl_self_ms = None
    if state["last_final"] is not None and state["last_word_end_s"] is not None:
        end_of_speech_wall = audio_start + state["last_word_end_s"]
        upl_self_ms = (state["last_final"] - end_of_speech_wall) * 1000.0
    upl_ms = upl_self_ms
    upl_anchor = "self" if model else "realtime"
    if model and state["last_final"] is not None and sample.last_word_end_s is not None:
        upl_ms = (state["last_final"] - audio_start - sample.last_word_end_s) * 1000.0
        upl_anchor = "realtime"

    evidence = {
        "model": model,
        "partial_count": state["partial_count"],
        "final_count": len(state["finals"]),
        "completed": state["completed"],
        "result_statuses": state["result_statuses"],
        "cancellations": state["cancellations"],
        "audio_bytes_sent": bytes_sent,
        "audio_bytes_expected": len(sample.pcm16_mono_16k),
        "pace": pace,
        "chunk_ms": chunk_ms,
        "finalization": "SDK stream close at end of recording",
        "setup_ms": (audio_start - call_start) * 1000.0,
        "first_text_s": (state["first_text"] - audio_start
                         if state["first_text"] is not None else None),
        "first_partial_s": (state["first_partial"] - audio_start
                            if state["first_partial"] is not None else None),
        "last_final_s": (state["last_final"] - audio_start
                         if state["last_final"] is not None else None),
        "end_of_audio_s": end_of_audio - audio_start,
    }
    hypothesis = " ".join(text.strip() for text in state["finals"])
    return AsrResult(service_name, hypothesis,
                     first_lat_ms if not state["error"] else None,
                     lbl_ms if not state["error"] else None, state["error"],
                     request_id="; ".join(state["request_ids"]) or None,
                     session_id=state["session_id"],
                     upl_ms=upl_ms if not state["error"] else None,
                     upl_self_ms=upl_self_ms if not state["error"] else None,
                     upl_anchor=upl_anchor if upl_ms is not None else None,
                     first_word_start_s=state["first_word_start_s"],
                     last_word_end_s=state["last_word_end_s"],
                     words=state["words"] or None,
                     streaming_evidence=evidence, segments=segments)


def transcribe_realtime(sample: Sample, pace: bool = True,
                        phrases: tuple[str, ...] = (),
                        phrase_weight: float = 1.0,
                        chunk_ms: int = REALTIME_CHUNK_MS) -> AsrResult:
    return _run_realtime(sample, service_name="realtime",
                         key=AZURE_SPEECH_KEY,
                         endpoint=AZURE_SPEECH_ENDPOINT,
                         region=AZURE_SPEECH_REGION,
                         pace=pace, post_refinement=False,
                         segmentation_silence_ms=SEGMENTATION_SILENCE_MS,
                         phrases=phrases, phrase_weight=phrase_weight, chunk_ms=chunk_ms)


def transcribe_realtime_refine(sample: Sample, pace: bool = True,
                               phrases: tuple[str, ...] = (),
                               phrase_weight: float = 1.0,
                               chunk_ms: int = REALTIME_CHUNK_MS) -> AsrResult:
    return _run_realtime(sample, service_name="realtime_refine",
                         key=REFINE_AZURE_KEY,
                         endpoint=REFINE_AZURE_ENDPOINT,
                         region=REFINE_AZURE_REGION,
                         pace=pace, post_refinement=True,
                         segmentation_silence_ms=SEGMENTATION_SILENCE_MS,
                         phrases=phrases, phrase_weight=phrase_weight, chunk_ms=chunk_ms)


def transcribe_realtime_mai_2(sample: Sample, pace: bool = True,
                              phrases: tuple[str, ...] = (),
                              phrase_weight: float = 1.0,
                              chunk_ms: int = REALTIME_CHUNK_MS) -> AsrResult:
    return _run_realtime(sample, service_name="realtime_mai_2",
                         key=AZURE_SPEECH_KEY,
                         endpoint=AZURE_SPEECH_ENDPOINT,
                         region=AZURE_SPEECH_REGION,
                         pace=pace, post_refinement=False,
                         model=MAI_STREAMING_MODEL,
                         phrases=phrases, phrase_weight=phrase_weight, chunk_ms=chunk_ms)


def transcribe_realtime_diarization(sample: Sample, pace: bool = True,
                                    phrases: tuple[str, ...] = (), phrase_weight: float = 1.0,
                                    chunk_ms: int = REALTIME_CHUNK_MS) -> AsrResult:
    if phrases or phrase_weight != 1.0:
        return AsrResult("realtime_diarization", "", None, None,
                         "Phrase biasing is not verified for SDK diarization")
    return _run_realtime(sample, service_name="realtime_diarization",
                         key=AZURE_SPEECH_KEY, endpoint=AZURE_SPEECH_ENDPOINT,
                         region=AZURE_SPEECH_REGION, pace=pace, post_refinement=False,
                         chunk_ms=chunk_ms)
