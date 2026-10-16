from __future__ import annotations

from .asr_fast import _streaming_upload, add_phrase_list
from .config import (LLM_API_VERSION, LLM_AZURE_KEY, SEGMENTATION_SILENCE_MS,
                     AsrResult, Sample, llm_rest_base_url)


def _endpoint() -> str:
    return f"{llm_rest_base_url()}?api-version={LLM_API_VERSION}"


def transcribe_fast_llm(sample: Sample, pace: bool = True,
                        phrases: tuple[str, ...] = (),
                        phrase_weight: float = 1.0) -> AsrResult:
    definition = {
        "enhancedMode": {
            "enabled": True,
            "task": "transcribe",
        }
    }
    add_phrase_list(definition, phrases, phrase_weight)
    vad_end_s = None
    if sample.last_word_end_s is not None:
        vad_end_s = sample.last_word_end_s + SEGMENTATION_SILENCE_MS / 1000.0
    return _streaming_upload(sample.pcm16_mono_16k, definition,
                             _endpoint(), pace, "fast_llm", sample,
                             key=LLM_AZURE_KEY, vad_end_s=vad_end_s)


def transcribe_fast_mai_1_5(sample: Sample, pace: bool = True,
                            phrases: tuple[str, ...] = (),
                            phrase_weight: float = 1.0) -> AsrResult:
    locale_short = sample.locale.split("-")[0]
    definition = {
        "locales": [locale_short],
        "enhancedMode": {
            "enabled": True,
            "model": "MAI-Transcribe-1.5",
        },
    }
    add_phrase_list(definition, phrases, phrase_weight)
    vad_end_s = None
    if sample.last_word_end_s is not None:
        vad_end_s = sample.last_word_end_s + SEGMENTATION_SILENCE_MS / 1000.0
    return _streaming_upload(sample.pcm16_mono_16k, definition,
                             _endpoint(), pace, "fast_mai_1_5", sample,
                             key=LLM_AZURE_KEY, vad_end_s=vad_end_s)


def transcribe_fast_mai_2(sample: Sample, pace: bool = True,
                          phrases: tuple[str, ...] = (),
                          phrase_weight: float = 1.0) -> AsrResult:
    locale_short = sample.locale.split("-")[0]
    definition = {
        "locales": [locale_short],
        "enhancedMode": {
            "enabled": True,
            "model": "MAI-Transcribe-2",
        },
    }
    add_phrase_list(definition, phrases, phrase_weight)
    vad_end_s = None
    if sample.last_word_end_s is not None:
        vad_end_s = sample.last_word_end_s + SEGMENTATION_SILENCE_MS / 1000.0
    return _streaming_upload(sample.pcm16_mono_16k, definition,
                             _endpoint(), pace, "fast_mai_2", sample,
                             key=LLM_AZURE_KEY, vad_end_s=vad_end_s)
