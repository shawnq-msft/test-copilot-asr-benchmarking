import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
AZURE_SPEECH_ENDPOINT = (os.getenv("AZURE_SPEECH_ENDPOINT") or "").rstrip("/")

LLM_AZURE_KEY = os.getenv("LLM_AZURE_SPEECH_KEY") or AZURE_SPEECH_KEY
LLM_AZURE_ENDPOINT = ((os.getenv("LLM_AZURE_SPEECH_ENDPOINT") or "").rstrip("/")
                      or AZURE_SPEECH_ENDPOINT)
LLM_AZURE_REGION = os.getenv("LLM_AZURE_SPEECH_REGION") or AZURE_SPEECH_REGION

REFINE_AZURE_KEY = os.getenv("REFINE_AZURE_SPEECH_KEY") or AZURE_SPEECH_KEY
REFINE_AZURE_ENDPOINT = ((os.getenv("REFINE_AZURE_SPEECH_ENDPOINT") or "").rstrip("/")
                         or AZURE_SPEECH_ENDPOINT)
REFINE_AZURE_REGION = os.getenv("REFINE_AZURE_SPEECH_REGION") or AZURE_SPEECH_REGION

FAST_API_VERSION = "2024-11-15"
LLM_API_VERSION = "2025-10-15"

# Default locales for the bundled sample testset.
# Override via --languages on the CLI or pass your own list to run_full.py.
LOCALES = ("de-DE", "en-GB", "es-ES", "fr-FR", "it-IT")

TARGET_SR = 16000
TARGET_CHANNELS = 1
TARGET_SAMPWIDTH = 2

REALTIME_CHUNK_MS = 100
SEGMENTATION_SILENCE_MS = 500

SERVICES = (
    "fast_default",
    "fast_llm",
    "fast_mai",
    "fast_mai_1_5",
    "fast_mai_2",
    "realtime",
    "realtime_refine",
    "whisper_v3",
)


def rest_base_url() -> str:
    if AZURE_SPEECH_ENDPOINT:
        return f"{AZURE_SPEECH_ENDPOINT}/speechtotext/transcriptions:transcribe"
    return (f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com"
            f"/speechtotext/transcriptions:transcribe")


def llm_rest_base_url() -> str:
    if LLM_AZURE_ENDPOINT:
        return f"{LLM_AZURE_ENDPOINT}/speechtotext/transcriptions:transcribe"
    return (f"https://{LLM_AZURE_REGION}.api.cognitive.microsoft.com"
            f"/speechtotext/transcriptions:transcribe")


def rest_host() -> str:
    if AZURE_SPEECH_ENDPOINT:
        return urlparse(AZURE_SPEECH_ENDPOINT).hostname or ""
    return f"{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com"


def llm_rest_host() -> str:
    if LLM_AZURE_ENDPOINT:
        return urlparse(LLM_AZURE_ENDPOINT).hostname or ""
    return f"{LLM_AZURE_REGION}.api.cognitive.microsoft.com"


@dataclass
class Sample:
    sample_id: str
    dataset: str
    locale: str
    pcm16_mono_16k: bytes
    reference: str
    duration_s: float
    first_word_start_s: float | None = None
    last_word_end_s: float | None = None


@dataclass
class AsrResult:
    service: str
    hypothesis: str
    first_latency_ms: float | None
    lbl_ms: float | None
    error: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    upl_ms: float | None = None
    upl_self_ms: float | None = None
    upl_anchor: str | None = None
    first_word_start_s: float | None = None
    last_word_end_s: float | None = None
    words: list[dict] | None = None
    boundary_fix: dict | None = None
    vad_truncated_s: float | None = None


def assert_credentials():
    if not AZURE_SPEECH_KEY:
        raise SystemExit("AZURE_SPEECH_KEY must be set in .env (see .env.example)")
    if not AZURE_SPEECH_ENDPOINT and not AZURE_SPEECH_REGION:
        raise SystemExit("AZURE_SPEECH_ENDPOINT or AZURE_SPEECH_REGION must be set in .env")
