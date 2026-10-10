"""Generate synthetic WAV test samples using Azure Text-to-Speech.

Reads reference commands from testdata/<locale>/<dataset>/trans.txt,
synthesizes speech for any missing WAV files using Azure TTS, and saves
them as 16 kHz mono PCM WAV — exactly the format the benchmark expects.

trans.txt format (one entry per line):
  - Tab-separated:  filename.wav<TAB>Reference text
  - Plain text:     Reference text   (filename auto-derived from text)

Lines starting with '#' are treated as comments and ignored.

Usage:
    # All locales and all datasets
    python scripts/generate_testset.py

    # One locale, all its datasets
    python scripts/generate_testset.py --locale en-GB

    # Specific locale + dataset
    python scripts/generate_testset.py --locale en-GB --dataset smart-commands

    # Force re-synthesis even if WAV already exists
    python scripts/generate_testset.py --locale en-GB --force

Environment (from .env or system):
    AZURE_SPEECH_KEY      — Azure Speech resource key
    AZURE_SPEECH_REGION   — Azure region (e.g. eastus)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from benchmark.datasets_loader import _load_trans

load_dotenv()

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")

# Default neural voice per locale. Override with --voice on the CLI.
DEFAULT_VOICES: dict[str, str] = {
    "en-GB": "en-GB-SoniaNeural",
    "en-US": "en-US-JennyNeural",
    "de-DE": "de-DE-KatjaNeural",
    "es-ES": "es-ES-ElviraNeural",
    "fr-FR": "fr-FR-DeniseNeural",
    "it-IT": "it-IT-ElsaNeural",
    "ja-JP": "ja-JP-NanamiNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "nb-NO": "nb-NO-PernilleNeural",
    "nl-NL": "nl-NL-ColetteNeural",
    "pl-PL": "pl-PL-AgnieszkaNeural",
    "sv-SE": "sv-SE-SofieNeural",
}

DATA_ROOT = Path(__file__).resolve().parents[1] / "testdata"


def _read_trans(trans_path: Path) -> list[tuple[str, str]]:
    return _load_trans(trans_path)


def _synthesize(text: str, voice: str, out_path: Path) -> bool:
    """Synthesize text → 16 kHz mono PCM WAV using Azure TTS SDK."""
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        print("ERROR: azure-cognitiveservices-speech not installed.")
        print("       Run: python -m pip install -r requirements.txt")
        return False

    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION,
    )
    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(out_path))
    synth = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )
    result = synth.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return True
    try:
        details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
        print(f"    [TTS error] {details.reason}: {details.error_details}")
    except Exception:
        print(f"    [TTS error] reason={result.reason}")
    return False


def process_dataset(locale: str, dataset: str, voice: str, force: bool = False,
                    limit: int = 0, seed: int = 42) -> None:
    ds_dir = DATA_ROOT / locale / dataset
    trans_path = ds_dir / "trans.txt"
    if not trans_path.exists():
        raise ValueError(f"No trans.txt found at {trans_path}; custom recordings are not synthesized")

    entries = _read_trans(trans_path)
    if not entries:
        raise ValueError(f"Empty reference file: {trans_path}")
    if limit and len(entries) > limit:
        entries = random.Random(seed).sample(entries, limit)

    print(f"\n[{locale}/{dataset}] synthesizing {len(entries)} entries (voice={voice})")
    n_ok = n_skip = n_fail = 0
    for filename, reference in entries:
        wav_path = ds_dir / filename
        if wav_path.exists() and not force:
            n_skip += 1
            continue
        print(f"  → {filename}")
        if _synthesize(reference, voice, wav_path):
            n_ok += 1
        else:
            print(f"    FAILED: {reference!r}")
            n_fail += 1

    status = f"{n_ok} synthesized, {n_skip} skipped (already exist)"
    if n_fail:
        status += f", {n_fail} FAILED"
    print(f"[{locale}/{dataset}] done: {status}")
    if n_fail:
        raise RuntimeError(f"{locale}/{dataset}: {n_fail} TTS failures; validate audio before benchmarking")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate synthetic WAV test samples from trans.txt using Azure TTS"
    )
    p.add_argument("--locale", default=None,
                   help="Locale to process (e.g. en-GB). Default: all locales in testdata/.")
    p.add_argument("--dataset", default=None,
                   help="Dataset name to process (e.g. smart-commands). Default: all datasets.")
    p.add_argument("--voice", default=None,
                   help="Azure TTS neural voice name. Defaults to the locale's default voice.")
    p.add_argument("--force", action="store_true",
                   help="Re-synthesize WAVs even if they already exist.")
    p.add_argument("--testdata-dir", type=Path, help="Custom testdata root")
    p.add_argument("--max-per-dataset", type=int, default=0,
                   help="Select the same sample subset as the benchmark (default: 0, all)")
    p.add_argument("--seed", type=int, default=42, help="Match the benchmark's subsampling seed")
    args = p.parse_args()
    global DATA_ROOT
    if args.testdata_dir:
        DATA_ROOT = args.testdata_dir
    if args.dataset and not args.locale:
        p.error("--dataset requires --locale")
    if args.max_per_dataset < 0:
        p.error("--max-per-dataset must be >=0")
    selected_locales = [args.locale] if args.locale else [p.name for p in DATA_ROOT.iterdir() if p.is_dir()]
    if not args.voice and any(locale not in DEFAULT_VOICES for locale in selected_locales):
        p.error("No default TTS voice for a selected locale; supply --voice explicitly")

    if not AZURE_SPEECH_KEY:
        print("ERROR: AZURE_SPEECH_KEY not set.")
        print("       Copy .env.example to .env and fill in your Azure Speech credentials.")
        sys.exit(1)

    if args.locale and args.dataset:
        voice = args.voice or DEFAULT_VOICES.get(args.locale, "en-US-JennyNeural")
        process_dataset(args.locale, args.dataset, voice, args.force, args.max_per_dataset, args.seed)

    elif args.locale:
        locale_dir = DATA_ROOT / args.locale
        if not locale_dir.exists():
            print(f"ERROR: {locale_dir} does not exist")
            sys.exit(1)
        voice = args.voice or DEFAULT_VOICES.get(args.locale, "en-US-JennyNeural")
        for ds_dir in sorted(locale_dir.iterdir()):
            if ds_dir.is_dir():
                process_dataset(args.locale, ds_dir.name, voice, args.force, args.max_per_dataset, args.seed)

    else:
        if not DATA_ROOT.exists():
            print(f"ERROR: {DATA_ROOT} does not exist")
            sys.exit(1)
        for locale_dir in sorted(DATA_ROOT.iterdir()):
            if not locale_dir.is_dir() or locale_dir.name.startswith("."):
                continue
            locale = locale_dir.name
            voice = args.voice or DEFAULT_VOICES.get(locale, "en-US-JennyNeural")
            for ds_dir in sorted(locale_dir.iterdir()):
                if ds_dir.is_dir():
                    process_dataset(locale, ds_dir.name, voice, args.force, args.max_per_dataset, args.seed)


if __name__ == "__main__":
    main()
