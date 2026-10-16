from __future__ import annotations

import random
import re
import hashlib
import json
import math
from pathlib import Path

from .audio_utils import load_audio_as_pcm16_mono_16k, write_pcm16_wav
from .config import Sample
from .eval_options import EvaluationOptions, validate_options
from .metrics import normalize_text


def load_manifest(path: Path) -> list[Sample]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    samples = []
    identities = set()
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict) or entry.get("schema_version") != 1:
            raise ValueError(f"Manifest line {number}: schema_version must be 1")
        sample_id = entry.get("sample_id")
        dataset = entry.get("dataset", "user-audio")
        if not all(isinstance(value, str) and value.strip() for value in (sample_id, dataset, entry.get("audio"))):
            raise ValueError(f"Manifest line {number}: audio/sample_id/dataset must be nonempty strings")
        identity = (dataset, sample_id)
        if identity in identities:
            raise ValueError(f"Duplicate manifest identity: {identity}")
        identities.add(identity)
        audio_path = path.parent / entry["audio"]
        try:
            pcm, duration = load_audio_as_pcm16_mono_16k(str(audio_path))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"{sample_id}: cannot decode manifest audio {audio_path}: {exc}") from exc
        if duration < .2:
            raise ValueError(f"{sample_id}: recording must be at least 0.2 seconds")
        scenario = entry.get("scenario", "single-language")
        if scenario not in ("single-language", "language-switching"):
            raise ValueError(f"{sample_id}: invalid scenario")
        locale = entry.get("requested_language", "")
        if not isinstance(locale, str):
            raise ValueError(f"{sample_id}: requested_language must be a string")
        reference = entry.get("reference")
        text = ""
        segments = None
        language = None
        available = False
        if reference is not None:
            if not isinstance(reference, dict):
                raise ValueError(f"{sample_id}: reference must be an object")
            language = reference.get("language")
            if language is not None and (not isinstance(language, str) or not language.strip()):
                raise ValueError(f"{sample_id}: reference language must be a nonempty string")
            segments = reference.get("segments")
            if segments is not None:
                if not isinstance(segments, list) or not segments:
                    raise ValueError(f"{sample_id}: reference segments must be a nonempty list")
                last_start = -1.0
                for segment in segments:
                    if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
                        raise ValueError(f"{sample_id}: segment text must be a string")
                    for field in ("speaker", "language"):
                        value = segment.get(field)
                        if field == "speaker" and isinstance(value, int) and not isinstance(value, bool):
                            segment[field] = str(value)
                            continue
                        if value is not None and (not isinstance(value, str) or not value.strip()):
                            raise ValueError(f"{sample_id}: {field} must be a nonempty string")
                    start, end = segment.get("start_s"), segment.get("end_s")
                    if start is not None or end is not None:
                        if (not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                                    and math.isfinite(v) for v in (start, end))
                                or not 0 <= start < end <= duration + .001 or start < last_start):
                            raise ValueError(f"{sample_id}: invalid/unordered segment timestamps")
                        last_start = start
                text = " ".join(segment["text"] for segment in segments)
                available = True
            if "text" in reference:
                if not isinstance(reference["text"], str):
                    raise ValueError(f"{sample_id}: reference text must be a string")
                if segments and normalize_text(text) != normalize_text(reference["text"]):
                    raise ValueError(f"{sample_id}: reference text disagrees with segments")
                text = reference["text"]
                available = True
        samples.append(Sample(sample_id, dataset, locale, pcm, text, duration,
                              reference_available=available, reference_segments=segments,
                              reference_language=language, scenario=scenario, manifest_hash=digest,
                              manifest_path=str(path.resolve()), audio_path=str(audio_path.resolve())))
    if not samples:
        raise ValueError("Manifest contains no samples")
    return samples


def prepare_evaluation(samples: list[Sample], services, options: EvaluationOptions, *, phrases=()) -> None:
    if not samples:
        raise ValueError("Evaluation selection contains no samples")
    validate_options(options, services, [s.locale for s in samples], phrases=phrases)
    if options.lid_strategy == "continuous" and "realtime" in services:
        from .config import AZURE_SPEECH_ENDPOINT
        if not AZURE_SPEECH_ENDPOINT.endswith("/speech/universal/v2"):
            raise ValueError("Continuous SDK LID requires an explicitly configured Speech universal/v2 endpoint")
    for sample in samples:
        if options.score_wder:
            segments = sample.reference_segments
            if (not sample.reference_available or not segments
                    or not any(normalize_text(s["text"]) for s in segments)
                    or any(not s.get("speaker") or s["speaker"].strip().lower() in ("unknown", "unassigned")
                           for s in segments if normalize_text(s["text"]))):
                raise ValueError(f"{sample.sample_id}: wDER requires user-provided speaker-labeled text")
            last_end = 0.0
            for segment in segments:
                if segment.get("start_s") is not None:
                    if segment["start_s"] < last_end:
                        raise ValueError(f"{sample.sample_id}: scored overlap is not supported")
                    last_end = segment["end_s"]
        sample.evaluation_options = options

# Root of the test data directory. Override at module level before calling
# iter_samples() or list_datasets() to use a different location.
# e.g.:  import benchmark.datasets_loader as loader; loader.DATA_ROOT = Path("mydata")
DATA_ROOT = Path(__file__).resolve().parent.parent / "testdata"

_NUMBERED_PREFIX = re.compile(r"^\d+\.")


def _scenario_dirs(locale: str, scenario: str) -> list[Path]:
    """Find all dataset directories for a given locale + scenario name.

    Matches directories whose name equals the scenario string exactly, or
    ends with '-<scenario>' (for compatibility with legacy naming like
    '1l_de-DE_female-DT1').
    """
    locale_dir = DATA_ROOT / locale
    if not locale_dir.exists():
        return []
    dirs = []
    for d in sorted(locale_dir.iterdir()):
        if d.is_dir() and (d.name == scenario or d.name.endswith(f"-{scenario}")):
            dirs.append(d)
    return dirs


def _load_trans(trans_path: Path) -> list[tuple[str, str]]:
    """Load (filename, reference) pairs from a trans.txt file.

    Supports two formats:
      - Tab-separated:  filename.wav<TAB>Reference text
      - Plain text:     Reference text  (filename derived via _ref_to_filename)
    Lines starting with '#' are treated as comments and skipped.
    """
    entries = []
    text = trans_path.read_text(encoding="utf-8-sig")
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            entries.append((parts[0], parts[1]))
        else:
            fn = _ref_to_filename(line) + ".wav"
            entries.append((fn, line))
    return entries


def _ref_to_filename(text: str) -> str:
    """Convert a reference text to a safe WAV filename stem."""
    s = text.lower()
    s = re.sub(r"['''\u2019]", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s[:70].rstrip("_")


def _ref_from_filename(name: str) -> str:
    """Derive reference text from a wav filename: strip .wav, strip optional NNN. prefix."""
    stem = name.rsplit(".", 1)[0] if name.lower().endswith(".wav") else name
    stem = _NUMBERED_PREFIX.sub("", stem)
    return stem.replace("_", " ")


def _load_from_wavs(directory: Path) -> list[tuple[str, str]]:
    """Discover samples by deriving reference text from WAV filenames."""
    entries = []
    for wav in sorted(directory.glob("*.wav")):
        ref = _ref_from_filename(wav.name)
        if ref:
            entries.append((wav.name, ref))
    return entries


def iter_samples(dataset: str, limit: int | None = 30,
                 seed: int = 42, *, strict: bool = False) -> list[Sample]:
    """Load audio samples for a dataset identifier like 'en-GB_smart-commands'.

    Args:
        dataset: '<locale>_<scenario>' string, e.g. 'en-GB_smart-commands'.
        limit: Maximum number of samples to return (random subset if exceeded).
        seed: Random seed for reproducible subsampling.

    Returns:
        List of Sample objects with loaded PCM audio.
    """
    parts = dataset.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid dataset name: {dataset!r}. "
            "Expected format: '<locale>_<scenario>' (e.g. 'en-GB_smart-commands')"
        )
    locale, scenario = parts

    dirs = _scenario_dirs(locale, scenario)
    if not dirs:
        raise FileNotFoundError(
            f"No directories found for {locale}/{scenario} under {DATA_ROOT}. "
            "Run: python scripts/generate_testset.py --locale {locale}"
        )

    all_entries: list[tuple[Path, str, str, bool]] = []
    for d in dirs:
        trans_path = d / "trans.txt"
        if trans_path.exists():
            pairs = _load_trans(trans_path)
        else:
            pairs = _load_from_wavs(d)
        for filename, reference in pairs:
            wav_path = d / filename
            if wav_path.exists() or strict:
                all_entries.append((wav_path, reference, f"{d.name}/{filename}", trans_path.exists()))

    if not all_entries:
        raise FileNotFoundError(
            f"No WAV files found for {locale}/{scenario}. "
            "Run: python scripts/generate_testset.py --locale {locale} --dataset {scenario}"
        )

    if limit and len(all_entries) > limit:
        rng = random.Random(seed)
        all_entries = rng.sample(all_entries, limit)

    samples = []
    for wav_path, reference, sample_id, has_reference in all_entries:
        if strict and not wav_path.is_file():
            raise FileNotFoundError(f"Missing referenced audio: {wav_path}")
        try:
            pcm, duration = load_audio_as_pcm16_mono_16k(str(wav_path))
        except Exception as e:
            if strict:
                raise ValueError(f"Cannot decode {wav_path}: {e}") from e
            print(f"[skip] {wav_path}: {e}")
            continue
        if duration < 0.2:
            if strict:
                raise ValueError(f"Audio shorter than 0.2 seconds: {wav_path}")
            continue
        samples.append(Sample(
            sample_id=sample_id,
            dataset=dataset,
            locale=locale,
            pcm16_mono_16k=pcm,
            reference=reference,
            duration_s=duration,
            reference_available=has_reference,
            reference_language=locale if has_reference else None,
            audio_path=str(wav_path.resolve()),
        ))
    return samples


def list_datasets(locale: str | None = None) -> list[str]:
    """List all available dataset identifiers under DATA_ROOT.

    Args:
        locale: If given, only list datasets for that locale.

    Returns:
        List of '<locale>_<dataset>' strings.
    """
    if not DATA_ROOT.exists():
        return []
    if locale:
        locales = [locale]
    else:
        locales = [d.name for d in sorted(DATA_ROOT.iterdir()) if d.is_dir()]
    datasets = []
    for loc in locales:
        loc_dir = DATA_ROOT / loc
        if not loc_dir.exists():
            continue
        for d in sorted(loc_dir.iterdir()):
            if d.is_dir():
                datasets.append(f"{loc}_{d.name}")
    return datasets
