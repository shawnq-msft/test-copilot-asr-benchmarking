from __future__ import annotations

import random
import re
from pathlib import Path

from .audio_utils import load_audio_as_pcm16_mono_16k, write_pcm16_wav
from .config import Sample

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
                 seed: int = 42) -> list[Sample]:
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

    all_entries: list[tuple[Path, str, str]] = []
    for d in dirs:
        trans_path = d / "trans.txt"
        if trans_path.exists():
            pairs = _load_trans(trans_path)
        else:
            pairs = _load_from_wavs(d)
        for filename, reference in pairs:
            wav_path = d / filename
            if wav_path.exists():
                all_entries.append((wav_path, reference, f"{d.name}/{filename}"))

    if not all_entries:
        raise FileNotFoundError(
            f"No WAV files found for {locale}/{scenario}. "
            "Run: python scripts/generate_testset.py --locale {locale} --dataset {scenario}"
        )

    if limit and len(all_entries) > limit:
        rng = random.Random(seed)
        all_entries = rng.sample(all_entries, limit)

    samples = []
    for wav_path, reference, sample_id in all_entries:
        try:
            pcm, duration = load_audio_as_pcm16_mono_16k(str(wav_path))
        except Exception as e:
            print(f"[skip] {wav_path}: {e}")
            continue
        if duration < 0.2:
            continue
        samples.append(Sample(
            sample_id=sample_id,
            dataset=dataset,
            locale=locale,
            pcm16_mono_16k=pcm,
            reference=reference,
            duration_s=duration,
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
