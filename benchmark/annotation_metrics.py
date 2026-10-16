"""Versioned speaker and language metrics; absent evidence is never a zero score."""
from __future__ import annotations

from collections import Counter

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from .metrics import normalize_text

SCORING_VERSION = "aligned-speaker-v1"


def speaker_label(value) -> str | None:
    if value is None or str(value).strip().lower() in ("", "unknown", "unassigned"):
        return None
    return str(value).strip()


def tokens(segments: list[dict]) -> list[dict]:
    return [{"text": word, "speaker": speaker_label(segment.get("speaker")),
             "language": segment.get("language")}
            for segment in segments for word in normalize_text(segment["text"]).split()]


def aligned_pairs(reference: list[dict], hypothesis: list[dict]) -> list[tuple[int, int]]:
    pairs = []
    for tag, a, b, c, d in Levenshtein.opcodes(
            [t["text"] for t in reference], [t["text"] for t in hypothesis]):
        if tag in ("equal", "replace"):
            pairs.extend(zip(range(a, b), range(c, d)))
    return pairs


def speaker_agreement(reference: list[dict], hypothesis: list[dict]) -> dict:
    ref, hyp = tokens(reference), tokens(hypothesis)
    pairs = aligned_pairs(ref, hyp)
    ref_ids = sorted({t["speaker"] for t in ref if t["speaker"] is not None})
    hyp_ids = sorted({t["speaker"] for t in hyp if t["speaker"] is not None})
    counts = Counter((ref[a]["speaker"], hyp[b]["speaker"]) for a, b in pairs)
    mapping = {}
    if ref_ids and hyp_ids:
        weights = np.array([[counts[r, h] for h in hyp_ids] for r in ref_ids])
        ri, hi = linear_sum_assignment(weights, maximize=True)
        mapping = {hyp_ids[h]: ref_ids[r] for r, h in zip(ri, hi) if weights[r, h] > 0}
    correct = sum(ref[a]["speaker"] is not None and hyp[b]["speaker"] is not None
                  and mapping.get(hyp[b]["speaker"]) == ref[a]["speaker"] for a, b in pairs)
    total = len(pairs)
    return {
        "speaker_errors": total - correct, "aligned_words": total,
        "rate": (total - correct) / total if total else None,
        "reference_words": len(ref), "hypothesis_words": len(hyp),
        "alignment_coverage": total / len(ref) if ref else None,
        "speaker_mapping": mapping, "reference_speakers": len(ref_ids),
        "hypothesis_speakers": len(hyp_ids),
        "unmatched_hypothesis_speakers": sorted(set(hyp_ids) - mapping.keys()),
    }


def describe_segments(segments: list[dict]) -> dict:
    words = tokens(segments)
    labels = [speaker_label(s.get("speaker")) for s in segments]
    languages = [s.get("language") for s in segments]
    return {
        "speakers": sorted({label for label in labels if label is not None}),
        "segments": len(segments),
        "speaker_switches": sum(a is not None and b is not None and a != b
                                for a, b in zip(labels, labels[1:])),
        "unknown_speaker_word_rate": (sum(t["speaker"] is None for t in words) / len(words)
                                      if words else None),
        "detected_languages": sorted({language for language in languages if language}),
        "language_switches": sum(bool(a and b and a != b) for a, b in zip(languages, languages[1:])),
    }


def language_scores(reference_language: str | None, reference: list[dict],
                    hypothesis: list[dict], strategy: str | None) -> dict:
    if strategy is None:
        return {"status": "not_applicable_monolingual", "clip_correct": None, "segment_accuracy": None}
    languages = sorted({s["language"] for s in hypothesis if s.get("language")})
    clip = None
    locale = None
    if reference_language and languages:
        clip = len({lang.split("-")[0].lower() for lang in languages}) == 1 and (
            languages[0].split("-")[0].lower() == reference_language.split("-")[0].lower())
        locale = len(languages) == 1 and languages[0].lower() == reference_language.lower()
    result = {"status": "available" if languages else "detected_language_unavailable",
              "reference_language": reference_language, "detected_languages": languages,
              "clip_correct": clip, "clip_locale_exact": locale,
              "segment_accuracy": None, "segment_coverage": None, "confusion_seconds": []}
    if strategy == "at-start":
        return result
    labeled = [s for s in reference if s.get("language")]
    if not labeled or any(s.get("start_s") is None or s.get("end_s") is None for s in labeled):
        result["segment_status"] = "reference_timing_or_language_unavailable"
        return result
    predicted = [s for s in hypothesis if s.get("language") and s.get("start_s") is not None
                 and s.get("end_s") is not None]
    for timeline in (labeled, predicted):
        ordered = sorted(timeline, key=lambda s: s["start_s"])
        if any(b["start_s"] < a["end_s"] for a, b in zip(ordered, ordered[1:])):
            result["segment_status"] = "overlapping_labels_not_scorable"
            return result
    total = sum(s["end_s"] - s["start_s"] for s in labeled)
    covered = correct = 0.0
    confusion = Counter()
    for ref in labeled:
        for hyp in predicted:
            overlap = max(0, min(ref["end_s"], hyp["end_s"]) - max(ref["start_s"], hyp["start_s"]))
            if overlap:
                a, b = ref["language"].split("-")[0].lower(), hyp["language"].split("-")[0].lower()
                covered += overlap
                correct += overlap if a == b else 0
                confusion[a, b] += overlap
    result.update(segment_accuracy=correct / total if total else None,
                  segment_coverage=covered / total if total else None,
                  reference_seconds=total, correct_seconds=correct, covered_seconds=covered,
                  segment_status="available" if predicted else "hypothesis_timing_or_language_unavailable",
                  confusion_seconds=[{"reference": a, "hypothesis": b, "seconds": seconds}
                                     for (a, b), seconds in sorted(confusion.items())])
    if not predicted:
        result["segment_accuracy"] = None
    return result
