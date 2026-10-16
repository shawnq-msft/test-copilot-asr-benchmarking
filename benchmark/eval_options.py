"""Explicit request modes. Capabilities are conservative, not automatic fallbacks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class EvaluationOptions:
    language_mode: str = "monolingual"
    lid_strategy: str | None = None
    candidates: tuple[str, ...] = ()
    diarization: bool = False
    max_speakers: int | None = None
    score_wder: bool = False

    def to_dict(self) -> dict:
        return {**asdict(self), "candidates": list(self.candidates)}

    @property
    def variant_id(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


# Conservative verified configuration surface, not a claim of exhaustive service support.
CAPABILITIES = {
    "fast_default": ({"monolingual", "at-start", "native"}, True),
    "fast_llm": ({"monolingual", "native"}, True),
    "fast_mai_1_5": ({"monolingual"}, False),
    "fast_mai_2": ({"monolingual", "native"}, True),
    "realtime": ({"monolingual", "at-start", "continuous"}, False),
    "realtime_refine": ({"monolingual"}, False),
    "realtime_diarization": ({"monolingual"}, True),
    "realtime_mai_2": ({"native"}, False),
    "realtime_mai_2_api": ({"monolingual", "native"}, False),
    "whisper_v3": ({"monolingual"}, False),
}
VERIFIED_LOCALES = frozenset(("en-US", "en-GB", "de-DE", "es-ES", "fr-FR", "it-IT", "zh-CN"))


def validate_options(options: EvaluationOptions, services, locales, *, phrases=()) -> None:
    if options.language_mode not in ("monolingual", "multilingual"):
        raise ValueError("language_mode must be monolingual or multilingual")
    if options.language_mode == "monolingual":
        if options.lid_strategy or options.candidates:
            raise ValueError("LID strategy/candidates require --language-mode multilingual")
        strategy = "monolingual"
        checked_locales = list(locales)
    else:
        strategy = options.lid_strategy
        if strategy not in ("at-start", "continuous", "native"):
            raise ValueError("Multilingual mode requires an explicit --lid-strategy")
        checked_locales = list(options.candidates)
        if strategy == "native":
            if options.candidates:
                raise ValueError("Native multilingual mode does not accept candidate filtering")
        elif not 2 <= len(options.candidates) <= (4 if strategy == "at-start" else 10):
            raise ValueError("At-start LID requires 2-4 candidates; continuous LID requires 2-10")
        bases = [locale.split("-")[0] for locale in options.candidates]
        if len(set(bases)) != len(bases):
            raise ValueError("LID candidates must have distinct base languages")
    if not services:
        raise ValueError("Select at least one service")
    if any(locale not in VERIFIED_LOCALES for locale in checked_locales):
        raise ValueError("Locale is outside the verified evaluation set: "
                         + ", ".join(sorted(set(checked_locales) - VERIFIED_LOCALES)))
    if options.max_speakers is not None:
        if not options.diarization or not 2 <= options.max_speakers <= 35:
            raise ValueError("--max-speakers requires diarization and must be between 2 and 35")
    if options.score_wder and not options.diarization:
        raise ValueError("--score-wder requires --diarization")
    for service in services:
        if service not in CAPABILITIES:
            raise ValueError(f"Unknown service: {service}")
        strategies, speakers = CAPABILITIES[service]
        if strategy not in strategies:
            raise ValueError(f"{service}: {strategy} is unsupported/unverified; "
                             f"supported configurations: {', '.join(sorted(strategies))}")
        if options.diarization and not speakers:
            raise ValueError(f"{service}: diarization is unsupported/unverified")
        if service == "realtime_diarization":
            if not options.diarization:
                raise ValueError("realtime_diarization requires --diarization")
            if options.max_speakers is not None or phrases:
                raise ValueError("SDK diarization does not expose verified max-speakers/phrase settings")


def add_evaluation_arguments(parser) -> None:
    parser.add_argument("--manifest", type=str, help="Version-1 JSONL user audio/annotation manifest")
    parser.add_argument("--language-mode", choices=("monolingual", "multilingual"), default="monolingual")
    parser.add_argument("--lid-strategy", choices=("at-start", "continuous", "native"))
    parser.add_argument("--lid-candidates", nargs="+", default=[])
    parser.add_argument("--diarization", action="store_true")
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--score-wder", action="store_true",
                        help="Require user-supplied speaker references and score aligned-word wDER")


def options_from_args(args) -> EvaluationOptions:
    return EvaluationOptions(args.language_mode, args.lid_strategy, tuple(args.lid_candidates),
                             args.diarization, args.max_speakers, args.score_wder)
