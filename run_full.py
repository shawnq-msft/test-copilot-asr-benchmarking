"""ASR Benchmark — entry point for running full benchmark suites."""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

from benchmark.config import DEFAULT_SERVICES, LOCALES, SERVICES, assert_credentials
import benchmark.datasets_loader as _loader
from benchmark.report import build_report
from benchmark.runner import run_benchmark
from benchmark.eval_options import add_evaluation_arguments, options_from_args
from scripts.validate_benchmark import (check_selection, check_resume, collect_samples,
                                        make_metadata, metadata_path, validate_artifacts)


def main():
    p = argparse.ArgumentParser(
        description="Run ASR accuracy and latency benchmarks against Azure Speech services.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test — 5 samples, 2 services, en-GB only
  python run_full.py --languages en-GB --max-per-dataset 5 --no-pace --services fast_default realtime

  # Monolingual run, all locales, compatible service subset
  python run_full.py --services fast_default realtime

  # Resume a partial run
  python run_full.py --languages en-GB --resume results/asr-bench_en-GB_20260527_120000.csv

  # Custom testdata directory
  python run_full.py --testdata-dir /path/to/myaudio --tag myproject
""",
    )
    p.add_argument("--languages", nargs="+", default=list(LOCALES),
                   help="Locales to benchmark (default: all in LOCALES config). "
                        "Example: --languages en-GB de-DE")
    p.add_argument("--max-per-dataset", type=int, default=30,
                   help="Maximum samples per dataset (default: 30)")
    p.add_argument("--workers", type=int, default=4,
                   help="Thread pool size for parallel samples (default: 4)")
    p.add_argument("--no-pace", action="store_true",
                   help="Skip real-time pacing — runs faster but latency numbers are unrealistic")
    p.add_argument("--chunk-ms", type=int, default=100,
                   help="Streaming audio chunk duration in ms (default: 100; REST unchanged)")
    p.add_argument("--services", nargs="+", choices=SERVICES, default=list(DEFAULT_SERVICES),
                   help="Services to benchmark; choose a mode-compatible subset. Automatic-only realtime_mai_2 "
                        "requires explicit multilingual/native mode. "
                        "Example: --services fast_default fast_mai_2 realtime realtime_mai_2")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to an existing CSV to resume from (skips already-completed rows)")
    p.add_argument("--tag", type=str, default="asr-bench",
                   help="Prefix for output filenames (default: asr-bench)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible dataset subsampling (default: 42)")
    p.add_argument("--testdata-dir", type=str, default=None,
                   help="Path to testdata directory (default: testdata/ next to this script). "
                        "Override to use your own audio files.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for result files (default: results/ next to this script)")
    p.add_argument("--phrase-list", nargs="*", default=[],
                   help="Runtime phrase-list hints applied to supported Speech endpoints")
    p.add_argument("--phrase-weight", type=float, default=1.0,
                   help="Phrase-list biasing weight from 0.0 to 2.0 (default: 1.0)")
    add_evaluation_arguments(p)
    args = p.parse_args()
    options = options_from_args(args)

    if not 0.0 <= args.phrase_weight <= 2.0:
        p.error("--phrase-weight must be between 0.0 and 2.0")
    if args.chunk_ms <= 0:
        p.error("--chunk-ms must be positive")

    try:
        warnings = check_selection(args.services, args.languages, args.max_per_dataset,
                                   args.workers, args.phrase_list)
        assert_credentials(tuple(args.services))
    except ValueError as exc:
        p.error(str(exc))
    for warning in warnings:
        print(f"[warning] {warning}")
    if args.resume and not args.manifest and len(args.languages) != 1:
        p.error("--resume requires exactly one --languages locale")

    # Override DATA_ROOT if --testdata-dir was given
    if args.testdata_dir:
        _loader.DATA_ROOT = Path(args.testdata_dir)
        print(f"[config] testdata-dir: {_loader.DATA_ROOT}")

    try:
        if args.manifest and any(arg.split("=")[0] in ("--languages", "--testdata-dir", "--max-per-dataset")
                                 for arg in sys.argv[1:]):
            raise ValueError("--manifest is an exact selection; do not combine with locale/root/limit flags")
        groups = ({"manifest": _loader.load_manifest(Path(args.manifest))} if args.manifest else
                  {lang: collect_samples([lang], args.max_per_dataset, args.seed) for lang in args.languages})
        for samples in groups.values():
            _loader.prepare_evaluation(samples, args.services, options, phrases=args.phrase_list)
    except (ValueError, OSError) as exc:
        p.error(str(exc))

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang, samples in groups.items():
        print(f"\n{'='*60}")
        print(f"  Language: {lang}")
        print(f"{'='*60}\n")

        if args.resume:
            csv_path = Path(args.resume)
            ts = csv_path.stem.split("_", 2)[-1]
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            csv_path = output_dir / f"{args.tag}_{lang}_{options.variant_id}_{ts}.csv"
        md_path = csv_path.with_name(csv_path.stem + "_report.md")

        try:
            metadata = make_metadata(samples, args.services, pace=not args.no_pace,
                                     seed=args.seed, limit=0 if args.manifest else args.max_per_dataset,
                                     workers=args.workers, phrases=args.phrase_list,
                                     phrase_weight=args.phrase_weight, chunk_ms=args.chunk_ms,
                                     evaluation_options=options)
            if args.resume:
                check_resume(csv_path, metadata)
            elif csv_path.exists() or metadata_path(csv_path).exists():
                raise ValueError(f"Output already exists: {csv_path}; choose a new --tag")
            else:
                metadata_path(csv_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except (ValueError, OSError) as exc:
            p.error(str(exc))

        active_services = tuple(args.services)
        print(f"[run] {len(samples)} samples × {len(active_services)} services, workers={args.workers}")
        run_benchmark(samples, csv_path, services=active_services,
                      workers=args.workers, pace=not args.no_pace, retries=2,
                      sequential=(args.workers <= 1),
                      phrases=tuple(args.phrase_list),
                      phrase_weight=args.phrase_weight, chunk_ms=args.chunk_ms,
                      evaluation_options=options)
        validation = validate_artifacts(csv_path, require_reports=False)
        validation_path = csv_path.with_name(csv_path.stem + "_validation.json")
        validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        if not validation["ok"]:
            print(json.dumps(validation, indent=2))
            raise SystemExit(f"Benchmark validation failed; partial results preserved at {csv_path}")
        build_report(csv_path, md_path, tag=f"{args.tag}_{lang}")
        print(f"\nCSV:    {csv_path}")
        print(f"Report: {md_path}")

        from scripts.error_analysis import main as error_main
        ea_path = csv_path.with_name(csv_path.stem + "_error_analysis.md")
        error_main(csv_path, ea_path)
        print(f"Error analysis: {ea_path}")
        validation = validate_artifacts(csv_path)
        validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        if not validation["ok"]:
            raise SystemExit(f"Artifact validation failed; see {validation_path}")
        print(f"Validation: {validation_path}")


if __name__ == "__main__":
    main()
