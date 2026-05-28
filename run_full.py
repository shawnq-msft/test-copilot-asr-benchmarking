"""ASR Benchmark — entry point for running full benchmark suites."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmark.config import LOCALES, SERVICES, assert_credentials
from benchmark.datasets_loader import list_datasets, iter_samples
import benchmark.datasets_loader as _loader
from benchmark.report import build_report
from benchmark.runner import run_benchmark


def main():
    p = argparse.ArgumentParser(
        description="Run ASR accuracy and latency benchmarks against Azure Speech services.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test — 5 samples, 2 services, en-GB only
  python run_full.py --languages en-GB --max-per-dataset 5 --no-pace --services fast_default realtime

  # Full run, all locales
  python run_full.py

  # Resume a partial run
  python run_full.py --languages en-GB --resume results/asr-bench_en-GB_20260527_120000.csv

  # Custom testdata directory
  python run_full.py --testdata-dir /path/to/myaudio --tag myproject
""",
    )
    p.add_argument("--languages", nargs="*", default=list(LOCALES),
                   help="Locales to benchmark (default: all in LOCALES config). "
                        "Example: --languages en-GB de-DE")
    p.add_argument("--max-per-dataset", type=int, default=30,
                   help="Maximum samples per dataset (default: 30)")
    p.add_argument("--workers", type=int, default=4,
                   help="Thread pool size for parallel samples (default: 4)")
    p.add_argument("--no-pace", action="store_true",
                   help="Skip real-time pacing — runs faster but latency numbers are unrealistic")
    p.add_argument("--services", nargs="*", default=list(SERVICES),
                   help="Services to benchmark (default: all). "
                        "Example: --services fast_default fast_mai realtime")
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
    args = p.parse_args()

    assert_credentials()

    # Override DATA_ROOT if --testdata-dir was given
    if args.testdata_dir:
        _loader.DATA_ROOT = Path(args.testdata_dir)
        print(f"[config] testdata-dir: {_loader.DATA_ROOT}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.languages:
        print(f"\n{'='*60}")
        print(f"  Language: {lang}")
        print(f"{'='*60}\n")

        if args.resume:
            csv_path = Path(args.resume)
            ts = csv_path.stem.split("_", 2)[-1]
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            csv_path = output_dir / f"{args.tag}_{lang}_{ts}.csv"
        md_path = csv_path.with_name(csv_path.stem + "_report.md")

        datasets = list_datasets(lang)
        if not datasets:
            print(f"[skip] no datasets found for {lang} under {_loader.DATA_ROOT}")
            print(f"       Run: python scripts/generate_testset.py --locale {lang}")
            continue
        print(f"[load] datasets: {datasets}")

        samples = []
        for ds in datasets:
            try:
                ds_samples = iter_samples(ds, limit=args.max_per_dataset, seed=args.seed)
                print(f"  {ds}: {len(ds_samples)} samples")
                samples.extend(ds_samples)
            except FileNotFoundError as e:
                print(f"  [skip] {ds}: {e}")

        if not samples:
            print(f"[skip] no samples loaded for {lang}")
            continue

        active_services = tuple(s for s in args.services if s in SERVICES)
        print(f"[run] {len(samples)} samples × {len(active_services)} services, workers={args.workers}")
        run_benchmark(samples, csv_path, services=active_services,
                      workers=args.workers, pace=not args.no_pace, retries=2,
                      sequential=(args.workers <= 1))
        build_report(csv_path, md_path, tag=f"{args.tag}_{lang}")
        print(f"\nCSV:    {csv_path}")
        print(f"Report: {md_path}")

        from scripts.error_analysis import main as error_main
        ea_path = csv_path.with_name(csv_path.stem + "_error_analysis.md")
        error_main(csv_path, ea_path)
        print(f"Error analysis: {ea_path}")


if __name__ == "__main__":
    main()
