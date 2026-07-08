---
mode: agent
description: Step 3 — Run the ASR benchmark.
---

Run the ASR benchmark to measure Word Error Rate (WER), Sentence Error Rate (SER),
and latency (LBL, UPL) across Azure Speech services.

## Quick smoke test (recommended first run)

Tests 5 samples × 2 services × 1 locale — completes in ~2 minutes:

```bash
python run_full.py --languages en-GB --max-per-dataset 5 --no-pace --services fast_default realtime
```

Check the output:
```bash
python -c "
from pathlib import Path
results = sorted(Path('results').glob('asr-bench_en-GB_*.csv'))
if results:
    import csv
    rows = list(csv.DictReader(open(results[-1], encoding='utf-8')))
    print(f'Latest CSV: {results[-1].name}')
    print(f'Rows: {len(rows)}')
    wers = [float(r[\"wer\"]) for r in rows if r[\"wer\"]]
    if wers:
        print(f'WER range: {min(wers):.3f} – {max(wers):.3f}  mean: {sum(wers)/len(wers):.3f}')
else:
    print('No CSV found yet.')
"
```

## Full benchmark options

**One locale, all services, 30 samples each:**
```bash
python run_full.py --languages en-GB
```

**All locales (takes 20-60 min depending on network):**
```bash
python run_full.py
```

**Specific services only:**
```bash
python run_full.py --languages en-GB --services fast_default fast_mai fast_mai_1_5 realtime
```

**Resume a partial run:**
```bash
python run_full.py --languages en-GB --resume results/asr-bench_en-GB_<timestamp>.csv
```

**Custom testdata or output directory:**
```bash
python run_full.py --testdata-dir /path/to/myaudio --output-dir /path/to/output --tag myproject
```

## Available CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--languages` | all locales | Locales to test (space-separated) |
| `--max-per-dataset` | 30 | Max samples per dataset |
| `--workers` | 4 | Parallel thread pool size |
| `--no-pace` | off | Skip real-time pacing (faster, unrealistic latency) |
| `--services` | all 6 | Services to benchmark |
| `--resume` | none | Path to partial CSV to resume |
| `--tag` | asr-bench | Prefix for output filenames |
| `--seed` | 42 | Random seed for reproducible sampling |
| `--testdata-dir` | testdata/ | Custom audio data root |
| `--output-dir` | results/ | Where to write CSV and reports |

## Output per run

After the run, `results/` will contain:
- `asr-bench_<locale>_<timestamp>.csv` — raw per-sample metrics
- `asr-bench_<locale>_<timestamp>_report.md` — aggregate WER/latency table
- `asr-bench_<locale>_<timestamp>_error_analysis.md` — best/worst/hallucinations
- `asr-bench_<locale>_<timestamp>_words.jsonl` — per-word timestamps (realtime)

## Next steps

- **Step 4** — `.github/prompts/04-loose-match.prompt.md` (normalize compound-word WER)
- **Step 5** — `.github/prompts/05-analyze.prompt.md` (re-generate error analysis if needed)
- **Step 6** — `.github/prompts/06-justify.prompt.md` (write human interpretation)
