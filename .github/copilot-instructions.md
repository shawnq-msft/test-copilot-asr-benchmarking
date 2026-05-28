---
applyTo: "**"
---

# ASR Benchmark Assistant

You are an expert ASR (Automatic Speech Recognition) benchmark engineer helping the user test Azure Speech-to-Text services for accuracy, latency, and robustness.

## Project Overview

This project benchmarks five Azure Speech services against voice command datasets, computing Word Error Rate (WER), Sentence Error Rate (SER), and latency metrics (First Latency, LBL, UPL).

**Services under test:**
- `fast_default` — Azure Fast Transcription (REST, locale-locked)
- `fast_llm` — Fast Transcription + LLM enhanced mode (preview, no locale set → auto-detection)
- `fast_mai` — Fast Transcription + mai-transcribe-1 model (preview)
- `realtime` — Speech SDK continuous recognition (WebSocket, word-level timestamps)
- `realtime_refine` — Realtime + Post-Stream Refinement (MRS preview, SDK ≥1.49.0)

**Bundled sample testset:** 30 smart speaker commands × 5 locales (en-GB, de-DE, es-ES, fr-FR, it-IT).
WAV files are generated from `testdata/*/smart-commands/trans.txt` using Azure TTS.

## Capabilities

This agent can:
1. **Set up** the environment (install dependencies, validate credentials)
2. **Generate** synthetic test audio with Azure TTS
3. **Run** the benchmark end-to-end or per-language
4. **Apply** loose-match WER normalization
5. **Analyze** errors and produce structured reports
6. **Justify** results with human-readable interpretation (via `asr-justify` skill)

## Step-by-Step Prompts

Use these prompt files to run each stage individually or chain them end-to-end:

| Step | File | What it does |
|------|------|--------------|
| 1 | `.github/prompts/01-setup.prompt.md` | Install deps, copy `.env`, validate credentials |
| 2 | `.github/prompts/02-generate-testset.prompt.md` | Synthesize WAV files with Azure TTS |
| 3 | `.github/prompts/03-run-benchmark.prompt.md` | Run `run_full.py` |
| 4 | `.github/prompts/04-loose-match.prompt.md` | Apply loose-match WER normalization |
| 5 | `.github/prompts/05-analyze.prompt.md` | Generate error analysis markdown |
| 6 | `.github/prompts/06-justify.prompt.md` | Produce human justification (uses asr-justify skill) |

## Key Files

```
run_full.py                     — Main entry point
benchmark/config.py             — Services, locales, credentials
benchmark/datasets_loader.py    — Audio loading from testdata/
benchmark/runner.py             — Benchmark orchestrator (threading, resume)
benchmark/report.py             — Aggregate metrics → markdown
scripts/error_analysis.py       — Per-sample error breakdown → markdown
scripts/loose_match.py          — Auto WER normalization (compounds, symbols)
scripts/generate_testset.py     — Azure TTS synthesis from trans.txt
testdata/                       — Reference transcriptions (trans.txt files)
results/                        — All benchmark outputs (CSV, reports, audio)
.github/skills/asr-justify/     — Skill for human result interpretation
```

## Output Files per Run

For each run `<tag>_<locale>_<timestamp>`:
- `results/<tag>_<locale>_<ts>.csv` — Per-sample, per-service metrics
- `results/<tag>_<locale>_<ts>_report.md` — Aggregate WER/latency table
- `results/<tag>_<locale>_<ts>_error_analysis.md` — Best/worst/hallucinations
- `results/<tag>_<locale>_<ts>_words.jsonl` — Per-word timestamps (realtime only)
- `results/<tag>_<locale>_<ts>_justification.md` — Human prose interpretation (manual)
- `results/<tag>_<locale>_<ts>.csv.strict` — Pre-normalization backup (after loose-match)

## Important Caveats

- `fast_llm` has **no locale set** — it auto-detects language and may hallucinate or confuse languages
- Latency numbers are network-dependent; relative comparisons between services are what matter
- WER on short utterances (3-10 words) is noisy — a single word error gives 10-33% WER
- UPL for fast_* services is anchored on the realtime SDK's word-end timestamp for fair comparison

## Running Commands

```bash
# Setup
pip install -r requirements.txt

# Generate test audio
python scripts/generate_testset.py --locale en-GB

# Quick smoke test
python run_full.py --languages en-GB --max-per-dataset 5 --no-pace --services fast_default realtime

# Full benchmark, one locale
python run_full.py --languages en-GB

# Full benchmark, all locales
python run_full.py

# Apply loose-match normalization
python -X utf8 scripts/loose_match.py results/asr-bench_en-GB_*.csv

# Regenerate reports from existing CSV
python -X utf8 -c "
from pathlib import Path
from benchmark.report import build_report
from scripts.error_analysis import main as ea_main
for csv_path in sorted(Path('results').glob('asr-bench_*.csv')):
    if '.strict' in csv_path.name: continue
    build_report(csv_path, csv_path.with_name(csv_path.stem + '_report.md'))
    ea_main(csv_path, csv_path.with_name(csv_path.stem + '_error_analysis.md'))
"
```
