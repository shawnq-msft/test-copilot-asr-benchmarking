# ASR Benchmark — Azure Speech Services

A reusable benchmarking framework for Azure Speech-to-Text services. Measures **Word Error Rate (WER)**, **Sentence Error Rate (SER)**, and latency (**LBL**, **UPL**) across five services and multiple languages.

Comes with a bundled 30-command smart speaker testset in five locales (en-GB, de-DE, es-ES, fr-FR, it-IT). Generate audio with one command, run the benchmark, and get detailed markdown reports.

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd test-copilot-asr-benchmarking
pip install -r requirements.txt
```

### 2. Set up credentials

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Edit `.env` and fill in your Azure Speech resource details:

```ini
AZURE_SPEECH_KEY=your_key_here
AZURE_SPEECH_REGION=eastus
AZURE_SPEECH_ENDPOINT=https://eastus.api.cognitive.microsoft.com
```

> For `fast_llm` and `fast_mai` (LLM/MAI preview services), you may need a separate Speech resource. Add `LLM_AZURE_SPEECH_KEY`, `LLM_AZURE_SPEECH_REGION`, `LLM_AZURE_SPEECH_ENDPOINT` if so.

### 3. Generate test audio

The testset reference texts are included. Generate the WAV files using Azure TTS:

```bash
# All 5 locales (takes ~3 minutes)
python scripts/generate_testset.py

# Just English to get started
python scripts/generate_testset.py --locale en-GB
```

### 4. Run a quick smoke test

Test 5 samples × 2 services × 1 locale to verify everything works (~1-2 minutes):

```bash
python run_full.py --languages en-GB --max-per-dataset 5 --no-pace --services fast_default realtime
```

### 5. View the results

Open the generated report:

```
results/asr-bench_en-GB_<timestamp>_report.md
results/asr-bench_en-GB_<timestamp>_error_analysis.md
```

---

## Services Under Test

| Service | Description |
|---------|-------------|
| `fast_default` | Azure Fast Transcription (REST, api-version 2024-11-15). Locale-locked. |
| `fast_llm` | Fast Transcription + LLM enhanced mode (preview). **No locale set — auto-detects language.** May hallucinate or confuse languages. |
| `fast_mai` | Fast Transcription + mai-transcribe-1 model (preview). Requires LLM Speech preview enabled. |
| `realtime` | Speech SDK continuous recognition (WebSocket, word-level timestamps). |
| `realtime_refine` | Realtime + Post-Stream Refinement (MRS preview, requires SDK ≥ 1.49.0). |

---

## Bundled Sample Testset

The `testdata/` directory contains 30 smart speaker commands per locale:

| Locale | Language | Example commands |
|--------|----------|-----------------|
| en-GB | English (UK) | "Set a timer for five minutes", "Turn off the living room lights" |
| de-DE | German | "Stell einen Timer für fünf Minuten", "Garagentor öffnen" |
| es-ES | Spanish | "Pon un temporizador de cinco minutos", "Buenas noches" |
| fr-FR | French | "Mets un minuteur de cinq minutes", "Bonne nuit" |
| it-IT | Italian | "Imposta un timer di cinque minuti", "Buona notte" |

WAV files are generated from the `trans.txt` reference files using Azure TTS
(voice: `en-GB-SoniaNeural`, `de-DE-KatjaNeural`, etc.).

---

## Running Benchmarks

### Full run

```bash
# One locale
python run_full.py --languages en-GB

# All locales (takes 20–60 min depending on network)
python run_full.py

# Specific services
python run_full.py --languages en-GB --services fast_default fast_mai realtime
```

### Resume a partial run

If a run was interrupted, resume it without re-testing completed samples:

```bash
python run_full.py --languages en-GB --resume results/asr-bench_en-GB_<timestamp>.csv
```

### Custom data

Bring your own audio files in the same structure:

```bash
python run_full.py --testdata-dir /path/to/myaudio --output-dir /path/to/output --tag myproject
```

Your audio directory should contain `<locale>/<dataset-name>/` folders with WAV files and an optional `trans.txt`.

### All CLI options

```
--languages          Locales to test (default: all in config)
--max-per-dataset    Samples per dataset (default: 30)
--workers            Thread pool size (default: 4)
--no-pace            Skip real-time pacing (faster, but latency unrealistic)
--services           Services to benchmark (default: all 5)
--resume             Path to partial CSV to resume from
--tag                Output filename prefix (default: asr-bench)
--seed               Random seed for reproducible sampling (default: 42)
--testdata-dir       Custom audio data root (default: testdata/)
--output-dir         Where to write results (default: results/)
```

---

## Output Files

For each run `<tag>_<locale>_<timestamp>`:

| File | Description |
|------|-------------|
| `results/<run>.csv` | Per-sample, per-service metrics (WER, SER, INS/DEL/SUB, latency) |
| `results/<run>_report.md` | Aggregate WER/latency table with endpoint and dataset details |
| `results/<run>_error_analysis.md` | Best/worst/hallucinations with `[▶]` audio playback links |
| `results/<run>_words.jsonl` | Per-word timestamps for realtime samples (for latency audit) |
| `results/<run>_justification.md` | Human prose interpretation (written manually with the asr-justify skill) |
| `results/<run>.csv.strict` | Pre-normalization backup (created after running `loose_match.py`) |

---

## Latency Metrics

Three latency metrics are computed for each sample:

- **First Latency** — how quickly the first partial result appears after the user *starts* speaking (realtime only; REST has no partial results)
- **LBL** (Last-final Beyond Last-chunk) — server flush time relative to the last audio chunk sent; can be negative when the server runs ahead of audio I/O
- **UPL** (User-Perceived Latency) — how late the final transcript arrives after the user *stops* speaking; anchored to the realtime SDK's word-end timestamp for fair cross-service comparison

---

## WER Normalization

WER is computed after NFKC + lowercase + punctuation-strip normalization. An additional
**loose-match** pass handles systematic false positives:

```bash
# Apply loose matching after a benchmark run
python -X utf8 scripts/loose_match.py results/asr-bench_en-GB_<timestamp>.csv

# Regenerate reports after normalization
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

Loose matching catches: compound-word splits (`stummschalten` ↔ `stumm schalten`),
alphanumeric spacing (`2D` ↔ `2 D`), and degree/percent symbol variants.

---

## Using with GitHub Copilot

This project is configured as a Copilot Agent (`.github/copilot-instructions.md`).
Copilot can run each step for you or guide you through the full pipeline.

### Step-by-step prompts

Use the prompt files in `.github/prompts/` — invoke them from the Copilot Chat panel:

| Step | Prompt file | What it does |
|------|-------------|--------------|
| 1 | `01-setup.prompt.md` | Install dependencies, configure `.env`, validate credentials |
| 2 | `02-generate-testset.prompt.md` | Synthesize WAV files with Azure TTS |
| 3 | `03-run-benchmark.prompt.md` | Run `run_full.py` with your chosen options |
| 4 | `04-loose-match.prompt.md` | Normalize WER for compound words and symbol variants |
| 5 | `05-analyze.prompt.md` | Regenerate error analysis from existing CSV |
| 6 | `06-justify.prompt.md` | Write human-readable interpretation (uses `asr-justify` skill) |

### asr-justify skill

The `.github/skills/asr-justify/SKILL.md` defines a 9-step procedure for writing
a prose interpretation of benchmark results — covering where WER is misleading,
fast_llm hallucinations, genuine service disagreements, latency QA, and recommendations.

---

## Adding Your Own Test Data

1. Create a directory: `testdata/<locale>/<dataset-name>/`
2. Add `trans.txt` (one reference command per line, or tab-separated `filename.wav<TAB>reference`)
3. Either:
   - Place existing WAV files in the same directory
   - Run `python scripts/generate_testset.py --locale <locale>` to synthesize them
4. Run: `python run_full.py --languages <locale>`

---

## Project Structure

```
.
├── .env.example                  # Credential template
├── .github/
│   ├── copilot-instructions.md   # Copilot agent persona and capabilities
│   ├── prompts/                  # Step-by-step runnable prompt files
│   │   ├── 01-setup.prompt.md
│   │   ├── 02-generate-testset.prompt.md
│   │   ├── 03-run-benchmark.prompt.md
│   │   ├── 04-loose-match.prompt.md
│   │   ├── 05-analyze.prompt.md
│   │   └── 06-justify.prompt.md
│   └── skills/asr-justify/
│       └── SKILL.md              # Human result interpretation skill
├── benchmark/
│   ├── config.py                 # Credentials, service list, data classes
│   ├── runner.py                 # Benchmark orchestrator (threading, resume)
│   ├── report.py                 # Aggregate metrics → markdown report
│   ├── metrics.py                # WER/SER computation
│   ├── datasets_loader.py        # Audio loading from testdata/
│   ├── audio_utils.py            # PCM/WAV conversion, resampling
│   ├── ping.py                   # TCP ping + IP geolocation for report metadata
│   ├── boundary_fix.py           # Speech boundary heuristics for realtime
│   ├── asr_fast.py               # Azure Fast Transcription (REST)
│   ├── asr_realtime.py           # Azure Speech SDK continuous recognition
│   ├── asr_llm_speech.py         # fast_llm and fast_mai (LLM/MAI preview)
│   └── asr_whisper.py            # Whisper v3 stub (not yet implemented)
├── scripts/
│   ├── generate_testset.py       # Azure TTS synthesis from trans.txt
│   ├── error_analysis.py         # Per-sample error breakdown
│   └── loose_match.py            # Automatic WER normalization
├── testdata/                     # Reference transcriptions (WAVs git-ignored)
│   ├── en-GB/smart-commands/trans.txt
│   ├── de-DE/smart-commands/trans.txt
│   ├── es-ES/smart-commands/trans.txt
│   ├── fr-FR/smart-commands/trans.txt
│   └── it-IT/smart-commands/trans.txt
├── results/                      # Benchmark outputs (git-ignored)
├── run_full.py                   # Main entry point
└── requirements.txt
```

---

## Important Notes

- **`fast_llm` caveat**: this service has no locale set and relies on language auto-detection. It may produce text in the wrong language, especially for non-English audio. This is intentional — it's included specifically to benchmark the auto-detection behavior.
- **TTS-synthesized audio**: the bundled testset uses Azure TTS, which produces clean, consistent audio. WER will be lower than on real human recordings. Use your own recordings for more realistic results.
- **Latency is network-dependent**: absolute UPL/LBL numbers depend heavily on your distance to the Azure region. Use relative comparisons between services, not absolute values.
- **Short utterances amplify WER**: smart speaker commands are 3–10 words. A single word error gives 10–33% WER. Consider SER (Sentence Error Rate) alongside WER for a fuller picture.
