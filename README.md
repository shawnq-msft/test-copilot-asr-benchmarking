# Copilot ASR Benchmark for Azure Speech

A reusable Copilot ASR benchmark framework for Azure Speech-to-Text services. It measures **Word Error Rate (WER)**, **Sentence Error Rate (SER)**, command recognition accuracy, and latency (**LBL**, **UPL**) across Azure Speech Fast Transcription, Speech SDK realtime recognition, LLM enhanced modes, and multiple languages.

中文：这是一个用于 Azure Speech-to-Text 服务的 ASR Benchmark 框架，可通过 GitHub Copilot 辅助运行，适用于评估智能语音命令、Voice Agent 和多语言语音识别场景中的准确率、实时转写、错误类型和延迟表现。

日本語: これは Azure Speech-to-Text サービス向けの ASR ベンチマークフレームワークで、GitHub Copilot による実行支援に対応し、スマート音声コマンド、Voice Agent、多言語音声認識シナリオの精度、リアルタイム文字起こし、エラー傾向、レイテンシを評価できます。

한국어: 이 도구는 Azure Speech-to-Text 서비스를 위한 ASR 벤치마크 프레임워크로, GitHub Copilot의 실행 지원을 활용해 스마트 음성 명령, Voice Agent, 다국어 음성 인식 시나리오의 정확도, 실시간 전사, 오류 유형, 지연 시간을 평가할 수 있습니다.

In a recent full validation run, the benchmark completed **604 test cases** across **5 recognition modes** without service invocation failures. The results showed Azure Speech was stable and usable for this English home-command scenario, with high command-level recognition accuracy in both `fast_llm` and `realtime` modes.

This tool fully integrates with Copilot to run ASR accuracy and latency tests across six Azure services, synthesize test audio, analyze errors, and write human-readable benchmark justifications — all via Copilot Chat.

Measures **Word Error Rate (WER)**, **Sentence Error Rate (SER)**, and latency (**LBL**, **UPL**) across six services and multiple languages. Comes with a bundled 30-command smart speaker testset in five locales (en-GB, de-DE, es-ES, fr-FR, it-IT).

## Why this benchmark matters

This project is designed for teams building Copilot-style voice experiences, voice agents, smart device commands, car assistants, and speech-driven product workflows. It helps answer practical ASR questions:

- Can Azure Speech reliably recognize short voice commands?
- Which recognition mode works best for command-and-control scenarios?
- Where do errors come from: ASR quality, text normalization, reference labels, or pronunciation variants?
- How do `fast_default`, `fast_llm`, `fast_mai`, `realtime`, and `realtime_refine` compare?
- What are the recurring issues in numbers, word forms, homophones, and command phrase normalization?

## Field validation notes

The benchmark has been used to analyze realistic command recognition behavior, including:

- Number and formatting mismatches, such as `one` versus `1`, or `ten` versus `10`
- Near-homophone and phrase errors, such as `filter` versus `future`, `fan` versus `phone/fast`, and `auto` versus `autumn`
- Command phrase variations, such as `start` versus `star`
- Reference text and audio mismatches that can distort accuracy statistics
- Product-specific command phrases that may need domain adaptation, such as `turn on`, `filter`, `turbo`, `oscillation`, `dehumidification`, `firefly light effect`, and `running lights effect`

These notes make the benchmark useful beyond a raw WER score: it can separate model behavior, dataset quality, reference normalization, and product vocabulary issues.

---

## Quick Start with Copilot

### **Use GitHub Copilot to run the entire pipeline:**

1. **Set up** — Open Copilot Chat and run prompt `.github/prompts/01-setup.prompt.md`
   - Installs dependencies, configures `.env`, validates credentials

2. **Generate audio** — Run `.github/prompts/02-generate-testset.prompt.md`
   - Synthesizes WAV files from trans.txt using Azure TTS (all 5 locales or select one)

3. **Run benchmark** — Run `.github/prompts/03-run-benchmark.prompt.md`
   - Executes full benchmark or quick smoke test; generates CSV + reports

4. **Normalize WER** — Run `.github/prompts/04-loose-match.prompt.md`
   - Applies loose-match normalization for compound words and symbols

5. **Analyze errors** — Run `.github/prompts/05-analyze.prompt.md`
   - Re-runs error analysis on existing CSV

6. **Write justification** — Run `.github/prompts/06-justify.prompt.md`
   - Invokes `asr-justify` skill to write human prose interpretation of results

**Each prompt is independent** — run any step in any order, or invoke the full pipeline end-to-end.

---

## Copilot Agent & Skills

This workspace is a **GitHub Copilot Agent** configured with two key components:

### Copilot Agent (`.github/copilot-instructions.md`)

Declares the agent persona, capabilities, and available skills for Copilot Chat. The agent:
- Guides you through the 6-step benchmark workflow
- Explains each service's capabilities and caveats
- References the `asr-justify` skill for result interpretation
- Provides examples of how to invoke individual steps

### Step-by-Step Prompts (`.github/prompts/`)

Self-contained, runnable prompt files that Copilot executes:

| # | Prompt | What it does |
|---|--------|-------------|
| 1 | `01-setup.prompt.md` | Install deps, copy `.env`, validate credentials |
| 2 | `02-generate-testset.prompt.md` | Synthesize WAV files with Azure TTS for all/specific locales |
| 3 | `03-run-benchmark.prompt.md` | Run `run_full.py` with options, save results |
| 4 | `04-loose-match.prompt.md` | Normalize WER, regenerate reports |
| 5 | `05-analyze.prompt.md` | Re-run error analysis on CSV |
| 6 | `06-justify.prompt.md` | Invoke `asr-justify` skill for human interpretation |

### `asr-justify` Skill (`.github/skills/asr-justify/SKILL.md`)

Custom Copilot skill that reads a benchmark CSV + auto reports and writes a **9-step human prose interpretation**:

1. Recap aggregate metrics from report
2. Identify where WER is misleading (short utterances, hallucinations)
3. Flag fast_llm language confusion (no locale set, auto-detection failures)
4. Summarize genuine service disagreements
5. Analyze error shape (INS/DEL/SUB breakdown)
6. Interpret latency with network context
7. Provide actionable recommendations
8. QA pass on latency outliers
9. Output formatted justification markdown

**Usage via Copilot:**
```
Justify the results in results/asr-bench_en-GB_20260527_120000.csv
```

The skill automatically loads the report, error analysis, and word timings to produce a complete interpretation.

---

## Manual CLI (without Copilot)

If you prefer command-line control:

```bash
# Clone and install
git clone https://github.com/shawnq-msft/test-copilot-asr-benchmarking.git
cd test-copilot-asr-benchmarking
pip install -r requirements.txt

# Copy and fill in .env
cp .env.example .env
# Edit .env with your Azure Speech credentials

# Generate audio for one locale
python scripts/generate_testset.py --locale en-GB

# Run benchmark (quick test: 5 samples, 1 locale, 2 services)
python run_full.py --languages en-GB --max-per-dataset 5 --no-pace --services fast_default realtime

# View results
cat results/asr-bench_en-GB_<timestamp>_report.md
cat results/asr-bench_en-GB_<timestamp>_error_analysis.md
```

> The repository was renamed from `test-copliot-asr-benchmarking` to `test-copilot-asr-benchmarking` to fix the Copilot spelling. Existing GitHub links to the old URL continue to redirect to this repository.

### Full CLI options

```
--languages          Locales to test (default: all)
--max-per-dataset    Samples per dataset (default: 30)
--workers            Thread pool size (default: 4)
--no-pace            Skip real-time pacing (faster, unrealistic latency)
--services           Services to benchmark (default: all 6)
--resume             Resume from partial CSV
--tag                Output filename prefix (default: asr-bench)
--seed               Random seed (default: 42)
--testdata-dir       Custom audio root (default: testdata/)
--output-dir         Output directory (default: results/)
```

---

## Services Under Test

| Service | Description |
|---------|-------------|
| `fast_default` | Azure Fast Transcription (REST, api-version 2024-11-15). Locale-locked. |
| `fast_llm` | Fast Transcription + LLM enhanced mode (preview). **No locale set — auto-detects.** May hallucinate or confuse languages (intentional for benchmarking). |
| `fast_mai` | Fast Transcription + mai-transcribe-1 model (preview). Requires LLM Speech preview. |
| `fast_mai_1_5` | Fast Transcription + mai-transcribe-1.5 model (preview). Supports phrase list and transcribe style in MAI 1.5 preview. |
| `realtime` | Speech SDK continuous recognition (WebSocket, word-level timestamps). |
| `realtime_refine` | Realtime + Post-Stream Refinement (MRS preview, SDK ≥ 1.49.0). |

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

WAV files are generated from `trans.txt` using Azure TTS with per-locale neural voices (e.g., `en-GB-SoniaNeural`, `de-DE-KatjaNeural`).

---

## Output Files

For each run `<tag>_<locale>_<timestamp>`:

| File | Description |
|------|-------------|
| `results/<run>.csv` | Per-sample, per-service metrics (WER, SER, INS/DEL/SUB, latency) |
| `results/<run>_report.md` | Aggregate WER/latency table with endpoint and dataset info |
| `results/<run>_error_analysis.md` | Best/worst/hallucinations with `[▶]` audio playback links |
| `results/<run>_words.jsonl` | Per-word timestamps for realtime samples (latency audit) |
| `results/<run>_justification.md` | Human prose interpretation (written with asr-justify skill) |
| `results/<run>.csv.strict` | Pre-normalization CSV backup (created after loose_match) |
| `results/audio/<dataset>/<sample_id>.wav` | Extracted audio clips for manual review |

---

## Latency Metrics

Three latency values are computed per sample:

- **First Latency** — how quickly the first partial appears after user *starts* speaking (realtime only)
- **LBL** (Last-final Beyond Last-chunk) — server flush time after last audio chunk; can be negative
- **UPL** (User-Perceived Latency) — delay of final transcript after user *stops* speaking; anchored to realtime SDK's word-end for fair cross-service comparison

Full definitions and ASCII diagrams are in the auto-generated `_report.md`.

---

## WER Normalization

Normalization steps:
1. **NFKC** — unicode normalization
2. **Lowercase** — case-insensitive comparison
3. **Punctuation strip** — remove punctuation
4. **Space collapse** — normalize whitespace

Optional **loose-match** post-processing fixes systematic false positives:

```bash
python scripts/loose_match.py results/asr-bench_en-GB_<timestamp>.csv
```

Catches: compound-word splits (`stummschalten` ↔ `stumm schalten`), alphanumeric spacing (`2D` ↔ `2 D`), symbol variants (`360°` ↔ `360 gradi`).

Regenerate reports after normalization:
```bash
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

---

## Adding Custom Test Data

1. Create: `testdata/<locale>/<dataset-name>/`
2. Add `trans.txt` — one reference command per line (or `filename.wav<TAB>reference`)
3. Either place WAV files or run: `python scripts/generate_testset.py --locale <locale>`
4. Run benchmark: `python run_full.py --languages <locale>`

---

## Project Structure

```
.
├── .env.example                     # Credential template
├── .github/
│   ├── copilot-instructions.md      # Copilot agent persona
│   ├── prompts/                     # 6 step-by-step prompt files
│   │   ├── 01-setup.prompt.md
│   │   ├── 02-generate-testset.prompt.md
│   │   ├── 03-run-benchmark.prompt.md
│   │   ├── 04-loose-match.prompt.md
│   │   ├── 05-analyze.prompt.md
│   │   └── 06-justify.prompt.md
│   └── skills/asr-justify/
│       └── SKILL.md                 # Human result interpretation skill
├── benchmark/                       # Core benchmarking framework
│   ├── config.py                    # Services, locales, credentials
│   ├── runner.py                    # Orchestrator (threading, resume)
│   ├── report.py                    # Aggregate metrics → markdown
│   ├── metrics.py                   # WER/SER computation
│   ├── datasets_loader.py           # Audio loading from testdata/
│   ├── audio_utils.py               # PCM/WAV conversion
│   ├── ping.py                      # TCP ping + IP geolocation
│   ├── boundary_fix.py              # Speech boundary heuristics
│   ├── asr_fast.py                  # Azure Fast Transcription
│   ├── asr_realtime.py              # Speech SDK continuous
│   ├── asr_llm_speech.py            # LLM and MAI models
│   └── asr_whisper.py               # Whisper v3 stub
├── scripts/
│   ├── generate_testset.py          # Azure TTS synthesis
│   ├── error_analysis.py            # Error breakdown analysis
│   └── loose_match.py               # WER normalization
├── testdata/                        # Reference trans.txt (WAVs git-ignored)
│   ├── en-GB/smart-commands/trans.txt
│   ├── de-DE/smart-commands/trans.txt
│   ├── es-ES/smart-commands/trans.txt
│   ├── fr-FR/smart-commands/trans.txt
│   └── it-IT/smart-commands/trans.txt
├── results/                         # Benchmark outputs (git-ignored)
├── run_full.py                      # Main entry point
└── requirements.txt
```

---

## Important Notes

- **`fast_llm` caveat**: no locale set, relies on language auto-detection. May produce wrong-language text, especially for non-English audio. Intentional for benchmarking auto-detection behavior.
- **TTS audio**: bundled testset uses Azure TTS (clean, consistent). WER will be lower than on real human recordings.
- **Latency is network-dependent**: absolute UPL/LBL numbers depend on distance to Azure region. Use *relative* comparisons between services.
- **Short utterances amplify WER**: smart speaker commands are 3–10 words. Single word error = 10–33% WER. Consider SER alongside WER.

---

## Examples

### Run a smoke test with Copilot

1. Open Copilot Chat in VS Code
2. Copy the `.github/prompts/03-run-benchmark.prompt.md` file
3. Ask: *"Run a quick benchmark: 5 samples, en-GB only, fast_default and realtime services"*
4. Copilot executes the benchmark and generates reports
5. Ask: *"Justify the results in the generated CSV"*
6. Copilot invokes the `asr-justify` skill and produces a prose interpretation

### Resume a partial benchmark (CLI)

```bash
python run_full.py --languages en-GB --resume results/asr-bench_en-GB_20260527_120000.csv
```

### Use your own test data (Copilot)

1. Add audio files to `testdata/my-locale/my-dataset/`
2. Add `trans.txt` with references
3. In Copilot, ask: *"Benchmark my custom testdata: my-dataset, my-locale, all services"*
4. Copilot runs the benchmark against your audio

---

## Reference

For detailed latency definitions, speech boundary fixes, and WER computation formulas, see the auto-generated `*_report.md` after a benchmark run.
