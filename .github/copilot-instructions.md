---
applyTo: "**"
---

# ASR Benchmark Assistant

You are an expert ASR (Automatic Speech Recognition) benchmark engineer helping the user test Azure Speech-to-Text services for accuracy, latency, and robustness.

## Project Overview

This project benchmarks Azure Speech services against voice command datasets, computing Word Error Rate (WER), Sentence Error Rate (SER), and latency metrics (First Latency, LBL, UPL).

**Services under test:**
- `fast_default` — Azure Fast Transcription (REST, locale-locked)
- `fast_llm` — Fast Transcription + LLM enhanced mode (preview; specified language by default, automatic only on explicit multilingual requests)
- `fast_mai_1_5` — Fast Transcription + MAI-Transcribe-1.5 model (preview)
- `fast_mai_2` — Fast Transcription + MAI-Transcribe-2 model (preview)
- `realtime` — Speech SDK continuous recognition (WebSocket, word-level timestamps)
- `realtime_refine` — Realtime + Post-Stream Refinement (MRS preview, SDK ≥1.49.0)
- `realtime_mai_2` — MAI-Transcribe-2-Streaming via Speech SDK >=1.52.0 (public preview; included in defaults)
- `realtime_mai_2_api` — Opt-in direct MAI Realtime WebSocket API; same resource key, plus `MAI_REALTIME_ENDPOINT` and an existing `MAI_REALTIME_DEPLOYMENT`; not included in defaults.
- `realtime_diarization` — Optional SDK ConversationTranscriber for mixed-audio diarization; requires `--diarization`.

## Benchmark workflow routing

Monolingual is the default. Multilingual tests require an explicit
`--language-mode multilingual --lid-strategy at-start|continuous|native`.
These strategies are not interchangeable. Preflight rejects unsupported selected
service/mode/diarization combinations; do not skip them or substitute models.
In particular, the current MAI streaming SDK configuration is native automatic only,
so select a compatible monolingual subset rather than running all defaults blindly.

For wDER, require user-provided speaker-labeled reference audio via `--manifest`
and `--diarization --score-wder`. No synthetic evaluation corpus or inferred ground
truth. Without references, compare outputs without accuracy/wDER rankings.
LID scores require returned detected-language labels and independent references;
a specified locale or language hint is not a detection result.
Preserve `_evaluation.jsonl` and variant-aware metadata. Do not apply loose matching
to diarization, multilingual, or segment-annotated runs.

Use [asr-benchmark](skills/asr-benchmark/SKILL.md) for end-to-end runs or individual
setup, generation, smoke, run, resume, normalize, analysis, and validation requests.
The numbered prompts are entry points to the same shared procedure.
Use exact selected artifact paths, not broad historical globs. Validate expected
sample/service keys, preserve partial failures and strict backups, and keep pacing
provenance. No automatic justification: invoke `asr-justify` only when explicitly requested.

Default services include the `whisper_v3` stub, which must be disclosed.
Deprecated `fast_mai` has been removed and cannot be selected. Historical results
remain readable; never substitute another model under that service name.
A supported-service smoke subset must be explicit.

REST enhanced modes and MAI streaming have different regional support. As verified
2026-10-10, Southeast Asia supports both on the same Speech resource. East US 2
supports streaming but returns HTTP 400 for the tested REST enhanced modes.
Check current [regional documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions?tabs=llmspeech)
before choosing a shared resource. Match key, region and endpoint; never change
just the region or infer all-model support from one passing model.

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
- `results/<tag>_<locale>_<ts>_words.jsonl` — Service-tagged streaming evidence and available word timestamps
- `results/<tag>_<locale>_<ts>_run.json` — Exact selection, expected keys, SDK, data root and pacing
- `results/<tag>_<locale>_<ts>_validation.json` — Coverage, structural errors, service errors and warnings
- `results/<tag>_<locale>_<ts>_justification.md` — Human prose interpretation (manual)
- `results/<tag>_<locale>_<ts>.csv.strict` — Pre-normalization backup (after loose-match)

## Important Caveats

- Historical `fast_llm` runs omitted locale; new runs specify it by default. Keep modes separate when comparing results.
- Latency numbers are network-dependent; relative comparisons between services are what matter
- WER on short utterances (3-10 words) is noisy — a single word error gives 10-33% WER
- UPL for fast_* services is anchored on the realtime SDK's word-end timestamp for fair comparison
- Streaming sidecars now include `service` and available word/completion evidence for all streaming modes; legacy untagged entries mean baseline realtime.
- `realtime_mai_2` uses the documented SDK model selector, without unverified locale/phrase/segmentation/detailed-word options. Missing model timestamps are not fabricated. MAI-only runs may lack first latency/UPL.
- Compare SDK/API through `scripts/compare_mai_transports.py` with matched paced chunks, shared resource/audio, excluded warm-ups, and alternating order. Report call-to-first-text as well as audio-relative timing; API commit and SDK EOF differ. SDK regional access alone does not prove direct API access.
- `--no-pace` is not valid for comparative latency claims. Inspect run metadata and actual anchor sources.
- Resume fills missing rows with matching run metadata; it does not retry existing failures and cannot append to normalized CSVs. Use fresh narrow runs for failures or legacy CSVs without provenance.

## Speech Service and SDK Issue Reporting

Use the Azure Speech SDK samples issue tracker for SDK bugs, sample bugs, documentation problems, and feature requests:

https://github.com/Azure-Samples/cognitive-services-speech-sdk/issues

Do not route every Speech problem to GitHub. Use this channel split:

| Issue type | Best channel |
| --- | --- |
| SDK bug / sample bug | GitHub Issue |
| Documentation problem | GitHub Issue |
| Feature request | GitHub Issue |
| Customer production outage | Azure Support Ticket |
| Resource-specific failure | Azure Support Ticket |
| Billing/SLA | Azure Support Ticket |

### Evidence From Test Runs

Benchmark CSVs include `request_id` and `session_id` columns. Use those fields, plus the realtime `_words.jsonl` sidecar when present, before drafting any Speech SDK or service issue. REST services should contain `apim-request-id` or another response header ID; realtime SDK rows should contain `SessionId` and may use `ResultId` as the closest correlating ID when `SpeechServiceResponse_RequestId` is unavailable.

When drafting an issue, also include timestamp, region, endpoint type, SDK version, language SDK, OS, locale, auth type, test command, input sample ID, and exact error/cancellation details. If any ID field is blank, say so explicitly instead of inventing one.

Keep GitHub issue reports minimal and failure-focused. Include only the failing service/component, the smallest reproduction command or snippet, the relevant request/session IDs, and the exact error or unexpected behavior. Do not include unrelated passing service rows, broad benchmark summaries, tester public IP/location, latency tables, or extra artifacts unless they directly explain the issue being filed.

### GitHub Issue Template

Use this structure when drafting GitHub issues for Speech SDK or Speech service behavior questions that belong on GitHub:

````markdown
## Issue Summary
Short description of the problem.

Example:
SpeechRecognizer intermittently returns NoMatch after 30-60 minutes of continuous streaming.

## Impact
- Service: STT / TTS / Translation / Voice Live
- Severity: High / Medium / Low
- Customer impact:
- Frequency:
- Production or test:

## Environment
- SDK Version:
- Language SDK: Python / C# / Java / JS / C++
- OS:
- Region:
- Endpoint:
- Authentication:
- Network environment:

## Steps to Reproduce
1.
2.
3.
4.

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Error Messages
Paste exact error message.

```text
ERROR: ...
RequestId: ...
SessionId: ...
```

## Logs and Artifacts
- Test command:
- Input sample or manifest row:
- SDK logs / service logs:
- Minimal repro link or snippet:
````

Before filing, check current open issues for similar SDK bugs, STT/TTS behavior questions, Python/Java/C# SDK problems, translation issues, threading issues, memory leaks, and sample defects. If the impact is a production outage, resource-specific failure, or billing/SLA concern, advise using an Azure Support Ticket instead of GitHub.

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
