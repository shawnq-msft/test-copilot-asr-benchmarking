# Copilot ASR Benchmark for Azure Speech

A reusable Copilot ASR benchmark framework for Azure Speech-to-Text services. It measures **Word Error Rate (WER)**, **Sentence Error Rate (SER)**, command recognition accuracy, and latency (**LBL**, **UPL**) across Azure Speech Fast Transcription, Speech SDK realtime recognition, LLM enhanced modes, and multiple languages.

## New: MAI streaming and validated Copilot workflows

- **MAI-Transcribe-2-Streaming** (`realtime_mai_2`) joins the benchmark through
  **Speech SDK >=1.52.0**, with incremental transcripts, final-result validation,
  request/session IDs, and streaming timing evidence. It is distinct from the
  REST model `fast_mai_2` and remains a **public preview**.
- **Optional direct MAI Realtime API** (`realtime_mai_2_api`) is available through
  `--services`, alongside the SDK adapter. It uses the same resource key with a
  MAI Realtime endpoint and deployed model, and is **excluded from default runs**.
- **One Copilot skill, full pipeline or individual stages:** the
  [asr-benchmark skill](.github/skills/asr-benchmark/SKILL.md) routes setup, audio
  generation, smoke tests, full runs, resume, normalization, analysis, and validation.
  Existing numbered prompts remain available; justification is explicit-only.
- **Verifiable results:** exact sample/service coverage, persistent run settings,
  strict-score backups, service-tagged streaming evidence, and actionable failures.
  Missing timestamps are never invented, and unpaced runs are not latency benchmarks.

> **Recommended testing region: Southeast Asia (`southeastasia`).**
> As verified on **October 10, 2026**, one shared Speech resource there supports
> all seven default implemented, non-retired benchmark modes, including REST enhanced
> transcription and MAI streaming. Use that resource's matching **key, region,
> and endpoint**; no separate MAI credentials are required.
> **East US 2 is not a drop-in replacement for combined tests:** MAI streaming
> worked there, but REST enhanced-mode requests returned HTTP 400.
> The optional direct Realtime API additionally requires a deployed model and
> accessible API route; the seven-mode recommendation does not verify that route.
> Regional availability can change; check the current
> [REST enhanced-mode regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions?tabs=llmspeech)
> and [MAI streaming regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming#availability-and-regions).

After [setup](.github/prompts/01-setup.prompt.md) and
[preparing test audio](.github/prompts/02-generate-testset.prompt.md), smoke-test
the six implemented monolingual configurations with paced audio:

```powershell
.\.venv\Scripts\python.exe -X utf8 run_full.py --languages en-GB --max-per-dataset 2 --workers 1 --services fast_default fast_llm fast_mai_1_5 fast_mai_2 realtime realtime_refine
```

This explicit service list excludes the `whisper_v3` stub,
opt-in `realtime_mai_2_api` / `realtime_diarization` / `azure_whisper`, and `realtime_mai_2`.
The SDK MAI service can also be selected in monolingual mode using base language codes.
Deprecated MAI-Transcribe-1 (`fast_mai`) has been removed; MAI 1.5 and 2 remain available.
The latest full-local-Xiaomi check covered **one recording across all seven modes**:
7/7 successful service calls, not a statistically meaningful accuracy or latency ranking.

中文：这是一个用于 Azure Speech-to-Text 服务的 ASR Benchmark 框架，可通过 GitHub Copilot 辅助运行，适用于评估智能语音命令、Voice Agent 和多语言语音识别场景中的准确率、实时转写、错误类型和延迟表现。

日本語: これは Azure Speech-to-Text サービス向けの ASR ベンチマークフレームワークで、GitHub Copilot による実行支援に対応し、スマート音声コマンド、Voice Agent、多言語音声認識シナリオの精度、リアルタイム文字起こし、エラー傾向、レイテンシを評価できます。

한국어: 이 도구는 Azure Speech-to-Text 서비스를 위한 ASR 벤치마크 프레임워크로, GitHub Copilot의 실행 지원을 활용해 스마트 음성 명령, Voice Agent, 다국어 음성 인식 시나리오의 정확도, 실시간 전사, 오류 유형, 지연 시간을 평가할 수 있습니다.

In a recent full validation run, the benchmark completed **604 test cases** across **5 recognition modes** without service invocation failures. The results showed Azure Speech was stable and usable for this English home-command scenario, with high command-level recognition accuracy in both `fast_llm` and `realtime` modes.

This tool fully integrates with Copilot to run ASR accuracy and latency tests across Azure services, synthesize test audio, analyze errors, and write human-readable benchmark justifications — all via Copilot Chat.

Measures **Word Error Rate (WER)**, **Sentence Error Rate (SER)**, and latency (**LBL**, **UPL**) across multiple services and languages. Comes with a bundled 30-command smart speaker testset in five locales (en-GB, de-DE, es-ES, fr-FR, it-IT).

## Why this benchmark matters

This project is designed for teams building Copilot-style voice experiences, voice agents, smart device commands, car assistants, and speech-driven product workflows. It helps answer practical ASR questions:

- Can Azure Speech reliably recognize short voice commands?
- Which recognition mode works best for command-and-control scenarios?
- Where do errors come from: ASR quality, text normalization, reference labels, or pronunciation variants?
- How do `fast_default`, `fast_llm`, `fast_mai_1_5`, `fast_mai_2`, `realtime`, `realtime_refine`, and `realtime_mai_2` compare?
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

Use the discoverable [asr-benchmark skill](.github/skills/asr-benchmark/SKILL.md)
to run a full workflow or just the requested stage. For example:

```text
Run end-to-end for en-GB, 5 samples per dataset, comparing realtime and fast_default.
Reuse existing audio, preserve strict results, and apply loose matching.

Validate this exact CSV without rerunning recognition.

Analyze this run using D:\audio\testdata; do not normalize or justify it.
```

The skill tracks exact output paths and checks sample/service coverage. Plain
end-to-end requests reuse valid audio, generate only missing requested audio,
run paced recognition, validate, preserve strict scoring, normalize, and refresh
reports. Justification remains explicit-only. See the
[stage procedures](.github/skills/asr-benchmark/references/stages.md).

### MAI-Transcribe-2-Streaming

`realtime_mai_2` is included in default runs, separate from REST `fast_mai_2`.
It uses [the documented Speech SDK model selector](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming-speech-sdk)
and requires **azure-cognitiveservices-speech >=1.52.0**. This is a public preview,
not a production SLA. It uses the same `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`,
and `AZURE_SPEECH_ENDPOINT` as baseline realtime. No separate MAI key or endpoint
is required: use a Speech resource in a documented supported region, with its
matching key and endpoint.
Monolingual mode sets `speech_recognition_language` to the base language code
(`zh-CN` -> `zh`, `en-GB` -> `en`), not the full locale. A paced live check with
SDK 1.52.0 on 2026-10-22 confirmed outgoing `speech.context` contained
`phraseDetection.language="zh"` and the selected MAI model, with a final transcript.
This verifies an explicitly sent/accepted configuration, not an absolute language
lock or a guarantee of transcription accuracy. The streaming sidecar records the
requested locale and SDK language code. Explicit multilingual/native mode leaves
the language unset; it is not a fallback for monolingual failures.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\validate_benchmark.py preflight --languages en-GB --services realtime_mai_2 --language-mode multilingual --lid-strategy native --max-per-dataset 2 --workers 1
.\.venv\Scripts\python.exe -X utf8 run_full.py --languages en-GB --services realtime_mai_2 --language-mode multilingual --lid-strategy native --max-per-dataset 2 --workers 1 --tag streaming-smoke
```

The adapter does not enable unverified phrase-list,
segmentation or detailed-word options. Explicit phrase hints with this service
are rejected. Baseline realtime boundaries anchor first latency/UPL when available.
In a MAI-only run, absent model word timing means these metrics are unavailable;
LBL and final transcript evidence can still be collected. Do not infer zero latency.

The retired `fast_mai` service is no longer selectable. Historical results remain
readable and are never mapped to another model. The unimplemented
`whisper_v3` stub remains in defaults and is disclosed. Choose an explicit service
subset for a smoke test; a default run with service failures fails validation.

### Azure OpenAI Whisper (optional)

Select `--services azure_whisper` for an existing Azure OpenAI `whisper`
deployment. This is **not** the unimplemented `whisper_v3` service; no historical
results are relabeled. Configure `WHISPER_AZURE_KEY`, `WHISPER_AZURE_ENDPOINT`,
`WHISPER_AZURE_DEPLOYMENT`, and the actual `WHISPER_AZURE_MODEL_VERSION` (default
`001`) using the same resource. Run metadata and reports retain the deployment,
endpoint, model version, API version and requested base language code.

The adapter sends a PCM WAV multipart upload to the Azure audio transcription API.
Pacing is enabled by default; first latency is whole-request time to the final
JSON response, not a first partial. LBL starts at the end of upload, and UPL
requires an available baseline anchor. Word timestamps and detected language are
not inferred. Files over 25 MB, phrase hints, multilingual/LID and diarization
are rejected rather than silently split or substituted. The adapter is opt-in
and does not change default service selection.
`WHISPER_AZURE_REQUESTS_PER_MINUTE` defaults to 3; set it to the actual deployment
quota. Requests within one runner process are spaced before latency measurement
starts. Do not run multiple processes against that quota without coordinating
their traffic. At 3 requests/minute, 5,000 recordings require at least 27.8 hours.

### Avoiding REST enhanced-mode HTTP 400 errors

REST enhanced transcription and MAI streaming have different regional support.
On 2026-10-10, a Xiaomi test using one Southeast Asia resource verified
`fast_default`, `fast_llm`, `fast_mai_1_5`, `fast_mai_2`, `realtime`,
`realtime_refine`, and `realtime_mai_2`. The same REST enhanced-mode definitions
returned HTTP 400 in East US 2, while MAI streaming worked there.

Use a shared Speech resource in **Southeast Asia (`southeastasia`)** for combined
tests, with its matching key and endpoint. No MAI-specific credentials are needed.
Check the current [REST enhanced-mode region table](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions?tabs=llmspeech)
and [MAI streaming regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming#availability-and-regions).
Preflight warns on regions absent from the documented lists; it does not reroute.

MAI-Transcribe-1 was deprecated August 20, 2026 and its `fast_mai` adapter has been
removed. `--services fast_mai` is rejected locally; no legacy request is sent.

### Comparing MAI Realtime API with Speech SDK

The opt-in `realtime_mai_2_api` service uses the
[documented Realtime WebSocket API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming-realtime),
while `realtime_mai_2` remains the SDK implementation. Defaults are unchanged.
The Realtime API requires a **deployed MAI-Transcribe-2-Streaming model** and the
resource's **Azure MAI Realtime API v1.0** endpoint. Set `MAI_REALTIME_ENDPOINT`
and `MAI_REALTIME_DEPLOYMENT`, using the same resource/key as the SDK.
Do not assume that a resource which accepts SDK requests also exposes the
Realtime API route: a Southeast Asia test resource returned `404 API Not Found`,
while the East US 2 resource with a deployed model accepted the route (2026-10-12).
This is resource-level evidence, not a new regional availability guarantee.

To select this endpoint in a normal benchmark, configure these optional values
in `.env` using your resource's actual endpoint and deployment name:

```dotenv
MAI_REALTIME_ENDPOINT=https://your-resource-name.services.ai.azure.com
MAI_REALTIME_DEPLOYMENT=MAI-Transcribe-2-Streaming
```

Keep `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, and `AZURE_SPEECH_ENDPOINT`
matched to that resource; there is no separate MAI API key. Then preflight and run:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\validate_benchmark.py preflight --languages en-GB --services realtime realtime_mai_2_api --max-per-dataset 2 --workers 1 --chunk-ms 20
.\.venv\Scripts\python.exe -X utf8 run_full.py --languages en-GB --services realtime realtime_mai_2_api --max-per-dataset 2 --workers 1 --chunk-ms 20 --tag mai-api-smoke
```

Baseline `realtime` supplies shared word-boundary latency anchors. You can select
only `--services realtime_mai_2_api`, but speech-anchored first latency and UPL
will be unavailable without those anchors. Neither selection changes defaults.

The guide recommends 10-20 ms chunks and explicit `input_audio_buffer.commit`.
Neither guide establishes that the Realtime API is inherently faster than SDK.
Compare **matched** chunk sizes and include connection/setup time; do not compare
a preconnected API session with an SDK cold start and call the difference model speed.
API commit at EOF and SDK stream-close at EOF can still have different server
finalization semantics.
Auto-detection omits the optional language field, as in the guide's Python sample;
the tested service rejected the JSON example's explicit `"language": null`.

For a reproducible paired experiment using existing audio:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\compare_mai_transports.py --output-dir results\mai-transport-experiment --commands 5 --repetitions 5 --chunk-ms 100 20
```

This uses all local Xiaomi recordings plus five seeded en-GB smart commands,
freezes baseline-realtime speech boundaries, performs excluded warm-ups, and
alternates SDK/API order within each matched chunk-size condition. It calls the
existing benchmark runner with one worker, paced PCM, fresh sessions, and no retries.
The output directory must be new. With one Xiaomi recording, this is 120 measured
calls plus six anchor and four warm-up calls. It makes paid service calls.

Outputs include exact experiment selection and PCM hashes, per-pair CSVs/run
metadata/streaming evidence/validation, `trials.jsonl`, `comparison.json`, and
`comparison.md`. Raw failures remain visible; summaries use successful matched
pairs. No normalization or automatic human justification is applied. First
available text, speech-relative first text, call-to-first-text, setup, LBL, UPL,
and strict WER are distinguished. Repeated trials of six recordings do not
establish a general performance ranking.

`run_full.py --chunk-ms 20` changes streaming chunk size only (REST stays unchanged);
the default remains 100 ms. Optional API service ordering follows the requested
streaming order after baseline realtime, which still runs first for shared anchors.

### Speaker diarization and explicit language modes

**Monolingual is now the default:** the request specifies the sample's declared
language. Use `--language-mode multilingual` only when testing LID/multilingual
recognition, and choose `--lid-strategy at-start|continuous|native` explicitly.
At-start identifies one language; continuous supports changes between utterances;
native uses the model's own multilingual behavior. These are not equivalent.
SDK continuous LID is not a promise of within-sentence code-switch detection.
Configured languages/hints are never scored as detected-language output.

The current conservative configuration surface is below. Unsupported or unverified
combinations fail preflight before paid calls; there is no skipping or model fallback.

| Service | Monolingual | Multilingual strategy | Diarization |
|---|---|---|---|
| `fast_default` | Specified locale | `at-start`, `native` | Yes |
| `fast_llm` | Specified locale | `native` | Yes |
| `fast_mai_1_5` | Specified base language | Not enabled | Not enabled |
| `fast_mai_2` | Specified base language | `native` | Yes |
| `realtime` | Specified locale | `at-start`, `continuous` | Use the separate service below |
| `realtime_refine` | Specified locale | Not enabled | Not enabled |
| `realtime_diarization` (optional) | Specified locale | Not enabled | SDK ConversationTranscriber |
| `realtime_mai_2` | Specified base language code; not a hard lock | `native` | Not verified |
| `realtime_mai_2_api` (optional) | Documented language hint, not a hard lock | `native` | Not verified |
| `azure_whisper` (optional) | Specified base language code | Not enabled | Not enabled |

`fast_mai` has been removed; `whisper_v3` remains an unimplemented stub.
Service defaults are unchanged, including the failing `whisper_v3` stub.
Use an explicit compatible subset; do not mix language modes. Historical automatic
`fast_llm` runs must not be mislabeled as new monolingual runs.

Initial fixed-language/candidate configuration is conservatively limited to
`en-US`, `en-GB`, `de-DE`, `es-ES`, `fr-FR`, `it-IT`, and `zh-CN`; this is not an exhaustive
Azure language-support claim. Candidate LID accepts distinct base languages
(2-4 at-start; 2-10 continuous). SDK continuous LID additionally requires a matching
resource's documented `/speech/universal/v2` endpoint to be configured explicitly.
The SDK diarization path currently rejects LID, phrase hints, and max-speaker controls
until their combined behavior is verified. REST `--max-speakers` accepts 2-35.
No new live capability smoke tests are implied by these offline-validated integrations.

#### User audio and optional annotations

Use `--manifest` for an exact, versioned JSONL selection. Audio paths are relative
to the manifest. Do not combine it with `--languages`, `--testdata-dir`, or an explicit
`--max-per-dataset`; each record
declares its own requested language and scenario. No evaluation audio is synthesized.

Example **schema**, not an included annotated dataset:

```json
{"schema_version":1,"sample_id":"meeting-1","audio":"meeting.wav","requested_language":"en-GB","scenario":"single-language","reference":{"language":"en-GB","segments":[{"text":"Hello there","speaker":"speaker-a","start_s":0.2,"end_s":1.1},{"text":"Good morning","speaker":"speaker-b","start_s":1.3,"end_s":2.2}]}}
```

- Omit `reference` entirely for no-ground-truth comparison. Filenames are not references.
- Flat `reference.text` enables WER. Ordered `reference.segments` with text/speaker
  labels enable wDER; segments may also be individual words. Real timestamps are
  optional for wDER and required for duration-weighted segment LID scoring.
- `reference.language` is the clip-level truth for single-language audio.
  For switching audio, use `scenario: "language-switching"` and reference segment
  language tags. Do not pass reference labels as automatic candidate settings.
- Reference text must agree with reference segments when both are supplied.
  Explicit wDER scoring rejects missing/unknown speaker labels and timed overlap.
- No-reference reports show transcripts, speaker counts/switches, unknown labels,
  detected languages, and aligned pairwise disagreement. They do not claim accuracy,
  wDER, or a winning endpoint.

With user-provided data, preflight and run diarization as follows:

```powershell
.\.venv\Scripts\python.exe scripts\validate_benchmark.py preflight --manifest D:\audio\meeting.jsonl --services fast_default realtime_diarization --diarization --score-wder --workers 1
.\.venv\Scripts\python.exe run_full.py --manifest D:\audio\meeting.jsonl --services fast_default realtime_diarization --diarization --score-wder --workers 1 --tag speakers
```

Omit `--score-wder` for descriptive comparison without speaker references.
For explicit LID testing:

```powershell
.\.venv\Scripts\python.exe run_full.py --manifest D:\audio\languages.jsonl --services fast_default --language-mode multilingual --lid-strategy at-start --lid-candidates en-GB de-DE --workers 1
```

#### Metric and artifact contract

Versioned evaluations retain the ordinary ASR reporting contract, not just
speaker/language evidence: source/set composition, per-set/configuration accuracy,
latency mean/median/p90 with observation counts and anchor definitions,
reference-backed best/median/worst cases with audio links, concrete cross-model
differences and edit patterns, and explicit failures/provenance. Reference-free
comparisons remain descriptive. Negative LBL is explained as a final event before
audio upload ended, not negative wall-clock latency.

`build_evaluation_report` regenerates one exact run locally.
Main reports use the fixed **classic-v1** Python renderer, based on the historical
benchmark template: Endpoints under test, Datasets under test, Results, Speech
boundaries, and Latency definitions with the original timeline. The Results table
keeps its original columns and mean/p90 formatting. Versioned runs append the
evaluation details (CER, timing counts/medians, variants, examples and failures)
instead of replacing the main template. REST request-to-response timing is not
displayed as streaming First Latency. Missing/reference-free scores stay
unavailable; variants are never pooled. Local evaluation rendering does not probe
the current network or present current adapter configuration as historical fact.
Agents invoke the renderer; they must not rewrite the main report or prepend
ad-hoc tables/notes. Interpretation belongs in a separately requested justification.
Template changes require regression tests; rendering does not alter pacing,
transcripts, scores or recorded measurements.

`build_combined_evaluation_report` accepts an explicit list of CSV paths, keeping
configurations separate; optional `retry_csv_paths` appear only as a separate
appendix and never replace original rows. Smoke CSVs are not implicitly included.
Neither function changes scores or calls Speech services. Full report content and
arithmetic still require the skill's verification gate; generation alone does not
set `report_verified`.

- Chinese evaluations use CER as primary accuracy, plus character-normalized SER
  and latency. CER applies NFKC, lowercase, and removal of punctuation/whitespace;
  reports provide micro/macro CER and CSVs retain character edit counts. Existing
  whitespace WER remains explicitly labeled and is not Chinese word-segmented WER.
  CER and Chinese SER always preserve this strict character policy even when permitted WER loose
  matching is applied; it is not silently recomputed with compound-word rules.

- wDER = speaker errors / aligned correct-or-substituted word pairs, after one
  optimal one-to-one speaker mapping per recording. Insertions/deletions remain
  in WER and alignment coverage, not the wDER denominator. Zero aligned pairs
  means unavailable, not 0. Reports show micro/macro wDER and counts.
- Tokenization uses the existing whitespace-based WER normalization, retaining
  speaker ownership. This does not provide a new tokenizer for unsegmented scripts.
- Clip LID accuracy distinguishes base language from exact locale. Continuous/native
  segment accuracy is correct labeled overlap duration / labeled reference duration;
  missing coverage is explicit. Missing returned labels or required timing means
  unavailable, not a fabricated score.
- Version-2 run metadata fingerprints options, requested languages, PCM, and
  references. CSV and sidecars include configuration identities. Resume cannot
  mix modes or changed input/reference data; historical artifacts remain readable.
- `_evaluation.jsonl` retains final speaker/language segments and scoring evidence
  for both REST and SDK. Reports expose strict/reference-free status, failures,
  missing labels, and correlation IDs; evaluation report generation is local.
- Full selected audio is sent for manifest, diarization, and multilingual evaluation.
  No reference-derived trimming is applied to these requests.
- Loose matching remains available for ordinary unannotated monolingual ASR,
  but is rejected for diarization, multilingual, or speaker/segment-annotated runs.

### Validation and resume

- New runs persist `_run.json` (selection, expected keys, SDK, data root and pacing)
  and `_validation.json` (structural errors, service errors, warnings, exact coverage).
- `_words.jsonl` contains service-tagged streaming entries, available words, IDs,
  and partial/final completion evidence. Legacy entries without `service` mean `realtime`.
- `run_full.py` rejects invalid/empty selections and missing referenced audio,
  exits nonzero on service failures, and preserves partial artifacts. Reports are
  generated automatically only after row validation passes.
- Resume requires one locale, matching run metadata and CSV schema. It fills missing
  rows without rerunning completed rows; existing errors require a fresh narrow run.
  Normalized results and legacy CSVs without run metadata cannot be resumed safely.
  Legacy reports/analysis remain readable.
- `--max-per-dataset 0` means all samples; negative limits are invalid.
- `--no-pace` is functional/accuracy-only. Latency comparison requires paced audio
  and verified compatible anchors.

For an exact selected CSV, run:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\validate_benchmark.py artifacts $csv
```

`--rows-only` validates before reports exist. `--allow-legacy` permits historical
artifact inspection while disclosing that exact expected coverage/pacing cannot
be verified. The validator makes no Azure calls and imposes no invented WER threshold.
Normalization changes scoring only, preserves `.csv.strict`, and requires explicit
report regeneration. Never resume into an already normalized run.

Focused offline regression tests (SDK/network mocked):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_streaming_workflows.py
```

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

**Each prompt is independently invokable** with its required inputs. Requested
end-to-end execution follows the validated stage order; standalone stages do not
implicitly authorize paid recognition or modifications to other runs.

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
3. Flag language confusion in explicitly selected multilingual/LID runs
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
--services           Services to benchmark (default: all configured)
--resume             Resume from partial CSV
--tag                Output filename prefix (default: asr-bench)
--seed               Random seed (default: 42)
--testdata-dir       Custom audio root (default: testdata/)
--output-dir         Output directory (default: results/)
--phrase-list        Runtime phrase hints (space-separated; quote multiword phrases)
--phrase-weight      Phrase-list bias from 0.0 to 2.0 (default: 1.0)
```

For example, to bias all supported endpoints toward an unabbreviated place name:

```bash
python run_full.py --languages en-US --services fast_default fast_llm fast_mai_1_5 fast_mai_2 realtime realtime_refine --phrase-list "Ionia, Michigan" --phrase-weight 2.0
```

Phrase lists influence recognition but are not a guaranteed output-formatting rule. In particular, display-form inverse text normalization can still abbreviate words. The legacy `mai-transcribe-1` model rejects `phraseList`; the benchmark therefore leaves that endpoint unadapted while applying phrases to supported endpoints.

---

## Services Under Test

| Service | Description |
|---------|-------------|
| `fast_default` | Azure Fast Transcription (REST, api-version 2024-11-15). Locale-locked. |
| `fast_llm` | Fast Transcription + LLM enhanced mode (preview). Specified locale by default; native auto-detection only when multilingual mode is requested. |
| `fast_mai_1_5` | Fast Transcription + MAI-Transcribe-1.5 model (preview). Supports phrase list and transcribe style. |
| `fast_mai_2` | Fast Transcription + MAI-Transcribe-2 model (preview). Supports 60 languages, keyword biasing, diarization, word timestamps, and configurable transcription style. |
| `realtime` | Speech SDK continuous recognition (WebSocket, word-level timestamps). |
| `realtime_refine` | Realtime + Post-Stream Refinement (MRS preview, SDK ≥ 1.49.0). |
| `realtime_mai_2` | MAI-Transcribe-2-Streaming preview via Speech SDK >=1.52.0; included in default runs. |
| `realtime_mai_2_api` | Optional MAI-Transcribe-2-Streaming Realtime WebSocket API; requires endpoint/deployment settings and the shared Speech key. Excluded from defaults. |
| `realtime_diarization` | Optional Speech SDK ConversationTranscriber for mixed-audio speaker labeling. Requires `--diarization`; excluded from defaults. |
| `whisper_v3` | Unimplemented stub; selecting it produces an explicit service failure. |

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
| `results/<run>_words.jsonl` | Service-tagged streaming evidence and available word timestamps |
| `results/<run>_run.json` | Selected sample/service keys, SDK, data root, pacing and run settings |
| `results/<run>_validation.json` | Exact coverage, structural errors, service failures and warnings |
| `results/<run>_justification.md` | Human prose interpretation (written with asr-justify skill) |
| `results/<run>.csv.strict` | Pre-normalization CSV backup (created after loose_match) |
| `results/audio/<dataset>/<sample_id>.wav` | Extracted audio clips for manual review |

---

## Latency Metrics

Three latency values are computed per sample:

- **First Latency** — how quickly the first partial appears after user *starts* speaking (streaming services with trustworthy speech-start timing)
- **LBL** (Last-final Beyond Last-chunk) — server flush time after last audio chunk; can be negative
- **UPL** (User-Perceived Latency) — delay of final transcript after user *stops* speaking; inspect actual anchor sources before comparing. REST and MAI streaming use baseline realtime boundaries when available; refinement retains its own word timing.

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

- **Language-mode caveat**: old `fast_llm` runs omitted locale; new runs specify it by default. Test automatic detection explicitly and do not pool historical and new modes.
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
