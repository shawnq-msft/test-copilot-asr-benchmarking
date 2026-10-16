# Stage procedures (PowerShell)

Run from the repository root. The examples use the workspace virtual environment;
use the actual selected interpreter if different. Substitute exact user-selected
paths and scope; never run angle-bracket placeholders literally.

## Setup

1. Inspect the selected interpreter and installed dependencies. Check imports and
   SDK version before installing anything:

   ```powershell
   .\.venv\Scripts\python.exe -c "import azure.cognitiveservices.speech as s; import rapidfuzz, soundfile, requests, dotenv, numpy, tqdm; print(s.__version__); print('model selector:', hasattr(s.SpeechConfig, 'model'))"
   ```

2. If dependencies are missing/outdated, use that interpreter's
   `-m pip install -r requirements.txt`. If no environment exists, create/select one
   first. Do not assume activation persists between tool calls.
3. If `.env` is absent, `Copy-Item .env.example .env`; never overwrite an existing
   file. Have the user populate it locally. Never read credentials into chat.
4. `realtime_mai_2` uses the same primary `AZURE_SPEECH_KEY`,
   `AZURE_SPEECH_REGION`, and `AZURE_SPEECH_ENDPOINT` as baseline realtime.
   No separate MAI credentials are required; use a Speech resource in a documented
   supported region. Existing `LLM_AZURE_SPEECH_*` and `REFINE_AZURE_SPEECH_*`
   overrides for other modes remain optional and fall back to primary configuration.
   Choose a region supporting all requested services: Southeast Asia supports both
   REST enhanced mode and MAI streaming (verified 2026-10-10), whereas East US 2
   supports streaming but not the tested REST enhanced modes. Check the current
   [REST regions table](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions?tabs=llmspeech)
   and [streaming overview](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming).
   Do not change only the region while retaining another resource's key/endpoint.
5. Validate only the selected services' configuration. Presence is not live access:

   ```powershell
   .\.venv\Scripts\python.exe -c "from benchmark.config import assert_credentials; assert_credentials(('realtime', 'realtime_mai_2')); print('Local configuration present; authentication not yet verified')"
   ```

## Inventory / generate

Use reference filenames from the loader, not WAV counts. The strict preflight
checks missing referenced audio, selected sample decoding, short clips, and duplicate
sample identities. For a complete audio inventory use `--max-per-dataset 0`.

If requested references exist but WAVs are missing, use the existing generator for
only the requested locale/dataset; it skips existing files:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\generate_testset.py --locale en-GB --dataset smart-commands --max-per-dataset 2 --seed 42
```

For custom data add `--testdata-dir D:\audio\testdata`; preserve that root in all
later stages. Match the benchmark's `--max-per-dataset` and `--seed` so a small
run does not synthesize unrequested samples. Zero selects all references.
Supply an explicit `--voice` when no locale default exists.
Do not use `--force` to fix corrupt custom recordings. Stop and report corrupt or
missing recordings when synthesis was not requested/appropriate.
Generation failures are blockers even if some files were written. Revalidate
audio after generation, not just the exit code or file count.

## Preflight / run

For large supplied archives, inventory audio and reference identities first. Agree
on full-corpus versus sampled scope before paid calls. Persist a fixed-seed sampled
manifest, source manifest hash, per-dataset counts and allocation policy; distinguish
dataset-balanced sampling from corpus-proportional estimates. Validate the exact
selected audio, not necessarily every unselected archive member. Preserve explicitly
included challenge subsets and report them separately.

For Chinese samples, use micro/macro CER and character-normalized SER as primary
accuracy outputs. Whitespace WER is only a diagnostic, not segmented Chinese WER.
Permitted loose WER matching must not change CER or Chinese SER. Keep native
multilingual runs strict; do not silently combine their settings with fixed-language
runs or fabricate missing MAI language labels/word-end anchors.

Use the same selection flags for preflight and run:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\validate_benchmark.py preflight --languages en-GB --services realtime fast_default --max-per-dataset 2 --workers 1
.\.venv\Scripts\python.exe -X utf8 run_full.py --languages en-GB --services realtime fast_default --max-per-dataset 2 --workers 1 --tag streaming-smoke
```

Inspect preflight JSON and paid scope before recognition. Ordinary MAI-2 tests
require an explicit target language; follow the language gate below. Do not select
`realtime_mai_2 --language-mode multilingual --lid-strategy native` merely because
it is the only currently supported SDK configuration. That combination requires
an explicitly authorized auto-language diagnostic/comparison. Baseline anchors
may be absent; compare actual timestamp availability, not assumptions.

`--no-pace` is appropriate only for functional/accuracy smoke tests, not latency
claims. Explicitly label any such output. Do not silently change user pacing.

`run_full.py` writes CSV, `_run.json` (selection/pacing/SDK and expected keys),
`_words.jsonl` (service-tagged streaming evidence), and `_validation.json`.
It generates `_report.md` and `_error_analysis.md` only after row validation passes.
A failed run retains CSV/evidence/validation and exits nonzero.

The runner does not automatically normalize or justify. Those are separate stages.

For diarization and LID, use the versioned user-manifest schema in the README.
No annotated dataset is generated automatically. `--score-wder` requires independent
speaker references; without them, use descriptive comparison without accuracy claims.
Example with user-provided files:

```powershell
.\.venv\Scripts\python.exe scripts\validate_benchmark.py preflight --manifest D:\audio\meeting.jsonl --services fast_default realtime_diarization --diarization --score-wder --workers 1
.\.venv\Scripts\python.exe run_full.py --manifest D:\audio\meeting.jsonl --services fast_default realtime_diarization --diarization --score-wder --workers 1
```

Default monolingual requests specify the declared language. Multilingual/LID requires
an explicit strategy; do not conflate at-start, continuous, and native behavior.
Run metadata and `_evaluation.jsonl` retain configuration identity, reference/input
hashes, final segment labels, coverage and scores. Missing returned language/speaker
labels stay unavailable. Never score a configured locale as detected language.

### MAI-2 language preflight

1. Require a declared target locale for ordinary MAI-2 testing, using the selected
   language or each manifest row's `requested_language`, and monolingual mode.
   Preserve the reference language independently; do not relabel Cantonese audio
   as Mandarin to satisfy configuration checks.
2. Use the transport's supported language field, not a universal literal locale:
   - REST `fast_mai_2`: the existing monolingual adapter maps the requested locale
     into `locales` (Chinese `zh-CN` becomes `["zh"]`).
   - Direct `realtime_mai_2_api`: set
     `session.audio.input.transcription.language` to `zh` for Chinese, before
     sending audio. Keep `zh-CN` as the benchmark selection locale, not the
     literal API field. Verify supported values for other requested languages;
     do not silently guess a replacement for an unsupported language.
   - SDK `realtime_mai_2`: fixed-language operation remains unverified and is not
     exposed by this adapter. Block an ordinary fixed-language run before paid
     calls; ask for an approved compatible route/subset or a separate explicit
     native-auto experiment. A generic SDK locale assignment is not verification.
3. For the direct API, verify the effective request and the
   `streaming_evidence.negotiated_input.transcription.language` from
   `session.updated`. Require an exact match before audio. Record any rejection,
   mismatch or absent confirmation; never reinterpret it as successful language
   selection. This confirmation is not a detected-language label for LID scoring.
4. An explicitly requested auto comparison omits the API language field; it does
   not send `null` or an invalid locale as a substitute. Freeze the same audio,
   resource/model and pacing across conditions; preserve all pairs and failures.
   Test literal `zh-CN` only as an explicitly requested invalid-value diagnostic,
   not as an ordinary third accuracy condition.

For an existing approved API deployment and user-provided Mandarin manifest:

```powershell
.\.venv\Scripts\python.exe scripts\validate_benchmark.py preflight --manifest D:\audio\mandarin.jsonl --services realtime_mai_2_api --language-mode monolingual --workers 1
.\.venv\Scripts\python.exe run_full.py --manifest D:\audio\mandarin.jsonl --services realtime_mai_2_api --language-mode monolingual --workers 1
```

Each manifest row must declare the selected target, e.g. `"requested_language":
"zh-CN"`. An approved Cantonese-under-Chinese challenge may also request `zh-CN`
while retaining `reference.language: yue-CN`; label it as a challenge, not proof
of fixed-Cantonese support.

Local evidence, 2026-10-14 (East US 2, same direct API deployment):
- [Literal-value test](../../../../results/mai-api-zh-vs-zh-CN_20261014/summary.md):
  `zh` was confirmed and transcribed successfully. Literal `zh-CN` returned
  `invalid_value` for `session.audio.input.transcription.language` before audio.
  It did not silently fall back to auto.
- [All 18 Cantonese challenge pairs](../../../../results/tesla-yue-api-language18_20261014/summary.md):
  auto versus `zh` produced 16 differing normalized transcripts; `zh` had lower
  CER on 10 recordings, higher on 2, and equal on 6. Micro CER was 76.09% versus
  42.39%. Cases Y04/Y13 changed non-Chinese-form outputs to the correct Chinese
  transcript. Some gains were only traditional/simplified character differences.
  These single-repetition challenge results motivate explicit language selection,
  not an absolute language lock, universal improvement or a Mandarin CER claim.

These are local ignored result artifacts; the findings above remain explicit if
the artifacts are absent in another checkout. Keep new run evidence with its own
metadata/verification; never reuse historical confirmation as proof of a new run.

### SDK versus direct MAI Realtime API

Use the same region/resource/key for both transports. Configure the resource's
`MAI_REALTIME_ENDPOINT` and existing `MAI_REALTIME_DEPLOYMENT`; do not create a
deployment or change saved shared configuration without approval. Direct API
availability must be smoke-tested separately from SDK availability.

The current transport harness uses native-auto for both transports. It is an
exception to the ordinary MAI-2 language requirement, not a fixed-language test.
Only after explicit approval of that auto configuration and the paid scope can
a controlled comparison use:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\compare_mai_transports.py --output-dir results\mai-transport-comparison --commands 5 --repetitions 5 --chunk-ms 100 20
```

With one Xiaomi recording, this is 60 paired trials / 120 measured requests,
plus six baseline anchor calls and four excluded warm-up requests. The output
directory must not exist. The harness uses the existing runner and stops before
measurement if anchors or warm-ups fail. It does not resume interrupted trials
or erase failures. Inspect exact per-pair validation files, the experiment
manifest, cumulative trials, final validation, and comparison JSON/Markdown.
First-text timing includes finalized deltas as well as provisional text; do not
confuse it with the standard partial-only first-latency column. Compare matched
chunks and report both audio-relative and call-to-first-text timing, strict WER,
and the explicit API commit versus SDK EOF finalization caveat.

## Resume

Read the selected `_run.json` and CSV. Resume requires one locale, the same
selection/data root/seed/limit/pacing/phrase options, matching schema, and no strict
backup indicating normalization. Legacy CSVs without metadata require a fresh run;
they remain readable for analysis.

```powershell
.\.venv\Scripts\python.exe -X utf8 run_full.py --languages en-GB --services realtime fast_default --max-per-dataset 2 --workers 1 --resume $csv
```

Here `$csv` is an exact path resolved from the user request. Completed rows are
not called again, including existing error rows. Resume fills missing rows; it is
not "retry failed." Recommend a fresh narrowly selected run for errors. Never
delete historical error rows or mix normalized/new strict rows automatically.

## Validate artifacts

After assigning `$csv` to the exact completed CSV path:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\validate_benchmark.py artifacts $csv
```

Use `--rows-only` for a failed/partial run before reports exist. `--allow-legacy`
permits inspection of older artifacts, but `coverage_verified` remains false
without run metadata. Missing legacy sidecars are disclosed, not invented.
The validator is local and does not require Azure credentials.

The JSON distinguishes structural errors, service errors, and warnings. Verify
exact keys, not simply row totals. Correlation-ID warnings do not imply a
successful request had an invented ID. Persist final validation JSON alongside
the run; do not replace valid JSON with mixed stdout from another stage.

The current validator checks report existence/nonzero size, not semantic
completeness. Its `ok` result does not replace the verification gate below.

## Normalize and refresh reports

Diarization, multilingual, and segment-annotated evaluations retain strict token
ownership; loose matching is rejected. Generate their evaluation reports directly.
Their report path is local; historical report generation still makes network context probes.
The versioned renderer includes ASR set-level analysis and ranked cases alongside
annotation evidence. For an explicitly selected multi-run report, use
`benchmark.evaluation.build_combined_evaluation_report(csv_paths, output,
retry_csv_paths=...)`. Pass only measured CSVs in `csv_paths`; smoke is excluded
and retry CSVs stay in a separate appendix. Never merge configurations or replace
an original failed row with a retry. Reverify all regenerated outputs below.

Only normalize the exact completed, validated run. `loose_match.py` creates a
first `.csv.strict` backup and adjusts scoring columns; it does **not** regenerate
reports itself. Never overwrite the strict backup or normalize a still-running run.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\loose_match.py $csv
```

Refresh reports for that one CSV with the same custom data root, if used:

```powershell
$env:ASR_CSV = $csv
$env:ASR_DATA_ROOT = (Resolve-Path .\testdata).Path
@'
import os
from pathlib import Path
import benchmark.datasets_loader as loader
from benchmark.report import build_report
from scripts.error_analysis import main
loader.DATA_ROOT = Path(os.environ["ASR_DATA_ROOT"])
p = Path(os.environ["ASR_CSV"])
build_report(p, p.with_name(p.stem + "_report.md"))
main(p, p.with_name(p.stem + "_error_analysis.md"))
'@ | .\.venv\Scripts\python.exe -X utf8 -
```

Keep variable setup and execution in the same shell call. `build_report` performs
existing network context probes; do not describe report generation as offline.
For error analysis alone, use:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\error_analysis.py $csv $analysis --testdata-dir $dataRoot
```

Resolve `$analysis` to the matching `_error_analysis.md` and `$dataRoot` to the
original data root. No ASR/TTS invocation is needed for these stages.

Revalidate, preserving UTF-8 JSON and checking the exit code:

```powershell
$validation = .\.venv\Scripts\python.exe -X utf8 scripts\validate_benchmark.py artifacts $csv
$status = $LASTEXITCODE
$validationPath = [IO.Path]::Combine([IO.Path]::GetDirectoryName($csv), [IO.Path]::GetFileNameWithoutExtension($csv) + "_validation.json")
[IO.File]::WriteAllText($validationPath, ($validation -join "`n"), [Text.UTF8Encoding]::new($false))
if ($status -ne 0) { throw "Artifact validation failed: $validationPath" }
```

Compare strict/loose scoring by `(dataset, sample_id, service)` and confirm that
transcripts, IDs, error status, and timing evidence were not changed. Never
represent normalized WER as raw strict WER.

## Verification before delivery

This is an explicit agent-performed stage, not a new CLI flag. Use the existing
artifact validator for structural checks and direct artifact inspection/local
calculations for the checks below. Run it after all requested report generation,
normalization or refresh operations. Do not invoke ASR/TTS or `asr-justify`.

### 1. Bind and validate the evidence

- Resolve exact CSVs and their matching run metadata, validation JSON, streaming
  and evaluation sidecars, report/error-analysis files and strict backups when
  present. Record paths and SHA256 hashes for the artifacts actually verified.
  Do not select historical files with broad globs.
- Reconcile exact expected sample/service/configuration keys, duplicates, missing
  and unexpected rows, successes, failures and warnings. Separate smoke, measured
  runs and retries; never replace an original failure with a successful retry.
- Verify selection against the manifest: per-set counts must sum to the selected
  total. When sampling, show known source counts, selected counts, seed and
  allocation policy. State unavailable source counts rather than inventing them.
  Keep challenge subsets and different corpora separately identifiable.
- On failed validation, follow the failure gate: no automatic normalization or
  downstream work. Explicitly requested analysis may report failed/partial data,
  retaining the failed validation and all missing-coverage/service-error evidence.

### 2. Verify report contents, not just files or headings

Read both the main report and error analysis. A concise summary may link to them,
but cannot replace them. Check applicable content even when `_evaluation.jsonl`
selects a specialized renderer:

| Required content | Verification evidence |
|---|---|
| Scope and dataset/set composition | Source, split, selected sets/counts, sampling policy, reference availability and challenge subsets match manifests/inventory. |
| Models and configuration | All selected models are represented; fixed-language versus native/at-start/continuous modes, locales, region, SDK, pacing and chunks match run metadata. Disclose excluded stubs. |
| Accuracy by set and model/configuration | Counts, failures, metric denominators, primary metric and scoring policy are explicit. Chinese uses micro/macro CER and character-normalized SER; whitespace WER is diagnostic. Do not silently pool configurations or corpora. |
| Latency statistics and definitions | Show available First/LBL/UPL mean, median and p90, with valid observation counts. Define timing origins and anchor sources; distinguish REST final-response timing from streaming first text. Explain missing values and evidence-backed negative LBL; never clamp or reinterpret them as a speedup. |
| Best/median/worst examples | For every reference-backed set/model/configuration group, select actual successful cases by the declared primary metric, with a deterministic tie rule. Include sample ID, reference, hypothesis, score/edit counts and resolvable audio links where available. Identify missing audio and groups without scorable rows. |
| Cross-service differences | Include concrete aligned sample comparisons, scores and representative insertion/deletion/substitution or hallucination-like output examples supported by evidence. Mere transcript-equality flags or a dump of all pairs is insufficient. Explicitly distinguish configuration differences; report absence of observed disagreements when applicable. |
| Failures and limitations | Counts and available IDs match validation; include missing timestamps/language labels, unavailable metrics, failed retries, denominator differences and sampling limitations. |
| Reproducibility and navigation | Link the exact input selection, CSV, strict backup if present, metadata, validation, sidecars and detailed reports. State strict/loose policy and preserve source attribution/license for public datasets. |

Do not require invented labels or scores: wDER requires speaker references; LID
accuracy requires reference and returned labels. Without reference text, mark
accuracy/ranked-case checks not applicable and provide descriptive comparisons.
Not applicable requires a concrete reason; a renderer omitting an otherwise
applicable section is a failure, not a reason to waive it.

### 3. Recompute and reconcile

- Recompute every reported aggregate from its exact source rows: numerator,
  denominator, error exclusions, CER micro/macro or WER, SER and latency summaries.
  Compare at the displayed precision using the repository metric definitions.
  Check both set-level and overall tables; macro and micro averages are not
  interchangeable. Verify examples and ranking against their source rows too.
- For normalized artifacts, compare strict/current identities, reference and
  hypothesis text, IDs, error status and timing. Only the permitted normalization
  fields may differ; CER and Chinese SER must remain unchanged. Verify the backup
  exists and was not overwritten. Native/annotated evaluations remain strict.
- Inspect latency provenance before comparative claims. Missing anchors or
  unpaced/incompatible configurations must be disclosed, not treated as comparable
  zero-valued observations. If a timing anomaly cannot be explained from evidence,
  mark the interpretation unverified and block the corresponding claim.

### 4. Persist the result and enforce the gate

Write `<csv-stem>_verification.json` beside each verified run. For a combined
summary, also write `<summary-stem>_verification.json` binding every constituent
run and the summary. Keep these separate from `_validation.json`.

The record must contain:
- `schema_version`, `verified_utc`, and `verification_method` (`agent_artifact_review`);
- `artifacts`: exact paths, roles and SHA256 hashes of inspected sources/outputs;
- `data_validation_ok`, `coverage_verified`, and `report_verified` as separate facts;
- `checks`: each check's name, `passed`/`failed`/`not_applicable` status, concrete
  evidence (artifact location, compared values/counts), and any exclusion reason;
- `discrepancies` and `limitations`, including unverified claims.

`report_verified` may be true only when every applicable report check passes.
It does not turn failed service calls or incomplete coverage into successful data.
Never fill a record with assumed passes, or claim the existing CLI generated it.
An explicit partial/failure analysis can pass report verification only when its
scope, missing data and failed data validation are accurately disclosed.

If verification fails, preserve original measurements and strict backups. Fix
the report generator/template, regenerate only the affected exact reports, rerun
structural validation and content verification, then update the verification
record. If repair is outside the requested scope, stop with the missing sections
or mismatches and label reporting incomplete. Do not silently rerun paid tests.
Final delivery links the full reports and verification record and states execution,
service-error, coverage and report-verification outcomes separately.

## Justify (explicit-only)

Invoke `asr-justify` only when requested. Pass the exact CSV and matching artifacts;
include strict scoring, `_run.json`, validation warnings, and service-tagged words.
The justification skill reads evidence and writes prose; it must not repair data,
rerun services, normalize, or trigger itself upon completion.
