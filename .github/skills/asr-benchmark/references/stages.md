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

Use the same selection flags for preflight and run:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\validate_benchmark.py preflight --languages en-GB --services realtime realtime_mai_2 --max-per-dataset 2 --workers 1
.\.venv\Scripts\python.exe -X utf8 run_full.py --languages en-GB --services realtime realtime_mai_2 --max-per-dataset 2 --workers 1 --tag streaming-smoke
```

Inspect preflight JSON and paid scope before recognition. For a MAI-only run select
`--services realtime_mai_2`; baseline anchors will be absent unless the model
returns usable word timings. Compare the actual availability, not assumptions.

`--no-pace` is appropriate only for functional/accuracy smoke tests, not latency
claims. Explicitly label any such output. Do not silently change user pacing.

`run_full.py` writes CSV, `_run.json` (selection/pacing/SDK and expected keys),
`_words.jsonl` (service-tagged streaming evidence), and `_validation.json`.
It generates `_report.md` and `_error_analysis.md` only after row validation passes.
A failed run retains CSV/evidence/validation and exits nonzero.

The runner does not automatically normalize or justify. Those are separate stages.

## Resume

Read the selected `_run.json` and CSV. Resume requires one locale, the same
selection/data root/seed/limit/pacing/phrase options, matching schema, and no strict
backup indicating normalization. Legacy CSVs without metadata require a fresh run;
they remain readable for analysis.

```powershell
.\.venv\Scripts\python.exe -X utf8 run_full.py --languages en-GB --services realtime realtime_mai_2 --max-per-dataset 2 --workers 1 --resume $csv
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

## Normalize and refresh reports

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

## Justify (explicit-only)

Invoke `asr-justify` only when requested. Pass the exact CSV and matching artifacts;
include strict scoring, `_run.json`, validation warnings, and service-tagged words.
The justification skill reads evidence and writes prose; it must not repair data,
rerun services, normalize, or trigger itself upon completion.
