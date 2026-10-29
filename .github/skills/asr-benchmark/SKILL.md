---
name: asr-benchmark
description: Run Azure ASR benchmark workflows end-to-end or perform a specific stage: setup, synthesize missing test audio, smoke-test, compare selected services including MAI-Transcribe-2-Streaming, resume, validate, normalize WER, or analyze an existing run. Preserve exact scope and validate artifacts before claiming success.
---

# ASR benchmark workflow

Use this skill for benchmark operations, not for general software testing.
The existing Python commands are the execution engine; do not write a second runner.

## Route the request

| Intent | Stages |
|---|---|
| Set up / check environment | Setup only |
| Generate audio | Inventory, missing-audio generation, audio validation |
| Smoke-test / run / compare services | Preflight, requested benchmark, artifact validation, verification of generated reports |
| End-to-end | Setup check, inventory, missing-audio generation if needed, preflight, paced run, validation, permitted loose matching, refreshed reports, final validation and verification |
| Resume | Inspect exact run metadata and CSV, validate selection, resume, validate, verify refreshed reports |
| Normalize | Exact completed CSV, strict backup, permitted normalization, refreshed reports, validation and verification |
| Analyze / regenerate reports | Exact CSV and matching data root, report/error analysis, verification; no recognition |
| Validate results | Artifact validation and verification of requested existing reports; no recognition or TTS |
| Justify | Invoke `asr-justify` only if explicitly requested, including as part of an end-to-end request |

Read [stage procedures](references/stages.md) only for the requested stages.
Do not advance to unrequested stages except prerequisites needed for the request.
Existing numbered prompts remain independent entry points into this procedure.

## Fixed report rendering

Generate main reports with `benchmark.report.build_report` (or
`benchmark.evaluation.build_evaluation_report` for versioned runs). Both use the
fixed classic-v1 main template: Endpoints under test, Datasets under test,
Results, Speech boundaries, and Latency definitions with the original timeline.
Keep the original Results columns and mean/p90 display. Versioned evidence adds
an evaluation appendix, not an alternative main layout. Error analysis and
explicit combined analysis retain their detailed reference-aware layout.
Do not hand-author, replace or prepend tables/notes in the main report. Fix the
renderer and its regression tests when necessary; keep interpretation in a
separately requested justification. Never change pacing or measurements to fix
report presentation.

## Establish scope once

Language mode defaults to monolingual (specified language), never automatic LID.
For multilingual requests require an explicit at-start, continuous, or native
strategy, plus candidate locales where required. Unsupported selected combinations
must fail preflight; do not silently skip or change modes. The MAI streaming SDK
adapter maps monolingual locales to base language codes (`zh-CN` -> `zh`).
Its presence in defaults does not authorize an automatic-language exception.

### MAI-2: explicit target language is mandatory

For ordinary MAI-2 smoke, accuracy and end-to-end tests, require an explicitly
selected target locale and monolingual mode. A dataset locale alone is not proof
that the service received a language setting. Never silently omit the setting or
switch to native-auto to make a selected model run.

- `fast_mai_2`: retain the requested locale in selection/metadata and use the
  existing monolingual adapter's REST `locales` mapping.
- `realtime_mai_2_api`: set `session.audio.input.transcription.language` before
  the first audio append. Chinese selection `zh-CN` maps to API **`zh`**.
  Literal API **`zh-CN` is invalid**: live tests on 2026-10-14 returned
  `invalid_value` before audio, not an automatic-language fallback.
  Require `session.updated` to confirm the requested API language; a missing,
  rewritten or rejected value blocks fixed-language testing.
- `realtime_mai_2` (Speech SDK): set `speech_recognition_language` to the base
  language code, not the full locale. A paced SDK 1.52.0 live check on 2026-10-22
  verified outgoing `speech.context` with `phraseDetection.language="zh"` and a
  successful MAI final transcript. Preserve the requested locale and actual code
  in streaming evidence. This verifies sent/accepted configuration, not an
  absolute language lock. Never silently fall back to automatic detection.
- Unset-language/native-auto runs are exceptions only for explicitly requested
  multilingual, language-setting or transport comparisons. Record them as separate
  configurations, not as satisfying this fixed-language rule.

Preserve the requested locale, actual API language, returned session configuration
and reference language separately. `zh` is not a guarantee of correct transcription,
exclusive Chinese output, or a Cantonese-specific lock. See
[MAI-2 language procedure and evidence](references/stages.md#mai-2-language-preflight).

### Other scope requirements

For mixed-audio speaker evaluation use `--diarization` and supported REST services
or optional `realtime_diarization`. wDER additionally requires `--score-wder` and
a user-provided JSONL manifest with reference speaker labels. Do not synthesize
evaluation data or infer labels from endpoint output. Without ground truth, report
only cross-endpoint differences; no wDER, LID accuracy, or best-service claims.
Manifest selection is exact, independent of legacy directory/locale/limit flags.
Use final segment evidence, preserve mode/configuration identity and data hashes,
and distinguish returned language IDs from requested language hints.
No automatic loose matching on diarization, multilingual, or annotated evaluations.

Carry locales, service keys, data/output roots, sample limit, seed, workers,
pacing, phrase settings, and exact CSV paths through the workflow. Ask about
missing consequential choices; do not repeatedly ask for values already supplied.

- Plain end-to-end uses configured locales/services, up to 30 samples per dataset,
  seed 42, four workers, and real-time pacing. Explain selected sample/service counts
  before paid work. A smoke test normally selects one locale and a small explicit
  service subset; it does not authorize a full all-locale run.
- Defaults include `realtime_mai_2` and the unimplemented `whisper_v3` stub.
  Deprecated `fast_mai` has been removed; reject requests to run it.
  Warn clearly; do not silently exclude or replace a requested service.
  Offer an explicit supported-service subset when appropriate.
- Check the region supports every selected model, not just streaming. REST
  enhanced modes are not documented in East US 2. Southeast Asia supports both
  REST enhanced mode and MAI streaming (verified 2026-10-10). Prefer one shared
  Speech resource there for combined tests; verify current documentation and
  actual responses. Do not silently change resource, region, or model.
- `realtime_mai_2` is the SDK streaming model, not REST `fast_mai_2`. Requires
  SDK >=1.52.0. Monolingual mode sends a base language code; phrase grammar,
  segmentation and detailed-word options are not enabled; explicit phrase hints fail. Apply
  the mandatory language gate above before selecting it for ordinary tests.
- `azure_whisper` is opt-in Azure OpenAI Whisper, not the `whisper_v3` stub.
  Verify the deployed model/version and matching endpoint/key; retain these in
  metadata. Do not substitute Whisper `001` for v3 without user approval.
- `realtime_mai_2_api` is an opt-in direct WebSocket adapter, not a default or an
  SDK replacement. It uses the primary Speech key and requires the same resource's
  MAI Realtime API endpoint plus a deployed model (`MAI_REALTIME_ENDPOINT`,
  `MAI_REALTIME_DEPLOYMENT`). SDK availability does not prove this route exists.
  For transport comparisons, use `scripts/compare_mai_transports.py` with a new
  exact output directory and agreed sample/repetition scope. It uses the existing
  runner, frozen baseline anchors, excluded warm-ups, alternating order, and
  matched paced chunks. Compare call-to-first-text as well as audio-relative
  timing; API explicit commit and SDK EOF have different finalization semantics.
  Keep failures and strict WER, and never infer a general speedup from one run.
- `--max-per-dataset 0` selects all available samples. Negative limits, duplicate
  services/locales, unknown services, and zero workers are invalid.
- No synthetic regeneration of user recordings and no `--force` without explicit
  instruction. Missing custom audio without synthesis references is a blocker.
- Bind subsequent stages to the exact paths printed by the run, one per locale.
  Never mutate all historical CSVs through a glob. Resolve ambiguous "latest" requests
  by matching locale/tag and asking when more than one run fits.

## Verification gate (mandatory before delivery)

Follow [report verification](references/stages.md#verification-before-delivery)
after every report generation or refresh, including manifest-based and
speaker/language evaluation paths. A summary table, nonempty report file, or
successful CSV validation is not proof that the reporting contract is satisfied.

- Verify data coverage, report completeness, and numerical consistency separately.
  Read the actual reports and reconcile their values and examples with the exact
  CSV, strict backup, metadata, validation and sidecars; headings alone do not pass.
- For MAI-2, verify the effective language setting, not just the locale column.
  API runs require matching `session.updated` language evidence. Explicit auto
  exceptions and configuration rejections must remain visibly separate.
- Require dataset/set composition, applicable per-set/model results, latency
  definitions/statistics, best/median/worst reference-backed cases, meaningful
  cross-service differences, failures and provenance. A specialized evaluation
  template must preserve these applicable ASR sections, not replace them with
  speaker/language evidence alone.
- Persist a separate `_verification.json` with check-level status, evidence,
  discrepancies and justified not-applicable items. This is an agent-performed
  verification record, not an output currently produced by the artifact validator.
  Do not overwrite `_validation.json` or change scores to make verification pass.
- Missing required content or mismatched values blocks "report complete". Repair
  only the reporting path, regenerate from existing evidence, then verify again.
  If blocked, deliver the exact gap and label the report incomplete; do not rerun
  paid recognition, hide failures, or replace full reports with a short summary.

## Success and failure contract

- Local configuration checks do not prove authentication, region availability, or
  model access. A successful live final response is separate evidence.
- Run the validator, inspect its JSON, and persist the outcome. `ok` requires both
  structural checks and zero service failures. `coverage_verified` separately states
  whether expected sample/service keys were verified.
  Report verification is a separate gate: report files being present/nonempty
  does not verify their required contents or calculations.
- Preserve partial outputs and show failures with available request/session IDs.
  Missing IDs stay explicitly missing. Never print keys or authentication headers.
- Stop normalization and automatic downstream work when a run fails validation.
  An explicit analyze request may inspect failed results without presenting them
  as a successful benchmark.
- Never call an unpaced run a latency benchmark. Check `_run.json` and `upl_anchor`.
  Missing timestamps/partials are unavailable, not zero; self-anchored latency is
  not equivalent to shared baseline-realtime anchoring.
- Quality thresholds are user-defined. Do not invent a universal acceptable WER
  or latency threshold or claim one is enforced by the structural validator.
- Finish with exact artifact links, selection, expected/actual rows, service errors,
  validation warnings, strict-versus-loose status, pacing caveats, verification
  outcome and blockers. Distinguish execution complete, service success, and report
  verified; an accurately verified failure report is not an all-success benchmark.
  Do not automatically justify, file issues, commit, or run another paid benchmark.
