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
| Smoke-test / run / compare services | Preflight, requested benchmark, artifact validation |
| End-to-end | Setup check, inventory, missing-audio generation if needed, preflight, paced run, validation, loose matching, refreshed reports, final validation |
| Resume | Inspect exact run metadata and CSV, validate selection, resume, validate |
| Normalize | Exact completed CSV, strict backup, normalization, refreshed reports, validation |
| Analyze / regenerate reports | Exact CSV and matching data root, report/error analysis only |
| Validate results | Artifact validator only; no recognition or TTS |
| Justify | Invoke `asr-justify` only if explicitly requested, including as part of an end-to-end request |

Read [stage procedures](references/stages.md) only for the requested stages.
Do not advance to unrequested stages except prerequisites needed for the request.
Existing numbered prompts remain independent entry points into this procedure.

## Establish scope once

Carry locales, service keys, data/output roots, sample limit, seed, workers,
pacing, phrase settings, and exact CSV paths through the workflow. Ask about
missing consequential choices; do not repeatedly ask for values already supplied.

- Plain end-to-end uses configured locales/services, up to 30 samples per dataset,
  seed 42, four workers, and real-time pacing. Explain selected sample/service counts
  before paid work. A smoke test normally selects one locale and a small explicit
  service subset; it does not authorize a full all-locale run.
- Defaults include `realtime_mai_2` and the unimplemented `whisper_v3` stub.
  Retired `fast_mai` is explicit-only for legacy diagnostics, not a default.
  Warn clearly; do not silently exclude or replace a requested service.
  Offer an explicit supported-service subset when appropriate.
- Check the region supports every selected model, not just streaming. REST
  enhanced modes are not documented in East US 2. Southeast Asia supports both
  REST enhanced mode and MAI streaming (verified 2026-10-10). Prefer one shared
  Speech resource there for combined tests; verify current documentation and
  actual responses. Do not silently change resource, region, or model.
- `realtime_mai_2` is the SDK streaming model, not REST `fast_mai_2`. Requires
  SDK >=1.52.0. Language locking, phrase grammar, segmentation and detailed-word
  options are not enabled for it; unsupported explicit phrase hints fail.
- `--max-per-dataset 0` selects all available samples. Negative limits, duplicate
  services/locales, unknown services, and zero workers are invalid.
- No synthetic regeneration of user recordings and no `--force` without explicit
  instruction. Missing custom audio without synthesis references is a blocker.
- Bind subsequent stages to the exact paths printed by the run, one per locale.
  Never mutate all historical CSVs through a glob. Resolve ambiguous "latest" requests
  by matching locale/tag and asking when more than one run fits.

## Success and failure contract

- Local configuration checks do not prove authentication, region availability, or
  model access. A successful live final response is separate evidence.
- Run the validator, inspect its JSON, and persist the outcome. `ok` requires both
  structural checks and zero service failures. `coverage_verified` separately states
  whether expected sample/service keys were verified.
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
  validation warnings, strict-versus-loose status, pacing caveats, and blockers.
  Do not automatically justify, file issues, commit, or run another paid benchmark.
