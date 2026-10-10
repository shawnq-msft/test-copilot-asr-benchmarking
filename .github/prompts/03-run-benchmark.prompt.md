---
mode: agent
description: Run an ASR smoke test, selected benchmark, resume, or end-to-end workflow with exact artifact validation.
---

Use the [asr-benchmark skill](../skills/asr-benchmark/SKILL.md) and
[stage procedures](../skills/asr-benchmark/references/stages.md).

Route the actual request: run/smoke/resume does not implicitly authorize the full
pipeline. Explicit end-to-end includes missing-audio generation, paced recognition,
validation, strict-preserving loose matching, refreshed reports, and final validation.
Justification remains explicit-only.

Carry all selection flags through preflight and run. `realtime_mai_2` is included in
defaults; retired `fast_mai` is explicit-only, and the default `whisper_v3` stub must
be disclosed. Verify regional support for both REST enhanced modes and streaming;
Southeast Asia supports both, unlike East US 2 (verified 2026-10-10).
Validate exact sample/service keys and capture the printed paths.
Preserve partial failures and IDs; never claim success from row counts alone.
