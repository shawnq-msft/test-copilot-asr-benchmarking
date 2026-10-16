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
defaults; deprecated `fast_mai` has been removed, and the default `whisper_v3` stub must
be disclosed. Verify regional support for both REST enhanced modes and streaming;
Southeast Asia supports both, unlike East US 2 (verified 2026-10-10).
Validate exact sample/service keys and capture the printed paths.
Preserve partial failures and IDs; never claim success from row counts alone.

Default to monolingual; only use multilingual/LID on explicit request, preserving
at-start/continuous/native distinctions. Reject unsupported combinations (including
the current native-only MAI SDK adapter in a monolingual run) before calls.
MAI-2 ordinary tests must explicitly set the target language, not merely label
the dataset locale. For the direct API, Chinese selection `zh-CN` must send
`language="zh"` and verify matching `session.updated`; literal API `zh-CN` is
rejected. Do not silently switch the SDK to auto; ask for an approved compatible
route or explicit auto diagnostic. Follow the skill's MAI-2 language gate.
Diarization wDER requires
user-provided speaker references; without them compare outputs without accuracy claims.
Do not generate evaluation audio or normalize diarization/multilingual annotations.
