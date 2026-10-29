---
mode: agent
description: Analyze an exact ASR CSV or refresh its reports without rerunning services or changing scores.
---

Use the [asr-benchmark skill](../skills/asr-benchmark/SKILL.md), analyze/validate
stages in its [stage procedures](../skills/asr-benchmark/references/stages.md).

Resolve the CSV and original data root. Analyze its existing strict or normalized
scoring state; do not normalize unless requested. Include selected-service
best/median/worst cases and cross-service disagreements, including `realtime_mai_2`.
Disclose service failures, missing audio/IDs/timestamps, and unknown/unpaced provenance.
No recognition or TTS calls are needed. Report generation retains network context
probes, so do not call that operation offline. Do not auto-justify.

Use the fixed classic-v1 main report template through the Python renderer.
Preserve the historical sections, Results columns and latency timeline;
versioned evaluation details are an appendix. Do not rewrite or prepend
agent-authored report tables/notes. Keep detailed error analysis separate.

For versioned speaker/language evaluations, read the matching `_evaluation.jsonl`.
Separate modes/strategies and labeled versus no-reference results. No-reference
comparisons have no wDER/LID accuracy or best-service rankings. Configured locales
are not detected language IDs. This evaluation report path is local; the network
context caveat above applies to historical reports.

Before delivery, perform the skill's
[verification gate](../skills/asr-benchmark/references/stages.md#verification-before-delivery).
Read the generated report/error analysis, verify required contents and recompute
reported aggregates/examples from the exact artifacts. Preserve applicable ASR
analysis when using the specialized evaluation renderer. Persist separate
verification JSON; file existence or artifact-validator success alone cannot
establish report completeness. Do not substitute a summary table for full reports.
