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
