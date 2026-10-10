---
mode: agent
description: Normalize one explicitly selected completed ASR run, preserve strict scoring, and refresh its reports.
---

Use the [asr-benchmark skill](../skills/asr-benchmark/SKILL.md), normalize/refresh
stages in its [stage procedures](../skills/asr-benchmark/references/stages.md).

Resolve one exact CSV (or an explicitly requested batch), never a broad historical
glob. Validate completion first. Preserve `.csv.strict`, run loose matching, then
regenerate both report and error analysis once with the original data root.
The normalization script itself does not regenerate reports.
Validate row identities and unchanged non-scoring evidence and persist the outcome.
Do not resume into normalized results or invoke justification automatically.
