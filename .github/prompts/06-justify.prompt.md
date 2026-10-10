---
mode: agent
description: Explicitly invoke asr-justify for an exact benchmark run and its matching evidence.
---

Invoke the [asr-justify skill](../skills/asr-justify/SKILL.md) for the requested CSV.
Resolve ambiguous run selection before writing. Include the report, error analysis,
strict backup when present, run metadata, validation outcome, and service-tagged
word/streaming evidence. Missing optional artifacts must be disclosed.

Derive service rankings from actual results, including `realtime_mai_2` when present.
Do not make latency comparisons for unpaced runs or incompatible anchors.
This step reads evidence and writes justification only; it does not rerun,
normalize, or repair artifacts.
