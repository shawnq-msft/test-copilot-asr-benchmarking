---
mode: agent
description: Generate only missing requested ASR test audio and validate exact reference/audio coverage.
---

Use the [asr-benchmark skill](../skills/asr-benchmark/SKILL.md), inventory/generation
stage, and its [stage procedures](../skills/asr-benchmark/references/stages.md).

Preserve the requested locale, dataset, data root and voice. Reuse existing audio;
never force-overwrite custom recordings. Verify filenames against references and
decode generated WAVs rather than comparing counts. Stop on synthesis failures.
Do not start recognition or mutate results unless separately requested.
