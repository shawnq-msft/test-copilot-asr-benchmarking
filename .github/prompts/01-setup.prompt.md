---
mode: agent
description: Set up and validate the selected ASR benchmark environment without running recognition.
---

Use the [asr-benchmark skill](../skills/asr-benchmark/SKILL.md), setup stage only.
Read its [stage procedures](../skills/asr-benchmark/references/stages.md).

Inspect the selected interpreter and dependencies before installing. Preserve any
existing `.env`; never expose keys. Validate configuration for the requested
services, including Speech SDK >=1.52.0 for `realtime_mai_2`.
Distinguish local credential presence from verified remote authentication.
Report readiness/blockers; do not generate audio or start paid recognition.
