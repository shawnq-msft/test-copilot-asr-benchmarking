---
mode: agent
description: Step 6 — Write a human justification of benchmark results using the asr-justify skill.
---

Produce a human-readable interpretation of a completed benchmark run by invoking
the `asr-justify` skill defined in `.github/skills/asr-justify/SKILL.md`.

Read that skill file now and follow its procedure precisely.

## Inputs

Given a run tag like `asr-bench_en-GB_20260527_120000`, the input artifacts are:

| File | Contents |
|------|----------|
| `results/<run>.csv` | Per-sample, per-service metrics |
| `results/<run>_report.md` | Aggregate WER/latency table |
| `results/<run>_error_analysis.md` | Best/worst/hallucinations with audio links |
| `results/<run>_words.jsonl` | Per-word timestamps (realtime samples) |
| `results/<run>.csv.strict` | Pre-normalization backup (if loose-match was applied) |

## How to invoke

Tell me which run to justify, for example:
- "justify results/asr-bench_en-GB_<timestamp>.csv"
- "write justification for the latest en-GB run"

I will then:
1. Read the skill file at `.github/skills/asr-justify/SKILL.md`
2. Read the four input artifacts for that run
3. Follow the 9-step justification procedure in the skill
4. Write the output to `results/<run>_justification.md`

## Finding the latest run

```bash
python -c "
from pathlib import Path
csvs = sorted(Path('results').glob('asr-bench_*.csv'))
csvs = [c for c in csvs if '.strict' not in c.name]
for c in csvs[-3:]:
    import csv as csvmod
    rows = list(csvmod.DictReader(open(c, encoding='utf-8')))
    wers = [float(r['wer']) for r in rows if r['wer']]
    print(f'{c.stem}  ({len(rows)} rows, mean WER {sum(wers)/len(wers):.3f})')
"
```

## Skill reference

Full procedure and style guide: `.github/skills/asr-justify/SKILL.md`

The skill output (`_justification.md`) is a prose document — not bullet soup.
It covers:
- **Headline numbers** — recap of the aggregate table
- **One-line take** — best service for accuracy, best for latency, biggest caveat
- **Where WER is misleading** — artifacts, short refs, style deltas, compound words
- **fast_llm hallucinations** — count, language confusion pattern
- **Genuine service disagreements** — residue after removing artifacts
- **INS/DEL/SUB shape** — what kind of errors each engine makes
- **Latency** — QA-passed numbers, per-service interpretation
- **Recommendations** — 4–6 actionable bullets
