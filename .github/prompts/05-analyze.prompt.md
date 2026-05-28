---
mode: agent
description: Step 5 — (Re-)generate the error analysis markdown from a benchmark CSV.
---

Generate or regenerate the error analysis markdown file for a benchmark run.
This produces per-(dataset, service) best/median/worst WER samples, detects
`fast_llm` hallucinations, lists top cross-service disagreements, and writes
`[▶]` audio links for manual review.

Run this step **after** applying loose-match normalization (Step 4) to get the
normalized WER numbers in the analysis.

## Generate error analysis for a specific CSV

```bash
python -X utf8 scripts/error_analysis.py \
    results/asr-bench_en-GB_<timestamp>.csv \
    results/asr-bench_en-GB_<timestamp>_error_analysis.md
```

## Regenerate for all CSVs in results/

```bash
python -X utf8 -c "
from pathlib import Path
from scripts.error_analysis import main as ea_main
for csv_path in sorted(Path('results').glob('asr-bench_*.csv')):
    if '.strict' in csv_path.name:
        continue
    ea_path = csv_path.with_name(csv_path.stem + '_error_analysis.md')
    print(f'--- {csv_path.name}')
    ea_main(csv_path, ea_path)
"
```

## Regenerate both report and error analysis

```bash
python -X utf8 -c "
from pathlib import Path
from benchmark.report import build_report
from scripts.error_analysis import main as ea_main
for csv_path in sorted(Path('results').glob('asr-bench_*.csv')):
    if '.strict' in csv_path.name:
        continue
    print(f'--- {csv_path.name}')
    build_report(csv_path, csv_path.with_name(csv_path.stem + '_report.md'))
    ea_main(csv_path, csv_path.with_name(csv_path.stem + '_error_analysis.md'))
"
```

## What the error analysis contains

- **Datasets under test** — summary of each dataset in the run
- **Speech boundaries** — realtime boundary-fix action counts
- **Results table** — aggregate WER/SER/INS/DEL/SUB/LBL/UPL per (dataset, service)
- **Worst errors** — top 10 highest-WER rows with audio links
- **Most common substitution patterns** — word-level ref→hyp substitutions
- **fast_llm hallucinations** — WER ≥ 0.8 with ≤ 1 word overlap (likely language confusion)
- **Top disagreements** — samples where fast_default and realtime WER differ most

## Audio links

The `[▶]` links in the error analysis point to `results/audio/<dataset>/<sample_id>.wav`.
These are written automatically when `error_analysis.py` runs.
Open the markdown file in VS Code (with Markdown Preview) to click-play clips.

## Next step

Proceed to **Step 6** — `.github/prompts/06-justify.prompt.md`
to write a human-readable interpretation of the results.
