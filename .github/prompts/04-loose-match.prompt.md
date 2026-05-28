---
mode: agent
description: Step 4 — Apply loose-match WER normalization to a benchmark CSV.
---

Apply automatic WER normalization to catch systematic false-positive errors caused
by formatting differences rather than genuine recognition mistakes:
- Compound-word splits: `stummschalten` ↔ `stumm schalten`
- Alphanumeric spacing: `2D` ↔ `2 D`
- Degree/unit variants: `20°` ↔ `20 degrees` ↔ `20 gradi`

This zeroes out WER on affected rows, backs up the original to `.csv.strict`,
then regenerates the report and error analysis markdown files.

## Find the CSV to normalize

```bash
python -c "
from pathlib import Path
csvs = sorted(Path('results').glob('asr-bench_*.csv'))
csvs = [c for c in csvs if '.strict' not in c.name]
for c in csvs[-5:]:
    import csv as csvmod
    rows = list(csvmod.DictReader(open(c, encoding='utf-8')))
    wers = [float(r['wer']) for r in rows if r['wer']]
    mean_wer = sum(wers)/len(wers) if wers else 0
    print(f'{c.name}  rows={len(rows)}  mean_wer={mean_wer:.3f}')
"
```

## Apply loose matching

**One CSV:**
```bash
python -X utf8 scripts/loose_match.py results/asr-bench_en-GB_<timestamp>.csv
```

**All CSVs in results/**:
```bash
python -X utf8 -c "
from pathlib import Path
from scripts.loose_match import main
for csv_path in sorted(Path('results').glob('asr-bench_*.csv')):
    if '.strict' in csv_path.name:
        continue
    print(f'--- {csv_path.name}')
    main(csv_path)
"
```

## Regenerate reports after normalization

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

## Verify the effect

```bash
python -c "
from pathlib import Path
import csv as csvmod
for strict in sorted(Path('results').glob('*.csv.strict')):
    loose = strict.with_suffix('')  # remove .strict
    if not loose.exists(): continue
    strict_rows = {(r['dataset'], r['sample_id'], r['service']): r
                   for r in csvmod.DictReader(open(strict, encoding='utf-8'))}
    loose_rows  = {(r['dataset'], r['sample_id'], r['service']): r
                   for r in csvmod.DictReader(open(loose,  encoding='utf-8'))}
    changed = [(k, strict_rows[k]['wer'], loose_rows[k]['wer'])
               for k in strict_rows
               if strict_rows[k]['wer'] != loose_rows[k].get('wer','')]
    print(f'{loose.name}: {len(changed)} rows normalized')
    for k, old, new in changed[:5]:
        ds, sid, svc = k
        print(f'  {svc}/{sid}: WER {old} -> {new}')
"
```

## Next step

Proceed to **Step 5** — `.github/prompts/05-analyze.prompt.md`
or skip to **Step 6** — `.github/prompts/06-justify.prompt.md`
