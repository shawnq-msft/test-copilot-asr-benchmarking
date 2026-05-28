---
mode: agent
description: Step 2 — Generate synthetic WAV test samples using Azure TTS.
---

Generate WAV audio files for the benchmark testset using Azure Text-to-Speech.
This reads reference texts from `testdata/<locale>/smart-commands/trans.txt` and
synthesizes 16 kHz mono PCM WAV files using Azure's neural TTS voices.

## Check what's available

```bash
python -c "
from pathlib import Path
root = Path('testdata')
for locale_dir in sorted(root.iterdir()):
    if not locale_dir.is_dir(): continue
    for ds_dir in sorted(locale_dir.iterdir()):
        if not ds_dir.is_dir(): continue
        wav_count = len(list(ds_dir.glob('*.wav')))
        txt_path = ds_dir / 'trans.txt'
        ref_count = sum(1 for l in txt_path.read_text(encoding='utf-8-sig').splitlines() if l.strip() and not l.startswith('#')) if txt_path.exists() else 0
        status = 'READY' if wav_count >= ref_count and ref_count > 0 else f'needs synthesis ({ref_count} refs, {wav_count} wavs)'
        print(f'  {locale_dir.name}/{ds_dir.name}: {status}')
"
```

## Generate audio (choose one)

**All locales at once** (may take a few minutes):
```bash
python scripts/generate_testset.py
```

**One locale** (fastest for testing):
```bash
python scripts/generate_testset.py --locale en-GB
```

**Specific locale + dataset**:
```bash
python scripts/generate_testset.py --locale en-GB --dataset smart-commands
```

**With a specific voice** (overrides the default for that locale):
```bash
python scripts/generate_testset.py --locale en-GB --voice en-GB-RyanNeural
```

**Re-synthesize existing files** (force overwrite):
```bash
python scripts/generate_testset.py --locale en-GB --force
```

## Default voices per locale

| Locale | Default voice |
|--------|--------------|
| en-GB  | en-GB-SoniaNeural |
| de-DE  | de-DE-KatjaNeural |
| es-ES  | es-ES-ElviraNeural |
| fr-FR  | fr-FR-DeniseNeural |
| it-IT  | it-IT-ElsaNeural |

## Verify synthesis succeeded

```bash
python -c "
from benchmark.datasets_loader import list_datasets, iter_samples
datasets = list_datasets()
print(f'Available datasets: {datasets}')
for ds in datasets[:2]:
    try:
        samples = iter_samples(ds, limit=5)
        print(f'  {ds}: {len(samples)} samples loaded OK (durations: {[round(s.duration_s,1) for s in samples]}s)')
    except Exception as e:
        print(f'  {ds}: ERROR - {e}')
"
```

## Next step

Proceed to **Step 3** — `.github/prompts/03-run-benchmark.prompt.md`
