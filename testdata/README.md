# testdata/ — Sample Test Set

This directory contains the reference transcriptions for the bundled sample testset.
WAV audio files are **not** committed to git — they must be generated locally using
Azure Text-to-Speech.

## Structure

```
testdata/
├── en-GB/smart-commands/trans.txt   — 30 English smart speaker commands
├── de-DE/smart-commands/trans.txt   — 30 German commands
├── es-ES/smart-commands/trans.txt   — 30 Spanish commands
├── fr-FR/smart-commands/trans.txt   — 30 French commands
└── it-IT/smart-commands/trans.txt   — 30 Italian commands
```

Each `trans.txt` is tab-separated:  `filename.wav<TAB>Reference text`

## Generating WAV files

**Prerequisites:** `.env` file with `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION`.

```bash
# Generate all locales
python scripts/generate_testset.py

# One locale only
python scripts/generate_testset.py --locale en-GB

# One dataset only
python scripts/generate_testset.py --locale en-GB --dataset smart-commands

# Force re-synthesis (overwrite existing WAVs)
python scripts/generate_testset.py --locale en-GB --force
```

The script uses the Azure TTS neural voices listed in `scripts/generate_testset.py`
(`DEFAULT_VOICES`). Override with `--voice` to use a different voice:

```bash
python scripts/generate_testset.py --locale en-GB --voice en-GB-RyanNeural
```

## Adding your own test data

1. Create a directory: `testdata/<locale>/<dataset-name>/`
2. Add a `trans.txt` file (tab-separated or plain-text, one command per line)
3. Place your own WAV files in the same directory **or** run `generate_testset.py`
   to synthesize them from the reference texts
4. Run the benchmark:
   ```bash
   python run_full.py --languages <locale>
   ```

The benchmark auto-discovers all `testdata/<locale>/*/` directories.
