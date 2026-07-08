---
mode: agent
description: Step 1 — Set up the ASR benchmark environment.
---

Set up the ASR benchmark environment by running these steps in order:

## 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

Verify the key packages installed correctly:
```bash
python -c "import azure.cognitiveservices.speech; import rapidfuzz; import soundfile; print('All dependencies OK')"
```

## 2. Configure credentials

If `.env` does not exist, copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

Then open `.env` and fill in the required values:
- `AZURE_SPEECH_KEY` — your Azure Speech resource key
- `AZURE_SPEECH_REGION` — your Azure region (e.g. `eastus`)
- `AZURE_SPEECH_ENDPOINT` — your endpoint URL (e.g. `https://eastus.api.cognitive.microsoft.com`)

For `fast_llm`, `fast_mai`, and `fast_mai_1_5` you may need a separate Speech resource with LLM/MAI preview enabled.
Set `LLM_AZURE_SPEECH_KEY`, `LLM_AZURE_SPEECH_REGION`, `LLM_AZURE_SPEECH_ENDPOINT` if so.

## 3. Validate credentials

```bash
python -c "from benchmark.config import assert_credentials; assert_credentials(); print('Credentials OK')"
```

If this fails, check that `.env` exists and contains `AZURE_SPEECH_KEY`.

## 4. Check testdata is present

```bash
python -c "
from benchmark.datasets_loader import list_datasets
datasets = list_datasets()
if datasets:
    print(f'Found {len(datasets)} datasets: {datasets}')
else:
    print('No datasets found. Run prompt 02-generate-testset to create test audio.')
"
```

## Summary

After completing these steps you are ready to:
- Proceed to **Step 2** (`.github/prompts/02-generate-testset.prompt.md`) to generate test audio
- Or if you already have audio files, jump to **Step 3** (`.github/prompts/03-run-benchmark.prompt.md`)
