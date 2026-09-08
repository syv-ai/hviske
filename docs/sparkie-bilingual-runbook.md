# Sparkie bilingual runbook

This runbook is for the private `syvai/hviske-v6` Cohere run. It deliberately keeps
training and publication separate.

## Prerequisites and authentication

Install the project with `uv sync --all-extras`, ensure Sparkie has a compatible CUDA
and Transformers installation, and log in without putting credentials in this checkout:

```bash
hf auth login
uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual --cfg job
```

The Hugging Face account must have accepted access to the gated
`CohereLabs/cohere-transcribe-03-2026` model, read access to private
`syvai/p1-transcripts`, and write access to create and upload the private
`syvai/hviske-v6` model. Do not put tokens in this runbook, shell history, manifests,
or Hydra configuration.

Before resolving the preset, set the three private transcript join columns after
checking the private schema:

```bash
export P1_AUDIO_JOIN_COLUMN=...
export P1_TRANSCRIPT_JOIN_COLUMN=...
export P1_TRANSCRIPT_TEXT_COLUMN=...
```

The values above are intentionally unverified placeholders. The p1 audio remains
streaming; only the compact transcript side is indexed.

## Local manifests and validation

Build compact manifests from the existing WAV/VTT directories. This records paths and
cue offsets and does not copy audio:

```bash
uv run python src/scripts/build_vtt_manifest.py \
  --source-dir "$HOME/drtv-asr-dataset/data/drtv" \
  --output "$HOME/drtv-asr-dataset/data/drtv/drtv-manifest.jsonl" \
  --language da
uv run python src/scripts/build_vtt_manifest.py \
  --source-dir "$HOME/drtv-asr-dataset/data/youtube" \
  --output "$HOME/drtv-asr-dataset/data/youtube/youtube-manifest.jsonl" \
  --language da
```

Resolve the fixed Hydra entry point with `--config-name` (rather than trying to pass a
second positional config name), then run the focused checks. The full run validates Hub
access and dataset schemas while it streams, so do not remove the validation step:

```bash
uv run python src/scripts/finetune_asr_model.py \
  --config-name sparkie_bilingual --cfg job
uv run pytest tests/test_sparkie_config.py tests/test_private_hub.py -q
```

The resolved preset has 12 ordered sources and explicit probabilities: 60% Danish and
40% English. It uses only CoRal v3 `read_aloud` and `conversation`; no older CoRal
source is included. Local audio, manifests, Hydra output, checkpoints, and tracking
artefacts are never Hub upload inputs.

## GPU smoke, pilot, and long run

Stop `qwen38-ar` only immediately before the GPU smoke or training process, so another
Sparkie service is not displaced during preparation. First run a 2-10 step smoke:

```bash
# Stop qwen38-ar immediately before this command, not during preparation.
tmux new -s hviske-smoke \
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual max_steps=2 save_steps=2 eval_steps=2'
```

Inspect the resolved logs and checkpoints, then run a bounded pilot (for example
2,000 steps) before committing to the long run. Keep publication disabled; the preset
sets `push_to_hub=false`, `private=true`, `private_only=true`, and disables external
experiment tracking:

```bash
tmux new -s hviske-pilot \
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual max_steps=2000'
```

Launch the approved long run in its own tmux session:

```bash
tmux new -s hviske-v6 \
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual'
```

Monitor with `tmux attach -t hviske-v6`, `nvidia-smi`, and `df -h`. Keep at least the
bounded checkpoint retention configured by the preset and stop if disk space, GPU
memory, or streaming errors become unsafe.

## Explicit private publication

Only after reviewing the pilot/long-run output, publish the saved model as a separate,
explicit command. The command requires `--private`; the uploader refuses public
repositories, creates missing repositories as private, verifies visibility before and
after upload, and uploads only model/tokeniser files plus bilingual model-card metadata:

```bash
uv run python src/scripts/publish_private_model.py \
  models/hviske-v6 syvai/hviske-v6 \
  --finetuned-from CohereLabs/cohere-transcribe-03-2026 \
  --language da --language en --private
```

Never use a public destination or upload raw audio, local manifests, Hydra output,
checkpoints, or tracking artefacts.
