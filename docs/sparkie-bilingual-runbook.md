# Sparkie bilingual runbook

This runbook is for the private `syvai/hviske-v6` Cohere run. Training and
publication are deliberately separate.

## Prerequisites and authentication

Install the project, confirm the pinned preset resolves, and authenticate without
putting credentials in this checkout:

```bash
uv sync --python 3.11 --all-extras
hf auth login
hf auth whoami
```

The Hugging Face account must have accepted access to
`CohereLabs/cohere-transcribe-03-2026`, read access to private `syvai/p1` and
`syvai/p1-transcripts`, and read access to every public source. It also needs write
access to create and upload the private `syvai/hviske-v6` model. Never put tokens in
this runbook, shell history, manifests, or Hydra configuration.

Set the exact private transcript revision and join columns after checking the pinned
private schema:

```bash
export P1_TRANSCRIPT_REVISION=<full-p1-transcripts-commit-sha>
export P1_AUDIO_JOIN_COLUMN=<p1-audio-key-column>
export P1_TRANSCRIPT_JOIN_COLUMN=<p1-transcript-key-column>
export P1_TRANSCRIPT_TEXT_COLUMN=<p1-transcript-text-column>
```

All four values are required. The p1 audio side remains streaming; the bounded
preflight also streams the transcript side, while training indexes the compact
transcript side for the join.

## Fixed production mix

Do not reorder or rebalance the preset. Its 15 streams are sampled as follows:

| Language | Source | Configuration and split | Probability |
| --- | --- | --- | ---: |
| Danish | `syvai/p1` + `syvai/p1-transcripts` | `train` joined by the configured key | 0.075 |
| Danish | local DRTV manifest | `train` | 0.075 |
| Danish | local YouTube manifest | `train` | 0.075 |
| Danish | `CoRal-project/coral-v3` | `read_aloud` / `train` | 0.075 |
| Danish | `CoRal-project/coral-v3` | `conversation` / `train` | 0.075 |
| Danish | `alexandrainst/ftspeech` | `train` | 0.075 |
| Danish | `alexandrainst/nota` | `train` | 0.075 |
| Danish | `alexandrainst/nst-da` | `train` | 0.075 |
| English | `MLCommons/peoples_speech` | `clean` / `train` | 0.16 |
| English | `edinburghcstr/ami` | `sdm` / `train` | 0.04 |
| English | `edinburghcstr/ami` | `ihm` / `train` | 0.03 |
| English | `facebook/voxpopuli` | `en` / `train` | 0.09 |
| English | `openslr/librispeech_asr` | `clean` / `train.100` | 0.01 |
| English | `openslr/librispeech_asr` | `clean` / `train.360` | 0.025 |
| English | `openslr/librispeech_asr` | `other` / `train.500` | 0.045 |

The total is 60% Danish and 40% English. FLEURS `en_us` is evaluation-only.
Common Voice, GigaSpeech, SPGISpeech, older CoRal data, and CoRal TTS are not part of
this run.

## Local manifests and bounded preflight

Build compact manifests from the existing WAV/VTT directories. This records paths and
cue offsets without copying audio:

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

Run the preflight while `qwen38-ar` is still serving. It resolves every required
environment variable, checks Hub authentication and gated model access, validates all
pinned dataset coordinates and schemas, validates each local manifest and its first
referenced WAV, and consumes at most one streamed row per training, transcript, and
validation source. It does not load the ASR model, download background noise, or start
training.

```bash
uv run python src/scripts/finetune_asr_model.py \
  --config-name sparkie_bilingual --cfg job
uv run python src/scripts/preflight_finetuning_data.py \
  --config-name sparkie_bilingual
uv run pytest tests/test_sparkie_config.py \
  tests/test_preflight_finetuning_data.py -q
```

Do not proceed if any preflight check fails. Fix access, schema, revision, or local-file
errors and rerun the complete preflight.

## GPU smoke, pilot, and full run

Stop `qwen38-ar` only after the preflight passes and immediately before starting the GPU
smoke. Do not displace it during data preparation. Run each training phase in its own
tmux session from the repository root.

Start with a two-step smoke:

```bash
# Stop qwen38-ar now, immediately before launching this session.
tmux new -s hviske-smoke \
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual max_steps=2 save_steps=2 eval_steps=2'
```

Inspect the resolved log, GPU memory, and checkpoint before running a bounded 2,000-step
pilot. Publication remains disabled in the preset.

```bash
tmux new -s hviske-pilot \
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual max_steps=2000'
```

Review pilot loss, validation metrics, throughput, checkpoint resumption, and disk use.
Only then launch the approved full run:

```bash
tmux new -s hviske-v6 \
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual'
```

Monitor with:

```bash
tmux attach -t hviske-v6
nvidia-smi
df -h
```

The preset retains at most three checkpoints. Stop the run if free disk space, GPU
memory, temperatures, or repeated streaming failures become unsafe. Preserve enough
free space for the next checkpoint and final model save.

## Explicit private publication

Publish only after reviewing the selected pilot or full-run checkpoint. The separate
command requires `--private`, refuses a public destination, verifies private visibility
before and after upload, and stages only recognised model, tokeniser, processor, and
model-card files. Trainer-side publication remains disabled.

Supply the complete set of Hub-backed training sources as model-card metadata. The two
local Danish manifest sources must additionally be described in the model-card review;
their audio and manifests must never be uploaded.

```bash
uv run python src/scripts/publish_private_model.py \
  models/hviske-v6 syvai/hviske-v6 \
  --finetuned-from CohereLabs/cohere-transcribe-03-2026 \
  --language da --language en --private \
  --training-dataset-id syvai/p1 \
  --training-dataset-id syvai/p1-transcripts \
  --training-dataset-id CoRal-project/coral-v3 \
  --training-dataset-id alexandrainst/ftspeech \
  --training-dataset-id alexandrainst/nota \
  --training-dataset-id alexandrainst/nst-da \
  --training-dataset-id MLCommons/peoples_speech \
  --training-dataset-id edinburghcstr/ami \
  --training-dataset-id facebook/voxpopuli \
  --training-dataset-id openslr/librispeech_asr \
  --evaluation-status 'Pilot and full-run evaluation reviewed before publication.'
```

Never use a public destination or upload raw audio, local manifests, Hydra output,
checkpoints, caches, or experiment-tracking artefacts.
