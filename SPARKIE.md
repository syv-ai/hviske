# Sparkie bilingual runbook

This runbook is for the private `syvai/hviske-v6.0` Cohere run. Training and
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
`CohereLabs/cohere-transcribe-03-2026`, read access to the manually gated
`syvai/p1-segments` dataset, and read access to every public source. It also needs
write access to create and upload the private `syvai/hviske-v6.0` model. Never put
tokens in this runbook, shell history, manifests, or Hydra configuration.

Set the two immutable data-gate revisions:

```bash
export P1_SEGMENTS_REVISION=<full-40-hex-p1-segments-commit-sha>
# Use the completed immutable overlay commit; branches and short SHAs are rejected.
export HVISKE_OVERLAY_REVISION=<full-40-hex-overlay-commit-sha>
```

Both values are required. P1 is loaded directly as a normal Hub dataset with `audio`
and `text` columns; no transcript join or join-column environment variables are used.
The source revision is intentionally not hardcoded while the manually gated dataset is
still being finalised. Branches, short SHAs, and non-hex revisions are rejected.

## Fixed production mix

Do not reorder or rebalance the preset. Its 16 streams are sampled as follows:

| Language | Source | Configuration and split | Probability |
| --- | --- | --- | ---: |
| Danish | `syvai/p1-segments` | `train` | 0.102857 |
| Danish | local DRTV manifest | `train` | 0.128571 |
| Danish | local YouTube manifest | `train` | 0.09 |
| Danish | `syvai/danish-asr-unified` + v5-tiny overlay | `source=coral_read_aloud` / `train` | 0.017143 |
| Danish | `syvai/danish-asr-unified` + v5-tiny overlay | `source=coral_conversation` / `train` | 0.128572 |
| Danish | `syvai/danish-asr-unified` + v5-tiny overlay | `source=ftspeech` / `train` | 0.040909 |
| Danish | `syvai/danish-asr-unified` + v5-tiny overlay | `source=nota` / `train` | 0.021428 |
| Danish | `syvai/danish-asr-unified` + v5-tiny overlay | `source=nst` / `train` | 0.021429 |
| Danish | `syvai/danish-asr-unified` + v5-tiny overlay | `source=voxpopuli` / `train` | 0.049091 |
| English | `MLCommons/peoples_speech` | `clean` / `train` | 0.16 |
| English | `edinburghcstr/ami` | `sdm` / `train` | 0.04 |
| English | `edinburghcstr/ami` | `ihm` / `train` | 0.03 |
| English | `facebook/voxpopuli` | `en` / `train` | 0.09 |
| English | `openslr/librispeech_asr` | `clean` / `train.100` | 0.01 |
| English | `openslr/librispeech_asr` | `clean` / `train.360` | 0.025 |
| English | `openslr/librispeech_asr` | `other` / `train.500` | 0.045 |

These are source-sampling probabilities chosen for style and acoustic balance, not
weights proportional to row count. Large formal or read-aloud corpora are deliberately
capped. The Danish total is exactly 60% and the English total 40%. FLEURS `en_us` is
evaluation-only. The unified Danish repository is filtered into six separately weighted
streams and uses the `HVISKE_OVERLAY_REVISION` v5-tiny metadata overlay; its rows are not
loaded directly from the obsolete source repositories. Common Voice,
GigaSpeech, SPGISpeech, older CoRal data, and CoRal TTS are not part of this run.

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
referenced WAV, and consumes one direct P1 example. It fully consumes every overlay to
prove positional length, duplicate, equality, and action/text integrity before lazy
training starts. Overlay preflight casts audio with decoding disabled, so this gate
never decodes the audio stream. The preflight does not load the ASR model, download
background noise, or start training.

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
tmux new-session -d -s hviske-smoke \
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual max_steps=2 save_steps=2 eval_steps=2 max_validation_samples_per_dataset=32'
```

Inspect the resolved log, GPU memory, and checkpoint before running a bounded 2,000-step
pilot. Publication remains disabled in the preset.

```bash
tmux new-session -d -s hviske-pilot \
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual max_steps=2000 max_validation_samples_per_dataset=256'
```

Review pilot loss, validation metrics, throughput, checkpoint resumption, and disk use.
Only then launch the approved full run:

```bash
tmux new-session -d -s hviske-v6-0 \
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual max_validation_samples_per_dataset=1000'
```

These command-only sessions exit when the job finishes. Attach or capture logs while a
job is running; completed ephemeral sessions are not available afterwards.

Monitor a running job with:

```bash
tmux attach -t hviske-v6-0
nvidia-smi
df -h
```

The preset retains at most three checkpoints. Stop the run if free disk space, GPU
memory, temperatures, or repeated streaming failures become unsafe. Preserve enough
free space for the next checkpoint and final model save.

## Olmix anchor calibration matrix

**Blocked: do not run the current Olmix launcher or anchor configurations.**

The existing read-heavy and spontaneous-heavy anchors classify P1, FTSpeech, and
VoxPopuli incorrectly. Their tests validate that stale taxonomy rather than the
corrected one. They are retained only as implementation history and must not be used
for GPU runs.

After the v6.0 source probabilities are frozen:

1. regenerate all calibration anchors with P1 under broadcast/conversation and
   FTSpeech and VoxPopuli under parliament;
2. decide how the mixed People's Speech corpus maps into the optimisation domains;
3. update the source map, launcher, and tests;
4. review the resolved probabilities and 60/40 language totals; and
5. restore reviewed smoke and matrix commands to this runbook.

Follow [`docs/olmix-benchmark-plan.md`](docs/olmix-benchmark-plan.md) for the experiment
design. No Olmix command is approved until this blocked section is replaced.

## Explicit private publication

Publish only after reviewing the selected pilot or full-run checkpoint. The separate
command requires `--private`, refuses a public destination, verifies private visibility
immediately before and after the single staged upload, and stages only a complete
reloadable Cohere package plus a strict model card. Trainer-side publication remains
disabled. It resolves the preset automatically, including both local Danish manifest
sources (without their paths), every Hub source, revisions, splits, languages,
probabilities, licence, base model, and evaluation status. A reviewed card may be
supplied only after it has been checked against that metadata; local audio and manifests
must never be uploaded.

```bash
uv run python src/scripts/publish_private_model.py \
  models/hviske-v6.0 syvai/hviske-v6.0 --private \
  --evaluation-status 'Pilot and full-run evaluation reviewed before publication.'
```

Never use a public destination or upload raw audio, local manifests, Hydra output,
checkpoints, caches, or experiment-tracking artefacts.
