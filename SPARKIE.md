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
preflight builds the complete compact transcript index and consumes one joined audio
example, while training uses the same index-plus-streamed-audio join path.

## Fixed production mix

Do not reorder or rebalance the preset. Its 16 streams are sampled as follows:

| Language | Source | Configuration and split | Probability |
| --- | --- | --- | ---: |
| Danish | `syvai/p1` + `syvai/p1-transcripts` | `train` joined by the configured key | 0.08 |
| Danish | local DRTV manifest | `train` | 0.10 |
| Danish | local YouTube manifest | `train` | 0.07 |
| Danish | `CoRal-project/coral-v3` | `read_aloud` / `train` | 0.04 |
| Danish | `CoRal-project/coral-v3` | `conversation` / `train` | 0.10 |
| Danish | `alexandrainst/ftspeech` | `train` | 0.05 |
| Danish | `alexandrainst/nota` | `train` | 0.05 |
| Danish | `alexandrainst/nst-da` | `train` | 0.05 |
| Danish | `syvai/danish-asr-unified` | `default` / `train`, `source=voxpopuli` | 0.06 |
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
evaluation-only. The unified Danish repository is filtered to `source=voxpopuli`, so its
ftspeech, CoRal, NST, and Nota rows do not enter through this stream. Common Voice,
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
referenced WAV, and consumes one joined P1 example. It does not prove complete P1 audio
key coverage without scanning the full 1.9 TB audio stream; the compact transcript index
is intentionally built in memory. It does not load the ASR model, download background
noise, or start training.

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
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual max_steps=2 save_steps=2 eval_steps=2 max_validation_samples_per_dataset=32'
```

Inspect the resolved log, GPU memory, and checkpoint before running a bounded 2,000-step
pilot. Publication remains disabled in the preset.

```bash
tmux new -s hviske-pilot \
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual max_steps=2000 max_validation_samples_per_dataset=256'
```

Review pilot loss, validation metrics, throughput, checkpoint resumption, and disk use.
Only then launch the approved full run:

```bash
tmux new -s hviske-v6 \
  'uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual max_validation_samples_per_dataset=1000'
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

## Olmix anchor calibration matrix

The Olmix benchmark is local-only and serial. It compares the two model configs
`whisper-xxsmall` (`openai/whisper-tiny`) and `hviske-v5-tiny`
(`syvai/hviske-v5-tiny`) against the production, read-speech-heavy, and
spontaneous/conversational-heavy anchors. The anchors preserve the 16-source order,
60/40 Danish/English split, and non-zero representation of every source. The launcher
writes one unique directory per job containing `metadata.json`, `run.log`,
the step-tagged `evaluation_metrics.jsonl`, and the final model output. It records the
commit, immutable model revision, command, timings, and evaluation metrics; no
credentials are written.

Run these commands inside the externally supplied training container. This repository
does not build or configure that image: provide the project checkout, model/cache and
background-noise mounts, plus the usual runtime environment (including GPU access and
an existing Hugging Face login) according to the deployment. The first command is
credential-free: use that existing login rather than putting a token in an environment
variable, command, or file.

```bash
docker exec -it <hviske-container> bash
cd /workspace/hviske
uv sync --python 3.11 --all-extras
uv run python src/scripts/finetune_asr_model.py \
  --config-name sparkie_bilingual +anchors=olmix_baseline --cfg job
uv run pytest tests/test_olmix_benchmark.py -q
```

Run both two-step model smokes in an interactive tmux session and inspect their logs:

```bash
tmux new-session -s olmix-smoke \
  'set -euo pipefail; cd /workspace/hviske && uv run python src/scripts/run_olmix_benchmark.py \
   --smoke --output-root runs/olmix'
```

After both smokes succeed, run the six jobs serially with `--skip-smoke`. If the
matrix is launched without that flag, it repeats the two-model smoke immediately
before the matrix. Each full job uses
3,000 steps, caps every validation stream at 500 examples, evaluates at steps 250,
500, 1,000, 2,000, and 3,000, and disables Hub publication and experiment tracking.

```bash
tmux new-session -s olmix-matrix \
  'set -euo pipefail; cd /workspace/hviske && uv run python src/scripts/run_olmix_benchmark.py \
   --matrix --skip-smoke --output-root runs/olmix'
tmux capture-pane -pt olmix-matrix:0 -S -200
```

For one selected calibration job, use the same launcher with `--model` and
`--anchor`, for example:

```bash
tmux new-session -s olmix-read \
  'set -euo pipefail; cd /workspace/hviske && uv run python src/scripts/run_olmix_benchmark.py \
   --model hviske-v5-tiny --anchor olmix_read_speech_heavy \
   --output-root runs/olmix'
```

The requested evaluation steps are non-uniform, so `eval_steps` alone cannot express
this schedule. The launcher passes the schedule through `evaluation_steps`; the
training callback suppresses evaluations at all other steps and writes exact
step-tagged records to `evaluation_metrics.jsonl`. These are evaluation steps, not
retained checkpoints: only the final step is saved and one final checkpoint is kept.

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
  models/hviske-v6 syvai/hviske-v6 --private \
  --evaluation-status 'Pilot and full-run evaluation reviewed before publication.'
```

Never use a public destination or upload raw audio, local manifests, Hydra output,
checkpoints, caches, or experiment-tracking artefacts.
