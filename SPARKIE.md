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
uv run wandb login
uv run wandb login --verify
```

W&B uses the authenticated user's default/personal workspace; do not set an entity
unless an explicit workspace override is required. The production project is `hviske`.
`uv run wandb login --verify` must pass while the GPU service is still running. Never
put a W&B API key in this runbook, shell history, Hydra configuration, or a command.

The Sparkie preset requires a working CUDA device. The lockfile resolves matching
PyTorch and torchaudio 2.10 releases; Linux aarch64 installs the CUDA 13.0 wheels from
the explicit PyTorch index, while other platforms use the normal PyPI wheels. The
training entrypoint checks `torch.cuda.is_available()` before W&B, model, or dataset
initialisation and reports the installed torch version and CUDA build when it fails.
Check the driver before starting a run:

```bash
nvidia-smi
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
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

Do not reorder or rebalance the preset. Its 15 active streams are sampled as follows:

| Language | Source | Configuration and split | Probability |
| --- | --- | --- | ---: |
| Danish | `syvai/p1-segments` | `train` | 0.102857 |
| Danish | local DRTV manifest | `train` | 0.128571 |
| Danish | local YouTube manifest | `train` | 0.09 |
| Danish | `syvai/danish-asr-unified` + v5-tiny overlay | `source=coral_read_aloud` / `train` | 0.017143 |
| Danish | `syvai/danish-asr-unified` + v5-tiny overlay | `source=coral_conversation` / `train` | 0.128572 |
| Danish | `syvai/danish-asr-unified` + v5-tiny overlay | `source=ftspeech` / `train` | 0.09 |
| Danish | `syvai/danish-asr-unified` + v5-tiny overlay | `source=nota` / `train` | 0.021428 |
| Danish | `syvai/danish-asr-unified` + v5-tiny overlay | `source=nst_da` / `train` | 0.021429 |
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
evaluation-only. The unified Danish repository is filtered into five separately weighted
active streams and uses the `HVISKE_OVERLAY_REVISION` v5-tiny metadata overlay; its rows
are not loaded directly from the obsolete source repositories. Each active stream resolves
only its inclusive source-contiguous shard range at both immutable revisions:
`nota` 354--367, `ftspeech` 367--563, `coral_read_aloud` 563--623,
`coral_conversation` 623--653, and `nst_da` 653--689. The reusable Danish VoxPopuli
configuration remains available, but is not selected: at unified audio revision
`5a3a49ee981baab6e1e37ddd2c45f9943c27d08f` and positional overlay shard range `0--354`,
its first 100 joined clips are OGG mono 16 kHz (duration min 16.15, median 30, max 30),
so none survive the strict `1 < duration < 10` filter. Keeping it active would scan
1.745 million rows without yielding an example. Boundary shards deliberately appear in
both adjacent ranges so filters retain complete source coverage. Empty shard numbers may
be absent, but the resolved base and overlay basename order must match.
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

Materialise the five positional overlays before preflight. The command resolves the
pinned mirrored base and overlay files, applies the same strict action, text, equality,
and row-count semantics one physical shard at a time, and stores only embedded
compressed audio bytes, final text, and source. It is bounded by one shard, retains a
10 GiB free-disk reserve by default, resumes validated shard receipts, and publishes a
deterministic manifest plus `COMPLETE` marker only after every checksum and schema
passes. Never point training at an incomplete directory.

```bash
export HVISKE_MATERIALISED_OVERLAYS_ROOT="$HOME/hviske-v6-overlays"
uv run python src/scripts/materialise_finetuning_overlays.py \
  --config-name sparkie_bilingual \
  --output-root "$HVISKE_MATERIALISED_OVERLAYS_ROOT"
uv run python src/scripts/finetune_asr_model.py \
  --config-name sparkie_bilingual --cfg job
uv run python src/scripts/preflight_finetuning_data.py \
  --config-name sparkie_bilingual
uv run pytest tests/test_materialised_overlays.py tests/test_sparkie_config.py \
  tests/test_wandb_setup.py tests/test_preflight_finetuning_data.py -q
```

The preflight validates the same complete marker, manifest, immutable revisions,
source/file provenance, Parquet schemas, counts, receipts, and checksums before any
materialised stream is opened. It also checks Hub authentication and the remaining
remote sources without logging signed URLs. Sparkie fails closed when
`HVISKE_MATERIALISED_OVERLAYS_ROOT` is absent or invalid; it never falls back to the
remote positional join.

The preset deliberately uses `dataset_num_workers=1` and
`dataloader_num_workers=4`. For regular datasets, `dataset_num_workers=1` maps to
in-process preprocessing (`num_proc=None`), not a one-worker child process. Validation
filtering happens while Hub and `httpx` connections may already be open; forking
preprocessing workers can inherit one of those sockets and deadlock in `CLOSE-WAIT`.
Keep preprocessing serial for this streaming campaign. The local Parquet artefact has
independent physical shards, so three spawned DataLoader workers can interleave shards
without sharing a positional offset and can restart deterministically. Four workers drove
the GB10 to sustained 85–86°C operation; three retain parallel loading with safer thermal
headroom. Keep positional
equality checks strict during materialisation and do not relax them: generic keyed and
non-materialised positional overlays remain supported by the reusable loader.
The per-source streaming shuffle is applied after the local materialised shards are
validated and opened, but before audio decoding and duration filtering. The global
buffer is 1 for already-sharded sources: deterministic physical-shard shuffling remains
intact, while the source probabilities still interleave all streams. The one-shard
local DRTV and YouTube manifests override it with 128 rows, and the five materialised
unified sources use 16 rows for local mixing. Every buffer therefore shuffles metadata
rather than decoded audio. The reviewed shard-bounded, decode-free graph with the old
global 128-row value still took 127 minutes, read 90 GB, and reached 19 GB worker RSS
before its first batch. This ordering is important: shuffling before a positional join
would corrupt base/overlay alignment. The shard bounds avoid the previous full 273 GB
scan in each of five source-filtered streams while keeping the source filters as runtime
validation.

The finetuning entrypoint selects PyTorch's `spawn` start method before experiment
tracking, Hub data loading, or Trainer/DataLoader construction, preventing workers from
inheriting Hub HTTP clients and sockets. An already-selected `spawn` method is reused; a
conflicting method fails clearly instead of silently falling back to `fork`.
Do not raise `dataset_num_workers` for Hub streams. Every smoke, pilot, full-run, and
interruption-recovery command below inherits these values from
`config/sparkie_bilingual.yaml`; do not add a preprocessing-worker override.

Remote `hf://` reads use the project-owned retry policy in
`config/sparkie_bilingual.yaml`: six retries after the initial request, exponential
backoff from 1 second capped at 30 seconds, and up to 0.5 seconds of jitter. It retries
only transport failures, HTTP 429, and HTTP 5xx responses. A closed Hub HTTP client is
reset in that worker before retrying. Authentication, not-found, schema, and data errors
remain fatal, and local/materialised Parquet reads do not use this policy. Keep these
bounds conservative for multi-week runs; repeated exhaustion is an operational failure
to investigate rather than an invitation to increase the retry budget.

Do not proceed if any preflight check fails. Fix access, schema, revision, or local-file
errors and rerun the complete preflight. If a run hangs during validation materialisation,
stop that tmux session, leave `dataset_num_workers` at 1 (in-process preprocessing), and
restart the same command (or the resume command below after a checkpoint exists); do not
retry by increasing preprocessing workers.

## GPU smoke, pilot, and full run

Stop `qwen38-ar` only after the preflight passes and immediately before starting the GPU
smoke. Do not displace it during data preparation. Run every training phase in its own
tmux session from the repository root. The commands below use `env` assignments so the
immutable data revisions and non-secret W&B identity come from this shell, not an old
tmux server environment.

Create a private local state directory and fresh IDs. `resume=never` makes an accidental
ID collision fail rather than append to an old campaign. This directory is local state,
not a credential store:

```bash
wandb_state_dir="$PWD/.hviske-wandb"
mkdir -p "$wandb_state_dir"
new_wandb_id() {
  uv run python -c 'import wandb; print(wandb.util.generate_id())'
}
persist_wandb_id() {
  printf '%s\n' "$2" > "$1"
}
export WANDB_PROJECT=hviske
p1_revision_q=$(printf '%q' "$P1_SEGMENTS_REVISION")
overlay_revision_q=$(printf '%q' "$HVISKE_OVERLAY_REVISION")
materialised_overlay_root_q=$(printf '%q' "$HVISKE_MATERIALISED_OVERLAYS_ROOT")
wandb_project_q=$(printf '%q' "$WANDB_PROJECT")
```

Start with a two-step smoke. Generate and persist its ID before launching it, and keep
its model and metrics in a phase-specific directory:

```bash
smoke_id=$(new_wandb_id)
persist_wandb_id "$wandb_state_dir/smoke.id" "$smoke_id"
smoke_dir_q=$(printf '%q' "$PWD/runs/smoke")
smoke_metrics_q=$(printf '%q' "$PWD/runs/smoke/metrics.jsonl")
tmux new-session -d -s hviske-smoke \
  "env -u WANDB_API_KEY -u WANDB_ENTITY -u WANDB_BASE_URL -u WANDB_RUN_ID -u WANDB_RESUME -u WANDB_NAME -u WANDB_RUN_GROUP P1_SEGMENTS_REVISION=$p1_revision_q HVISKE_OVERLAY_REVISION=$overlay_revision_q HVISKE_MATERIALISED_OVERLAYS_ROOT=$materialised_overlay_root_q WANDB_PROJECT=$wandb_project_q WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual experiment_tracking.name_experiment=hviske experiment_tracking.name_group=v6.0 experiment_tracking.name_run=v6.0-smoke experiment_tracking.id=$smoke_id experiment_tracking.mode=online experiment_tracking.resume=never model_dir=$smoke_dir_q evaluation_steps='[2]' evaluation_metrics_path=$smoke_metrics_q stop_after_steps=2 save_steps=2 max_validation_samples_per_dataset=32"
```

Inspect the resolved log, GPU memory, and checkpoint before starting the pilots. The
learning-rate pilots are owner-waived for the current campaign; if they are requested as
optional diagnostics, run the `5e-6` pilot to completion first, then `1e-5`. They have
independent fresh W&B IDs, model directories, and metric files. Optional pilots retain
their 100,000-step cosine-scheduler horizon while the tested stop callback ends training
at 2,000 steps. Each evaluation is exactly at steps 250, 500, 1,000, and 2,000.

```bash
pilot_5e6_id=$(new_wandb_id)
persist_wandb_id "$wandb_state_dir/pilot-5e-6-seed-4242.id" "$pilot_5e6_id"
pilot_5e6_dir_q=$(printf '%q' "$PWD/runs/pilot-5e-6-seed-4242")
pilot_5e6_metrics_q=$(printf '%q' "$PWD/runs/pilot-5e-6-seed-4242/metrics.jsonl")
tmux new-session -d -s hviske-pilot-5e-6 \
  "env -u WANDB_API_KEY -u WANDB_ENTITY -u WANDB_BASE_URL -u WANDB_RUN_ID -u WANDB_RESUME -u WANDB_NAME -u WANDB_RUN_GROUP P1_SEGMENTS_REVISION=$p1_revision_q HVISKE_OVERLAY_REVISION=$overlay_revision_q HVISKE_MATERIALISED_OVERLAYS_ROOT=$materialised_overlay_root_q WANDB_PROJECT=$wandb_project_q WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual experiment_tracking.name_experiment=hviske experiment_tracking.name_group=v6.0-pilot experiment_tracking.name_run=v6.0-pilot-5e-6-seed-4242 experiment_tracking.id=$pilot_5e6_id experiment_tracking.mode=online experiment_tracking.resume=never model.learning_rate=5e-6 seed=4242 model_dir=$pilot_5e6_dir_q evaluation_steps='[250,500,1000,2000]' evaluation_metrics_path=$pilot_5e6_metrics_q stop_after_steps=2000 max_steps=100000 save_steps=500 max_validation_samples_per_dataset=256"
```

After the `5e-6` session exits and its results are reviewed, run the independent `1e-5`
pilot:

```bash
pilot_1e5_id=$(new_wandb_id)
persist_wandb_id "$wandb_state_dir/pilot-1e-5-seed-4242.id" "$pilot_1e5_id"
pilot_1e5_dir_q=$(printf '%q' "$PWD/runs/pilot-1e-5-seed-4242")
pilot_1e5_metrics_q=$(printf '%q' "$PWD/runs/pilot-1e-5-seed-4242/metrics.jsonl")
tmux new-session -d -s hviske-pilot-1e-5 \
  "env -u WANDB_API_KEY -u WANDB_ENTITY -u WANDB_BASE_URL -u WANDB_RUN_ID -u WANDB_RESUME -u WANDB_NAME -u WANDB_RUN_GROUP P1_SEGMENTS_REVISION=$p1_revision_q HVISKE_OVERLAY_REVISION=$overlay_revision_q HVISKE_MATERIALISED_OVERLAYS_ROOT=$materialised_overlay_root_q WANDB_PROJECT=$wandb_project_q WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual experiment_tracking.name_experiment=hviske experiment_tracking.name_group=v6.0-pilot experiment_tracking.name_run=v6.0-pilot-1e-5-seed-4242 experiment_tracking.id=$pilot_1e5_id experiment_tracking.mode=online experiment_tracking.resume=never model.learning_rate=1e-5 seed=4242 model_dir=$pilot_1e5_dir_q evaluation_steps='[250,500,1000,2000]' evaluation_metrics_path=$pilot_1e5_metrics_q stop_after_steps=2000 max_steps=100000 save_steps=500 max_validation_samples_per_dataset=256"
```

Review the `5e-6` pilot after its tmux session exits before launching the `1e-5` pilot;
these pilots are intentionally serial, not concurrent. Review both results and set
`winner_lr` to exactly `5e-6` or `1e-5`. Repeat the winner with seed 4243 in another
isolated run before full training:

```bash
winner_lr=5e-6  # Change only after reviewing both seed-4242 pilots.
repeat_id=$(new_wandb_id)
persist_wandb_id "$wandb_state_dir/pilot-winner-seed-4243.id" "$repeat_id"
repeat_dir_q=$(printf '%q' "$PWD/runs/pilot-winner-seed-4243")
repeat_metrics_q=$(printf '%q' "$PWD/runs/pilot-winner-seed-4243/metrics.jsonl")
tmux new-session -d -s hviske-pilot-seed-4243 \
  "env -u WANDB_API_KEY -u WANDB_ENTITY -u WANDB_BASE_URL -u WANDB_RUN_ID -u WANDB_RESUME -u WANDB_NAME -u WANDB_RUN_GROUP P1_SEGMENTS_REVISION=$p1_revision_q HVISKE_OVERLAY_REVISION=$overlay_revision_q HVISKE_MATERIALISED_OVERLAYS_ROOT=$materialised_overlay_root_q WANDB_PROJECT=$wandb_project_q WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual experiment_tracking.name_experiment=hviske experiment_tracking.name_group=v6.0-pilot experiment_tracking.name_run=v6.0-pilot-$winner_lr-seed-4243 experiment_tracking.id=$repeat_id experiment_tracking.mode=online experiment_tracking.resume=never model.learning_rate=$winner_lr seed=4243 model_dir=$repeat_dir_q evaluation_steps='[250,500,1000,2000]' evaluation_metrics_path=$repeat_metrics_q stop_after_steps=2000 max_steps=100000 save_steps=500 max_validation_samples_per_dataset=256"
```

The current campaign proceeds directly from the smoke to the full run; the owner-waived
pilots are not a prerequisite and no pilot checkpoint is required. Start from the pinned
Cohere base with a fresh persisted ID and its own output directory:

```bash
full_id=$(new_wandb_id)
persist_wandb_id "$wandb_state_dir/full.id" "$full_id"
full_dir_q=$(printf '%q' "$PWD/runs/hviske-v6.0")
tmux new-session -d -s hviske-v6-0 \
  "env -u WANDB_API_KEY -u WANDB_ENTITY -u WANDB_BASE_URL -u WANDB_RUN_ID -u WANDB_RESUME -u WANDB_NAME -u WANDB_RUN_GROUP P1_SEGMENTS_REVISION=$p1_revision_q HVISKE_OVERLAY_REVISION=$overlay_revision_q HVISKE_MATERIALISED_OVERLAYS_ROOT=$materialised_overlay_root_q WANDB_PROJECT=$wandb_project_q WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual experiment_tracking.name_experiment=hviske experiment_tracking.name_group=v6.0 experiment_tracking.name_run=v6.0-full experiment_tracking.id=$full_id experiment_tracking.mode=online experiment_tracking.resume=never model_dir=$full_dir_q eval_steps=2000 max_steps=200000 max_validation_samples_per_dataset=1000"
```

If the full run is interrupted, preserve the local checkpoint and read the persisted
full ID. `resume=must` rejects a missing or wrong remote run instead of silently creating
another one; Trainer restores the same local optimiser and scheduler state:

```bash
full_id=$(<"$wandb_state_dir/full.id")
checkpoint="$PWD/runs/hviske-v6.0/checkpoint-<step>"
checkpoint_q=$(printf '%q' "$checkpoint")
tmux new-session -d -s hviske-v6-0-resume \
  "env -u WANDB_API_KEY -u WANDB_ENTITY -u WANDB_BASE_URL -u WANDB_RUN_ID -u WANDB_RESUME -u WANDB_NAME -u WANDB_RUN_GROUP P1_SEGMENTS_REVISION=$p1_revision_q HVISKE_OVERLAY_REVISION=$overlay_revision_q HVISKE_MATERIALISED_OVERLAYS_ROOT=$materialised_overlay_root_q WANDB_PROJECT=$wandb_project_q WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false uv run python src/scripts/finetune_asr_model.py --config-name sparkie_bilingual experiment_tracking.name_experiment=hviske experiment_tracking.name_group=v6.0 experiment_tracking.name_run=v6.0-full experiment_tracking.id=$full_id experiment_tracking.mode=online experiment_tracking.resume=must resume_from_checkpoint=$checkpoint_q model_dir=$full_dir_q eval_steps=2000 max_steps=200000 max_validation_samples_per_dataset=1000"
```

The full ID file is the persistence record needed for interruption recovery; never put a
W&B API key in it. The `WANDB_LOG_MODEL=false` and `WANDB_WATCH=false` settings keep
checkpoints and model artefacts local while metrics and configuration remain online.
Pass the revision, identity, and policy in every session; never rely on variables
exported into an existing tmux server. These command-only sessions exit when the job
finishes. Attach or capture logs while a job is running; completed ephemeral sessions
are not available afterwards.

Monitor a running job with:

```bash
tmux attach -t hviske-v6-0
nvidia-smi
df -h
```

The preset retains at most three checkpoints. Stop the run if free disk space, GPU
memory, temperatures, or repeated streaming failures become unsafe. Preserve enough
free space for the next checkpoint and final model save.

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
