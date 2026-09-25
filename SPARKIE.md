# Sparkie bilingual runbook

This runbook covers the private Parakeet TDT campaign, including integrated Hub
publication. The production preset uses the pinned TDT base, 1--8-second
audio, per-device batch 6, effective batch 60, and two DataLoader workers.

## Prerequisites

Install the project and authenticate without putting credentials in this checkout:

```bash
uv sync --python 3.11 --all-extras
hf auth login
hf auth whoami
uv run wandb login --verify
nvidia-smi
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The Sparkie preset requires CUDA and access to the manually gated
`syvai/p1-segments` dataset. It also requires write access to the private model
repository. Never put tokens in this runbook, shell history, configuration, or
commands.

Resolve the production configuration before launching. The immutable P1 revision is
already pinned in `config/datasets/p1.yaml`:

```bash
uv run python src/scripts/finetune_asr_model.py --config-name bilingual --cfg job
```

P1 is loaded directly with its `audio` and `text` columns. The exact archival source
for the retired producer is `syvai/p1-segments`, path
`archives/hviske-p1-pipeline/f3dcf16/`, Hub commit
`44284e5849b6b1d96b874891c579654a644e0e2f`. The published dataset is manually gated.

## Fixed production mix

Do not reorder or rebalance the 15 configured streams. Source probabilities are chosen
for acoustic and linguistic balance, not row count:

| Language | Source | Probability |
| --- | --- | ---: |
| Danish | `syvai/p1-segments` | 0.102857 |
| Danish | DRTV local VTT | 0.128571 |
| Danish | YouTube local VTT | 0.09 |
| Danish | CoRal read-aloud | 0.017143 |
| Danish | CoRal conversation | 0.128572 |
| Danish | FTSpeech | 0.09 |
| Danish | Nota | 0.021428 |
| Danish | NST | 0.021429 |
| English | People's Speech clean | 0.16 |
| English | AMI SDM | 0.04 |
| English | AMI IHM | 0.03 |
| English | VoxPopuli | 0.09 |
| English | LibriSpeech clean 100 | 0.01 |
| English | LibriSpeech clean 360 | 0.025 |
| English | LibriSpeech other 500 | 0.045 |

The preset retains a 60/40 Danish/English split and filters examples to the configured
1--8-second duration range. All dataset revisions, including P1, are pinned in the
configuration.

## Smoke, pilot, and full run

Use a fresh W&B ID and a phase-specific output directory for each run. Keep the
training launch in a `tmux` session and scrub inherited W&B identity:

```bash
export WANDB_PROJECT=hviske
wandb_id=$(uv run python -c 'import wandb; print(wandb.util.generate_id())')
run_dir="$PWD/runs/smoke"
tmux new-session -d -s hviske-smoke \
  "env -u WANDB_API_KEY -u WANDB_ENTITY -u WANDB_BASE_URL -u WANDB_RUN_ID -u WANDB_RESUME -u WANDB_NAME -u WANDB_RUN_GROUP WANDB_PROJECT=$WANDB_PROJECT WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false uv run python src/scripts/finetune_asr_model.py --config-name bilingual experiment_tracking.name_run=v6.0-smoke experiment_tracking.id=$wandb_id experiment_tracking.resume=never model_dir=$run_dir evaluation_steps='[2]' stop_after_steps=2 max_validation_samples_per_dataset=32"
```

Inspect the resolved configuration, loss, evaluation metrics, checkpoint, GPU memory,
and temperature before starting a longer run. Optional 2,000-step learning-rate
pilots use fresh IDs and separate directories. The full run uses the production
`max_steps=200000` horizon and `eval_steps=2000` cadence.

If a run is interrupted, resume its local checkpoint with the same persisted W&B ID:

```bash
uv run python src/scripts/finetune_asr_model.py \
  --config-name bilingual resume_from_checkpoint=runs/hviske-v6.0/checkpoint-<step> \
  experiment_tracking.resume=must
```

Do not increase worker counts to work around a data failure. Stop for non-finite loss,
missing data, failed checkpoint reload, unsafe temperature or insufficient disk space.

## Serial Parakeet-TDT campaign

The campaign is a serial go/no-go ladder, not a set of concurrent jobs. Run each
preset only after reviewing the prior run's loss, language-appropriate validation,
and early checkpoints; every stage starts from the pinned base with a new output
folder and W&B ID. Never resume an earlier stage. The presets all keep
`push_to_hub=false`, `resume_from_checkpoint=false`, total batch 60, per-device
batch 4, zero workers, 1--8-second audio, and 500-step evaluation/save milestones.

Use these config names in order:

```text
parakeet_tdt_danish_leaderboard  -> parakeet_tdt_danish_05
  -> parakeet_tdt_danish_20 -> parakeet_tdt_danish_50
  -> parakeet_tdt_danish_100 -> parakeet_tdt_bilingual_10
  -> parakeet_tdt_bilingual_50 -> parakeet_tdt_bilingual_100
  -> parakeet_tdt_bilingual_65_35
```

The first five sources are ordered `coral_read_aloud`, `coral_conversation`,
`ftspeech`, `fleurs`, `common_voice_19`, capped at
`[299255, 147249, 995677, 1032, 3484]`. P1, DRTV, YouTube, Nota, and NST are
then added at 5%, 20%, 50%, and 100%; English follows only after full Danish.
The exact CV19 assumption is intentional: `fsicoli/common_voice_19_0`, Danish
`train`, revision `590c8abec6cf7c8d06e650f1438e60332a796e11` is the practical
stand-in because Sparkie has no CV27 MDC asset. Training excludes validation and
test splits, and caps are lazy after source/duration/text filtering.

At the observed ~403 optimizer steps/hour, the stages are approximately 59.8,
87.5, 170.6, 336.8, 613.7, 627.3, 681.6, 749.5, and 945.4 hours respectively
(2.5, 3.6, 7.1, 14.0, 25.6, 26.1, 28.4, 31.2, and 39.4 days). These estimates
exclude startup, validation, retries, and interruptions. The last preset is an
immutable 381,000-step 65/35 full bilingual composition equivalent to
`bilingual_v2`; the historical config remains unchanged.

An optional read-only evaluation of the old `checkpoint-37500` is allowed for
comparison in a separate results directory. Do not resume it or use it as a
training input.

## Private publication

Set the integrated publication options before the final run:

```bash
uv run python src/scripts/finetune_asr_model.py --config-name bilingual \
  push_to_hub=true private=true private_only=true
```

The training command generates the model card and verifies private Hub visibility.
Never upload raw audio, credentials, training outputs, caches, or experiment-tracking
artefacts.
