# Sparkie bilingual runbook

This runbook covers the private Parakeet TDT campaign. Training and publication are
separate operations. The production preset uses the pinned TDT base, 1--8-second
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

Set the immutable P1 revision before resolving the production configuration:

```bash
export P1_SEGMENTS_REVISION=<full-40-hex-p1-segments-commit-sha>
uv run python src/scripts/finetune_asr_model.py --config-name bilingual --cfg job
```

P1 is loaded directly with its `audio` and `text` columns. The exact archival source
for the retired producer is `syvai/p1-segments`, path
`archives/hviske-p1-pipeline/f3dcf16/`, Hub commit
`44284e5849b6b1d96b874891c579654a644e0e2f`. The published dataset is manually gated.

## Fixed production mix

Do not reorder or rebalance the 13 configured streams. Source probabilities are chosen
for acoustic and linguistic balance, not row count:

| Language | Source | Probability |
| --- | --- | ---: |
| Danish | `syvai/p1-segments` | 0.131626801667 |
| Danish | CoRal read-aloud | 0.021938013562 |
| Danish | CoRal conversation | 0.164534461864 |
| Danish | FTSpeech | 0.115173611422 |
| Danish | Nota | 0.027421557173 |
| Danish | NST | 0.027422836880 |
| English | People's Speech clean | 0.204753086973 |
| English | AMI SDM | 0.051188271743 |
| English | AMI IHM | 0.038391203807 |
| English | VoxPopuli | 0.115173611422 |
| English | LibriSpeech clean 100 | 0.012797067936 |
| English | LibriSpeech clean 360 | 0.031992669839 |
| English | LibriSpeech other 500 | 0.057586805712 |

The preset retains a 60/40 Danish/English split and filters examples to the configured
1--8-second duration range. Public dataset revisions are pinned in the configuration;
P1 is pinned at launch through `P1_SEGMENTS_REVISION`.

## Smoke, pilot, and full run

Use a fresh W&B ID and a phase-specific output directory for each run. Keep the
training launch in a `tmux` session and scrub inherited W&B identity:

```bash
export WANDB_PROJECT=hviske
wandb_id=$(uv run python -c 'import wandb; print(wandb.util.generate_id())')
run_dir="$PWD/runs/smoke"
tmux new-session -d -s hviske-smoke \
  "env -u WANDB_API_KEY -u WANDB_ENTITY -u WANDB_BASE_URL -u WANDB_RUN_ID -u WANDB_RESUME -u WANDB_NAME -u WANDB_RUN_GROUP P1_SEGMENTS_REVISION=$P1_SEGMENTS_REVISION WANDB_PROJECT=$WANDB_PROJECT WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false uv run python src/scripts/finetune_asr_model.py --config-name bilingual experiment_tracking.name_run=v6.0-smoke experiment_tracking.id=$wandb_id experiment_tracking.resume=never model_dir=$run_dir evaluation_steps='[2]' stop_after_steps=2 max_validation_samples_per_dataset=32"
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

## Private publication

After reviewing the final package and evaluation, publish it separately:

```bash
uv run python src/scripts/publish_model.py \
  runs/hviske-v6.0 syvai/hviske-v6.0 --private \
  --evaluation-status 'Full-run evaluation reviewed before publication.'
```

The command verifies private Hub visibility. Never upload raw audio, credentials,
training outputs, caches, or experiment-tracking artefacts.
