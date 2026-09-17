# Parakeet TDT validation runbook

This is an isolated validation path for the native Transformers
`nvidia/parakeet-tdt-0.6b-v3` preset. It does not change the active Cohere
configuration, campaign, W&B project, or production Sparkie commands. The commands
below are runbook templates only; review every gate before launching one.

## Fixed base and local prerequisites

The preset is pinned to revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`. Install the locked environment and
verify the native TDT classes before using a GPU:

```bash
uv sync --python 3.11 --all-extras
uv run python -c 'from transformers import AutoModelForTDT; print(AutoModelForTDT)'
uv run python src/scripts/finetune_asr_model.py --config-name asr_finetuning \
  model=parakeet-tdt --cfg job
```

Do not replace the revision with `main`, and do not pass the base revision when
loading a local saved model. Keep each experiment in its own output directory.
Never launch these commands against the active Cohere output directory.

## Gate 1: isolated zero-shot base evaluation

First authenticate to Hugging Face, confirm access to the pinned public checkpoint,
and record the resolved configuration. Then evaluate the untouched base with no
training and retain the result as the zero-shot baseline:

```bash
hf auth whoami
uv run python src/scripts/evaluate_model.py \
  model_id=nvidia/parakeet-tdt-0.6b-v3 \
  model_revision=541d1f99c6b0c3cd0b11a95167540bb8edefd82b \
  store_results=true
```

**Gate:** the pinned processor and model load natively, Danish `æ`, `ø`, and `å`
are encoded without vocabulary growth, and the baseline result is recorded before
any checkpoint is created.

## Gate 2: two-step smoke and lifecycle check

Use a fresh local directory and a separate W&B identity only if tracking is enabled.
The smoke must save at step 2; do not reuse a prior output directory:

```bash
smoke_dir="$PWD/runs/parakeet-tdt-smoke"
uv run python src/scripts/finetune_asr_model.py --config-name asr_finetuning \
  model=parakeet-tdt model_dir="$smoke_dir" stop_after_steps=2 \
  max_steps=2 save_steps=2 save_total_limit=1 evaluation_steps='[2]' \
  evaluation_metrics_path="$smoke_dir/evaluation-metrics.jsonl" \
  enable_experiment_tracking=false
```

After it exits, cleanly reload the saved model and processor, then resume from its
checkpoint in the same isolated directory. The resumed run must advance the global
step rather than overwrite step 2:

```bash
uv run python src/scripts/evaluate_model.py \
  model_id="$smoke_dir" store_results=false
uv run python src/scripts/finetune_asr_model.py --config-name asr_finetuning \
  model=parakeet-tdt model_dir="$smoke_dir" \
  resume_from_checkpoint="$smoke_dir/checkpoint-2" stop_after_steps=4 \
  max_steps=4 save_steps=2 save_total_limit=1 evaluation_steps='[4]' \
  evaluation_metrics_path="$smoke_dir/evaluation-metrics.jsonl" \
  enable_experiment_tracking=false
```

**Gate:** forward loss is finite, the saved processor reloads with `decoder_type=tdt`,
clean reload uses local files without the Hub base revision, and the resumed trainer
reports global step 4 (or a later explicitly requested step).

## Gate 3: bounded learning-rate pilots

Only after Gates 1-2 pass, run pilots serially in fresh directories. Review the
metrics at exactly steps 250, 500, 1000, and 2000 before selecting a learning rate.
The scheduler horizon remains bounded and no pilot is the active Cohere campaign:

```bash
for lr in 5e-6 1e-5; do
  run_dir="$PWD/runs/parakeet-tdt-pilot-$lr"
  uv run python src/scripts/finetune_asr_model.py --config-name asr_finetuning \
    model=parakeet-tdt model.learning_rate="$lr" model_dir="$run_dir" \
    evaluation_steps='[250,500,1000,2000]' stop_after_steps=2000 \
    max_steps=100000 save_steps=250 save_total_limit=8 \
    evaluation_metrics_path="$run_dir/evaluation-metrics.jsonl" \
    enable_experiment_tracking=false
 done
```

**Gate:** each pilot has independent checkpoints and metrics, reaches the requested
bounded stop without changing `config/model/cohere.yaml` or
`config/sparkie_bilingual.yaml`, and has a clean reload/resume record. Do not start a
longer run, publish a model, or alter Sparkie until the pilot comparison is reviewed.
