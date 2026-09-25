# Parakeet TDT validation runbook

This records the isolated validation path that qualified the native Transformers
`nvidia/parakeet-tdt-0.6b-v3` preset for production. The active bilingual campaign now
uses that pinned TDT base with 1--8-second audio and effective batch 60. The commands
below remain diagnostic templates only; never run them against the active campaign's
output directory or W&B identity.

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
Never launch these commands against the active production output directory.

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

This is a pre-campaign diagnostic. Its local resume check validates checkpoint
lifecycle handling only and does not authorise resuming any progressive campaign stage.

Use a fresh local directory and a separate W&B identity only if tracking is enabled.
The smoke must save at step 2; do not reuse a prior output directory. Keep two
checkpoints: Transformers can retain the tracked `best_model_checkpoint` while
rotating out the newest checkpoint when `save_total_limit=1`, which makes the
resume source or the newly completed step disappear. A limit of 2 guarantees that
both the best/source checkpoint and newest resume checkpoint survive:

```bash
smoke_dir="$PWD/runs/parakeet-tdt-smoke"
uv run python src/scripts/finetune_asr_model.py --config-name asr_finetuning \
  model=parakeet-tdt model_dir="$smoke_dir" stop_after_steps=2 \
  max_steps=2 save_steps=2 save_total_limit=2 evaluation_steps='[2]' \
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
  max_steps=4 save_steps=2 save_total_limit=2 evaluation_steps='[4]' \
  evaluation_metrics_path="$smoke_dir/evaluation-metrics.jsonl" \
  enable_experiment_tracking=false
```

When `evaluation_steps` contains the bounded `stop_after_steps`, the final validation
runs at that same global step immediately after `trainer.train()` returns. This ordering
lets spawned training DataLoader workers shut down before validation opens its own Hub
streams, avoiding a misleading worker `SIGABRT` during interpreter finalisation. The
terminal metrics still cover every configured validation dataset, use the same W&B run,
and are appended to `evaluation_metrics_path`; any earlier scheduled validations remain
in the training loop, and the terminal checkpoint is saved there as usual.

**Numerical gate:** processed training and validation examples must have non-empty
`labels` and positive `input_length`, and every observed forward loss must be finite.
Stop on any NaN or infinity and investigate the source example; do not mask the failure
with `nan_to_num` or continue the run.

**Gate:** forward loss is finite, the saved processor reloads with `decoder_type=tdt`,
clean reload uses local files without the Hub base revision, and the resumed trainer
reports global step 4 (or a later explicitly requested step).

## Gate 3: bounded learning-rate pilots

Only after Gates 1-2 pass, run pilots serially in fresh directories. Review the
metrics at exactly steps 250, 500, 1000, and 2000 before selecting a learning rate.
The scheduler horizon remains bounded and no pilot is the active production campaign:

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

Steps 250, 500, and 1000 evaluate in-loop. Step 2000 uses the post-training ordering
 described above and must appear exactly once in the metrics JSONL after the training
 workers have closed.

## Serial Danish-to-bilingual campaign

After the diagnostic gates, run the campaign presets serially. **This campaign is
fresh-run and local-only for now:** use a new output directory and W&B ID for every
stage, never resume a local checkpoint or W&B run, never pass a checkpoint from an
earlier stage, and never publish a model or create a Hub pull request. Keep
`resume_from_checkpoint=false`, `push_to_hub=false`, and
`experiment_tracking.log_model=false`. A failed or rejected go/no-go stage stops the
sequence; it is not silently skipped. The presets use total batch 60, per-device
batch 4, zero dataset/DataLoader workers, 1--8-second audio, and save/evaluate every
500 steps so early checkpoints are useful.

The source order (and therefore cap/probability order) is the five-source Danish
leaderboard baseline `coral_read_aloud`, `coral_conversation`, `ftspeech`, `fleurs`,
`common_voice_19`, followed by P1, DRTV, YouTube, Nota, NST. The Danish baseline
caps are `[299255, 147249, 995677, 1032, 3484]`; subsequent presets add the five
latter sources at 5%, 20%, 50%, and 100% caps. CV19 is an explicit practical
stand-in for the unavailable Sparkie CV27 MDC asset: it uses
`fsicoli/common_voice_19_0`, Danish `train`, revision
`590c8abec6cf7c8d06e650f1438e60332a796e11`. No validation or test split is used for
training. Caps are applied lazily only after source and duration/text filters.

Run these go/no-go stages in order (do not launch the next line until the prior
metrics and early checkpoints are accepted):

```text
parakeet_tdt_danish_leaderboard  24,112 steps   ~59.8 h (2.5 d)
parakeet_tdt_danish_05           35,274 steps   ~87.5 h (3.6 d)
parakeet_tdt_danish_20           68,758 steps   ~170.6 h (7.1 d)
parakeet_tdt_danish_50          135,726 steps   ~336.8 h (14.0 d)
parakeet_tdt_danish_100         247,340 steps   ~613.7 h (25.6 d)
parakeet_tdt_bilingual_10        252,810 steps   ~627.3 h (26.1 d)
parakeet_tdt_bilingual_50        274,690 steps   ~681.6 h (28.4 d)
parakeet_tdt_bilingual_100       302,040 steps   ~749.5 h (31.2 d)
parakeet_tdt_bilingual_65_35     381,000 steps   ~945.4 h (39.4 d)
```

Runtime estimates use the observed throughput of approximately 403 optimizer
steps/hour and exclude startup, validation, Hub retries, and interruptions. Danish
stages use the Danish leaderboard-aligned CoRal/FLEURS validation suite. English
validation is added only once English training sources appear. The final preset is
an immutable 65/35 full bilingual composition equivalent to `bilingual_v2`; the
historical config is not edited.

An old `checkpoint-37500` may optionally be evaluated for comparison using the
normal evaluation command and a separate results directory. Evaluation is read-only:
never resume training from that checkpoint and never use it as a campaign stage.
**Gate:** each pilot has finite training and validation losses at every reviewed step,
has independent checkpoints and metrics, reaches the requested bounded stop without
changing `config/bilingual.yaml`, and has a clean local checkpoint reload record. Do
not start a longer run, publish a model, or alter Sparkie until the pilot comparison is
reviewed.
