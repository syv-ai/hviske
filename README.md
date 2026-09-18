# Hviske

Hviske is a Danish ASR model and codebase, forked from the [CoRal repo](https://github.com/alexandrainst/coral).

______________________________________________________________________
[![License](https://img.shields.io/github/license/syv-ai/hviske)](https://github.com/syv-ai/hviske/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/syv-ai/hviske)](https://github.com/syv-ai/hviske/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-57%25-orange.svg)](https://github.com/syv-ai/hviske/tree/main/tests)

Author and maintainer:

- Dan Saattrup Smart (<dan@syv.dk>)

## Installation

1. Run `make install`, which installs `uv` (if it isn't already installed), sets up a
   Python 3.11 virtual environment and all project dependencies therein. It also
   provisions Python 3.12 for the Funcsort and Slopo quality tools.
2. Run `source .venv/bin/activate` to activate the virtual environment.
3. Run `make` to see a list of available commands.

## Usage

### Finetuning an Acoustic Model for Automatic Speech Recognition (ASR)

You can use the `finetune_asr_model` script to finetune your own ASR model:

```bash
uv run python src/scripts/finetune_asr_model.py [key=value]...
```

Here are some of the more important available keys:

- `model`: The base model to finetune. Supports the following values:
  - `wav2vec2-small`
  - `wav2vec2-medium`
  - `wav2vec2-large`
  - `whisper-xxsmall`
  - `whisper-xsmall`
  - `whisper-small`
  - `whisper-medium`
  - `whisper-large`
  - `whisper-large-turbo`
  - `cohere`
  - `parakeet-ctc`
  - `parakeet-rnnt`
  - `parakeet-tdt`

  The `parakeet-*` configs use the Transformers-native NVIDIA Parakeet CTC, RNNT,
  and TDT implementations. The TDT preset is pinned to a tested Hub revision.
  The Danish `parakeet-rnnt-da-dk` repository is NeMo-only and is not supported.
  Parakeet RNNT and TDT evaluation use native generation and do not pass Whisper
  language or task generation arguments. The Parakeet configs extend the native
  tokenizer with retained Danish characters (such as `æ`, `ø`, and `å`) when
  they would otherwise become unknown tokens; existing vocabulary and blank
  token IDs are preserved.

  The isolated TDT validation gates and bounded pilot commands are documented in
  [`docs/parakeet-tdt-runbook.md`](docs/parakeet-tdt-runbook.md). The `cohere` config
  fine-tunes `CohereLabs/cohere-transcribe-03-2026` with a
  Danish language and punctuation prompt at 16 kHz. The official Cohere checkpoint
  is gated on Hugging Face and requires accepted access and authentication. Native
  Transformers 5.5 loading is used without remote code; Danish fine-tuned
  checkpoints such as `syvai/hviske-v5.3` are also supported.
- `datasets`: The datasets to finetune the models on. Can be a single dataset or an
  array of datasets (written like [dataset1,dataset2,...]). Supports the following
  values:
  - `coral_read_aloud`
  - `coral_conversation`
  - `coral_tts`
  - `fleurs`
  - `ftspeech`
  - `nota`
  - `nst`
  - `coral_read_aloud`, `coral_conversation`, `ftspeech`, `nota`, `nst`, and
    `voxpopuli_da` (source-filtered rows from the pinned unified Danish dataset plus
    its v5-tiny metadata overlay)
  - `p1` (streaming audio and text from the pinned `syvai/p1-segments` dataset)
  - `peoples_speech_clean` (English People's Speech `clean` training split)
  - `ami_sdm` and `ami_ihm` (English AMI training splits)
  - `voxpopuli_en` (English VoxPopuli training split)
  - `librispeech_clean_train_100`, `librispeech_clean_train_360`, and
    `librispeech_other_train_500` (streaming English LibriSpeech)
  - `drtv_local` and `youtube_local` (local WAV/VTT manifests; both Danish)

  FLEURS `en_us` is evaluation-only in the Sparkie preset and has no training config.
  English Common Voice is not part of the production mix.
- `dataset_probabilities`: In case you are finetuning on several datasets, you need to
  specify the probability of sampling each one. This is an array of probabilities that
  need to sum to 1. If not set, the datasets are sampled uniformly. Production presets
  use source-sampling probabilities chosen for style and acoustic balance, not
  probabilities proportional to row count; large formal or read-aloud corpora are
  deliberately capped.
- `model_id`: The model ID of the finetuned model. Defaults to the model type along with
  a timestamp.
- `push_to_hub`, `hub_organisation` and `private`: Whether to push the finetuned model
  to the Hugging Face Hub, and if so, which organisation to push it to. If `private` is
  set to `True`, the model will be private. The default is not to push the model to the
  Hub. `private_only` hard-fails public destinations and verifies Hub visibility before
  and after upload. The production bilingual preset keeps publication off during
  training. Use `src/scripts/publish_model.py` only after reviewing the saved package;
  local manifest paths must never be included. The command defaults to TDT provenance;
  use `--config-name cohere_publication` only for the preserved Cohere baseline. Package
  validation rejects a model family that contradicts the selected provenance preset.
- `enable_experiment_tracking`: Whether training monitoring during training should be
  enabled. Defaults to false. You can also set `experiment_tracking` to either `wandb`
  or `mlflow` to specify which experiment tracking tool to use (`wandb` is used by
  default). The production bilingual preset uses the online W&B project `hviske` and
  the authenticated user's default workspace (no hardcoded entity). Supply the run group,
  name, ID, and resume policy at launch time. Generate and persist a fresh ID for every
  smoke, pilot, and full phase, using `resume=never` for new runs so an ID collision fails.
  Reuse only the persisted full-run ID with `experiment_tracking.resume=must` when
  resuming a local checkpoint. Keep
  `WANDB_LOG_MODEL=false` and `WANDB_WATCH=false` so checkpoints stay local while metrics
  and configuration are logged online. Authenticate with
  `uv run wandb login --verify`; the Python preflight never prompts or calls
  `wandb.login`. Never put API keys in commands or configuration.
- `per_device_batch_size` and `dataloader_num_workers`: The batch size and number of
  workers to use for training. Defaults to 8 and 4, respectively. Tweak these if you are
  running out of GPU memory.
- `model.learning_rate`, `total_batch_size`, `max_steps`, `warmup_steps`: Training
  parameters that you can tweak, although it shouldn't really be needed.

Dataset entries may set `language` to override the model-level Cohere prompt for
that source. A Hub audio source can still be joined to a compact transcript Hub dataset
without downloading the audio by setting `transcript_dataset_id`,
`transcript_subset`, `transcript_split`, `audio_join_column`,
`transcript_join_column`, and `transcript_text_column`. `revision` and
`transcript_revision` pin each side independently; `trust_remote_code` and
`transcript_trust_remote_code` default to false. The audio side remains streaming,
while the transcript side is indexed in memory. Column names are deliberately
configuration fields because private transcript schemas must be verified before use.
The supplied `voxpopuli_da` config selects only rows whose `source` is exactly
`voxpopuli`; the unified repository's ftspeech, CoRal, NST, and Nota rows remain
separately sourced. The supplied `p1` config reads `audio` and `text` directly from
`syvai/p1-segments` and requires `P1_SEGMENTS_REVISION`, a full immutable 40-character
commit SHA. Other dataset configs retain generic transcript-join support.

For local WAV/VTT data, first build a manifest without copying audio:

```bash
uv run python src/scripts/build_vtt_manifest.py \
  --source-dir "$HOME/drtv-asr-dataset/data/drtv" \
  --output "$HOME/drtv-asr-dataset/data/drtv/drtv-manifest.jsonl" \
  --language da
```

Use `config/datasets/drtv_local.yaml` or `youtube_local.yaml` as the dataset
configuration. Training seeks and reads only each cue from the original WAV when it
is consumed; the manifest stores paths, offsets, text, IDs, durations, and language.

The reproducible production bilingual preset is `config/bilingual.yaml`. Export
`P1_SEGMENTS_REVISION` and `HVISKE_OVERLAY_REVISION` with their immutable
40-character commits, then resolve the preset with the general finetuning entry point:

```bash
uv run python src/scripts/finetune_asr_model.py \
  --config-name bilingual --cfg job
```

The production preset caps validation materialisation at 1,000 examples per dataset.
Use `max_validation_samples_per_dataset=32` for a two-step smoke, `=256` for a pilot,
and `=1000` for the long run; the smoke must never materialise full validation.

The complete operational procedure, including smoke, pilot, full tmux run, monitoring,
and checkpoint retention, is in
[`SPARKIE.md`](SPARKIE.md).

See all the finetuning options in the `config/asr_finetuning.yaml` file.

### Evaluating an Automatic Speech Recognition (ASR) Model

You can use the `evaluate_model` script to evaluate an ASR model:

```bash
python src/scripts/evaluate_model.py [key=value]...
```

Here are some of the more important available keys:

- `model_id` (required): The Hugging Face model ID of the ASR model to evaluate.
- `dataset`: The ASR dataset to evaluate the model on. Can be any ASR dataset on the
  Hugging Face Hub. Note that subsets are separated with "::". Defaults to
  `CoRal-project/coral-v3::conversation`.
- `eval_split_name`: The dataset split to evaluate on. Defaults to `test`.
- `text_column`: The name of the column in the dataset that contains the text. Defaults
  to `text`.
- `audio_column`: The name of the column in the dataset that contains the audio. Defaults
  to `audio`.

See all the evaluation options in the `config/evaluation.yaml` file.

## Troubleshooting

If you're on MacOS and get an error saying something along the lines of "fatal error:
'lzma.h' file not found" then try the following and rerun `make install` afterwards:

```bash
export CPPFLAGS="-I$(brew --prefix)/include"
```

Another MacOS issue can happen if you get something like "fatal error: 'cstddef' file
not found" and/or "fatal error: 'climits' file not found". In this case, first ensure
that [you have Homebrew installed](https://brew.sh/), after which you run the following:

```bash
brew install cmake boost zlib eigen
```
