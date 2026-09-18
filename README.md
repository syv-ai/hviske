# Hviske

Hviske is a Danish automatic speech-recognition (ASR) codebase and model project.
It provides reusable data processing, training, evaluation, and publication tooling
under the `hviske` package.

## Installation

1. Run `make install` to install `uv`, Python 3.11, project dependencies, and quality
   tools.
2. Run `source .venv/bin/activate` if an activated environment is desired.
3. Run `make` to list the available commands.

For an existing installation, `uv sync --python 3.11 --all-extras` installs the
project and its optional `kenlm` dependency.

## Scripts

The supported entry points are:

- `src/scripts/finetune_asr_model.py`: Hydra training entry point.
- `src/scripts/evaluate_model.py`: Hydra model evaluation entry point.
- `src/scripts/fix_dot_env_file.py`: Makefile environment-file setup helper.

The project deliberately does not ship dataset downloaders, plotting utilities, P1
producer commands, or local caption-manifest builders.

## Usage

### Fine-tuning

```bash
uv run python src/scripts/finetune_asr_model.py [key=value]...
```

Supported model presets include Wav2Vec2, Whisper, Cohere, and the native Parakeet
CTC, RNNT, and TDT implementations. Parakeet RNNT and TDT inputs retain the exact
blank-prefixed decoder contract expected by the native Transformers models. The
Parakeet configs preserve existing vocabulary and blank-token IDs while adding Danish
characters when needed.

The reusable dataset configurations include CoRal, FLEURS, FTSpeech, Nota, NST,
VoxPopuli, People's Speech, AMI, LibriSpeech, and the published P1 dataset. P1 is
consumed directly from `syvai/p1-segments` with `audio` and `text` columns. Its
manually gated immutable revision must be supplied through
`P1_SEGMENTS_REVISION` before a production run.

P1 producer code is archived outside this repository. The exact source archive is
`syvai/p1-segments`, path `archives/hviske-p1-pipeline/f3dcf16/`, Hub commit
`44284e5849b6b1d96b874891c579654a644e0e2f`; the dataset remains manually gated.

The production bilingual preset is `config/bilingual.yaml`:

```bash
P1_SEGMENTS_REVISION=<full-40-hex-commit-sha> \
  uv run python src/scripts/finetune_asr_model.py --config-name bilingual --cfg job
```

See [`SPARKIE.md`](SPARKIE.md) for the GPU runbook and integrated checkpoint
publication procedure. See [`docs/parakeet-tdt-runbook.md`](docs/parakeet-tdt-runbook.md)
for focused Parakeet TDT validation.

### Evaluation

```bash
uv run python src/scripts/evaluate_model.py [key=value]...
```

The required `model_id` selects the Hugging Face model. Dataset, split, text-column,
and audio-column settings are defined in `config/evaluation.yaml`.

### Publication

Set `push_to_hub=true` in the fine-tuning configuration to publish the completed
model directly to the Hub. Set `private=true` and `private_only=true` for a private
repository. Publication generates the model card, validates the model package, and
rejects local data artefacts, credentials, and unsupported package contents.

## Development

```bash
make check
make test
```

The test suite uses Hugging Face datasets and short model runs. Network access,
FFmpeg, and Hugging Face authentication may be required for the complete suite.

## Troubleshooting

On macOS, install the missing system headers suggested by the compiler before rerunning
`make install`. For example, Homebrew users may need:

```bash
export CPPFLAGS="-I$(brew --prefix)/include"
```
