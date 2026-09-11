# Hviske

Hviske is a Python codebase for training and evaluating Danish automatic
speech-recognition models, with dataset and decoder tooling. It exposes the
`hviske` package.

## Stack

- Python `>=3.11,<3.13`, managed with `uv` and packaged with Hatchling.
- PyTorch, Hugging Face Datasets and Transformers, Hydra, and Click.
- Pytest, Ruff, Ty, Vulture, Slopo, and pre-commit for quality checks.
- Optional dependency groups include `kenlm`, `demo`, and `plotting`.

## Layout

- `src/hviske/`: data processing, models, metrics, training, and evaluation.
- `src/scripts/`: Hydra entry points and dataset, evaluation, and maintenance tools.
- `config/`: Hydra root configurations and configuration groups.
- `tests/`: pytest tests and shared fixtures.
- `.github/workflows/ci.yaml`: pull-request checks run on GitHub Actions.

## Setup and commands

Run commands from the repository root. The documented `src/scripts/...` paths
assume that launch directory.

```bash
make install
source .venv/bin/activate
make check
make test
```

`make install` keeps the project and test environment on Python 3.11. It also
provisions Python 3.12 for Funcsort and Slopo, whose commands run through
explicit `uv tool run --python 3.12` environments.

`make install` installs Python 3.11, syncs all extras, creates `.env`, installs
pre-commit, and runs `pre-commit autoupdate`. It also updates `uv` when `uv` is
already installed. These steps can change `.pre-commit-config.yaml` and global
tooling; inspect `git diff` afterwards.

For dependencies without project bootstrap, run:

```bash
uv sync --python 3.11 --all-extras
```

Hydra entry points accept `key=value` overrides. Common examples are:

```bash
uv run python src/scripts/finetune_asr_model.py model=wav2vec2-small
uv run python src/scripts/evaluate_model.py model_id=ORG/MODEL
uv run python src/scripts/run_asr_demo.py
uv run python src/scripts/train_ngram_decoder.py model=wav2vec2-small
```

Hydra `config_path` values such as `../../config` are relative to the declaring
script, not the launch CWD. CWD-relative inputs, outputs, and caches, including
the evaluation CSV and `.hviske-cache`, follow the launch CWD.

## Development posture

Hviske is currently used for dataset preparation and model experiments, not a
production deployment. Optimise for iteration speed, useful training-data breadth,
and speaker variation rather than production-grade hardening.

- Accept small errors, approximate alignments or transcriptions, and uncommon edge
  cases when they do not materially invalidate a model experiment.
- Do not add elaborate validation gates, production safeguards, or repeated
  builder-reviewer cycles for non-critical polish. A focused implementation and
  relevant tests are normally enough.
- For training datasets, prefer retaining useful examples with explicit best-effort
  semantics over rejecting data merely because every annotation cannot be proven.
- Remain strict about credentials and private data, destructive remote operations,
  unreadable or structurally corrupt artefacts, unbounded resource use, and changes
  that would make an experiment irreproducible.

## Testing and quality

The full test suite uses real Hugging Face datasets and short training runs.
Install FFmpeg and authenticate with Hugging Face using `HF_TOKEN`,
`HUGGINGFACE_HUB_TOKEN`, or an existing `hf auth login` session. Tests also
need network access and enough storage for model and dataset caches.

`make check` stages the working tree, runs all pre-commit hooks, and may rewrite
and stage files. If a `llama-server` process is already running, it also runs Slopo
in its explicit Python 3.12 tool environment; index, embed, and analyse failures
are fatal. When no server is running, it reports the normal conditional skip.
Ruff's configured hooks use `--fix` and `--unsafe-fixes`, so inspect `git diff` after
the run. `make test` runs pytest
and then `readme-cov`; the latter can rewrite `README.md`. Run a focused test with,
for example, `uv run pytest tests/test_package.py`.

Ruff uses 88-character lines, double quotes, import sorting, type annotations,
and Google-style docstrings. Keep Python changes compatible with Python 3.11.
Use Conventional Commit subjects such as `feat:`, `fix:`, or `docs:`.

## Configuration, outputs, and gotchas

- Preserve the `hviske` namespace; do not reintroduce `coral` imports or paths.
- Multi-GPU fine-tuning must use `accelerate launch`.
- `make` creates and includes a root `.env`; never inspect, paste, or commit it.
- Evaluation calls `evaluate()` first, then attempts cache cleanup, then writes a
  CSV only when `store_results=true`. Its current singular `model--...` glob
  ordinarily misses Hugging Face's `models--...` cache directories.
- N-gram training may run `sudo apt-get` when `apt-get` is present. It downloads
  and compiles KenLM under `cache_dir`, or `~/.cache` when that is unset.
- Training and evaluation can create caches, Hydra outputs, result files, and
  tracking directories. Do not commit generated artifacts. Slopo's database and
  reports under `.slopo/` are also generated and ignored.
- CI runs only for non-draft pull requests targeting `main`, uses Python 3.11 on
  Ubuntu, and checks out `main` rather than the pull request ref.
