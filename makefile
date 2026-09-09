# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: help install install-pre-commit install-dependencies install-quality-tools check test tree

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Force the installation to use position-independent code, which helps with the
# installation of the `samplerate` package, as it relies on an underlying C library.
export CFLAGS := -fPIC $(CFLAGS)
export CXXFLAGS := -fPIC $(CXXFLAGS)

# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

# Set the shell to bash, enabling the use of `source` statements
SHELL := /bin/bash

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing the 'Hviske' project..."
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet install-quality-tools
	@$(MAKE) --quiet setup-environment-variables
	@$(MAKE) --quiet install-pre-commit
	@echo "Installed the 'Hviske' project! You can now activate your virtual environment with 'source .venv/bin/activate'."
	@echo "Note that this is a 'uv' project. Use 'uv add <package>' to install new dependencies and 'uv remove <package>' to remove them."

install-uv:
	@if [ "$(shell which uv)" = "" ]; then \
		if [ "$(shell which rustup)" = "" ]; then \
			curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
			echo "Installed Rust."; \
		fi; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "Installed uv."; \
    else \
		echo "Updating uv..."; \
		uv self update || true; \
	fi

install-pre-commit:
	@uv run pre-commit install
	@uv run pre-commit autoupdate

install-dependencies:
	@uv python install 3.11
	@uv sync --python 3.11 --all-extras

install-quality-tools:
	@uv python install 3.12
	@uv tool install --python 3.12 --force git+https://github.com/saattrupdan/funcsort@v0.1.3
	@uv tool install --python 3.12 --force 'slopo>=0.5.0'

setup-environment-variables:
	@uv run python src/scripts/fix_dot_env_file.py

setup-environment-variables-non-interactive:
	@uv run python src/scripts/fix_dot_env_file.py --non-interactive

test:  ## Run tests
	@uv run pytest && uv run readme-cov

tree:  ## Print directory tree
	@tree -a --gitignore -I .git .

check:  ## Lint, format, and type-check the code
	@git add . && uv run pre-commit run --all-files
	@if command -v llama-server >/dev/null 2>&1 && pgrep -x llama-server >/dev/null; then \
		echo "Running Slopo code duplication detection..."; \
		export LITELLM_DROP_PARAMS=true; \
		export OPENAI_API_KEY=$${OPENAI_API_KEY:-sk-no-key}; \
		if ! INDEX_OUTPUT=$$(uv tool run --python 3.12 --from 'slopo>=0.5.0' slopo index 2>&1); then \
			echo "❌ Slopo index failed (Python 3.12 tool environment):"; echo "$$INDEX_OUTPUT"; exit 1; \
		fi; \
		echo "$$INDEX_OUTPUT" | grep -E "Indexed|unchanged|removed" || true; \
		if ! EMBED_OUTPUT=$$(uv tool run --python 3.12 --from 'slopo>=0.5.0' slopo embed 2>&1); then \
			echo "❌ Slopo embed failed (Python 3.12 tool environment):"; echo "$$EMBED_OUTPUT"; exit 1; \
		fi; \
		if ! ANALYSIS_OUTPUT=$$(uv tool run --python 3.12 --from 'slopo>=0.5.0' slopo analyze 2>&1); then \
			echo "❌ Slopo analyse failed (Python 3.12 tool environment):"; echo "$$ANALYSIS_OUTPUT"; exit 1; \
		fi; \
		echo "$$ANALYSIS_OUTPUT" | grep -E "Exact copies|Similarity ratio" || true; \
		DUPLICATE_COUNT=$$(echo "$$ANALYSIS_OUTPUT" | grep "Similarity ratio (including exact copies)" | sed -E 's/.*\(([0-9]+)\/.*/\1/'); \
		if [ "$$DUPLICATE_COUNT" -gt 0 ] 2>/dev/null; then \
			RATIO=$$(echo "$$ANALYSIS_OUTPUT" | grep "Similarity ratio (including exact copies)" | sed -E 's/.*: ([0-9.]+%).*/\1/'); \
			echo ""; \
			echo "❌ Slopo failed: duplicate code detected"; \
			printf "   Duplicate units: %s\\n" "$$DUPLICATE_COUNT"; \
			printf "   Similarity ratio: %s\\n" "$$RATIO"; \
			echo "   Full report: .slopo/report/index.md"; \
			echo "   To ignore reviewed duplicates, add their cluster hashes to .slopo/slopo.ignore.txt"; \
			echo ""; \
			exit 1; \
		fi; \
	else \
		echo "Slopo skipped (llama.cpp server not running)"; \
	fi

roest-315m-100k:  ## Train the Røst-315M model
	@OMP_NUM_THREADS=1 \
		uv run accelerate launch \
		--use-deepspeed \
		--zero-stage 2 \
		src/scripts/finetune_asr_model.py \
		model=wav2vec2-small \
		push_to_hub=true \
		model_id=roest-wav2vec2-315m-100k-steps-v2 \
		private=true \
		per_device_batch_size=64 \
		max_steps=100000 \
		datasets.coral_read_aloud.id=/work/asr-data/CoRal-project--coral-v3 \
		datasets.coral_conversation.id=/work/asr-data/CoRal-project--coral-v3

roest-315m-1m:  ## Train the Røst-315M model
	@OMP_NUM_THREADS=1 \
		uv run accelerate launch \
		--use-deepspeed \
		--zero-stage 2 \
		src/scripts/finetune_asr_model.py \
		model=wav2vec2-small \
		push_to_hub=true \
		model_id=roest-wav2vec2-315m-1m-steps \
		private=true \
		per_device_batch_size=64 \
		max_steps=1000000 \
		datasets.coral_read_aloud.id=/work/asr-data/CoRal-project--coral-v3 \
		datasets.coral_conversation.id=/work/asr-data/CoRal-project--coral-v3

roest-1.5b-30k:  ## Train the Røst-1.5B model
	@OMP_NUM_THREADS=1 \
		uv run accelerate launch \
		--use-deepspeed \
		--zero-stage 2 \
		src/scripts/finetune_asr_model.py \
		model=whisper-large \
		push_to_hub=true \
		model_id=roest-whisper-1.5b-30k-steps \
		private=true \
		per_device_batch_size=64 \
		max_steps=30000 \
		datasets.coral_read_aloud.id=/work/asr-data/CoRal-project--coral-v3 \
		datasets.coral_conversation.id=/work/asr-data/CoRal-project--coral-v3

roest-1.5b-100k:  ## Train the Røst-1.5B model
	@OMP_NUM_THREADS=1 \
		uv run accelerate launch \
		--use-deepspeed \
		--zero-stage 2 \
		src/scripts/finetune_asr_model.py \
		model=whisper-large \
		push_to_hub=true \
		model_id=roest-whisper-1.5b-100k-steps \
		private=true \
		per_device_batch_size=64 \
		max_steps=100000 \
		datasets.coral_read_aloud.id=/work/asr-data/CoRal-project--coral-v3 \
		datasets.coral_conversation.id=/work/asr-data/CoRal-project--coral-v3
