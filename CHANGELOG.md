# Changelog

## [Unreleased]

### Added

- Added reproducible PyTorch 2.10 CUDA resolution for Sparkie, with a fail-fast check
  that prevents the production preset from starting without a usable CUDA device.
- Added robust online W&B tracking for the production Sparkie v6.0 workflow, including
  credential-safe preflight, resumable run IDs, and local-only model checkpoints.
- Added reusable keyed and strict positional ASR dataset overlays with bounded SQLite
  indexing, shared preflight validation, and safe overlay provenance; configured the
  v6.0 Danish unified sources to use an environment-supplied immutable overlay revision.
- Added Transformers-native NVIDIA Parakeet CTC and RNNT fine-tuning.
- Added Danish vocabulary adaptation for native Parakeet tokenizers, preserving
  existing vocabulary and blank-token IDs while resizing CTC and RNNT heads.
- Added Parakeet CTC and RNNT model presets.

### Changed

- Switched production Sparkie P1 training to direct `syvai/p1-segments` loading with a
  required immutable `P1_SEGMENTS_REVISION`; generic transcript joins remain supported.
- Removed unsupported Parakeet TDT fine-tuning and the NeMo-only Danish RNNT preset.

### Removed

- Removed the inactive calibration launcher, anchor configurations, model preset, tests,
  and runbook instructions; retained the design plan for future reference.

### Fixed

- Bounded Sparkie startup by selecting only each Danish source's pinned base and
  positional-overlay shards, validating mirrored shard order, and shuffling joined
  metadata before audio decoding.
- Restored background-noise augmentation with torchaudio 2.10 by using a project-owned
  soundfile decoder while retaining torch-audiomentations resampling semantics.
- Recognised refreshed Danish overlay `review` actions while continuing to exclude
  them from training alongside the other disallowed actions.
- Removed package-import logging configuration so Hydra's policy controls both main
  and spawned DataLoader worker logs, keeping temporary signed dataset URLs out of
  training logs; CLI fallback logging no longer runs during spawned worker imports.
- Made local VTT and standardised streaming training graphs pickleable for PyTorch
  `spawn` workers without changing lazy audio loading or restartability.
- Normalised Sparkie `dataset_num_workers=1` to in-process filtering and mapping, and
  stopped requesting multiprocessing when materialising iterable validation datasets;
  the production positional-overlay graph now uses one training audio loader worker
  because multi-shard bases are incompatible with worker-local positional offsets.
  Generic keyed and non-positional graphs remain unchanged.
- Used the configured Hugging Face datasets cache when materialising validation
  streaming datasets without an explicit cache directory.
- Corrected the production NST source discriminator to `nst_da` across the base and
  overlay filters, restoring accepted NST rows.
- Restricted shared positional-overlay preflight to string source discriminators with
  unambiguous equality checks and non-colliding output columns.
- Projected overlay metadata before applying row filters, avoiding large unused model
  columns and audio requests during Sparkie v6 preflight while validating the original
  audio schema.
- Resolved features for untyped streaming datasets before filtering and overlaying,
  and made repeated streaming filters safe without materialising or decoding audio.
- Made W&B preflight non-interactive, scrubbed inherited Sparkie W&B identity, and
  redacted checkpoint paths from online configuration payloads.
- Separated bounded pilot stopping from the 100,000-step scheduler horizon, added
  resumable tracking finalisation with success and failure statuses, and hardened W&B
  preflight, IDs, environment policy, and configuration redaction.
- Restored the codebase MIT license and removed the P1 dataset license files.
- Preserved repeated reference tokens in Parakeet RNNT metrics.
