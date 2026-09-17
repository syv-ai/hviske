# Changelog

## [Unreleased]

### Added

- Added resumable, shard-bounded local materialisation for Sparkie's five positional
  overlays, with durable compressed audio, strict join parity, atomic receipts,
  checksummed provenance manifests, free-disk reserves, and shared preflight validation.
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

- Made the local positional-overlay artefact mandatory for Sparkie and enabled three
  spawned DataLoader workers while retaining serial dataset preprocessing and generic
  remote keyed/non-materialised overlay support; four workers exceeded the production
  thermal limit during a sustained GPU pilot.
- Tuned the production Sparkie shuffle buffers after the wf2t12vq smoke: a one-row
  global buffer for already-sharded Hub sources, 128 rows for local DRTV and YouTube,
  and 16 rows for the five positional unified sources. The old global 128-row graph took
  127 minutes, read 90 GB, and reached 19 GB worker RSS before its first batch; the
  200,000-step training horizon remains unchanged.
- Switched production Sparkie P1 training to direct `syvai/p1-segments` loading with a
  required immutable `P1_SEGMENTS_REVISION`; generic transcript joins remain supported.
- Removed unsupported Parakeet TDT fine-tuning and the NeMo-only Danish RNNT preset.

### Removed

- Removed the inactive calibration launcher, anchor configurations, model preset, tests,
  and runbook instructions; retained the design plan for future reference.

### Fixed

- Added bounded, jittered retries for transient Hugging Face Hub streaming reads,
  including process-local client recreation for closed-client failures; preserved
  upstream 416-as-EOF handling and policy reconfiguration; local and materialised
  file reads remain unchanged.
- Partitioned local VTT manifests into configurable metadata-only shards so Sparkie's
  spawned DataLoader workers can consume DRTV and YouTube examples in parallel.
- Replaced Trainer padding sentinels in generated token IDs before ASR metric decoding
  without mutating aggregated evaluation predictions.
- Matched floating-point Cohere generation features to the model inference dtype while
  preserving decoder IDs, attention masks, labels, and training inputs, including when
  evaluation uses standard distributed model wrappers.
- Removed the zero-yield Danish VoxPopuli stream from the active Sparkie mix and
  reassigned its 0.049091 probability to FTSpeech, preserving the 15-source 60/40
  Danish/English mixture; the reusable dataset configuration and unified provenance
  remain available for other sources.
- Restricted mirrored base and overlay shard validation to effective positional joins,
  preserving independent keyed-overlay shard selection in training and preflight.
- Hardened finetuning-data preflight transport logging before Hub access, preventing
  signed URLs and credentials from entering its logs.
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
