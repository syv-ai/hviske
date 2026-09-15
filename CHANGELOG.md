# Changelog

## [Unreleased]

### Added

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

### Fixed

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
