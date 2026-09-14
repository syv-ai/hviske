# Changelog

## [Unreleased]

### Added

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

- Restored the codebase MIT license and removed the P1 dataset license files.
- Preserved repeated reference tokens in Parakeet RNNT metrics.
